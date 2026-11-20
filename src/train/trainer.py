"""Training loop with the logging TRAINING_GUIDE.md section 4 requires.

Per gradient step (every `log_every`), one JSON line in log.jsonl:
    step, beta_k, W_mean, W_min_batch, W_max_batch, Lambda_max_batch,
    grad_norm_preclip, loss_train, lr
Every `val_every` steps: loss_val_1step, loss_val_halfT (rollout of
0.5 T_lambda on the validation trajectory).
Every `segment_log_every` steps: segments_<step>.npz with per-segment
    Lambda_i, grad_norm_i, W_i, mu_i, loss_i
computed by one backward pass per segment on a subsample, so that
log(grad_norm_i) can be plotted against the nominal teacher proxy Lambda_i
(exploratory association, not a verification of Lemma 1).

Loss normalization (sat schedule; see paper Section 6.5):
    per_segment : mean_i mse_i
    inv_W       : mean_i (W_bar / W_i) mse_i
    global      : sum_i N_i mse_i / sum_i N_i
MP-NODE: nominal window split into K chunks, each re-initialized from
data; loss = mean chunk mse + mp_penalty * mean squared end-of-chunk error
(the discontinuity between chunk k's end and chunk k+1's data start).
"""
from __future__ import annotations
import json
import math
import time
from pathlib import Path
import numpy as np
import torch

from src.models import NeuralODE, build_field
from src.train.config import RunConfig
from src.train.data import Dataset
from src.train.schedules_train import build_policy, run_name


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(cfg: RunConfig, ds: Dataset, device):
    m = cfg.model
    kw = dict(activation=m.activation)
    if m.kind == "mlp":
        kw.update(width=m.width, depth=m.depth)
    else:
        kw.update(channels=m.width, depth=m.depth, kernel=m.kernel)
    field = build_field(m.kind, ds.dim, ds.x_mean, ds.x_std, **kw)
    return NeuralODE(field, ds.dt * m.dt_mult).to(device)


def atomic_torch_save(value, path):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(value, tmp)
    tmp.replace(path)


class Trainer:
    def __init__(self, cfg: RunConfig, ds: Dataset, run_dir: Path, provenance=None):
        self.provenance = provenance
        self.last_record = None
        self.finished_summary = None
        self.completion_settings = {k: getattr(cfg.train, k) for k in
                                    ("n_steps", "min_steps", "early_stop_patience",
                                     "early_stop_metric", "early_stop_min_delta")}
        self.cfg = cfg
        self.ds = ds
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.device = pick_device(cfg.train.device)
        torch.manual_seed(cfg.train.seed)
        np.random.seed(cfg.train.seed)
        self.rng = np.random.default_rng(cfg.train.seed)

        self.model = build_model(cfg, ds, self.device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.train.lr,
                                     weight_decay=cfg.train.weight_decay)
        if cfg.train.lr_schedule == "cosine":
            self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.opt, T_max=cfg.train.n_steps, eta_min=cfg.train.lr_min)
        elif cfg.train.lr_schedule == "plateau":
            self.sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.opt, mode="min", factor=cfg.train.plateau_factor,
                patience=cfg.train.plateau_patience, threshold=cfg.train.early_stop_min_delta,
                threshold_mode="rel", min_lr=cfg.train.lr_min)
        else:
            raise ValueError(f"unknown lr_schedule {cfg.train.lr_schedule!r}")
        self.policy = build_policy(cfg.schedule, ds.T_lambda, ds.dt * cfg.model.dt_mult, cfg.train.n_steps)

        self.train_t = torch.as_tensor(ds.train, dtype=torch.float32, device=self.device)
        self.val_t = torch.as_tensor(ds.val, dtype=torch.float32, device=self.device)
        self.std_t = torch.as_tensor(ds.x_std, dtype=torch.float32, device=self.device)

        # Teacher rate with a small positive floor so that beta / mu is
        # finite and positive on contracting segments; those segments then
        # receive W_max through the explicit clip in the policy. With the
        # default quantile 0 the floor is the smallest positive FTLE in the
        # table (at least 1e-3), so W_max, not the floor, governs.
        f = ds.ftle_train
        pos = f[f > 0]
        self.mu_floor = float(max(np.quantile(pos, cfg.schedule.mu_floor_quantile), 1e-3))
        self.mu_train = np.maximum(f, self.mu_floor)
        self.k = int(cfg.model.dt_mult)          # data points per model step
        self.dtm = ds.dt * self.k                  # model step
        self.N_max = int(round(cfg.schedule.W_max * ds.T_lambda / self.dtm)) + 1
        self.n_nan = 0
        self.exceed = {"n": 0, "n_gt_1p1": 0, "n_gt_1p0": 0, "max_ratio": 0.0,
                       "n_warm": 0, "n_warm_gt_1p1": 0, "n_warm_gt_1p0": 0,
                       "seg": 0, "seg_wmin_bound": 0}
        self.train_time_prev = 0.0        # wallclock of earlier invocations (resume)
        self.integrated_steps = 0         # sum over steps of sum_i N_i (model steps unrolled)
        self.start_step = 0
        self.best_val = float("inf")
        self.best_step = -1
        self.bad_vals = 0
        self.peak_gn = 0.0
        ck = self.run_dir / "checkpoint_last.pt"
        if cfg.train.resume and ck.exists():
            self._load_full(ck)
            print(f"[resume] from step {self.start_step} (best {self.best_val:.3e} at {self.best_step})", flush=True)
        self.log_f = open(self.run_dir / "log.jsonl", "a")
        dev = str(self.device)
        if self.device.type == "cuda":
            dev += f" ({torch.cuda.get_device_name(self.device)})"
        n_par = sum(p.numel() for p in self.model.parameters())
        print(f"[setup] device {dev}; {n_par} params; {cfg.model.kind} field; "
              f"schedule {run_name(cfg.schedule)}; batch {cfg.train.batch_size}; "
              f"model dt {self.dtm:g} (= {self.k} x data dt); W_max {self.N_max - 1} model steps; validating every {cfg.train.val_every} steps; "
              f"lr_schedule {cfg.train.lr_schedule}", flush=True)
        if self.device.type != "cuda":
            print("[setup] WARNING: running on CPU. KS/L96 runs will be very slow; "
                  "request a GPU node.", flush=True)

    # ------------------------------------------------------------ checkpoints
    def _save_full(self, path, step):
        atomic_torch_save({
            "provenance": self.provenance, "last_record": self.last_record,
            "finished_summary": self.finished_summary,
            "completion_settings": self.completion_settings,
            "model": self.model.state_dict(), "opt": self.opt.state_dict(),
            "sched": self.sched.state_dict(), "lr_schedule": self.cfg.train.lr_schedule, "step": step,
            "best_val": self.best_val, "best_step": self.best_step,
            "bad_vals": self.bad_vals, "peak_gn": self.peak_gn, "n_nan": self.n_nan, "exceed": self.exceed,
            "train_time_prev": self.train_time_prev + (time.time() - self._t_fit) if hasattr(self, "_t_fit") else self.train_time_prev,
            "integrated_steps": self.integrated_steps,
            "rng": self.rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
        }, path)

    def _load_full(self, path):
        st = torch.load(path, map_location=self.device, weights_only=False)
        if self.provenance is not None and st.get("provenance") != self.provenance:
            raise ValueError("checkpoint provenance mismatch; use a fresh output directory")
        self.last_record = st.get("last_record")
        if st.get("completion_settings") == self.completion_settings:
            self.finished_summary = st.get("finished_summary")
        self.model.load_state_dict(st["model"])
        self.opt.load_state_dict(st["opt"])
        if st.get("lr_schedule", "cosine") == self.cfg.train.lr_schedule:
            self.sched.load_state_dict(st["sched"])
        else:
            print(f"[resume] checkpoint used lr_schedule={st.get('lr_schedule','cosine')}, "
                  f"config says {self.cfg.train.lr_schedule}; scheduler state not restored", flush=True)
        self.start_step = int(st["step"]) + 1
        self.best_val = st["best_val"]; self.best_step = st["best_step"]
        self.bad_vals = st["bad_vals"]; self.peak_gn = st["peak_gn"]; self.n_nan = st["n_nan"]
        self.exceed = st.get("exceed", self.exceed)
        self.train_time_prev = float(st.get("train_time_prev", 0.0))
        self.integrated_steps = int(st.get("integrated_steps", 0))
        self.rng.bit_generator.state = st["rng"]
        torch.set_rng_state(st["torch_rng"].cpu())

    # ------------------------------------------------------------ helpers
    def _sample_starts(self, B):
        hi = len(self.ds.train) - self.k * self.N_max - 1
        return self.rng.integers(0, hi, size=B)

    def _segment_mse(self, idx, N):
        """Per-segment normalized MSE for segments starting at idx, N steps."""
        idx_t = torch.as_tensor(idx, device=self.device)
        ar = torch.arange(N + 1, device=self.device) * self.k
        target = self.train_t[idx_t[:, None] + ar[None, :]]          # (B, N+1, dim)
        pred = self.model.rollout(target[:, 0], N)
        err = (pred[:, 1:] - target[:, 1:]) / self.std_t
        return (err ** 2).mean(dim=(1, 2))                            # (B,)

    def _grouped_losses(self, idx, N_i):
        """Returns per-segment mse (tensor, batch order) using one rollout per distinct N."""
        out = torch.empty(len(idx), device=self.device)
        for N in np.unique(N_i):
            m = np.where(N_i == N)[0]
            out[torch.as_tensor(m, device=self.device)] = self._segment_mse(idx[m], int(N))
        return out

    def _mpnode_loss(self, idx, N):
        K = self.policy.K
        n = max(1, N // K)
        idx_t = torch.as_tensor(idx, device=self.device)
        ar = torch.arange(n + 1, device=self.device) * self.k
        mses, ends = [], []
        for k in range(K):
            start = idx_t + k * n * self.k
            target = self.train_t[start[:, None] + ar[None, :]]
            pred = self.model.rollout(target[:, 0], n)
            err = (pred[:, 1:] - target[:, 1:]) / self.std_t
            mses.append((err ** 2).mean(dim=(1, 2)))
            ends.append((err[:, -1] ** 2).mean(dim=1))
        mse = torch.stack(mses, 1).mean(1)
        end = torch.stack(ends, 1).mean(1)
        return mse + self.cfg.schedule.mp_penalty * end, mse

    def _combine(self, mse_i, W_i, N_i):
        ln = self.cfg.schedule.loss_norm
        if self.policy.name != "sat" or ln == "per_segment":
            return mse_i.mean()
        if ln == "inv_W":
            w = torch.as_tensor(W_i.mean() / W_i, dtype=torch.float32, device=self.device)
            return (w * mse_i).mean()
        if ln == "global":
            w = torch.as_tensor(N_i / N_i.sum(), dtype=torch.float32, device=self.device)
            return (w * mse_i).sum()
        raise ValueError(ln)

    def _grad_norm(self):
        tot = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                tot += float(p.grad.detach().pow(2).sum())
        return math.sqrt(tot)

    # ------------------------------------------------------------ one step
    def step(self, step):
        cfg = self.cfg
        B = cfg.train.batch_size
        idx = self._sample_starts(B)
        mu = self.mu_train[idx]
        W_i, info = self.policy(step, mu)
        # Round DOWN to whole model steps so that mu_i * W_i <= beta_k holds exactly
        # (nearest rounding could exceed the budget by up to half a step). The only
        # remaining way to exceed the budget is the minimum-horizon floor W_min,
        # which binds when beta_k / mu_i < W_min; that case is counted below.
        n_min = int(cfg.schedule.W_min_steps)
        N_floor = np.floor(W_i / self.dtm + 1e-9).astype(int)
        N_i = np.maximum(n_min, N_floor)
        bk = info.get("beta_k", float("nan"))
        # Policies already clip to W_min, so test feasibility BEFORE that clip.
        wmin_bound = mu * (n_min * self.dtm) > bk + 1e-10
        W_i = N_i * self.dtm
        if self.policy.K > 1:
            # multi-window: the budget of a chunk is its own teacher rate at the
            # chunk start times the chunk length, not the segment budget / K.
            n_chunk = max(1, int(N_i[0]) // self.policy.K)
            starts = idx[:, None] + np.arange(self.policy.K)[None, :] * n_chunk * self.k
            Lambda_i = (self.mu_train[starts] * (n_chunk * self.dtm)).max(axis=1)
        else:
            Lambda_i = mu * W_i

        self.model.train()
        self.opt.zero_grad(set_to_none=True)
        if self.policy.name == "mpnode":
            loss_i, mse_i = self._mpnode_loss(idx, int(N_i[0]))
            loss = loss_i.mean()
        else:
            mse_i = self._grouped_losses(idx, N_i)
            loss = self._combine(mse_i, W_i, N_i)

        # Count actual training-forward model steps, including failed forwards;
        # exclude validation and per-segment diagnostic rollouts.
        self.integrated_steps += (B * self.policy.K * max(1, int(N_i[0]) // self.policy.K)
                                  if self.policy.name == "mpnode" else int(N_i.sum()))
        if not torch.isfinite(loss):
            self.n_nan += 1
            if self.n_nan > cfg.train.max_nan_steps:
                raise RuntimeError("too many non-finite losses; training diverged")
            return None

        loss.backward()
        gn = self._grad_norm()
        if cfg.train.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.train.grad_clip)
        self.opt.step()
        if cfg.train.lr_schedule == "cosine":
            self.sched.step()

        bk = info.get("beta_k", float("nan"))
        if np.isfinite(bk) and bk > 0:
            ratio = float(Lambda_i.max() / bk)
            k0 = max(1.0, cfg.schedule.warmup_frac * cfg.train.n_steps)
            self.exceed["seg"] += len(wmin_bound); self.exceed["seg_wmin_bound"] += int(wmin_bound.sum())
            if step < 3 * k0:
                self.exceed["n_warm"] += 1; self.exceed["n_warm_gt_1p1"] += int(ratio > 1.1)
                self.exceed["n_warm_gt_1p0"] += int(ratio > 1.0 + 1e-6)
            else:
                self.exceed["n"] += 1; self.exceed["n_gt_1p1"] += int(ratio > 1.1)
                self.exceed["n_gt_1p0"] += int(ratio > 1.0 + 1e-6)
                self.exceed["max_ratio"] = max(self.exceed["max_ratio"], ratio)
        rec = {
            "step": step,
            "t": time.time(),
            "beta_k": info.get("beta_k", float("nan")),
            "W_mean": float(W_i.mean() / self.ds.T_lambda),
            "W_min_batch": float(W_i.min() / self.ds.T_lambda),
            "W_max_batch": float(W_i.max() / self.ds.T_lambda),
            "Lambda_max_batch": float(Lambda_i.max()),
            "grad_norm_preclip": gn,
            "loss_train": float(loss.detach()),
            "mse_mean": float(mse_i.detach().mean()),
            "lr": self.opt.param_groups[0]["lr"],
        }
        return rec, (idx, mu, W_i, N_i, Lambda_i)

    # ------------------------------------------------------------ validation
    @torch.no_grad()
    def validate(self):
        self.model.eval()
        v = self.val_t
        k = self.k
        # one model step (= k data points)
        pred1 = self.model.step(v[:-k])
        l1 = float((((pred1 - v[k:]) / self.std_t) ** 2).mean())
        # half-Lyapunov-time rollouts from 64 evenly spaced starts
        n = int(round(0.5 * self.ds.T_lambda / self.dtm))
        starts = torch.linspace(0, len(v) - k * n - 2, 64, device=self.device).long()
        ar = torch.arange(n + 1, device=self.device) * k
        tgt = v[starts[:, None] + ar[None, :]]
        pred = self.model.rollout_nograd(tgt[:, 0], n)
        lh = float((((pred[:, 1:] - tgt[:, 1:]) / self.std_t) ** 2).mean())
        return {"loss_val_1step": l1, "loss_val_halfT": lh}

    # ------------------------------------------------------------ per-segment grads
    def log_segments(self, step, batch):
        idx, mu, W_i, N_i, Lambda_i = batch
        n = min(self.cfg.train.segment_log_n, len(idx))
        sel = self.rng.choice(len(idx), size=n, replace=False)
        params = [p for p in self.model.parameters() if p.requires_grad]
        gnorm = np.empty(n)
        lossv = np.empty(n)
        self.model.train()
        for j, s in enumerate(sel):
            if self.policy.name == "mpnode":
                li, _ = self._mpnode_loss(idx[s:s + 1], int(N_i[s]))
                li = li[0]
            else:
                li = self._segment_mse(idx[s:s + 1], int(N_i[s]))[0]
            grads = torch.autograd.grad(li, params, allow_unused=True)
            gnorm[j] = math.sqrt(sum(float(g.pow(2).sum()) for g in grads if g is not None))
            lossv[j] = float(li.detach())
        np.savez(self.run_dir / f"segments_{step:07d}.npz",
                 Lambda_i=Lambda_i[sel], grad_norm_i=gnorm,
                 W_i=W_i[sel], mu_i=mu[sel], loss_i=lossv, K=self.policy.K)
        self.opt.zero_grad(set_to_none=True)

    # ------------------------------------------------------------ main loop
    def fit(self):
        cfg = self.cfg
        if self.finished_summary is not None:
            self._load_best()
            self.log_f.close()
            return dict(self.finished_summary)
        t0 = time.time()
        self._t_fit = t0
        last = self.last_record
        stopped_early = False
        step = self.start_step - 1
        for step in range(self.start_step, cfg.train.n_steps):
            out = self.step(step)
            if out is None:
                continue
            rec, batch = out
            self.peak_gn = max(self.peak_gn, rec["grad_norm_preclip"])
            last = self.last_record = rec
            is_last = step == cfg.train.n_steps - 1
            if step % cfg.train.val_every == 0 or is_last:
                rec.update(self.validate())
                v = rec[cfg.train.early_stop_metric]
                if cfg.train.lr_schedule == "plateau" and np.isfinite(v):
                    self.sched.step(v)
                improved = np.isfinite(v) and (not np.isfinite(self.best_val)
                                               or v < self.best_val * (1.0 - cfg.train.early_stop_min_delta))
                if improved:
                    self.best_val, self.best_step, self.bad_vals = v, step, 0
                    atomic_torch_save(self.model.state_dict(), self.run_dir / "checkpoint_best.pt")
                else:
                    self.bad_vals += 1
                rec.update({"best_val": self.best_val, "best_step": self.best_step,
                            "bad_vals": self.bad_vals})
                print(f"[{self.ds.system}/{run_name(cfg.schedule)}/s{cfg.train.seed}] "
                      f"step {step} lr {rec['lr']:.1e} loss {rec['loss_train']:.3e} val1 {rec['loss_val_1step']:.3e} "
                      f"valH {rec['loss_val_halfT']:.3e} (best {self.best_val:.3e}@{self.best_step}, "
                      f"bad {self.bad_vals}) gn {rec['grad_norm_preclip']:.2e} "
                      f"W {rec['W_mean']:.3f}T Lmax {rec['Lambda_max_batch']:.2f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
            if step % cfg.train.log_every == 0 or is_last:
                self.log_f.write(json.dumps(rec) + "\n")
                self.log_f.flush()
            if cfg.train.progress_every and step % cfg.train.progress_every == 0:
                (self.run_dir / "RUNNING.lock").touch()      # heartbeat for other launchers
            if cfg.train.progress_every and step % cfg.train.progress_every == 0 and step % cfg.train.val_every != 0:
                el = time.time() - t0
                rate = (step + 1 - self.start_step) / max(el, 1e-9)
                print(f"    step {step}/{cfg.train.n_steps} loss {rec['loss_train']:.3e} "
                      f"gn {rec['grad_norm_preclip']:.2e} lr {rec['lr']:.1e} "
                      f"{rate:.2f} steps/s, {el:.0f}s elapsed", flush=True)
            if step % cfg.train.segment_log_every == 0:
                self.log_segments(step, batch)
            if cfg.train.ckpt_every and step and step % cfg.train.ckpt_every == 0:
                self._save_full(self.run_dir / "checkpoint_last.pt", step)
            if (self.bad_vals >= cfg.train.early_stop_patience and step >= cfg.train.min_steps):
                stopped_early = True
                print(f"[early stop] no improvement in {cfg.train.early_stop_metric} for "
                      f"{self.bad_vals} validations; stopping at step {step} "
                      f"(best {self.best_val:.3e} at step {self.best_step})", flush=True)
                break
        summary = {
            "wallclock_train_s": self.train_time_prev + (time.time() - t0),   # cumulative over resumes; includes validation and segment diagnostics
            "wallclock_this_invocation_s": time.time() - t0,
            "integrated_model_steps": self.integrated_steps,                    # training-forward steps across all batch elements
            "stopped_step": step,
            "stopped_early": stopped_early,
            "best_step": self.best_step,
            "best_val": self.best_val,
            "final_loss": last["loss_train"] if last else float("nan"),
            "final_mse": last["mse_mean"] if last else float("nan"),
            "final_grad_norm": last["grad_norm_preclip"] if last else float("nan"),
            "peak_grad_norm": self.peak_gn,
            "final_W_mean": last["W_mean"] if last else float("nan"),
            "final_Lambda_max": last["Lambda_max_batch"] if last else float("nan"),
            "n_nan_steps": self.n_nan,
            "mu_floor": self.mu_floor,
            # budget feasibility after step rounding and the W_min floor
            "budget_frac_gt_1p1_after_warmup": (self.exceed["n_gt_1p1"] / self.exceed["n"]) if self.exceed["n"] else float("nan"),
            "budget_frac_gt_1p0_after_warmup": (self.exceed["n_gt_1p0"] / self.exceed["n"]) if self.exceed["n"] else float("nan"),
            "budget_max_ratio_after_warmup": self.exceed["max_ratio"] if self.exceed["n"] else float("nan"),
            "budget_frac_gt_1p1_warmup": (self.exceed["n_warm_gt_1p1"] / self.exceed["n_warm"]) if self.exceed["n_warm"] else float("nan"),
            "budget_frac_gt_1p0_warmup": (self.exceed["n_warm_gt_1p0"] / self.exceed["n_warm"]) if self.exceed["n_warm"] else float("nan"),
            "wmin_bound_segment_frac": (self.exceed["seg_wmin_bound"] / self.exceed["seg"]) if self.exceed["seg"] else float("nan"),
        }
        self.finished_summary = summary
        atomic_torch_save(self.model.state_dict(), self.run_dir / "checkpoint_final.pt")
        self._save_full(self.run_dir / "checkpoint_last.pt", step)
        self.log_f.close()
        self._load_best()
        return summary

    def _load_best(self):
        best = self.run_dir / "checkpoint_best.pt"
        if best.exists():
            self.model.load_state_dict(torch.load(best, map_location=self.device, weights_only=True))
