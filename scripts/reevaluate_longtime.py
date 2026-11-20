"""Re-run ONLY the long-time evaluation of finished runs, one rollout at a time.

Why: `NeuralODE.rollout_nograd` checks finiteness of the whole batch at the end
of every 2000-step chunk and, if any trajectory has become non-finite, pads
EVERY trajectory of the batch with NaN from that chunk boundary on. The
long-time evaluator rolls its three trajectories out as one batch, so on a
system where rollouts leave the attractor (KS) a healthy trajectory is
recorded as having blown up at the chunk boundary after a companion's
failure, and its stable prefix is truncated. This script repeats
src.train.evaluate.long_time_stats exactly (same starts, same thresholds,
same statistics) except that each trajectory is rolled out on its own, so
every trajectory's blow-up time and stable prefix are its own, and it
records per-rollout values and the number of rollouts behind every statistic.

Training-side fields and the short-time valid-time fields of summary.json
are left untouched (valid time is unaffected: every recorded first crossing
precedes the first chunk boundary). The run's manifest, config and
checkpoints are not modified, so check_grid.py still verifies the run. The
summary records which evaluator produced the long-time fields and keeps the
previous batched values for comparison.

    python scripts/reevaluate_longtime.py runs_v12r1/ks_l22 datasets_v12r1
    python scripts/reevaluate_longtime.py runs_v12r1/ks_l22/ss_W0.5/seed0 datasets_v12r1
"""
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch
from src.train.config import RunConfig
from src.train.data import Dataset
from src.train.trainer import build_model, pick_device
from src.train.evaluate import _benettin_model, _mmd_rbf, _stride, _welch_psd
from src.train.provenance import atomic_json

SCRIPT_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@torch.no_grad()
def long_time_stats_independent(model, ds, n_long=3, lyap_T=100.0, psd_nperseg=1024,
                                mmd_n=2000, device=None, save_rollouts=None, min_stable_T=20.0):
    """src.train.evaluate.long_time_stats with each rollout integrated separately."""
    device = device or next(model.parameters()).device
    model.eval()
    k = _stride(model, ds)
    ref = ds.ref_long[::k]
    n_steps = len(ref) - 1
    steps_per_T = ds.T_lambda / model.dt
    rng = np.random.default_rng(123)                                   # identical starts to the batched evaluator
    starts = rng.choice(len(ds.val) - 1, size=n_long, replace=False)
    x0 = torch.as_tensor(ds.val[starts], dtype=torch.float32, device=device)
    roll = np.stack([model.rollout_nograd(x0[i:i + 1], n_steps, keep=True).cpu().numpy()[0]
                     for i in range(n_long)])                           # (n_long, T+1, dim), one at a time
    if save_rollouts is not None:
        np.savez_compressed(save_rollouts, rollouts=roll, starts=starts)

    lim = 5.0 * np.abs(ref).max()
    T_total = n_steps / steps_per_T
    blow_idx, blow_T = [], []
    for r in roll:
        bad = ~np.isfinite(r).all(axis=1) | (np.abs(r) > lim).any(axis=1)
        i = int(np.argmax(bad)) if bad.any() else n_steps + 1
        blow_idx.append(i)
        blow_T.append(min(i / steps_per_T, T_total))
    blow_T = np.array(blow_T)
    survived = np.array([i > n_steps for i in blow_idx])
    out = {
        "blowup_frac": float(1 - survived.mean()),
        "blowup_time_mean": float(blow_T.mean()),
        "blowup_time_min": float(blow_T.min()),
        "blowup_time_per_rollout": blow_T.round(3).tolist(),
        "T_long": float(T_total),
    }
    min_steps = int(min_stable_T * steps_per_T)
    coords = np.arange(min(8, ds.dim)) if ds.dim <= 8 else np.linspace(0, ds.dim - 1, 8).astype(int)
    mmd, psd, lam, used = [], [], [], []
    for j, (r, i, x) in enumerate(zip(roll, blow_idx, x0)):
        end = n_steps + 1 if i > n_steps else max(0, i - int(2 * steps_per_T))   # 2 T_lambda margin before the blow-up
        if end < min_steps:
            continue
        used.append(j)
        seg = r[:end]
        mmd.append(np.mean([_mmd_rbf(seg[:, c], ref[:, c], mmd_n, rng) for c in coords]))
        p = _welch_psd(seg, min(psd_nperseg, len(seg) // 4))
        rp = _welch_psd(ref, min(psd_nperseg, len(seg) // 4))
        psd.append(float(np.linalg.norm(p - rp) / np.linalg.norm(rp)))
        n_ly = int(min(lyap_T * steps_per_T, 0.5 * end))
        l = _benettin_model(model, x, n_ly, model.dt, device=device)
        if np.isfinite(l):
            lam.append(l)
    out["rollouts_used_for_stats"] = used
    out["n_rollouts_used_for_stats"] = len(used)
    out["n_rollouts_lambda1"] = len(lam)
    out["marginal_mmd_mean"] = float(np.mean(mmd)) if mmd else float("nan")
    out["marginal_mmd_std"] = float(np.std(mmd)) if mmd else float("nan")
    out["psd_rel_l2_mean"] = float(np.mean(psd)) if psd else float("nan")
    out["psd_rel_l2_std"] = float(np.std(psd)) if psd else float("nan")
    out["lambda1_model_mean"] = float(np.mean(lam)) if lam else float("nan")
    out["lambda1_model_std"] = float(np.std(lam)) if lam else float("nan")
    out["lambda1_true"] = ds.lambda1_true
    out["lambda1_ref"] = ds.lambda1_ref
    out["lambda1_abs_err_mean"] = float(np.mean([abs(l - ds.lambda1_true) for l in lam])) if lam else float("nan")
    return out


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "runs")
    dataset_dir = Path(sys.argv[2] if len(sys.argv) > 2 else "datasets")
    run_dirs = [root] if (root / "summary.json").exists() else sorted(p.parent for p in root.glob("**/summary.json"))
    device = pick_device("auto")
    ds_cache = {}
    for rd in run_dirs:
        cfg = RunConfig.load(rd / "config.yaml")
        ds = ds_cache.setdefault(cfg.data.system, Dataset.load(dataset_dir / f"{cfg.data.system}.npz"))
        ck = rd / "checkpoint_best.pt" if (rd / "checkpoint_best.pt").exists() else rd / "checkpoint_final.pt"
        model = build_model(cfg, ds, device)
        model.load_state_dict(torch.load(ck, map_location=device))
        s = json.loads((rd / "summary.json").read_text())
        if not s.get("evaluation_complete"):
            print(f"{rd}: evaluation not complete; skipped", flush=True)
            continue
        if s.get("longtime_evaluator", {}).get("independent_rollouts"):
            print(f"{rd}: already re-evaluated with independent rollouts; skipped", flush=True)
            continue
        t0 = time.time()
        old = {k: s.get(k) for k in ("blowup_frac", "blowup_time_mean", "blowup_time_min",
                                     "n_rollouts_used_for_stats", "lambda1_abs_err_mean",
                                     "marginal_mmd_mean", "psd_rel_l2_mean")}
        s.update(long_time_stats_independent(model, ds, n_long=cfg.eval.n_long_rollouts, lyap_T=cfg.eval.lyap_T,
                                             psd_nperseg=cfg.eval.psd_nperseg, mmd_n=cfg.eval.mmd_n,
                                             save_rollouts=rd / "rollouts_longtime.npz"))
        s["longtime_evaluator"] = {"script": "scripts/reevaluate_longtime.py", "sha256": SCRIPT_SHA256,
                                   "independent_rollouts": True, "checkpoint": ck.name,
                                   "previous_batched_values": old}
        s["wallclock_longtime_reeval_s"] = time.time() - t0
        atomic_json(rd / "summary.json", s)
        print(f"{rd}: blowup per rollout {s['blowup_time_per_rollout']} T_lambda (batched mean was "
              f"{old['blowup_time_mean']:.1f}), prefixes used {s['n_rollouts_used_for_stats']}/{cfg.eval.n_long_rollouts}, "
              f"lambda1 err {s['lambda1_abs_err_mean']:.4f} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
