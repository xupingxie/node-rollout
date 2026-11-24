"""Evaluation: valid time (per solver step), long-time statistics, lambda_1.

valid_time: first time at which NRMSE(t) = ||x_pred - x_true|| / ||std||
exceeds the threshold, evaluated at EVERY solver step over n_test
independent test trajectories. Reported in units of T_lambda. Evaluating
only every k-th step would quantize the reported values.

long_time_stats: n_long rollouts of T_long from attractor ICs;
  - marginal_mmd : RBF-kernel MMD^2 between per-coordinate marginals of
                   the model rollout and the reference trajectory, averaged
                   over (up to 8) coordinates, median-heuristic bandwidth
  - psd_rel_l2   : relative L2 distance of the coordinate-averaged Welch
                   PSD from the reference PSD
  - lambda1_model: Benettin on the learned model using autograd JVPs of the
                   RK4 step; compared with the true lambda_1 measured on
                   the data-generating system
  - blowup_frac  : fraction of rollouts that left the attractor
"""
from __future__ import annotations
import numpy as np
import torch

from src.train.data import Dataset


def _stride(model, ds):
    k = int(round(model.dt / ds.dt))
    assert k >= 1 and abs(k * ds.dt - model.dt) < 1e-9, "model dt must be an integer multiple of data dt"
    return k


def _val_windows(ds, horizon_T=5.0, n_windows=16):
    """Non-overlapping-start windows cut from the validation trajectory, shaped
    like ds.test, for model selection without touching the test set."""
    n_h = int(round(horizon_T * ds.T_lambda / ds.dt))
    n_h = max(2, min(n_h, len(ds.val) - 2))          # a short validation trajectory limits the horizon
    starts = np.linspace(0, len(ds.val) - n_h - 1, n_windows).astype(int)
    return np.stack([ds.val[s:s + n_h + 1] for s in starts])


@torch.no_grad()
def valid_time(model, ds: Dataset, thresholds=(0.3, 0.5), device=None, batch=25, split="test"):
    device = device or next(model.parameters()).device
    k = _stride(model, ds)
    std = torch.as_tensor(ds.x_std, dtype=torch.float32, device=device)
    arr = ds.test if split == "test" else _val_windows(ds)
    test = torch.as_tensor(arr[:, ::k], dtype=torch.float32, device=device)
    n, T1, _ = test.shape
    T = T1 - 1
    nrmse = torch.empty(n, T, device=device)
    model.eval()
    for a in range(0, n, batch):
        tgt = test[a:a + batch]
        pred = model.rollout_nograd(tgt[:, 0], T)
        e = ((pred[:, 1:] - tgt[:, 1:]) / std) ** 2
        nrmse[a:a + batch] = e.mean(dim=2).sqrt()
    nrmse = torch.nan_to_num(nrmse, nan=float("inf"))
    out = {}
    for thr in thresholds:
        exceed = nrmse > thr
        # column j is model step j+1 at time (j+1) dt; valid time = time of the
        # first violation (= T_horizon if it never occurs: right-censored)
        crossed = exceed.any(dim=1)
        first = torch.where(crossed, exceed.float().argmax(dim=1), torch.full((n,), T - 1, device=device))
        t = (first.float() + 1.0) * model.dt / ds.T_lambda
        key = f"{thr:g}".replace(".", "")
        out[f"valid_time_thr{key}_mean"] = float(t.mean())
        out[f"valid_time_thr{key}_std"] = float(t.std())
        out[f"valid_time_thr{key}_median"] = float(t.median())
        out[f"valid_time_thr{key}_censored_frac"] = float(1.0 - crossed.float().mean())
        out[f"valid_time_thr{key}_per_ic"] = t.cpu().numpy().round(4).tolist()
    out["valid_time_resolution"] = model.dt / ds.T_lambda
    out["valid_time_horizon"] = T * model.dt / ds.T_lambda
    return out


def _mmd_rbf(x, y, n=2000, rng=None):
    rng = rng or np.random.default_rng(0)
    x = rng.choice(x, size=min(n, len(x)), replace=False)
    y = rng.choice(y, size=min(n, len(y)), replace=False)
    z = np.concatenate([x, y])
    d = np.abs(z[:, None] - z[None, :])
    h = np.median(d[d > 0]) if np.any(d > 0) else 1.0
    k = np.exp(-(d ** 2) / (2 * h ** 2))
    nx, ny = len(x), len(y)
    kxx = k[:nx, :nx]; kyy = k[nx:, nx:]; kxy = k[:nx, nx:]
    return float((kxx.sum() - np.trace(kxx)) / (nx * (nx - 1))
                 + (kyy.sum() - np.trace(kyy)) / (ny * (ny - 1))
                 - 2 * kxy.mean())


def _welch_psd(x, nperseg):
    """Coordinate-averaged one-sided PSD via averaged Hann-windowed periodograms."""
    T, d = x.shape
    nperseg = int(min(nperseg, T))          # short series: one segment of its full length
    nseg = max(1, T // nperseg)
    w = np.hanning(nperseg)
    acc = np.zeros(nperseg // 2 + 1)
    for s in range(nseg):
        seg = x[s * nperseg:(s + 1) * nperseg]
        seg = seg - seg.mean(axis=0)
        f = np.fft.rfft(seg * w[:, None], axis=0)
        acc += (np.abs(f) ** 2).mean(axis=1)
    return acc / nseg


def _benettin_model(model, x0, n_steps, dt, qr_every=10, device=None):
    """lambda_1 of the learned flow via JVPs of the RK4 step."""
    x = x0.clone().unsqueeze(0)
    v = torch.randn_like(x)
    v = v / v.norm()
    log_sum = 0.0
    for k in range(n_steps):
        _, jv = torch.func.jvp(lambda z: model.step(z), (x,), (v,))
        x = model.step(x).detach()
        v = jv.detach()
        if not torch.isfinite(x).all():
            return float("nan")
        if (k + 1) % qr_every == 0:
            nv = v.norm()
            log_sum += float(torch.log(nv))
            v = v / nv
    return log_sum / (n_steps * dt)


def long_time_stats(model, ds: Dataset, n_long=3, lyap_T=100.0, psd_nperseg=1024,
                    mmd_n=2000, device=None, save_rollouts=None, min_stable_T=20.0):
    """Long-time statistics with explicit handling of unstable models.

    For each rollout, `blowup_time` is the first time (in T_lambda) at which
    the state is non-finite or exceeds 5x the reference amplitude; T_long if
    it never does. Statistics (MMD, PSD, lambda_1) are computed on the stable
    prefix of each rollout when that prefix is at least `min_stable_T`
    Lyapunov times long, so a model that is accurate for 80 T_lambda and
    then diverges still gets its attractor statistics measured on those 80.
    `blowup_frac` is the fraction of rollouts that did not survive T_long.
    """
    device = device or next(model.parameters()).device
    model.eval()
    k = _stride(model, ds)
    ref = ds.ref_long[::k]
    n_steps = len(ref) - 1
    steps_per_T = ds.T_lambda / model.dt
    rng = np.random.default_rng(123)
    starts = rng.choice(len(ds.val) - 1, size=n_long, replace=False)
    x0 = torch.as_tensor(ds.val[starts], dtype=torch.float32, device=device)
    roll = model.rollout_nograd(x0, n_steps, keep=True).cpu().numpy()   # (n_long, T+1, dim)
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
        "blowup_time_mean": float(blow_T.mean()),       # T_lambda units; == T_long if stable
        "blowup_time_min": float(blow_T.min()),
        "T_long": float(T_total),
    }
    min_steps = int(min_stable_T * steps_per_T)
    coords = np.arange(min(8, ds.dim)) if ds.dim <= 8 else np.linspace(0, ds.dim - 1, 8).astype(int)
    mmd, psd, lam = [], [], []
    ref_psd = _welch_psd(ref, psd_nperseg)
    n_used = 0
    for r, i, x in zip(roll, blow_idx, x0):
        # stable prefix, with a 2 T_lambda safety margin before the blowup
        end = n_steps + 1 if i > n_steps else max(0, i - int(2 * steps_per_T))
        if end < min_steps:
            continue
        n_used += 1
        seg = r[:end]
        mmd.append(np.mean([_mmd_rbf(seg[:, c], ref[:, c], mmd_n, rng) for c in coords]))
        p = _welch_psd(seg, min(psd_nperseg, len(seg) // 4))
        rp = _welch_psd(ref, min(psd_nperseg, len(seg) // 4))
        psd.append(np.linalg.norm(p - rp) / np.linalg.norm(rp))
        n_ly = int(min(lyap_T * steps_per_T, 0.5 * end))
        l = _benettin_model(model, x, n_ly, model.dt, device=device)
        if np.isfinite(l):
            lam.append(l)
    out["n_rollouts_used_for_stats"] = n_used
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
