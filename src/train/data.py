"""Dataset generation: trajectories, teacher FTLE table, true lambda_1.

`make_dataset` produces one .npz per system containing:
  train      (n_train+1, dim)  training trajectory
  val        (n_val+1, dim)    validation trajectory
  test       (n_test, n_horizon+1, dim)  short trajectories from distinct ICs
  ref_long   (n_long+1, dim)   reference trajectory for long-time stats
  ftle_train (n_train+1,)      teacher FTLE at every train index (interpolated
                               from every `ftle_stride`-th point)
  lambda1_true                 Benettin estimate on the true system
  dt, T_lambda, lambda1_ref, tau, x_mean, x_std, ftle_method

Disjointness. All pieces are consecutive, non-overlapping windows of ONE
long post-transient orbit, separated by gaps of `gap_T` Lyapunov times:

    [burn-in] [train] gap [val] gap [ref_long] gap [test_0] gap [test_1] ...

so no state appears in more than one piece. `check_disjoint` verifies this
by exact row comparison and `make_dataset` asserts it. (An earlier version
sampled the starting points 20 T_lambda apart along the orbit and then
integrated 200 T_lambda from each, which made validation and several test
trajectories sub-intervals of the training trajectory.)

Teacher FTLE. By default the defined largest FTLE,
log sigma_max(D phi^tau)/tau, from the full tangent matrix
(`ftle_method="svd"`). `ftle_method="single"` reproduces the earlier
single-random-vector directional proxy.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import time
import numpy as np

from src.systems import get_system
from src.ftle import ftle_largest


@dataclass
class Dataset:
    system: str
    dt: float
    T_lambda: float
    lambda1_ref: float
    lambda1_true: float
    tau: float
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    ref_long: np.ndarray
    ftle_train: np.ndarray
    x_mean: np.ndarray
    x_std: np.ndarray
    ftle_method: str = "svd"

    @property
    def dim(self):
        return self.train.shape[1]

    @staticmethod
    def load(path):
        z = np.load(path, allow_pickle=False)
        return Dataset(
            system=str(z["system"]), dt=float(z["dt"]), T_lambda=float(z["T_lambda"]),
            lambda1_ref=float(z["lambda1_ref"]), lambda1_true=float(z["lambda1_true"]),
            tau=float(z["tau"]), train=z["train"], val=z["val"], test=z["test"],
            ref_long=z["ref_long"], ftle_train=z["ftle_train"],
            x_mean=z["x_mean"], x_std=z["x_std"],
            ftle_method=str(z["ftle_method"]) if "ftle_method" in z.files else "single",
        )


def check_disjoint(train, val, test, ref_long) -> dict:
    """Exact-row overlap counts between the training trajectory and the other pieces."""
    trset = {r.tobytes() for r in train}
    out = {
        "val_in_train": int(sum(r.tobytes() in trset for r in val)),
        "ref_in_train": int(sum(r.tobytes() in trset for r in ref_long)),
        "test_rows_in_train": int(sum(r.tobytes() in trset for t in test for r in t)),
    }
    vs = {r.tobytes() for r in val}
    out["test_rows_in_val"] = int(sum(r.tobytes() in vs for t in test for r in t))
    return out


def _benettin_true(sys_obj, x0, T, dt, qr_every=10):
    """lambda_1 of the true system by Benettin's algorithm (long horizon, so
    the single vector is aligned; this is the asymptotic exponent)."""
    rng = np.random.default_rng(1)
    v = rng.standard_normal(sys_obj.dim)
    v /= np.linalg.norm(v)
    x = x0.copy()
    n = int(round(T / dt))
    log_sum = 0.0
    for k in range(n):
        x, v = sys_obj.tangent_step(x, v, dt)
        if (k + 1) % qr_every == 0:
            nv = np.linalg.norm(v)
            log_sum += np.log(nv)
            v /= nv
    return log_sum / (n * dt)


def make_dataset(system: str, T_train: float, T_val: float, n_test: int,
                 T_test: float, T_long: float, tau_factor: float = 0.5,
                 ftle_stride: int = 5, lyap_T: float = 1000.0, seed: int = 0,
                 gap_T: float = 5.0, ftle_method: str = "svd",
                 out_path: str | Path | None = None, verbose: bool = True) -> Dataset:
    t0 = time.time()
    sys_obj = get_system(system)
    dt = sys_obj.default_params.dt
    lam_ref = sys_obj.reference_lambda1
    T_lam = 1.0 / lam_ref
    rng = np.random.default_rng(seed)

    # One long post-transient orbit, sliced into disjoint windows with gaps.
    x_start = sys_obj.sample_attractor_ic(1, rng)[0]       # already burned in
    steps = lambda T: int(round(T * T_lam / dt))
    n_tr, n_va, n_lo, n_te, n_gap = steps(T_train), steps(T_val), steps(T_long), steps(T_test), steps(gap_T)
    total = (n_tr + 1) + n_gap + (n_va + 1) + n_gap + (n_lo + 1) + n_gap + n_test * (n_te + 1 + n_gap)
    if verbose:
        print(f"[data] {system}: dt={dt}, T_lambda={T_lam:.3f}; integrating one orbit of "
              f"{total} steps ({total*dt/T_lam:.0f} T_lambda) and slicing disjoint windows", flush=True)
    orbit = sys_obj.integrate(x_start, total * dt, dt=dt, return_traj=True)
    i = 0
    train = orbit[i:i + n_tr + 1]; i += n_tr + 1 + n_gap
    val = orbit[i:i + n_va + 1]; i += n_va + 1 + n_gap
    ref_long = orbit[i:i + n_lo + 1]; i += n_lo + 1 + n_gap
    test = []
    for _ in range(n_test):
        test.append(orbit[i:i + n_te + 1]); i += n_te + 1 + n_gap
    test = np.stack(test)
    assert i <= len(orbit)
    ov = check_disjoint(train, val, test, ref_long)
    assert all(v == 0 for v in ov.values()), f"split overlap detected: {ov}"

    x_mean = train.mean(axis=0)
    x_std = train.std(axis=0) + 1e-8

    # Teacher FTLE along the training trajectory.
    tau = tau_factor * T_lam
    idx = np.arange(0, len(train), ftle_stride)
    if verbose:
        print(f"[data] {system}: FTLE ({ftle_method}) at {len(idx)} train points, tau={tau:.3f}", flush=True)
    vals = np.empty(len(idx))
    for j, ii in enumerate(idx):
        res = ftle_largest(sys_obj, train[ii], tau, dt=dt, method=ftle_method,
                           rng=np.random.default_rng(seed + 1 + j))
        vals[j] = res.lambda_ftle if res.valid else np.nan
        if verbose and (j + 1) % max(1, len(idx) // 10) == 0:
            print(f"   ftle {j+1}/{len(idx)} ({time.time()-t0:.0f}s)", flush=True)
    ok = np.isfinite(vals)
    ftle_train = np.interp(np.arange(len(train)), idx[ok], vals[ok])

    if verbose:
        print(f"[data] {system}: Benettin lambda_1 over {lyap_T} time units", flush=True)
    lam_true = _benettin_true(sys_obj, ref_long[0], lyap_T, dt)

    ds = Dataset(system=system, dt=dt, T_lambda=T_lam, lambda1_ref=lam_ref,
                 lambda1_true=float(lam_true), tau=tau, train=train, val=val,
                 test=test, ref_long=ref_long, ftle_train=ftle_train,
                 x_mean=x_mean, x_std=x_std, ftle_method=ftle_method)
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_name(out_path.name + ".tmp.npz")
        np.savez_compressed(
            tmp, system=system, dt=dt, T_lambda=T_lam, lambda1_ref=lam_ref,
            lambda1_true=lam_true, tau=tau, train=train, val=val, test=test,
            ref_long=ref_long, ftle_train=ftle_train, x_mean=x_mean, x_std=x_std,
            ftle_method=ftle_method,
        )
        tmp.replace(out_path)
    if verbose:
        print(f"[data] {system}: done in {time.time()-t0:.0f}s; train {train.shape}, test {test.shape}; "
              f"overlap {ov}; FTLE mean {np.nanmean(vals):.4f} CV {np.nanstd(vals)/abs(np.nanmean(vals)):.3f} "
              f"nonpos {100*np.mean(vals[ok] <= 0):.1f}%; lambda_1 true {lam_true:.4f} (ref {lam_ref})", flush=True)
    return ds
