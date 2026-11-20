"""Finite-Time Lyapunov Exponent estimation via tangent-linear integration.

We compute, at each sampled point x_0 on the attractor and a chosen horizon tau:

    lambda_FTLE(x_0, tau) = (1/tau) * log sigma_max( D phi^tau(x_0) )

Two methods:

  method="svd" (default): propagate the full d x d tangent matrix
      M(tau) = D phi^tau(x_0) with the system's joint matrix step, rescaling
      M by a scalar periodically to avoid overflow (a scalar rescaling leaves
      sigma_max(M)/scale exact), and take log sigma_max(M(tau))/tau. This is
      the quantity defined above.

  method="single": propagate ONE random unit tangent vector with periodic
      renormalization (Benettin's single-vector scheme) and report
      log ||M v|| / tau. Over a short horizon the vector has not aligned with
      the most unstable direction, so this is a *directional stretching
      proxy*, not the largest FTLE: it mixes variation over attractor
      locations with variation over random orientations, and can be
      negative on segments whose sigma_max is well above one. It is kept
      for reproducing earlier results and as a cheap proxy; it must not be
      reported as the FTLE of the definition.

Two interfaces:
  - `ftle_largest`: FTLE at one point over horizon tau (method "svd" or "single")
  - `ftle_running_integral`: running log-stretch as a function of t

Both delegate to the system's joint state + tangent steps (`tangent_step`,
`tangent_step_matrix`), RK4 in the base class and ETDRK4 for KS.

NaN handling: if the joint step ever produces non-finite values (numerical
blowup) we abort the FTLE estimate at that point and mark it as invalid
rather than poisoning the rest of the table.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from src.systems.base import DynamicalSystem


@dataclass
class FTLEResult:
    """Result of an FTLE measurement at a single starting point."""
    x0: np.ndarray
    tau: float
    lambda_ftle: float       # estimated FTLE (NaN if integration blew up)
    log_stretch: float       # log of total stretch over horizon tau
    n_qr: int                # number of QR renormalizations used
    valid: bool = True       # False if integration encountered NaN/Inf


def ftle_largest_svd(
    sys: DynamicalSystem,
    x0: np.ndarray,
    tau: float,
    dt: float | None = None,
    rescale_interval_steps: int = 10,
) -> FTLEResult:
    """Largest FTLE at x0 over horizon tau from the full tangent matrix:
    log sigma_max(D phi^tau(x0)) / tau.

    The matrix is rescaled by its Frobenius norm every
    `rescale_interval_steps` steps (a scalar, so sigma_max is recovered
    exactly) to avoid overflow on long horizons.
    """
    dt = dt if dt is not None else sys.default_params.dt
    n_steps = int(np.round(tau / dt))
    x = x0.copy()
    M = np.eye(sys.dim)
    log_scale = 0.0
    valid = True
    n_resc = 0
    for k in range(n_steps):
        x, M = sys.tangent_step_matrix(x, M, dt)
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(M))):
            valid = False
            break
        if (k + 1) % rescale_interval_steps == 0:
            sc = np.linalg.norm(M)
            if sc == 0.0 or not np.isfinite(sc):
                valid = False
                break
            M /= sc
            log_scale += np.log(sc)
            n_resc += 1
    if valid:
        smax = np.linalg.svd(M, compute_uv=False)[0]
        log_stretch = np.log(smax) + log_scale
        lam = log_stretch / tau
    else:
        lam = float("nan")
        log_stretch = float("nan")
    return FTLEResult(x0=x0, tau=tau, lambda_ftle=lam, log_stretch=log_stretch,
                      n_qr=n_resc, valid=valid)


def ftle_largest(
    sys: DynamicalSystem,
    x0: np.ndarray,
    tau: float,
    dt: float | None = None,
    qr_interval_steps: int = 10,
    rng: np.random.Generator | None = None,
    method: str = "svd",
) -> FTLEResult:
    """Largest FTLE at x0 over horizon tau.

    method="svd" (default) computes log sigma_max(D phi^tau)/tau from the
    full tangent matrix (see `ftle_largest_svd`). method="single" is the
    single-random-vector directional proxy described in the module
    docstring.
    """
    if method == "svd":
        return ftle_largest_svd(sys, x0, tau, dt=dt, rescale_interval_steps=qr_interval_steps)
    if method != "single":
        raise ValueError(f"unknown FTLE method {method!r}")
    rng = rng if rng is not None else np.random.default_rng(0)
    dt = dt if dt is not None else sys.default_params.dt
    n_steps = int(np.round(tau / dt))

    x = x0.copy()
    v = rng.standard_normal(sys.dim)
    v = v / np.linalg.norm(v)

    log_stretch = 0.0
    n_qr = 0
    valid = True
    for k in range(n_steps):
        x, v = sys.tangent_step(x, v, dt)

        # NaN guard: abort cleanly if the joint step produced non-finite values.
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(v))):
            valid = False
            break

        if (k + 1) % qr_interval_steps == 0:
            nv = np.linalg.norm(v)
            if nv == 0.0 or not np.isfinite(nv):
                valid = False
                break
            log_stretch += np.log(nv)
            v = v / nv
            n_qr += 1

    if valid:
        # Account for any remaining stretch on the final partial interval.
        nv = np.linalg.norm(v)
        if nv > 0.0 and np.isfinite(nv):
            log_stretch += np.log(nv)
        lam = log_stretch / tau
    else:
        lam = float("nan")
        log_stretch = float("nan")

    return FTLEResult(
        x0=x0,
        tau=tau,
        lambda_ftle=lam,
        log_stretch=log_stretch,
        n_qr=n_qr,
        valid=valid,
    )


def ftle_running_integral(
    sys: DynamicalSystem,
    x0: np.ndarray,
    tau: float,
    dt: float | None = None,
    qr_interval_steps: int = 10,
    rng: np.random.Generator | None = None,
    method: str = "svd",
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (t_grid, Lambda_running), the running log-stretch along a
    trajectory.

    method="svd": Lambda_running[k] = log sigma_max(D phi^{t_k}(x0)), the
    log of the largest finite-time stretch factor up to time t_k, from the
    full tangent matrix. method="single": log ||D phi^{t_k} v|| for one
    random unit vector v (directional proxy; can be negative).

    If integration blows up, Lambda_running is filled with NaN beyond the
    point of failure.
    """
    dt = dt if dt is not None else sys.default_params.dt
    n_steps = int(np.round(tau / dt))
    if method == "svd":
        t_grid = np.arange(n_steps + 1) * dt
        Lam = np.full(n_steps + 1, np.nan)
        Lam[0] = 0.0
        x = x0.copy(); M = np.eye(sys.dim); log_scale = 0.0
        for k in range(n_steps):
            x, M = sys.tangent_step_matrix(x, M, dt)
            if not (np.all(np.isfinite(x)) and np.all(np.isfinite(M))):
                break
            if (k + 1) % qr_interval_steps == 0:
                sc = np.linalg.norm(M)
                if sc == 0.0 or not np.isfinite(sc):
                    break
                M /= sc; log_scale += np.log(sc)
            Lam[k + 1] = np.log(np.linalg.svd(M, compute_uv=False)[0]) + log_scale
        return t_grid, Lam
    if method != "single":
        raise ValueError(f"unknown FTLE method {method!r}")
    rng = rng if rng is not None else np.random.default_rng(0)

    x = x0.copy()
    v = rng.standard_normal(sys.dim)
    v = v / np.linalg.norm(v)

    t_grid = np.empty(n_steps + 1)
    Lambda = np.empty(n_steps + 1)
    t_grid[0] = 0.0
    Lambda[0] = 0.0

    log_stretch = 0.0
    blown_up = False
    for k in range(n_steps):
        if not blown_up:
            x, v = sys.tangent_step(x, v, dt)
            if not (np.all(np.isfinite(x)) and np.all(np.isfinite(v))):
                blown_up = True

        if blown_up:
            t_grid[k + 1] = (k + 1) * dt
            Lambda[k + 1] = float("nan")
            continue

        nv = np.linalg.norm(v)
        if nv > 0 and np.isfinite(nv):
            log_stretch += np.log(nv)
            v = v / nv
        else:
            blown_up = True
            t_grid[k + 1] = (k + 1) * dt
            Lambda[k + 1] = float("nan")
            continue

        t_grid[k + 1] = (k + 1) * dt
        Lambda[k + 1] = log_stretch

    return t_grid, Lambda


def ftle_table(
    sys: DynamicalSystem,
    n_points: int,
    tau: float,
    dt: float | None = None,
    rng_seed: int = 0,
    qr_interval_steps: int = 10,
    spacing: float | None = None,
    verbose: bool = True,
    method: str = "svd",
) -> dict:
    """Build a table of FTLE values across n_points samples on the attractor.
    `method` is passed to `ftle_largest` ("svd" = defined largest FTLE,
    "single" = directional proxy).

    Returns a dict with keys:
      - "x0": (n_points, dim) array of sampled initial conditions
      - "lambda_ftle": (n_points,) array of FTLE estimates (NaN where the
                      tangent-linear integration blew up)
      - "log_stretch": (n_points,) array of total log-stretches over [0, tau]
      - "tau": the horizon used
      - "valid_mask": (n_points,) bool array, True where the estimate is finite
      - "n_valid": int, number of successful estimates
      - "lambda_mean": mean over valid entries
      - "lambda_std": std over valid entries
      - "cv": coefficient of variation = std / mean over valid entries
    """
    rng = np.random.default_rng(rng_seed)
    if verbose:
        print(f"[ftle_table] {sys.name}: sampling {n_points} attractor points...",
              flush=True)
    x0s = sys.sample_attractor_ic(n_points, rng, spacing=spacing)

    lambdas = np.full(n_points, np.nan)
    log_stretches = np.full(n_points, np.nan)
    valid_mask = np.zeros(n_points, dtype=bool)

    if verbose:
        print(f"[ftle_table] {sys.name}: computing FTLE at tau={tau}...",
              flush=True)
    for i in range(n_points):
        sub = np.random.default_rng(rng_seed + 1 + i)
        res = ftle_largest(sys, x0s[i], tau, dt=dt,
                           qr_interval_steps=qr_interval_steps, rng=sub, method=method)
        if res.valid and np.isfinite(res.lambda_ftle):
            lambdas[i] = res.lambda_ftle
            log_stretches[i] = res.log_stretch
            valid_mask[i] = True

        if verbose and (i + 1) % max(1, n_points // 10) == 0:
            n_ok = int(valid_mask[: i + 1].sum())
            running_mean = float(np.nanmean(lambdas[: i + 1])) if n_ok > 0 else float("nan")
            print(f"  progress: {i+1}/{n_points} "
                  f"(valid={n_ok}, running mean={running_mean:.4f})",
                  flush=True)

    n_valid = int(valid_mask.sum())
    if n_valid > 0:
        valid_lambdas = lambdas[valid_mask]
        mean = float(np.mean(valid_lambdas))
        std = float(np.std(valid_lambdas))
        cv = std / abs(mean) if mean != 0 else float("nan")
    else:
        mean = float("nan")
        std = float("nan")
        cv = float("nan")

    if verbose:
        print(f"[ftle_table] {sys.name}: completed with {n_valid}/{n_points} "
              f"valid estimates (mean={mean:.4f}, cv={cv:.4f})", flush=True)

    return {
        "system": sys.name,
        "method": method,
        "x0": x0s,
        "lambda_ftle": lambdas,
        "log_stretch": log_stretches,
        "valid_mask": valid_mask,
        "n_valid": n_valid,
        "tau": tau,
        "lambda_mean": mean,
        "lambda_std": std,
        "cv": cv,
    }
