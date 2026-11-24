"""Estimate the largest Lyapunov exponent lambda_1 for each system.

Runs Benettin's algorithm on a long reference trajectory: integrate the
joint state + tangent system, renormalize the tangent every K solver
steps, and average log(stretch)/dt over the whole run.

This is the script that produces the lambda_1 values used to set
`reference_lambda1` in each src/systems/*.py file (currently hardcoded
to literature-or-empirical values). Running this script verifies that
the hardcoded values are correct for the chosen dt, integrator, and
attractor sampling.

The reported value is asymptotic (the FTLE at infinite tau converges
to lambda_1 by Oseledec's theorem). For finite-time estimates at
specific tau, use test_corollary3.py which computes the FTLE
distribution.

USAGE
-----
    python scripts/measure_lambda1.py \\
        --systems l63 rossler l96_n40 ks_l22 \\
        --t_total 2000.0 \\
        --n_burn 200.0 \\
        --qr_interval 10 \\
        --out results/lambda1_measurements.json

Expected runtimes (per system) on one CPU core:
  L63:        ~5 seconds
  Rossler:    ~30 seconds  (slow due to T_lambda ~ 15)
  L96 N=40:   ~30 seconds
  KS L=22:    ~3 minutes   (ETDRK4 is slow but stable)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.systems import get_system


def measure_lambda1(sys_obj, t_total: float, n_burn: float,
                    qr_interval: int = 10, seed: int = 0) -> dict:
    """Largest Lyapunov exponent via Benettin's algorithm.

    Returns a dict with:
      lambda_1        : the asymptotic estimate
      lambda_1_running: estimate computed from each window (for diagnostics)
      n_qr            : number of QR renormalizations performed
    """
    rng = np.random.default_rng(seed)
    dt = sys_obj.default_params.dt

    # Burn in to land on the attractor
    x = sys_obj._generic_ic(rng).copy()
    n_burn_steps = int(np.round(n_burn / dt))
    for _ in range(n_burn_steps):
        x = sys_obj._step(0.0, x, dt)
        if not np.all(np.isfinite(x)):
            raise RuntimeError(f"State blew up during burn-in for {sys_obj.name}")

    # Initialize random tangent
    v = rng.standard_normal(sys_obj.dim)
    v = v / np.linalg.norm(v)

    n_steps = int(np.round(t_total / dt))
    log_stretch_total = 0.0
    n_qr = 0
    running_estimates = []
    elapsed_t = 0.0

    for k in range(n_steps):
        x, v = sys_obj.tangent_step(x, v, dt)
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(v))):
            raise RuntimeError(f"State or tangent blew up at step {k} for {sys_obj.name}")

        if (k + 1) % qr_interval == 0:
            nv = float(np.linalg.norm(v))
            if nv <= 0 or not np.isfinite(nv):
                raise RuntimeError(f"Bad tangent norm {nv} at step {k} for {sys_obj.name}")
            log_stretch_total += np.log(nv)
            v = v / nv
            n_qr += 1
            elapsed_t = (k + 1) * dt
            # Report a running estimate every ~T_lambda
            if n_qr % max(1, int(round(1.0 / sys_obj.reference_lambda1 / (qr_interval * dt)))) == 0:
                running_estimates.append({
                    "t": elapsed_t,
                    "lambda_1_running": log_stretch_total / elapsed_t,
                })

    final = log_stretch_total / (n_steps * dt)
    return {
        "system": sys_obj.name,
        "t_total": t_total,
        "n_burn": n_burn,
        "dt": dt,
        "qr_interval": qr_interval,
        "n_qr": n_qr,
        "lambda_1": float(final),
        "lambda_1_reference": float(sys_obj.reference_lambda1),
        "T_lambda": float(1.0 / final) if final > 0 else float("inf"),
        "T_lambda_reference": float(1.0 / sys_obj.reference_lambda1),
        "lambda_1_running_history": running_estimates,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--systems", nargs="+",
                    default=["l63", "rossler", "l96_n40", "ks_l22"])
    ap.add_argument("--t_total", type=float, default=2000.0,
                    help="Total integration time in ODE units. Use larger "
                         "for high-T_lambda systems (Rossler, KS) for "
                         "stable averaging.")
    ap.add_argument("--n_burn", type=float, default=200.0,
                    help="Burn-in time before starting the FTLE accumulation.")
    ap.add_argument("--qr_interval", type=int, default=10,
                    help="QR renormalize every N solver steps.")
    ap.add_argument("--out", required=True, type=Path,
                    help="Path to write the JSON summary.")
    args = ap.parse_args()

    results = []
    for sys_name in args.systems:
        print(f"\n=== Measuring lambda_1 for {sys_name} ===")
        sys_obj = get_system(sys_name)

        # Scale t_total for systems with large T_lambda so we get
        # comparable averaging quality. Aim for ~100 T_lambda total.
        Tl_ref = 1.0 / sys_obj.reference_lambda1
        t_total = max(args.t_total, 100.0 * Tl_ref)
        n_burn = max(args.n_burn, 10.0 * Tl_ref)

        print(f"  t_total = {t_total:g}, n_burn = {n_burn:g}, dt = {sys_obj.default_params.dt}")
        result = measure_lambda1(sys_obj, t_total=t_total, n_burn=n_burn,
                                  qr_interval=args.qr_interval)
        print(f"  lambda_1 = {result['lambda_1']:.4f}  "
              f"(reference: {result['lambda_1_reference']:.4f}, "
              f"relative err: {abs(result['lambda_1'] - result['lambda_1_reference']) / result['lambda_1_reference']:.2%})")
        print(f"  T_lambda = {result['T_lambda']:.4f}  "
              f"(reference: {result['T_lambda_reference']:.4f})")
        results.append(result)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nWrote {len(results)} measurements to {args.out}")
    print("\n=== Summary ===")
    print(f"{'System':<12} {'lambda_1':>10} {'reference':>10} {'T_lambda':>10} {'rel err':>10}")
    for r in results:
        rel = abs(r['lambda_1'] - r['lambda_1_reference']) / r['lambda_1_reference']
        print(f"{r['system']:<12} {r['lambda_1']:>10.4f} {r['lambda_1_reference']:>10.4f} "
              f"{r['T_lambda']:>10.4f} {rel:>9.2%}")


if __name__ == "__main__":
    main()
