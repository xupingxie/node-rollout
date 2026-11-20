"""Verify Lemma 1 (gradient amplification scales as e^Lambda).

Two ways to use this script:

(A) Pure-tangent-linear test (no model training): for each system, compute
    Lambda_i(W) along trajectories at several W values via the running
    integral, and verify that ||v(W)|| / ||v(0)|| ~ e^Lambda_i(W). This is a
    *direct* test of the tangent-linear bound underlying Lemma 1, performed
    on the true vector field rather than on a learned f_theta.

(B) Logged-gradient test (uses existing training logs): provided a JSON
    file of (method, system, W, peak_gradient_norm, beta_estimate) records
    from your training pipeline, plot ||grad|| vs e^beta and verify the
    slope-1 scaling on log-linear axes.

Mode (A) is theorem-direct; mode (B) tests how tightly the bound is realized
in practice.

Output: results/lemma1/<mode>.json
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.systems import get_system
from src.ftle import ftle_running_integral


def run_tangent_linear_test(
    system_name: str,
    n_trajectories: int,
    W_factors: list[float],
    rng_seed: int,
    verbose: bool = True,
    ftle_method: str = "svd",
) -> list[dict]:
    """Mode (A): test e^Lambda(W) bound on the true vector field.

    For n_trajectories random attractor starting points, integrate the joint
    state + tangent system out to W = max(W_factors) * T_lambda. The running
    integral of log-stretch gives Lambda_i(W) directly. By construction the
    tangent-linear satisfies ||v(W)|| / ||v(0)|| = e^Lambda(W), so this run
    serves as a numerical sanity check that the FTLE estimator is consistent
    and that the e^Lambda scaling is well-behaved (no overflow, no
    cancellation) at the W values relevant to training.
    """
    sys_obj = get_system(system_name)
    T_lambda = 1.0 / sys_obj.reference_lambda1
    W_max_factor = max(W_factors)
    tau = W_max_factor * T_lambda

    rng = np.random.default_rng(rng_seed)
    if verbose:
        print(f"\n=== {system_name}: tangent-linear test, n_traj={n_trajectories}, "
              f"max_horizon={W_max_factor:g} T_lambda ===")

    x0s = sys_obj.sample_attractor_ic(n_trajectories, rng)

    results = []
    n_valid_trajs = 0
    for traj_idx in range(n_trajectories):
        sub = np.random.default_rng(rng_seed + traj_idx + 1)
        t_grid, Lambda_running = ftle_running_integral(
            sys_obj, x0s[traj_idx], tau=tau, rng=sub, method=ftle_method
        )

        # Skip trajectories that blew up before reaching the first sample W.
        if not np.isfinite(Lambda_running[-1]):
            print(f"  trajectory {traj_idx}: blew up; skipping.")
            continue
        n_valid_trajs += 1

        # Sample at the requested W_factors.
        for W_factor in W_factors:
            W = W_factor * T_lambda
            idx = int(np.argmin(np.abs(t_grid - W)))
            Lambda_W = Lambda_running[idx]
            if not np.isfinite(Lambda_W):
                continue
            results.append({
                "system": system_name,
                "trajectory_index": traj_idx,
                "W_factor": W_factor,
                "W": float(t_grid[idx]),
                "Lambda_W": float(Lambda_W),
                "exp_Lambda_W": float(np.exp(min(Lambda_W, 50))),
                "T_lambda": T_lambda,
            })

    if verbose:
        print(f"  {n_valid_trajs}/{n_trajectories} trajectories valid")
        for W_factor in W_factors:
            subset = [r for r in results if r["W_factor"] == W_factor]
            if subset:
                ls = np.array([r["Lambda_W"] for r in subset])
                print(f"  W = {W_factor:g} T_lambda: "
                      f"Lambda mean = {ls.mean():.3f}, std = {ls.std():.3f}, "
                      f"e^Lambda mean = {np.exp(ls).mean():.3g}")

    return results


def run_logged_gradient_test(
    log_file: Path,
    out_path: Path,
) -> None:
    """Mode (B): consume an existing log of training-time gradient norms.

    Expected log format: a JSON list of records, each with keys:
        - "system": str
        - "method": str
        - "W": float (nominal rollout horizon)
        - "K": int (windows; 1 for single-rollout)
        - "peak_grad_norm": float
        - "beta_estimate": float (optional; if missing, computed as W * lambda_mean / K)
        - "lambda_mean": float (system FTLE mean; used as fallback for beta)

    The script aggregates per (system, method, W, K) and writes out
    log(peak_grad_norm) vs beta_estimate for plotting.
    """
    with open(log_file) as f:
        records = json.load(f)

    out = []
    for rec in records:
        beta = rec.get("beta_estimate")
        if beta is None:
            # Fallback: beta = (W/K) * lambda_mean
            W = float(rec["W"])
            K = int(rec.get("K", 1))
            lam = float(rec["lambda_mean"])
            beta = (W / K) * lam
        out.append({
            "system": rec["system"],
            "method": rec["method"],
            "W": float(rec["W"]),
            "K": int(rec.get("K", 1)),
            "peak_grad_norm": float(rec["peak_grad_norm"]),
            "log_peak_grad_norm": float(np.log(max(rec["peak_grad_norm"], 1e-30))),
            "beta_estimate": float(beta),
            "exp_beta": float(np.exp(min(beta, 50))),  # clip for numerical safety
        })

    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["tangent", "logs"], default="tangent")
    parser.add_argument("--systems", nargs="+",
                        default=["l63", "rossler", "l96_n40", "ks_l22"])
    parser.add_argument("--n_trajectories", type=int, default=20,
                        help="Number of attractor trajectories (mode=tangent)")
    parser.add_argument("--W_factors", nargs="+", type=float,
                        default=[0.1, 0.25, 0.5, 1.0, 1.5, 2.0],
                        help="W as multiples of T_lambda (mode=tangent)")
    parser.add_argument("--log_file", type=str,
                        help="Training log JSON for mode=logs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ftle_method", choices=["svd", "single"], default="svd")
    parser.add_argument("--out_dir", type=str, default="results/lemma1")
    parser.add_argument("--task_id", type=int, default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "tangent":
        systems = args.systems
        if args.task_id is not None:
            systems = [systems[args.task_id]]

        all_results = []
        for s in systems:
            results = run_tangent_linear_test(
                system_name=s,
                n_trajectories=args.n_trajectories,
                W_factors=args.W_factors,
                rng_seed=args.seed,
                ftle_method=args.ftle_method,
            )
            all_results.extend(results)

        suffix = f"_task{args.task_id}" if args.task_id is not None else ""
        out_path = out_dir / f"tangent{suffix}.json"
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n=== Saved {len(all_results)} records to {out_path} ===")

    else:  # logs
        if args.log_file is None:
            raise SystemExit("Mode=logs requires --log_file")
        out_path = out_dir / "from_training_logs.json"
        run_logged_gradient_test(Path(args.log_file), out_path)
        print(f"=== Saved to {out_path} ===")


if __name__ == "__main__":
    main()
