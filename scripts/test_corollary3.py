"""Verify Corollary 3 (heterogeneity coverage gap).

For each benchmark system, compute the realized coverage ratio
    R = (W_bar_star - W_g_max) / W_g_max
where
    W_i_star(beta) = beta / mu_hat_i
    W_g_max       = min_i W_i_star(beta)

Corollary 3 predicts R is non-negative, vanishes when all mu_hat_i are equal,
and is bounded below by CV(mu_hat)^2, where CV is the coefficient of
variation of the *effective* (floor-clipped) instability mu_hat_i > 0 that
actually defines the schedule W_i = beta / mu_hat_i. The proof is two lines:
Jensen gives mean(1/mu) >= 1/mean(mu), and mu >= 0 gives
mean(mu^2) <= max(mu) * mean(mu), i.e. max(mu)/mean(mu) >= 1 + CV^2.

Because FTLE tables contain near-zero and negative entries (laminar
segments), the schedule is built from mu_hat_i = max(lambda_i, L) with a
floor L. The floor is equivalent to a maximum horizon W_max = beta / L.
The realized gap depends on this floor, so the script also sweeps the
floor over several percentiles of the positive FTLE values and records
the gap at each.

This test requires only an FTLE table per system (no training) and is the
single most valuable theorem-direct verification we can run.

Output: results/test_corollary3.json with per-system measurements.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

# Make src importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.systems import get_system
from src.ftle import ftle_table
from src.schedules import coverage_ratio


def run_system_coverage_test(
    system_name: str,
    n_points: int,
    tau_factors: list[float],
    rng_seed: int,
    out_dir: Path,
    save_ftle: bool = True,
    verbose: bool = True,
    ftle_method: str = "svd",
    clip_quantile: float = 0.10,
    clip_sweep_quantiles: tuple[float, ...] = (0.01, 0.05, 0.10, 0.25, 0.50),
) -> list[dict]:
    """Run Corollary 3 verification for a single system at multiple tau values.

    For each tau in [tau_factor * T_lambda for tau_factor in tau_factors],
    we build an FTLE table and compute the coverage ratio.
    """
    sys_obj = get_system(system_name)
    T_lambda = 1.0 / sys_obj.reference_lambda1

    results = []
    for tau_factor in tau_factors:
        tau = tau_factor * T_lambda
        if verbose:
            print(f"\n=== {system_name}: tau = {tau:.3f} ({tau_factor} T_lambda) ===")

        table = ftle_table(
            sys_obj,
            n_points=n_points,
            tau=tau,
            rng_seed=rng_seed,
            verbose=verbose,
            method=ftle_method,
        )

        # Validity check: if too few FTLE estimates succeeded, skip with
        # an informative record rather than crashing downstream consumers.
        if table["n_valid"] < max(10, int(0.5 * n_points)):
            print(f"  WARNING: only {table['n_valid']}/{n_points} valid FTLE "
                  f"estimates for {system_name} at tau={tau:g}; skipping.")
            results.append({
                "system": system_name,
                "tau_factor": tau_factor,
                "tau": tau,
                "n_points": n_points,
                "n_valid": table["n_valid"],
                "status": "insufficient_valid_estimates",
            })
            continue

        # Use FTLE estimates as mu_hat, restricted to valid entries.
        # Clip from below to avoid 1/mu blow-up from near-zero entries
        # (laminar/contracting segments). The clip threshold L_lower is the
        # analog of L_i in Proposition 4: we use the 10th percentile of
        # positive FTLE values, robust to outliers.
        lambdas_valid = table["lambda_ftle"][table["valid_mask"]]
        positive_mu = lambdas_valid[lambdas_valid > 0]
        if len(positive_mu) == 0:
            print(f"  WARNING: no positive FTLE values for {system_name} "
                  f"at tau={tau:g}; skipping.")
            results.append({
                "system": system_name,
                "tau_factor": tau_factor,
                "tau": tau,
                "n_points": n_points,
                "n_valid": table["n_valid"],
                "status": "no_positive_ftle",
            })
            continue
        L_lower = float(np.quantile(positive_mu, clip_quantile))
        L_lower = max(L_lower, 1e-3)
        mu_hat = np.maximum(lambdas_valid, L_lower)

        # Pick beta so that the resulting schedule is in a non-degenerate regime.
        # beta = tau * lambda_mean places the saturating schedule near the
        # FTLE-mean horizon, well inside the attractor's natural scale.
        beta = float(tau * table["lambda_mean"])

        cov = coverage_ratio(mu_hat, beta)

        # CV of the raw FTLE distribution (the pre-training diagnostic) and of
        # the effective clipped instability (the quantity the bound uses).
        cv = table["cv"]
        cv_eff = float(np.std(mu_hat) / np.mean(mu_hat))
        cv_sq_bound = cv_eff ** 2

        # Sensitivity of the realized gap to the floor percentile.
        clip_sweep = []
        for q in clip_sweep_quantiles:
            Lq = max(float(np.quantile(positive_mu, q)), 1e-3)
            mu_q = np.maximum(lambdas_valid, Lq)
            cov_q = coverage_ratio(mu_q, beta)
            cv_q = float(np.std(mu_q) / np.mean(mu_q))
            clip_sweep.append({
                "clip_quantile": q,
                "L_lower": Lq,
                "W_max": beta / Lq,
                "cv_eff": cv_q,
                "cv_sq_bound": cv_q ** 2,
                "ratio_minus_one": cov_q["ratio_minus_one"],
                "frac_at_floor": float(np.mean(lambdas_valid <= Lq)),
            })

        result = {
            "system": system_name,
            "ftle_method": ftle_method,
            "tau_factor": tau_factor,
            "tau": tau,
            "n_points": n_points,
            "n_valid": table["n_valid"],
            "beta": beta,
            "lambda_mean": table["lambda_mean"],
            "lambda_std": table["lambda_std"],
            "clip_quantile": clip_quantile,
            "L_lower_clip": L_lower,
            "W_max": beta / L_lower,
            "frac_at_floor": float(np.mean(lambdas_valid <= L_lower)),
            "frac_negative_ftle": float(np.mean(lambdas_valid <= 0)),
            "cv": cv,
            "cv_eff": cv_eff,
            "W_g_max": cov["W_g_max"],
            "W_bar_star": cov["W_bar_star"],
            "ratio_minus_one": cov["ratio_minus_one"],
            "cv_sq_bound": cv_sq_bound,
            "passes_lower_bound": cov["ratio_minus_one"] >= cv_sq_bound - 1e-6,
            "clip_sweep": clip_sweep,
            "status": "ok",
        }
        results.append(result)

        print(f"  CV(raw FTLE) = {cv:.4f}   CV(effective mu) = {cv_eff:.4f}")
        print(f"  Realized coverage gap    = {cov['ratio_minus_one']:.4f}")
        print(f"  CV^2 lower bound         = {cv_sq_bound:.4f}")
        print(f"  Passes lower bound: {result['passes_lower_bound']}")
        for c in clip_sweep:
            print(f"    floor q={c['clip_quantile']:.2f}: gap={c['ratio_minus_one']:.2f} "
                  f"cv_eff={c['cv_eff']:.3f} bound={c['cv_sq_bound']:.3f}")

        if save_ftle:
            ftle_path = out_dir / f"ftle_{system_name}_tau{tau_factor:g}{'' if ftle_method == 'svd' else '_single'}.npz"
            np.savez(
                ftle_path,
                lambda_ftle=table["lambda_ftle"],
                x0=table["x0"],
                tau=tau,
                lambda_mean=table["lambda_mean"],
                lambda_std=table["lambda_std"],
                cv=cv,
            )

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--systems", nargs="+",
                        default=["l63", "rossler", "l96_n40", "ks_l22"])
    parser.add_argument("--n_points", type=int, default=4000,
                        help="Attractor sample count per system")
    parser.add_argument("--tau_factors", nargs="+", type=float,
                        default=[0.5, 1.0, 2.0],
                        help="Multiples of T_lambda for the FTLE horizon")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ftle_method", choices=["svd", "single"], default="svd",
                        help="svd = defined largest FTLE (default); single = earlier directional proxy")
    parser.add_argument("--out_dir", type=str, default="results/corollary3")
    parser.add_argument("--task_id", type=int, default=None,
                        help="If set, run only the system at this index (for SLURM array)")
    parser.add_argument("--clip_quantile", type=float, default=0.10,
                        help="Percentile of positive FTLE values used as the "
                             "instability floor L (equivalently W_max = beta/L)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # SLURM job array support: each task processes one system.
    systems = args.systems
    if args.task_id is not None:
        systems = [systems[args.task_id]]

    all_results = []
    for s in systems:
        results = run_system_coverage_test(
            system_name=s,
            n_points=args.n_points,
            tau_factors=args.tau_factors,
            rng_seed=args.seed,
            out_dir=out_dir,
            ftle_method=args.ftle_method,
            clip_quantile=args.clip_quantile,
        )
        all_results.extend(results)

    # Save aggregate results
    suffix = f"_task{args.task_id}" if args.task_id is not None else ""
    out_path = out_dir / f"results{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n=== Saved {len(all_results)} results to {out_path} ===")


if __name__ == "__main__":
    main()
