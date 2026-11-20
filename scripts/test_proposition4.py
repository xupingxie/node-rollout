"""Test Proposition 4 in a linear teacher-rate proxy model.

Default uniform noise has a verified error bound epsilon over [0, W_max].
Gaussian noise is a separate stress test: epsilon is only a calibration level.
Neither experiment certifies learned-model sensitivity bounds.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.systems import get_system
from src.ftle import ftle_table
from src.schedules import (
    saturating_schedule,
    shrunken_schedule,
    coverage,
)


def run_system_robustness_test(
    system_name: str,
    n_points: int,
    sigma_eps_grid: list[float],
    confidence_multiplier: float,
    rng_seed: int,
    out_dir: Path,
    n_noise_realizations: int = 20,
    verbose: bool = True,
    ftle_method: str = "svd",
    noise_model: str = "uniform",
) -> list[dict]:
    """Verify Proposition 4 for one system across a noise sweep."""
    sys_obj = get_system(system_name)
    T_lambda = 1.0 / sys_obj.reference_lambda1
    tau = 0.5 * T_lambda

    if verbose:
        print(f"\n=== {system_name}: building clean FTLE table ===")

    table = ftle_table(sys_obj, n_points=n_points, tau=tau,
                       rng_seed=rng_seed, verbose=verbose, method=ftle_method)

    # Validity check: bail out gracefully if the FTLE table is unusable.
    if table["n_valid"] < max(10, int(0.5 * n_points)):
        print(f"  WARNING: only {table['n_valid']}/{n_points} valid FTLE "
              f"estimates for {system_name}; skipping all noise levels.")
        return [{
            "system": system_name,
            "n_points": n_points,
            "n_valid": table["n_valid"],
            "status": "insufficient_valid_estimates",
        }]

    lambdas_valid = table["lambda_ftle"][table["valid_mask"]]
    positive_mu = lambdas_valid[lambdas_valid > 0]
    if len(positive_mu) == 0:
        print(f"  WARNING: no positive FTLE values for {system_name}; "
              f"skipping all noise levels.")
        return [{
            "system": system_name,
            "n_points": n_points,
            "n_valid": table["n_valid"],
            "status": "no_positive_ftle",
        }]

    L_lower = float(np.quantile(positive_mu, 0.10))
    L_lower = max(L_lower, 1e-3)
    mu_true = np.maximum(lambdas_valid, L_lower)

    # Use the FTLE-mean horizon as the budget (same convention as Corollary 3).
    beta = float(tau * table["lambda_mean"])
    # Maximum W in the saturating schedule (used to translate Lambda-noise to
    # mu-noise so the test matches Proposition 4's |hat Lambda - Lambda| <= eps
    # statement uniformly over W in [0, W_max]).
    W_max = beta / L_lower

    # Oracle saturating schedule under the true mu.
    W_star = saturating_schedule(mu_true, beta)
    C_star = coverage(W_star)

    # Sum 1/L_i for the theoretical coverage-loss bound. Use mu_true as the
    # local instability rate; this is the conservative bound (uses each
    # segment's actual instability).
    sum_inv_L = float(np.sum(1.0 / mu_true))

    rng = np.random.default_rng(rng_seed)
    results = []

    for sigma_eps in sigma_eps_grid:
        # sigma_eps parameterizes eps; it is a standard deviation ONLY for
        # Gaussian noise. Uniform error has std eps/sqrt(3) at W_max.
        eps = confidence_multiplier * sigma_eps
        sigma_mu = sigma_eps / W_max
        if eps >= beta:
            if verbose:
                print(f"  Skipping sigma_eps={sigma_eps}: eps={eps} >= beta={beta}")
            continue

        feas_fracs = []
        cov_losses = []
        for r in range(n_noise_realizations):
            # Truncated Gaussian: clip mu_noisy to [L_lower, +inf) so the
            # estimator stays non-negative (matching Proposition 4's
            # non-decreasing continuous Lambda_hat).
            if noise_model == "uniform":
                # |hat Lambda - Lambda| <= eps holds EXACTLY: mu-error uniform in
                # [-eps/W_max, eps/W_max], so the Lambda-error over any W <= W_max
                # is at most eps. This is the hypothesis of Proposition 4.
                noise = rng.uniform(-eps / W_max, eps / W_max, size=mu_true.shape)
            else:
                # gaussian: eps = confidence_multiplier * sigma is a probabilistic
                # level, not a uniform bound; a stress test of the guarantee.
                noise = rng.normal(scale=sigma_mu, size=mu_true.shape)
            mu_noisy = np.maximum(mu_true + noise, L_lower)

            tilde_W = shrunken_schedule(mu_noisy, beta=beta, epsilon=eps)

            # Feasibility under TRUE mu.
            Lambda_at_tildeW = mu_true * tilde_W
            n_feasible = int(np.sum(Lambda_at_tildeW <= beta + 1e-9))
            feas_fracs.append(n_feasible / len(mu_true))

            cov_losses.append(C_star - coverage(tilde_W))

        feas_frac_mean = float(np.mean(feas_fracs))
        feas_frac_min = float(np.min(feas_fracs))
        cov_loss_mean = float(np.mean(cov_losses))
        cov_loss_max = float(np.max(cov_losses))

        # Theoretical bound: 2 eps sum_i 1/L_i. With L_i = mu_true_i.
        bound = 2.0 * eps * sum_inv_L

        result = {
            "system": system_name,
            "ftle_method": ftle_method,
            "noise_model": noise_model,
            "n_points": n_points,
            "tau": tau,
            "beta": beta,
            "L_lower": L_lower,
            "W_max": W_max,
            "sigma_eps": sigma_eps,
            "sigma_mu": sigma_mu if noise_model == "gaussian" else eps / (W_max * np.sqrt(3)),
            "epsilon": eps,
            "confidence_multiplier": confidence_multiplier,
            "n_noise_realizations": n_noise_realizations,
            "feasibility_fraction_mean": feas_frac_mean,
            "feasibility_fraction_min": feas_frac_min,
            "coverage_loss_mean": cov_loss_mean,
            "coverage_loss_max": cov_loss_max,
            "theoretical_bound": bound,
            "passes_feasibility": feas_frac_min == 1.0,
            "passes_coverage_bound": cov_loss_max <= bound + 1e-6,
        }
        results.append(result)

        if verbose:
            print(f"  sigma_eps={sigma_eps:.4g}  eps={eps:.4g}: "
                  f"feas={feas_frac_mean:.3f} (min={feas_frac_min:.3f}); "
                  f"cov_loss_max={cov_loss_max:.3g} vs bound={bound:.3g}; "
                  f"feas OK: {result['passes_feasibility']}, "
                  f"bound OK: {result['passes_coverage_bound']}")

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--systems", nargs="+",
                        default=["l63", "rossler", "l96_n40", "ks_l22"])
    parser.add_argument("--n_points", type=int, default=2000)
    parser.add_argument("--sigma_eps_grid", nargs="+", type=float,
                        default=[0.01, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8])
    parser.add_argument("--confidence_multiplier", type=float, default=2.0)
    parser.add_argument("--n_noise_realizations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ftle_method", choices=["svd", "single"], default="svd",
                        help="svd = defined largest FTLE (default); single = earlier directional proxy")
    parser.add_argument("--noise", choices=["uniform", "gaussian"], default="uniform",
                        help="uniform: bounded error, the proposition's hypothesis (verification); "
                             "gaussian: 2-sigma calibration (stress test)")
    parser.add_argument("--out_dir", type=str, default="results/proposition4")
    parser.add_argument("--task_id", type=int, default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    systems = args.systems
    if args.task_id is not None:
        systems = [systems[args.task_id]]

    all_results = []
    for s in systems:
        results = run_system_robustness_test(
            system_name=s,
            n_points=args.n_points,
            sigma_eps_grid=args.sigma_eps_grid,
            confidence_multiplier=args.confidence_multiplier,
            rng_seed=args.seed,
            out_dir=out_dir,
            n_noise_realizations=args.n_noise_realizations,
            ftle_method=args.ftle_method,
            noise_model=args.noise,
        )
        all_results.extend(results)

    suffix = f"_task{args.task_id}" if args.task_id is not None else ""
    if args.noise == "gaussian":
        suffix += "_gaussian"
    out_path = out_dir / f"results{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n=== Saved {len(all_results)} results to {out_path} ===")


if __name__ == "__main__":
    main()
