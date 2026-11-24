"""Aggregate and plot results from theorem-verification tests.

Generates the figures of the numerical checks:

  Corollary 3:      Coverage ratio vs CV across the four systems, with the
                    CV^2 lower bound (Corollary 3) overlaid.
  Lemma 1:          Lambda(W) running integrals across systems (tangent mode)
                    or log(grad_norm) vs beta scatter (logs mode).
  Proposition 4:    Coverage loss vs sigma_eps with the 2 eps sum 1/L_i bound.

Usage:
    python scripts/aggregate_and_plot.py \
        --corollary3_dir results/corollary3 \
        --proposition4_dir results/proposition4 \
        --lemma1_dir results/lemma1 \
        --out_dir plots
"""
from __future__ import annotations
import argparse
import json
from glob import glob
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt


def load_all_json(directory: Path) -> list[dict]:
    """Load and concatenate all results*.json files in a directory."""
    out: list[dict] = []
    for p in sorted(glob(str(directory / "results*.json"))) + sorted(glob(str(directory / "tangent*.json"))):
        with open(p) as f:
            data = json.load(f)
            if isinstance(data, list):
                out.extend(data)
            else:
                out.append(data)
    return out


def plot_corollary3(records: list[dict], out_path: Path) -> None:
    """Coverage ratio (W_bar_star / W_g_max - 1) vs CV across systems."""
    if not records:
        print("[plot_corollary3] No records found.")
        return

    # Group by system, take the tau_factor closest to 0.5 (paper convention).
    by_system: dict[str, dict] = {}
    for r in records:
        s = r["system"]
        # prefer tau_factor == 0.5
        if s not in by_system or abs(r["tau_factor"] - 0.5) < abs(by_system[s]["tau_factor"] - 0.5):
            by_system[s] = r

    systems = sorted(by_system.keys(), key=lambda s: by_system[s]["cv"])
    cvs = np.array([by_system[s]["cv"] for s in systems])
    realized = np.array([by_system[s]["ratio_minus_one"] for s in systems])
    bound = np.array([by_system[s].get("cv_eff", by_system[s]["cv"]) for s in systems]) ** 2

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(cvs, realized, s=80, c="C0", zorder=3,
               edgecolor="black", label=r"realized $(\bar W^\star - W_g^{\max})/W_g^{\max}$")

    # Annotate each point with its system name.
    for s, cv, r in zip(systems, cvs, realized):
        ax.annotate(s, (cv, r), xytext=(6, 6), textcoords="offset points", fontsize=9)

    # Theory lower bound line.
    cv_grid = np.linspace(0.0, max(2.0, cvs.max() * 1.1), 200)
    ax.plot(cv_grid, cv_grid ** 2,
            "k--", lw=1.5, alpha=0.7, label=r"lower bound $\mathrm{CV}^2$ (Corollary 3)")

    ax.set_xlabel(r"$\mathrm{CV}(\hat\lambda_{\mathrm{FTLE}})$")
    ax.set_ylabel(r"relative coverage gain")
    ax.set_title("Corollary 3: realized coverage gap vs CV")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot_corollary3] saved {out_path}")


def plot_proposition4(records: list[dict], out_path: Path) -> None:
    """Coverage loss vs sigma_eps for each system, with theoretical bound."""
    if not records:
        print("[plot_proposition4] No records found.")
        return

    by_system: dict[str, list[dict]] = {}
    for r in records:
        by_system.setdefault(r["system"], []).append(r)

    fig, axes = plt.subplots(1, len(by_system), figsize=(4 * len(by_system), 4),
                             sharey=False)
    if len(by_system) == 1:
        axes = [axes]
    for ax, (s, recs) in zip(axes, by_system.items()):
        recs = sorted(recs, key=lambda r: r["sigma_eps"])
        sigmas = np.array([r["sigma_eps"] for r in recs])
        loss = np.array([r["coverage_loss_max"] for r in recs])
        bound = np.array([r["theoretical_bound"] for r in recs])
        feas = np.array([r["feasibility_fraction_min"] for r in recs])

        ax.plot(sigmas, loss, "o-", c="C0", label="realized (max over realizations)")
        ax.plot(sigmas, bound, "k--", label=r"theory bound $2\epsilon \sum 1/L_i$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\sigma_\epsilon$")
        ax.set_ylabel("coverage loss")
        ax.set_title(f"{s}\nfeasibility {feas.min():.0%}-{feas.max():.0%}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    fig.suptitle("Proposition 4: robustness verification", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_proposition4] saved {out_path}")


def plot_lemma1_tangent(records: list[dict], out_path: Path) -> None:
    """log(e^Lambda(W)) = Lambda(W) vs W, by system. Should grow ~ linearly
    with rate ~ lambda_1, then deviate as the trajectory enters
    high-instability regions.
    """
    if not records:
        print("[plot_lemma1_tangent] No records found.")
        return

    by_system: dict[str, list[dict]] = {}
    for r in records:
        by_system.setdefault(r["system"], []).append(r)

    fig, axes = plt.subplots(1, len(by_system), figsize=(4 * len(by_system), 4),
                             sharey=False)
    if len(by_system) == 1:
        axes = [axes]
    for ax, (s, recs) in zip(axes, by_system.items()):
        T_lambda = recs[0]["T_lambda"]
        W_factors = sorted({r["W_factor"] for r in recs})
        means = []
        stds = []
        for wf in W_factors:
            ls = np.array([r["Lambda_W"] for r in recs if r["W_factor"] == wf])
            means.append(ls.mean())
            stds.append(ls.std())
        means = np.array(means)
        stds = np.array(stds)
        wfs = np.array(W_factors)

        ax.errorbar(wfs, means, yerr=stds, fmt="o-",
                    label=r"$\overline{\Lambda(W)}$", c="C0", capsize=3)
        # Reference: Lambda(W) ~ lambda_1 * W => W * lambda_1 = W * (1/T_lambda)
        ax.plot(wfs, wfs, "k--", lw=1.0, alpha=0.6,
                label=r"linear $\lambda_1 W$ reference")
        ax.set_xlabel(r"$W / T_\lambda$")
        ax.set_ylabel(r"$\Lambda(W)$")
        ax.set_title(s)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    fig.suptitle(r"Lemma 1 (mode A): $\Lambda(W)$ via running tangent integral", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_lemma1_tangent] saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corollary3_dir", type=str, default="results/corollary3")
    parser.add_argument("--proposition4_dir", type=str, default="results/proposition4")
    parser.add_argument("--lemma1_dir", type=str, default="results/lemma1")
    parser.add_argument("--out_dir", type=str, default="plots")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cor3 = load_all_json(Path(args.corollary3_dir))
    prop4 = load_all_json(Path(args.proposition4_dir))
    lem1 = load_all_json(Path(args.lemma1_dir))

    plot_corollary3(cor3, out_dir / "fig_corollary3_coverage_vs_cv.png")
    plot_proposition4(prop4, out_dir / "fig_proposition4_robustness.png")
    plot_lemma1_tangent(lem1, out_dir / "fig_lemma1_tangent.png")


if __name__ == "__main__":
    main()
