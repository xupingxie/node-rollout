"""Generate fig_cross_verification.png from theorem-verification JSON.

This is a bonus figure that connects Corollary 3 and Lemma 1: the
coefficient of variation CV(FTLE) measured from the Corollary 3
verification correlates very tightly (Pearson r ~ 0.96) with the
trajectory-to-trajectory standard deviation of Lambda(T_lambda)
measured independently from the Lemma 1 verification. The two
diagnostics measure FTLE heterogeneity from different angles and
agree quantitatively.

EXPECTED INPUTS
---------------
--corollary3_dir : directory of `results_task{0,1,2,3}.json` from
                   Corollary 3 verification runs. Each entry must have
                   `system`, `tau_factor`, `cv` fields.
--lemma1_dir     : directory of `tangent_task{0,1,2,3}.json` from
                   Lemma 1 verification runs (tangent mode). Each
                   record must have `system`, `W_factor`, `Lambda_W`
                   fields, with W_factor=1.0 present.

USAGE
-----
    python scripts/figures/plot_cross_verification.py \
        --corollary3_dir results/corollary3 \
        --lemma1_dir results/lemma1 \
        --out plots/fig_cross_verification.png

OUTPUTS both .png (default) and .pdf next to the requested path.
"""
import argparse
import json
from pathlib import Path
from glob import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DISPLAY = {"l63": "L63", "rossler": "Rössler",
           "l96_n40": "L96 ($N{=}40$)", "ks_l22": "KS"}
COLORS = {"l63": "#1f77b4", "rossler": "#ff7f0e",
          "l96_n40": "#2ca02c", "ks_l22": "#d62728"}


def load_cv_at_half(cor3_dir: Path) -> dict[str, float]:
    """Read CV at tau_factor=0.5 for each system."""
    cv: dict[str, float] = {}
    for path in sorted(cor3_dir.glob("results_task*.json")):
        # Skip Proposition 4 files (they have suffix __1_ or similar)
        if "__1_" in path.name or "_1_" in path.name:
            continue
        with open(path) as f:
            records = json.load(f)
        for r in records:
            if abs(r.get("tau_factor", -1) - 0.5) < 1e-6:
                cv[r["system"]] = r["cv"]
    return cv


def load_std_lambda_at_Tl(lem1_dir: Path) -> dict[str, float]:
    """Read std(Lambda) at W_factor=1.0 for each system."""
    std: dict[str, float] = {}
    for path in sorted(lem1_dir.glob("tangent_task*.json")):
        with open(path) as f:
            records = json.load(f)
        sys_name = records[0]["system"]
        ls = np.array([r["Lambda_W"] for r in records
                       if abs(r["W_factor"] - 1.0) < 1e-6])
        if len(ls) > 0:
            std[sys_name] = float(ls.std())
    return std


def plot(cv: dict[str, float], std_lam: dict[str, float], out: Path) -> None:
    systems = sorted(cv.keys(), key=lambda s: cv[s])
    cvs = np.array([cv[s] for s in systems])
    stds = np.array([std_lam[s] for s in systems])

    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    for x, y, s in zip(cvs, stds, systems):
        ax.scatter(x, y, s=140, color=COLORS.get(s, "gray"),
                   edgecolor="black", linewidth=0.8, zorder=3,
                   label=DISPLAY.get(s, s))
        ax.annotate(DISPLAY.get(s, s), (x, y),
                    xytext=(8, 5), textcoords="offset points", fontsize=10)

    # Linear fit and Pearson r
    if len(cvs) >= 2:
        slope, intercept = np.polyfit(cvs, stds, 1)
        xline = np.linspace(0, cvs.max() * 1.15, 50)
        ax.plot(xline, slope * xline + intercept, "--", color="gray",
                lw=1.5, alpha=0.7,
                label=rf"linear fit: $\sigma_\Lambda = {slope:.2f}\,$CV "
                      rf"$+ {intercept:.2f}$")
        r = float(np.corrcoef(cvs, stds)[0, 1])
        ax.text(0.05, 0.92, f"Pearson r = {r:.3f}", transform=ax.transAxes,
                fontsize=11,
                bbox=dict(facecolor="white", alpha=0.9, edgecolor="gray"))

    ax.set_xlabel(r"CV($\hat\lambda_{\mathrm{FTLE}}$) at $\tau = T_\lambda/2$"
                  "\n(from Corollary 3)", fontsize=10)
    ax.set_ylabel(r"$\sigma_\Lambda(T_\lambda)$ across 20 trajectories"
                  "\n(from Lemma 1)", fontsize=10)
    ax.set_title(r"Cross-verification: CV $\leftrightarrow \sigma_\Lambda$",
                 fontsize=11)
    ax.grid(True, alpha=0.25)
    ax.set_xlim(0, cvs.max() * 1.15)
    ax.set_ylim(0, stds.max() * 1.15)

    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"[plot_cross_verification] wrote {out} and .pdf")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corollary3_dir", required=True, type=Path)
    ap.add_argument("--lemma1_dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    cv = load_cv_at_half(args.corollary3_dir)
    std_lam = load_std_lambda_at_Tl(args.lemma1_dir)

    common = set(cv.keys()) & set(std_lam.keys())
    if len(common) < 2:
        raise SystemExit(
            f"Insufficient matched systems: cv={list(cv.keys())}, "
            f"std_lam={list(std_lam.keys())}"
        )
    cv = {k: v for k, v in cv.items() if k in common}
    std_lam = {k: v for k, v in std_lam.items() if k in common}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plot(cv, std_lam, args.out)


if __name__ == "__main__":
    main()
