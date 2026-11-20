"""Generate fig_corollary3_coverage_vs_cv.png from theorem-verification JSON.

Plots realized coverage gap bar{W}^star / W_g^max - 1 vs CV(FTLE),
with the reference curve CV^2 as a dashed line and the proven lower bound
CV_eff^2 (CV of the effective, floor-clipped instability) as hollow markers.
One point per (system, tau_factor) cell; three tau values per system
connected with thin lines.

EXPECTED INPUT
--------------
Directory of `results_task{0,1,2,3}.json` from
`scripts/test_corollary3.py` runs. Each record must have:
  - `system`     (string, one of l63/rossler/l96_n40/ks_l22)
  - `tau_factor` (float)
  - `cv`         (float, CV of the raw FTLE table; x-axis)
  - `cv_eff`     (float, CV of the effective clipped instability; the bound)
  - `ratio_minus_one` (float, the realized W_bar_star/W_g_max - 1)

USAGE
-----
    python scripts/figures/plot_corollary3.py \
        --corollary3_dir results/corollary3 \
        --out plots/fig_corollary3_coverage_vs_cv.png
"""
import argparse
import json
from glob import glob
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DISPLAY = {
    "l63": "Lorenz-63", "rossler": "Rössler",
    "l96_n40": "Lorenz-96 ($N{=}40$)", "ks_l22": "Kuramoto--Sivashinsky",
}
MARKERS = {"l63": "o", "rossler": "s", "l96_n40": "^", "ks_l22": "D"}
COLORS = {"l63": "#1f77b4", "rossler": "#ff7f0e",
          "l96_n40": "#2ca02c", "ks_l22": "#d62728"}


def load(d: Path):
    out = []
    for p in sorted(d.glob("results_task*.json")):
        # Skip Proposition 4 outputs that may live in the same dir
        if "__1_" in p.name or "_1_" in p.name:
            continue
        with open(p) as f:
            out.extend(json.load(f))
    return out


def plot(records, out: Path):
    fig, ax = plt.subplots(figsize=(6.0, 4.4))

    cv_grid = np.linspace(0.05, 2.2, 400)
    ax.plot(cv_grid, cv_grid ** 2,
            "--", color="gray", lw=1.5, alpha=0.85,
            label=r"reference curve $\mathrm{CV}^2$",
            zorder=1)

    for sys_name in ["l96_n40", "rossler", "ks_l22", "l63"]:
        recs = sorted([r for r in records if r["system"] == sys_name],
                      key=lambda r: r["tau_factor"])
        if not recs:
            continue
        cvs = np.array([r["cv"] for r in recs])
        gaps = np.array([r["ratio_minus_one"] for r in recs])
        ax.plot(cvs, gaps, "-", color=COLORS[sys_name], alpha=0.35, lw=1.0,
                zorder=2)
        ax.scatter(cvs, gaps,
                   marker=MARKERS[sys_name], s=85,
                   color=COLORS[sys_name], edgecolor="black", linewidth=0.7,
                   label=DISPLAY[sys_name], zorder=3)
        # Hollow markers: the bound evaluated at each cell's effective CV.
        if all("cv_eff" in r for r in recs):
            ax.scatter([r["cv"] for r in recs],
                       [r["cv_eff"] ** 2 for r in recs],
                       marker=MARKERS[sys_name], s=55, facecolors="none",
                       edgecolor=COLORS[sys_name], linewidth=1.2, zorder=2)
        for r in recs:
            offset = (5, -2) if r["tau_factor"] == 2.0 else (5, 5)
            ax.annotate(rf"$\tau{{=}}{r['tau_factor']:g}T_\lambda$",
                        (r["cv"], r["ratio_minus_one"]),
                        xytext=offset, textcoords="offset points",
                        fontsize=7, color=COLORS[sys_name], alpha=0.75)

    ax.set_yscale("log")
    ax.set_xlabel(r"$\mathrm{CV}(\hat\lambda_{\mathrm{FTLE}})$", fontsize=11)
    ax.set_ylabel(r"realized coverage gap $\bar{W}^\star/W_g^{\max} - 1$",
                  fontsize=11)
    ax.set_title("Corollary 3 verification across 4 systems × 3 horizons",
                 fontsize=11)
    ax.grid(True, which="both", alpha=0.25)
    ax.plot([], [], marker="o", ls="none", markerfacecolor="none",
            markeredgecolor="gray", label=r"$\mathrm{CV}_{\rm eff}^2$ per cell")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.95)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    fig.savefig(out.with_suffix(".pdf"))
    print(f"[plot_corollary3] wrote {out} and .pdf "
          f"({len(records)} points across {len({r['system'] for r in records})} systems)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corollary3_dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    records = load(args.corollary3_dir)
    if not records:
        raise SystemExit(f"No results_task*.json in {args.corollary3_dir}")
    plot(records, args.out)


if __name__ == "__main__":
    main()
