"""Generate fig_lemma1_tangent.png from Lemma 1 (tangent mode) JSON.

Four-panel grid: mean endpoint stretch Lambda_end(W) across N=20 trajectories
per system, plotted against W/T_lambda, with the reference line
lambda_1_ref * W (which becomes the identity line in W/T_lambda
units since lambda_1 * T_lambda = 1).

EXPECTED INPUT
--------------
Directory of `tangent_task{0,1,2,3}.json` from
`scripts/test_lemma1.py --mode tangent`. Each record:
  - `system`            (string)
  - `trajectory_index`  (int)
  - `W_factor`          (float)
  - `Lambda_W`          (float)
  - `T_lambda`          (float)

USAGE
-----
    python scripts/figures/plot_lemma1_tangent.py \
        --lemma1_dir results/lemma1 \
        --out plots/fig_lemma1_tangent.png
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DISPLAY = {
    "l63": "Lorenz-63", "rossler": "Rössler",
    "l96_n40": "Lorenz-96 ($N{=}40$)", "ks_l22": "Kuramoto--Sivashinsky",
}
# CV at tau=T_lambda/2 for the four systems (from Corollary 3 sweep)
CV_AT_HALF = {"l63": 0.75, "rossler": 0.53, "l96_n40": 0.11, "ks_l22": 0.12}


def load_by_system(d: Path):
    out = defaultdict(list)
    for p in sorted(d.glob("tangent_task*.json")):
        with open(p) as f:
            recs = json.load(f)
        for r in recs:
            out[r["system"]].append(r)
    return out


def plot(by_sys, out: Path):
    order = ["l63", "rossler", "l96_n40", "ks_l22"]
    available = [s for s in order if s in by_sys]
    if not available:
        raise SystemExit("No tangent_task*.json found")

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.0))
    for ax, sys_name in zip(axes.flat, available):
        recs = by_sys[sys_name]
        Tl = recs[0]["T_lambda"]
        W_factors = sorted(set(r["W_factor"] for r in recs))
        means, stds = [], []
        for wf in W_factors:
            ls = np.array([r["Lambda_W"] for r in recs
                           if abs(r["W_factor"] - wf) < 1e-9])
            means.append(ls.mean())
            stds.append(ls.std())
        W_factors = np.array(W_factors)
        means = np.array(means)
        stds = np.array(stds)

        # Reference line: lambda_1_ref * W = W_factor
        wf_grid = np.linspace(0, max(W_factors) * 1.05, 100)
        ax.plot(wf_grid, wf_grid, "--", color="gray", lw=1.5,
                label=r"$\lambda_1^{\rm ref} W$ reference line", zorder=1)

        # Per-trajectory points (semi-transparent)
        for r in recs:
            ax.scatter(r["W_factor"], r["Lambda_W"],
                       color="#1f77b4", s=14, alpha=0.18, zorder=2)

        ax.errorbar(W_factors, means, yerr=stds, fmt="o-",
                    color="#1f77b4", lw=2.0, ms=8,
                    markeredgecolor="black", markeredgewidth=0.6,
                    capsize=4, label=r"$\overline{\Lambda^{\rm end}(W)} \pm \sigma$",
                    zorder=3)

        ax.set_xlabel(r"$W / T_\lambda$", fontsize=10)
        ax.set_ylabel(r"$\Lambda^{\rm end}(W)$", fontsize=10)
        cv = CV_AT_HALF.get(sys_name)
        title = DISPLAY.get(sys_name, sys_name)
        if cv is not None:
            title += f"    (CV = {cv:.2f})"
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper left", fontsize=9, framealpha=0.95)
        ax.axhline(0, color="black", lw=0.5, alpha=0.4)

    for ax in axes.flat[len(available):]:
        ax.set_visible(False)

    fig.suptitle(r"Endpoint stretch $\Lambda^{\rm end}(W)$ against the reference line $\lambda_1^{\rm ref} W$ "
                 r"across 4 systems, 20 trajectories each",
                 fontsize=12, y=0.995)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"[plot_lemma1_tangent] wrote {out} and .pdf ({len(available)} systems)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lemma1_dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    by_sys = load_by_system(args.lemma1_dir)
    plot(by_sys, args.out)


if __name__ == "__main__":
    main()
