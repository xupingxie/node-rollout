"""Generate fig_longtime_marginals.png: coordinate marginal densities of
L63 long-time rollouts from trained models, compared against ground truth.

Three panels (x, y, z) each showing the kernel density estimate of the
coordinate over ~200 T_lambda of rollout, for each method evaluated.

EXPECTED INPUT
--------------
Per method: a JSON file containing concatenated long-time rollouts
shaped (K, T, 3) for K seeds, plus a config name.

    {
      "config": "MP-NODE (3x0.30)",
      "class": "mp_node",
      "rollouts": [
          [[x0,y0,z0], [x1,y1,z1], ...],   # seed 0, T points
          [[x0,y0,z0], ...],               # seed 1
          ...
      ]
    }

The reference trajectory (ground-truth L63) is a separate JSON of the
same shape but is treated as truth (filled gray, the others as colored
lines).

USAGE
-----
    python scripts/figures/plot_longtime_marginals.py \
        --reference data/l63_reference.json \
        --inputs logs/single_shoot.json logs/mp_node.json \
                 logs/global_lyapunov.json logs/adaptive.json \
        --out plots/fig_longtime_marginals.png
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


CLASS_COLORS = {
    "reference": "#333333",
    "single_shoot": "#888888",
    "mp_node": "#2ca02c",
    "global_lyapunov": "#1f77b4",
    "adaptive": "#d62728",
}


def infer_class(name: str) -> str:
    c = name.lower()
    if "single" in c or "shoot" in c:
        return "single_shoot"
    if "mp_node" in c or "mpnode" in c:
        return "mp_node"
    if "lyapunov" in c and "global" in c:
        return "global_lyapunov"
    if "ground" in c or "true" in c or "reference" in c:
        return "reference"
    return "adaptive"


def load(path: Path) -> dict:
    with open(path) as f:
        rec = json.load(f)
    arr = np.asarray(rec["rollouts"])  # (K, T, 3)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected (K,T,3) rollouts in {path}, got {arr.shape}")
    rec["arr"] = arr
    rec.setdefault("class", infer_class(rec.get("config", "")))
    return rec


def plot(reference: dict, curves: list[dict], out: Path):
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6), sharey=False)
    axis_names = ["x", "y", "z"]

    for j, name in enumerate(axis_names):
        ax = axes[j]
        ref_vals = reference["arr"][..., j].ravel()
        rmin, rmax = float(ref_vals.min()), float(ref_vals.max())
        pad = 0.1 * (rmax - rmin)
        grid = np.linspace(rmin - pad, rmax + pad, 400)

        # Reference as filled gray
        try:
            kde_ref = gaussian_kde(ref_vals)(grid)
            ax.fill_between(grid, 0, kde_ref, color="lightgray",
                            alpha=0.7, label="True L63" if j == 0 else None)
        except Exception:
            pass

        # Each method as a colored line
        for c in curves:
            vals = c["arr"][..., j].ravel()
            try:
                kde = gaussian_kde(vals)(grid)
            except Exception:
                continue
            color = CLASS_COLORS.get(c["class"], "black")
            label = c["config"] if j == 0 else None
            ax.plot(grid, kde, color=color, lw=1.5, label=label)

        ax.set_xlabel(name, fontsize=11)
        if j == 0:
            ax.set_ylabel("density", fontsize=11)
        ax.grid(True, alpha=0.2)

    axes[0].legend(loc="upper right", fontsize=8, framealpha=0.95)
    fig.suptitle(r"L63 long-time coordinate marginals "
                 r"($\sim$200 $T_\lambda$ rollout)", fontsize=11)
    fig.tight_layout()

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"[plot_longtime_marginals] wrote {out} and .pdf")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference", required=True, type=Path)
    ap.add_argument("--inputs", required=True, nargs="+", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    ref = load(args.reference)
    curves = [load(p) for p in args.inputs]
    plot(ref, curves, args.out)


if __name__ == "__main__":
    main()
