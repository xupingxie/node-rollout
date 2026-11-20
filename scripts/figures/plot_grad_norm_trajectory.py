"""Generate fig_grad_norm_trajectory.png: peak gradient norm over training
epochs for representative L63 configurations.

This figure shows the gradient norm trajectory during training across
four representative schedule classes: single-shoot at W=0.30, MP-NODE
at K=3, the global Lyapunov curriculum, and the saturating-budget
adaptive method (Phase 1 representative cell).

EXPECTED INPUT
--------------
For each configuration to plot, a JSON file with the per-epoch
gradient norm averaged over seeds, plus a band:

    {
      "config": "single_shoot_W=0.30",
      "epochs": [1, 2, 3, ..., 400],
      "grad_norm_mean": [12.3, 11.9, ..., 32.1],
      "grad_norm_std":  [1.1, 0.9, ..., 2.4],
      "n_seeds": 3
    }

Pass the JSON files via --inputs (one per configuration).

USAGE
-----
    python scripts/figures/plot_grad_norm_trajectory.py \
        --inputs logs/single_shoot.json logs/mp_node.json \
                 logs/global_lyapunov.json logs/adaptive.json \
        --out plots/fig_grad_norm_trajectory.png
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLASS_COLORS = {
    "single_shoot": "#888888",
    "mp_node": "#2ca02c",
    "global_lyapunov": "#1f77b4",
    "adaptive": "#d62728",
}


def infer_class(config: str) -> str:
    c = config.lower()
    if "single_shoot" in c or "shoot" in c:
        return "single_shoot"
    if "mp_node" in c or "mpnode" in c:
        return "mp_node"
    if "lyapunov" in c and "global" in c:
        return "global_lyapunov"
    return "adaptive"


def plot(curves: list[dict], out: Path):
    fig, ax = plt.subplots(figsize=(8.0, 4.5))

    for c in curves:
        epochs = np.asarray(c["epochs"])
        mean = np.asarray(c["grad_norm_mean"])
        std = np.asarray(c.get("grad_norm_std", np.zeros_like(mean)))
        cls = c.get("class") or infer_class(c["config"])
        color = CLASS_COLORS.get(cls, "black")
        style = "--" if cls == "adaptive" else "-"

        ax.plot(epochs, mean, style, color=color, lw=1.5,
                label=c["config"])
        ax.fill_between(epochs, mean - std, mean + std,
                        color=color, alpha=0.15)

    ax.set_yscale("log")
    ax.set_xlabel("Training epoch", fontsize=11)
    ax.set_ylabel(r"Gradient norm $\|\nabla_\theta \mathcal{L}\|$ (log scale)",
                  fontsize=11)
    ax.set_title("Gradient norm during L63 training\n"
                 "(mean over 3 seeds; shaded band $= \\pm 1\\sigma$)",
                 fontsize=11)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=9, framealpha=0.95)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"[plot_grad_norm_trajectory] wrote {out} and .pdf")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", required=True, nargs="+", type=Path,
                    help="One JSON file per configuration to overlay.")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    curves = []
    for p in args.inputs:
        with open(p) as f:
            curves.append(json.load(f))
    plot(curves, args.out)


if __name__ == "__main__":
    main()
