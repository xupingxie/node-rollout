"""Generate fig_ks_pilot_bar.png: bar chart of valid time / T_lambda
across methods on the Kuramoto--Sivashinsky pilot.

Shows MP-NODE, the saturating-budget adaptive single-rollout method,
the global Lyapunov curriculum, and short single-shoot at matched
horizon. Error bars are std over 5 seeds.

EXPECTED INPUT
--------------
A JSON file with one record per method:

    {
      "methods": [
        {
          "config": "MP-NODE (K=3)",
          "class": "mp_node",
          "valid_time_mean": 0.136,
          "valid_time_std": 0.011,
          "n_seeds": 5,
          "peak_grad_norm_mean": 0.04
        },
        ...
      ]
    }

USAGE
-----
    python scripts/figures/plot_ks_pilot_bar.py \
        --ks_summary results/ks_pilot/summary.json \
        --out plots/fig_ks_pilot_bar.png
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


def infer_class(name: str) -> str:
    c = name.lower()
    if "single" in c or "shoot" in c:
        return "single_shoot"
    if "mp_node" in c or "mpnode" in c:
        return "mp_node"
    if "lyapunov" in c and "global" in c:
        return "global_lyapunov"
    return "adaptive"


def plot(records, out: Path):
    records = sorted(records, key=lambda r: r["valid_time_mean"])
    n = len(records)
    fig, ax = plt.subplots(figsize=(8.0, max(3.5, 0.6 * n + 1.0)))

    y = np.arange(n)
    means = np.array([r["valid_time_mean"] for r in records])
    stds = np.array([r["valid_time_std"] for r in records])
    classes = [r.get("class") or infer_class(r["config"]) for r in records]
    colors = [CLASS_COLORS.get(c, "gray") for c in classes]
    labels = [r["config"] for r in records]

    bars = ax.barh(y, means, xerr=stds, color=colors,
                   edgecolor="black", linewidth=0.7,
                   error_kw=dict(ecolor="black", lw=0.8, capsize=4),
                   alpha=0.9)

    xmax = (means + stds).max() * 1.18
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(m + s + xmax * 0.01, i, f"{m:.3f} $\\pm$ {s:.3f}",
                va="center", fontsize=9)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel(r"valid time / $T_\lambda$ (mean $\pm$ std over seeds)",
                  fontsize=11)
    ax.set_title("Kuramoto--Sivashinsky pilot: valid time across methods",
                 fontsize=11)
    ax.set_xlim(0, xmax)
    ax.grid(True, axis="x", alpha=0.3)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"[plot_ks_pilot_bar] wrote {out} and .pdf")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ks_summary", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    with open(args.ks_summary) as f:
        data = json.load(f)
    records = data["methods"] if isinstance(data, dict) else data
    plot(records, args.out)


if __name__ == "__main__":
    main()
