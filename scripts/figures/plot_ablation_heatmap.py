"""Generate fig_ablation_heatmap.png from the L63 ablation summary JSON.

This is the 3x3 grid (bucketing x loss-normalization) of mean +/- std
valid time over 3 seeds, with the global Lyapunov baseline marked on
the colorbar. The 3x3 grid is the same one whose numerical values are
in Table 5 (`tab:l63_ablation_full`) of the paper.

EXPECTED INPUT
--------------
A JSON file with one record per (bucketing, loss_norm) cell:

    [
      {
        "config": "q4/inv_W",
        "bucketing": "q4",
        "loss_norm": "inv_W",
        "valid_time_mean": 0.121,
        "valid_time_std": 0.024,
        "n_seeds": 3
      },
      ...
    ]

Plus an optional record for the global Lyapunov baseline:

    {
        "config": "global_lyapunov_baseline",
        "valid_time_mean": 0.061,
        "valid_time_std": 0.012
    }

Bucketing levels (rows, top-to-bottom): "exact", "q16", "q4"
Loss-norm levels (cols, left-to-right): "global", "per_segment", "inv_W"

If you don't have this file but have the numbers from Table 5, you can
write it by hand in a few minutes.

USAGE
-----
    python scripts/figures/plot_ablation_heatmap.py \
        --ablation_summary results/l63_ablation/summary.json \
        --out plots/fig_ablation_heatmap_v2.png
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BUCKETING_ORDER = ["exact", "q16", "q4"]            # top-to-bottom in plot
LOSSNORM_ORDER = ["global", "per_segment", "inv_W"] # left-to-right
BUCKETING_LABELS = {
    "exact": "exact\n(per-segment)",
    "q16": "quantile (16)",
    "q4": "quantile (4)",
}
LOSSNORM_LABELS = {
    "global": "global\n(MSE / total)",
    "per_segment": "per_segment\n(mean per seg)",
    "inv_W": r"inv$\_$W (ours)",
}


def load_ablation(path: Path):
    """Return (3x3 mean, 3x3 std, baseline_mean)."""
    with open(path) as f:
        records = json.load(f)

    mean = np.full((3, 3), np.nan)
    std = np.full((3, 3), np.nan)
    baseline = None

    for r in records:
        cfg = r.get("config", "")
        if "baseline" in cfg.lower() or "lyapunov" in cfg.lower():
            baseline = r["valid_time_mean"]
            continue

        # Heatmap is only for the 3x3 adaptive grid; skip other records
        # (single_shoot, mp_node, etc.) that don't fit the bucketing x loss_norm grid.
        if r.get("class") and r["class"] not in ("adaptive", None):
            continue

        b = r.get("bucketing")
        ln = r.get("loss_norm")
        if b is None or ln is None:
            # Fall back to parsing config name; require "bucketing/loss_norm" form.
            parts = cfg.split("/")
            if len(parts) != 2:
                continue
            b, ln = parts
        # Normalize names that might appear differently in the summary
        b = b.replace("none", "exact")
        if b not in BUCKETING_ORDER or ln not in LOSSNORM_ORDER:
            continue
        i = BUCKETING_ORDER.index(b)
        j = LOSSNORM_ORDER.index(ln)
        mean[i, j] = r["valid_time_mean"]
        std[i, j] = r["valid_time_std"]

    return mean, std, baseline


def plot(mean, std, baseline, out: Path):
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    vmin = float(np.nanmin(mean)) if np.any(~np.isnan(mean)) else 0.0
    vmax = float(np.nanmax(mean)) if np.any(~np.isnan(mean)) else 1.0

    im = ax.imshow(mean, cmap="viridis", aspect="auto",
                   vmin=vmin, vmax=vmax)

    # Annotate cells: mean +/- std
    for i in range(3):
        for j in range(3):
            if np.isnan(mean[i, j]):
                txt = "—"
            else:
                txt = f"{mean[i,j]:.3f}\n$\\pm${std[i,j]:.3f}"
            ax.text(j, i, txt, ha="center", va="center",
                    color="white" if mean[i, j] < (vmin + vmax) / 2 else "black",
                    fontsize=10, fontweight="bold")

    # Star the winning cell
    if np.any(~np.isnan(mean)):
        i_star, j_star = np.unravel_index(np.nanargmax(mean), mean.shape)
        ax.scatter(j_star, i_star + 0.32, s=140, marker="*",
                   color="gold", edgecolor="black", linewidth=1.0, zorder=5)

    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels([LOSSNORM_LABELS[k] for k in LOSSNORM_ORDER], fontsize=10)
    ax.set_yticklabels([BUCKETING_LABELS[k] for k in BUCKETING_ORDER], fontsize=10)
    ax.set_xlabel("Loss normalization", fontsize=11)
    ax.set_ylabel("Bucketing of $W_i$", fontsize=11)
    ax.set_title("L63 adaptive-cell ablation: valid time / $T_\\lambda$\n"
                 "(mean $\\pm$ std over 3 seeds; baseline marked on colorbar)",
                 fontsize=10)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"valid time / $T_\lambda$", fontsize=10)
    if baseline is not None:
        cbar.ax.axhline(baseline, color="red", lw=1.5)
        cbar.ax.text(1.4, baseline, "global\nLyapunov\nbaseline",
                     transform=cbar.ax.get_yaxis_transform(),
                     fontsize=8, color="red", va="center")

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"[plot_ablation_heatmap] wrote {out} and .pdf")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ablation_summary", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    mean, std, baseline = load_ablation(args.ablation_summary)
    plot(mean, std, baseline, args.out)


if __name__ == "__main__":
    main()
