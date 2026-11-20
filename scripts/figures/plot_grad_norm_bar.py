"""Generate fig_grad_norm_bar.png: final gradient norm per method, on
L63 and KS, as a two-panel bar chart.

This is a REPLACEMENT for fig_grad_norm_trajectory.png. The original
trajectory figure showed gradient norm vs training epoch for 4 methods;
the per-epoch time series was not saved during the original training,
so it cannot be regenerated. This bar chart uses the FINAL gradient
norms (one number per method) which ARE preserved in
data/l63_ablation_summary.json and data/ks_pilot_summary.json. It
makes the same scientific point: schedule classes operate at
systematically different effective beta, manifesting as systematically
different gradient norms.

EXPECTED INPUTS
---------------
--l63_summary : data/l63_ablation_summary.json
--ks_summary  : data/ks_pilot_summary.json

USAGE
-----
    python scripts/figures/plot_grad_norm_bar.py \\
        --l63_summary data/l63_ablation_summary.json \\
        --ks_summary  data/ks_pilot_summary.json \\
        --out figures/fig_grad_norm_bar.png
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
CLASS_LABELS = {
    "single_shoot": "Single-rollout (single-shoot)",
    "mp_node": "Multi-window (MP-NODE)",
    "global_lyapunov": "Single-rollout (global Lyapunov)",
    "adaptive": "Single-rollout (adaptive, ours)",
}


def load_l63(path: Path):
    """Return list of (label, class, grad_mean, grad_std)."""
    with open(path) as f:
        records = json.load(f)
    # Pick the headline 4 configs that appear in Figure 3 of the paper
    wanted = [
        ("single_shoot_W=0.30", "Single-shoot W=0.30"),
        ("MP-NODE (3x0.30)",    "MP-NODE 3x0.30"),
        ("global_lyapunov_baseline", "Global Lyapunov"),
        ("q4/inv_W",            "Adaptive q4/inv_W"),
    ]
    out = []
    by_cfg = {r["config"]: r for r in records}
    for key, label in wanted:
        if key not in by_cfg:
            continue
        r = by_cfg[key]
        out.append((label, r["class"],
                    r.get("final_grad_norm_mean"),
                    r.get("final_grad_norm_std", 0.0)))
    return out


def load_ks(path: Path):
    """KS uses the slightly different schema from ks_pilot_summary.json."""
    with open(path) as f:
        records = json.load(f)
    name_map = {
        "ks_baseline_single_W2_v2":       ("Single-shoot W=2",  "single_shoot"),
        "ks_baseline_mpnode_v2":          ("MP-NODE 3xW",       "mp_node"),
        "ks_baseline_global_v2":          ("Global Lyapunov",   "global_lyapunov"),
        "ks_adaptive_q4_invW_v2":         ("Adaptive q4/inv_W", "adaptive"),
    }
    out = []
    by_cfg = {r["config_name"]: r for r in records}
    for key, (label, cls) in name_map.items():
        if key in by_cfg:
            r = by_cfg[key]
            # ks_pilot has no std for grad_norm — use 0
            out.append((label, cls, r["final_grad_norm_mean"], 0.0))
    return out


def plot_panel(ax, data, title):
    n = len(data)
    y = np.arange(n)
    means = np.array([d[2] for d in data])
    stds = np.array([d[3] if d[3] is not None else 0 for d in data])
    colors = [CLASS_COLORS.get(d[1], "gray") for d in data]
    labels = [d[0] for d in data]

    bars = ax.barh(y, means, xerr=stds, color=colors,
                   edgecolor="black", linewidth=0.7,
                   error_kw=dict(ecolor="black", lw=0.8, capsize=4),
                   alpha=0.92)
    xmax = (means + stds).max() * 1.4
    for i, (m, s) in enumerate(zip(means, stds)):
        annot = f"{m:.2f}" + (f" ± {s:.2f}" if s > 0 else "")
        ax.text(m + max(s, xmax * 0.005), i, annot,
                va="center", fontsize=8.5)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel(r"final gradient norm $\|\nabla_\theta\mathcal{L}\|$ (log scale)",
                  fontsize=10)
    ax.set_xscale("log")
    ax.set_title(title, fontsize=11)
    ax.grid(True, axis="x", which="both", alpha=0.3)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--l63_summary", required=True, type=Path)
    ap.add_argument("--ks_summary",  required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    l63 = load_l63(args.l63_summary)
    ks = load_ks(args.ks_summary)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.6))
    plot_panel(axes[0], l63, "Lorenz-63")
    plot_panel(axes[1], ks,  "Kuramoto--Sivashinsky")

    # Shared legend at top
    from matplotlib.patches import Patch
    classes_used = set(d[1] for d in l63) | set(d[1] for d in ks)
    handles = [Patch(facecolor=CLASS_COLORS[k], edgecolor="black",
                     label=CLASS_LABELS[k])
               for k in ["single_shoot", "mp_node",
                         "global_lyapunov", "adaptive"]
               if k in classes_used]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               fontsize=8, frameon=True, bbox_to_anchor=(0.5, -0.05))

    fig.suptitle("Final gradient norm across schedule classes "
                 "(Lemma 1 mode (B))", fontsize=11, y=1.02)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180, bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"[plot_grad_norm_bar] wrote {args.out} and .pdf")


if __name__ == "__main__":
    main()
