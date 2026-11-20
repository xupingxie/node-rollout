"""Generate fig_ablation_ranking.png: 14 L63 configurations ranked by
final valid time.

This pairs with the heatmap figure but shows all configurations
together (5 baselines + 3x3 adaptive grid = 14 configs) so the reader
can see how the adaptive cells rank against single-shoot, MP-NODE, and
the global Lyapunov curriculum baselines.

EXPECTED INPUT
--------------
Same JSON schema as plot_ablation_heatmap.py, augmented with the 5
baseline records. Each record:

    {
      "config": "single_shoot_W=0.30",
      "class": "single_shoot",
      "valid_time_mean": 0.306,
      "valid_time_std": 0.018,
      "n_seeds": 3
    }

The 9 adaptive grid records (q4/inv_W etc.) should set class="adaptive".
The 5 baseline records use class in {"single_shoot", "mp_node",
"global_lyapunov"}. Optional `wallclock_seconds` field is plotted as a
secondary annotation if present.

USAGE
-----
    python scripts/figures/plot_ablation_ranking.py \
        --ablation_summary results/l63_ablation/summary.json \
        --out plots/fig_ablation_ranking_v2.png
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
    "global_lyapunov": "Single-rollout: global Lyapunov curric.",
    "adaptive": "Single-rollout: adaptive (proposed)",
}


def infer_class(config: str) -> str:
    """Fallback inference if 'class' field is absent."""
    c = config.lower()
    if "single_shoot" in c or "shoot" in c:
        return "single_shoot"
    if "mp_node" in c or "mpnode" in c:
        return "mp_node"
    if "lyapunov" in c and "global" in c:
        return "global_lyapunov"
    return "adaptive"


def load(path: Path):
    with open(path) as f:
        recs = json.load(f)
    for r in recs:
        if "class" not in r:
            r["class"] = infer_class(r.get("config", ""))
    return recs


def plot(records, out: Path, recommended_config: str | None = None):
    records = sorted(records, key=lambda r: r["valid_time_mean"])
    n = len(records)
    fig, ax = plt.subplots(figsize=(9.0, max(4.0, 0.34 * n + 1.0)))

    y = np.arange(n)
    means = np.array([r["valid_time_mean"] for r in records])
    stds = np.array([r["valid_time_std"] for r in records])
    colors = [CLASS_COLORS.get(r["class"], "gray") for r in records]
    labels = [r["config"] for r in records]

    bars = ax.barh(y, means, xerr=stds, color=colors,
                   edgecolor="black", linewidth=0.5,
                   error_kw=dict(ecolor="black", lw=0.8, capsize=3),
                   alpha=0.9)

    # Star the recommended adaptive cell
    if recommended_config is not None:
        for i, r in enumerate(records):
            if r["config"] == recommended_config:
                bars[i].set_edgecolor("red")
                bars[i].set_linewidth(2.0)

    # Annotate values to the right of each bar
    xmax = (means + stds).max() * 1.10
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(m + s + xmax * 0.01, i, f"{m:.3f}", va="center", fontsize=8)

    # Global Lyapunov baseline as a vertical dashed line
    for r in records:
        if r["class"] == "global_lyapunov":
            ax.axvline(r["valid_time_mean"], color="#1f77b4",
                       lw=1.0, linestyle=":", alpha=0.5,
                       label="global Lyapunov baseline")
            break

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel(r"valid time / $T_\lambda$ (final, mean over seeds)",
                  fontsize=11)
    ax.set_title("L63: 14 configurations ranked by final valid time",
                 fontsize=11)
    ax.set_xlim(0, xmax)
    ax.grid(True, axis="x", alpha=0.3)

    # Legend for the four classes
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=CLASS_COLORS[k], edgecolor="black",
                     label=CLASS_LABELS[k])
               for k in ["single_shoot", "mp_node",
                         "global_lyapunov", "adaptive"]
               if any(r["class"] == k for r in records)]
    ax.legend(handles=handles, loc="lower right", fontsize=8)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"[plot_ablation_ranking] wrote {out} and .pdf")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ablation_summary", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--recommended_config", default="q4/inv_W",
                    help="Config to highlight with red border (default q4/inv_W)")
    args = ap.parse_args()

    records = load(args.ablation_summary)
    plot(records, args.out, recommended_config=args.recommended_config)


if __name__ == "__main__":
    main()
