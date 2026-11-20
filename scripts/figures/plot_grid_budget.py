"""Trained-model quality versus realized instability budget, one panel per system.

x: realized worst-case integrated instability Lambda_max (mean over seeds);
y: valid time (threshold 0.3) in T_lambda, mean +- std over seeds;
marker: schedule class. Same-budget pairs (adaptive vs constant window
at the same beta and cap) are joined by a thin line, so the comparison
Theorem 1 is about can be read off directly.

Legend labels follow the paper's terminology: "constant window" (fixed W),
"constant window, same budget" (W_g^max(beta)), "adaptive" (W_i^*(beta)).

    python scripts/figures/plot_grid_budget.py --data_dir data --out figures/fig_grid_budget.png
"""
import argparse
import json
import re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SYS = {"l63": "Lorenz-63", "l96_n40": "Lorenz-96 ($N{=}40$)", "ks_l22": "Kuramoto--Sivashinsky"}
STYLE = {
    "ss":      dict(marker="o", color="#4C72B0", label="constant window"),
    "const_g": dict(marker="s", color="#55A868", label="constant window, same budget"),
    "sat":     dict(marker="D", color="#C44E52", label="adaptive"),
    "ramp":    dict(marker="^", color="#8172B2", label="ramp"),
    "mpnode":  dict(marker="P", color="#CCB974", label="MP-NODE"),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_dir", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    files = [f for f in [a.data_dir / f"{s}_train_summary.json" for s in SYS] if f.exists()]
    fig, axes = plt.subplots(1, len(files), figsize=(4.6 * len(files), 3.8), squeeze=False)
    for ax, f in zip(axes[0], files):
        system = f.name.replace("_train_summary.json", "")
        recs = json.load(open(f))
        by = {r["config"]: r for r in recs}
        for r in recs:
            st = STYLE[r["schedule"]]
            capped = "Wmax0.5" in r["config"]
            ax.errorbar(r["Lambda_max_mean"], r["valid_time_mean"], yerr=r["valid_time_std"],
                        fmt="X" if "per_segment" in r["config"] else st["marker"], color=st["color"], mfc="white" if capped else st["color"],
                        ms=7, capsize=2, lw=1)
        # join same-budget pairs (adaptive vs constant window, same beta and cap)
        for r in recs:
            if r["schedule"] == "sat":
                cap = re.search(r"_Wmax[0-9.]+", r["config"])
                c = by.get(f"const_g_b{r['beta']:g}{cap.group(0) if cap else ''}")
                if c:
                    ax.plot([r["Lambda_max_mean"], c["Lambda_max_mean"]],
                            [r["valid_time_mean"], c["valid_time_mean"]], "-", color="gray", lw=0.8, alpha=0.6, zorder=0)
        ax.set_xscale("log")
        ax.set_xlabel(r"realized $\Lambda_{\max}$ (nats)")
        ax.set_ylabel(r"valid time / $T_\lambda$ (NRMSE 0.3)")
        ax.set_title(SYS[system])
        ax.grid(alpha=0.25, which="both")
    handles = [plt.Line2D([], [], marker=s["marker"], color=s["color"], ls="none", ms=7, label=s["label"])
               for s in STYLE.values()]
    handles.append(plt.Line2D([], [], marker="D", color=STYLE["sat"]["color"], mfc="white", ls="none", ms=7,
                              label=r"hollow: $W_{\max}=0.5T_\lambda$"))
    handles.append(plt.Line2D([], [], marker="X", color=STYLE["sat"]["color"], ls="none",
                              label="adaptive, unweighted"))
    axes[0][-1].legend(handles=handles, fontsize=7, loc="best")
    fig.tight_layout()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=180)
    fig.savefig(a.out.with_suffix(".pdf"))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
