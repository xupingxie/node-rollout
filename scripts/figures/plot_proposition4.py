"""Generate fig_proposition4_robustness.png from Proposition 4 JSON.

Four-panel grid: realized coverage loss vs uniform error bound epsilon
on log-log axes, with the theoretical bound 2*eps*sum_i 1/L_i
overlaid. Feasibility range annotated in each panel title.

EXPECTED INPUT
--------------
Directory of `results_task{0,1,2,3}.json` files from
`scripts/test_proposition4.py` runs. The Proposition 4 outputs share
the naming convention with Corollary 3 outputs, distinguished by the
extra fields `sigma_eps`, `epsilon`, `coverage_loss_max`,
`theoretical_bound`, `feasibility_fraction_min`. In your project the
Proposition 4 files have suffix `__1_`; if that's the case, point this
script at the directory containing those files.

USAGE
-----
    python scripts/figures/plot_proposition4.py \
        --proposition4_dir results/proposition4 \
        --out plots/fig_proposition4_robustness.png
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


def load_by_system(d: Path):
    out = defaultdict(list)
    for p in sorted(d.glob("*.json")):
        if "gaussian" in p.name:          # stress test, reported separately
            continue
        with open(p) as f:
            recs = json.load(f)
        for r in recs:
            # Must have the Proposition-4-specific fields
            if r.get("noise_model") != "uniform" or "coverage_loss_max" not in r:
                continue
            out[r["system"]].append(r)
    return out


def plot(by_sys, out: Path):
    order = ["l63", "rossler", "l96_n40", "ks_l22"]
    available = [s for s in order if s in by_sys]
    if not available:
        raise SystemExit("No Proposition 4 records found in input dir")

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.0))
    for ax, sys_name in zip(axes.flat, available):
        recs = sorted(by_sys[sys_name], key=lambda r: r["epsilon"])
        sigmas = np.array([r["epsilon"] for r in recs])
        realized = np.array([r["coverage_loss_max"] for r in recs])
        bound = np.array([r["theoretical_bound"] for r in recs])
        feas_min = np.array([r["feasibility_fraction_min"] for r in recs])

        ax.plot(sigmas, bound, "--", color="gray", lw=1.5,
                label=r"theory bound $2\epsilon \sum_i 1/L_i$")
        ax.plot(sigmas, realized, "o-", color="#1f77b4", lw=1.8, ms=8,
                markeredgecolor="black", markeredgewidth=0.7,
                label="realized (max over realizations)")

        ax.set_xscale("log")
        ax.set_yscale("log")
        import matplotlib.ticker as _t
        ax.xaxis.set_minor_formatter(_t.NullFormatter())
        ax.xaxis.set_major_formatter(_t.LogFormatterMathtext())
        ax.set_xlabel(r"uniform error bound $\epsilon$", fontsize=10)
        ax.set_ylabel("coverage loss", fontsize=10)
        feas_pct = (feas_min.min() * 100, feas_min.max() * 100)
        ax.set_title(f"{DISPLAY.get(sys_name, sys_name)}    "
                     f"(feas. {feas_pct[0]:.1f}%–{feas_pct[1]:.1f}%)",
                     fontsize=10)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(loc="upper left", fontsize=8, framealpha=0.95)

    # Hide unused panels
    for ax in axes.flat[len(available):]:
        ax.set_visible(False)

    fig.suptitle("Proposition 4: bounded-noise proxy experiment",
                 fontsize=12, y=0.995)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"[plot_proposition4] wrote {out} and .pdf ({len(available)} systems)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--proposition4_dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    by_sys = load_by_system(args.proposition4_dir)
    plot(by_sys, args.out)


if __name__ == "__main__":
    main()
