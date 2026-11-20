"""Exploratory association between per-segment gradient norm and the NOMINAL
teacher-side instability proxy Lambda_i = mu_i W_i.

This is NOT a verification of Lemma 1. The lemma bounds the gradient by
C exp(Lambda_i) with an unmeasured prefactor and with Lambda_i the learned
model's integrated instability over every subinterval; a fitted slope of a
binned percentile of log-gradient against a teacher proxy does not test that
pointwise inequality, and a slope above or below one neither refutes nor
confirms it. The figure shows how strongly gradients grow with the nominal
budget across the schedules that trained successfully.

Reads results/lemma1/segments_<system>.npz (from scripts/aggregate_runs.py) and
data/<system>_train_summary.json, and plots log ||grad L^(i)|| against
Lambda_i = mu_i W_i for segments logged during training.

Only runs that trained successfully are used (valid time >= --min_valid
T_lambda in the validation summary), because a diverged run produces enormous
gradients at every Lambda. Reported: the binned median and 95th percentile
of log-gradient against the nominal Lambda_i, and the least-squares slope
of the 95th-percentile curve, as descriptive statistics.

    python scripts/figures/plot_lemma1_segments.py --results_dir results --data_dir data \\
        --out figures/fig_lemma1_segments.png
"""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DISPLAY = {"l63": "Lorenz-63", "l96_n40": "Lorenz-96 ($N{=}40$)", "ks_l22": "Kuramoto--Sivashinsky"}
ORDER = ["l63", "l96_n40", "ks_l22"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_dir", type=Path, default=Path("results"))
    ap.add_argument("--data_dir", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min_valid", type=float, default=1.0, help="exclude runs below this validation valid time (T_lambda)")
    ap.add_argument("--n_bins", type=int, default=12)
    ap.add_argument("--min_per_bin", type=int, default=200)
    a = ap.parse_args()

    systems = [s for s in ORDER if (a.results_dir / "lemma1" / f"segments_{s}.npz").exists()]
    fig, axes = plt.subplots(1, len(systems), figsize=(4.4 * len(systems), 3.7), squeeze=False)
    report = {}
    for ax, system in zip(axes[0], systems):
        z = np.load(a.results_dir / "lemma1" / f"segments_{system}.npz", allow_pickle=True)
        L, g, run = z["Lambda_i"], z["grad_norm_i"], z["run"].astype(str)
        summ = {r["config"]: r for r in json.load(open(a.data_dir / f"{system}_train_summary.json"))}
        good_runs = {c for c, r in summ.items() if r.get("val_valid_time_mean", float("nan")) >= a.min_valid}
        ok = np.isfinite(L) & np.isfinite(g) & (g > 0) & np.isin(run, list(good_runs))
        L, lg = L[ok], np.log(g[ok])
        ax.scatter(L, lg, s=3, alpha=0.12, color="C0", rasterized=True)

        if not len(L):
            ax.set_title(DISPLAY.get(system, system))
            ax.text(0.5, 0.5, "No eligible validation runs", ha="center", transform=ax.transAxes)
            report[system] = dict(n_segments=0, n_runs=0, p95_slope=float("nan"), excluded_runs=sorted(summ))
            continue
        edges = np.linspace(0, L.max(), a.n_bins + 1)
        cen, p95, p50 = [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (L >= lo) & (L < hi)
            if m.sum() >= a.min_per_bin:
                cen.append(0.5 * (lo + hi)); p95.append(np.quantile(lg[m], 0.95)); p50.append(np.median(lg[m]))
        cen, p95, p50 = map(np.array, (cen, p95, p50))
        ax.plot(cen, p50, "s-", color="C1", ms=4, label="binned median")
        ax.plot(cen, p95, "o-", color="C3", ms=4, label="binned 95th pct")
        slope = float("nan")
        if len(cen) >= 3:
            A = np.vstack([cen, np.ones_like(cen)]).T
            slope, icpt = np.linalg.lstsq(A, p95, rcond=None)[0]
            xs = np.array([cen.min(), cen.max()])
            ax.plot(xs, slope * xs + icpt, "--", color="k", label=f"95th-pct slope {slope:.2f} (descriptive)")
        report[system] = dict(n_segments=int(ok.sum()), n_runs=len(set(run[ok])), p95_slope=slope,
                              excluded_runs=sorted(set(summ) - good_runs))
        ax.set_title(DISPLAY.get(system, system))
        ax.set_xlabel(r"nominal proxy budget $\hat\mu_i W_i$ (nats)")
        ax.set_ylabel(r"$\log \|\nabla_\theta \mathcal{L}^{(i)}\|$")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=180)
    fig.savefig(a.out.with_suffix(".pdf"))
    json.dump(report, open(a.out.with_suffix(".json"), "w"), indent=2)
    for s, r in report.items():
        print(f"{s:8s} segments {r['n_segments']:7d} from {r['n_runs']} run configs; 95th-pct slope {r['p95_slope']:.2f} (descriptive); "
              f"excluded {r['excluded_runs']}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
