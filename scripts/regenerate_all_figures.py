"""Master script: regenerate every figure in the paper.

Calls each `plot_*.py` script in `scripts/figures/` in turn, reading
from a canonical directory layout under `results/` and `data/`, and
writing to `figures/`.

EXPECTED INPUT LAYOUT
---------------------
    results/
        corollary3/results_task{0..3}.json     (from test_corollary3.py)
        proposition4/results_task{0..3}.json   (from test_proposition4.py)
        lemma1/tangent_task{0..3}.json         (from test_lemma1.py --mode tangent)
        l63_ablation/summary.json              (L63 ablation training summary)
        ks_pilot/summary.json                  (KS training summary)
        longtime/
            reference.json                      (ground-truth L63 rollout)
            *.json                              (one per method, long rollouts)
        training_logs/
            *.json                              (per-method gradient-norm series)

OUTPUT
------
    figures/
        fig_corollary3_coverage_vs_cv.{png,pdf}
        fig_proposition4_robustness.{png,pdf}
        fig_lemma1_tangent.{png,pdf}
        fig_cross_verification.{png,pdf}
        fig_ablation_heatmap_v2.{png,pdf}
        fig_ablation_ranking_v2.{png,pdf}
        fig_longtime_marginals.{png,pdf}
        fig_longtime_psd.{png,pdf}
        fig_ks_pilot_bar.{png,pdf}
        fig_grad_norm_trajectory.{png,pdf}

USAGE
-----
    python scripts/regenerate_all_figures.py --results_dir results --out_dir figures

Or run individual scripts with their own arguments; see each
plot_*.py file for details.

The script SKIPS figures whose input data is not present, and reports
a summary at the end indicating what was generated and what was
skipped.
"""
import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent / "figures"


def run(args: list[str], description: str) -> bool:
    """Run a python subprocess; return True if it succeeded."""
    print(f"\n=== {description} ===")
    print("  ", " ".join(str(a) for a in args))
    try:
        proc = subprocess.run([sys.executable] + args, check=False,
                              capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"  FAILED (exit {proc.returncode})")
            if proc.stdout:
                print("  stdout:", proc.stdout.strip())
            if proc.stderr:
                print("  stderr:", proc.stderr.strip())
            return False
        if proc.stdout.strip():
            print(proc.stdout.strip())
        return True
    except FileNotFoundError as e:
        print(f"  SKIPPED: {e}")
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results_dir", required=True, type=Path,
                    help="Top-level results directory (see layout in docstring)")
    ap.add_argument("--data_dir", default=None, type=Path,
                    help="Long-time / reference data directory (default: results_dir/longtime)")
    ap.add_argument("--out_dir", required=True, type=Path,
                    help="Output directory for figures")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = args.data_dir or (args.results_dir / "longtime")

    results = {}

    # 1. Corollary 3
    cor3_dir = args.results_dir / "corollary3"
    if cor3_dir.exists():
        results["cor3"] = run([
            str(SCRIPT_DIR / "plot_corollary3.py"),
            "--corollary3_dir", str(cor3_dir),
            "--out", str(args.out_dir / "fig_corollary3_coverage_vs_cv.png"),
        ], "Corollary 3 figure")
    else:
        results["cor3"] = False
        print(f"\nSKIPPED Corollary 3: {cor3_dir} not found")

    # 2. Proposition 4
    prop4_dir = args.results_dir / "proposition4"
    if prop4_dir.exists():
        results["prop4"] = run([
            str(SCRIPT_DIR / "plot_proposition4.py"),
            "--proposition4_dir", str(prop4_dir),
            "--out", str(args.out_dir / "fig_proposition4_robustness.png"),
        ], "Proposition 4 figure")
    else:
        results["prop4"] = False
        print(f"\nSKIPPED Proposition 4: {prop4_dir} not found")

    # 3. Lemma 1 (tangent)
    lem1_dir = args.results_dir / "lemma1"
    if lem1_dir.exists():
        results["lemma1"] = run([
            str(SCRIPT_DIR / "plot_lemma1_tangent.py"),
            "--lemma1_dir", str(lem1_dir),
            "--out", str(args.out_dir / "fig_lemma1_tangent.png"),
        ], "Lemma 1 tangent figure")
    else:
        results["lemma1"] = False
        print(f"\nSKIPPED Lemma 1: {lem1_dir} not found")

    # 4. Cross-verification
    if cor3_dir.exists() and lem1_dir.exists():
        results["cross"] = run([
            str(SCRIPT_DIR / "plot_cross_verification.py"),
            "--corollary3_dir", str(cor3_dir),
            "--lemma1_dir", str(lem1_dir),
            "--out", str(args.out_dir / "fig_cross_verification.png"),
        ], "Cross-verification figure")
    else:
        results["cross"] = False
        print("\nSKIPPED cross-verification (need both cor3 and lemma1)")

    # 5. Ablation heatmap
    abl = args.results_dir / "l63_ablation" / "summary.json"
    if abl.exists():
        results["abl_heatmap"] = run([
            str(SCRIPT_DIR / "plot_ablation_heatmap.py"),
            "--ablation_summary", str(abl),
            "--out", str(args.out_dir / "fig_ablation_heatmap_v2.png"),
        ], "L63 ablation heatmap")
        results["abl_ranking"] = run([
            str(SCRIPT_DIR / "plot_ablation_ranking.py"),
            "--ablation_summary", str(abl),
            "--out", str(args.out_dir / "fig_ablation_ranking_v2.png"),
        ], "L63 ablation ranking")
    else:
        results["abl_heatmap"] = results["abl_ranking"] = False
        print(f"\nSKIPPED L63 ablation figures: {abl} not found")

    # 6. KS pilot bar
    ks = args.results_dir / "ks_pilot" / "summary.json"
    if ks.exists():
        results["ks_bar"] = run([
            str(SCRIPT_DIR / "plot_ks_pilot_bar.py"),
            "--ks_summary", str(ks),
            "--out", str(args.out_dir / "fig_ks_pilot_bar.png"),
        ], "KS pilot bar chart")
    else:
        results["ks_bar"] = False
        print(f"\nSKIPPED KS pilot figure: {ks} not found")

    # 7. Long-time marginals + PSD
    lt_ref = data_dir / "reference.json"
    lt_methods = sorted(p for p in data_dir.glob("*.json") if p != lt_ref) if data_dir.exists() else []
    if lt_ref.exists() and lt_methods:
        results["lt_marginals"] = run([
            str(SCRIPT_DIR / "plot_longtime_marginals.py"),
            "--reference", str(lt_ref),
            "--inputs", *[str(p) for p in lt_methods],
            "--out", str(args.out_dir / "fig_longtime_marginals.png"),
        ], "Long-time marginals")
        results["lt_psd"] = run([
            str(SCRIPT_DIR / "plot_longtime_psd.py"),
            "--reference", str(lt_ref),
            "--inputs", *[str(p) for p in lt_methods],
            "--out", str(args.out_dir / "fig_longtime_psd.png"),
        ], "Long-time PSD")
    else:
        results["lt_marginals"] = results["lt_psd"] = False
        print(f"\nSKIPPED long-time figures: need {lt_ref} and method JSONs")

    # 8. Gradient norm trajectory
    tlog_dir = args.results_dir / "training_logs"
    tlog_files = sorted(tlog_dir.glob("*.json")) if tlog_dir.exists() else []
    if tlog_files:
        results["grad_norm"] = run([
            str(SCRIPT_DIR / "plot_grad_norm_trajectory.py"),
            "--inputs", *[str(p) for p in tlog_files],
            "--out", str(args.out_dir / "fig_grad_norm_trajectory.png"),
        ], "Gradient-norm trajectory")
    else:
        results["grad_norm"] = False
        print(f"\nSKIPPED gradient-norm figure: {tlog_dir} not found")

    # Summary
    print("\n" + "=" * 60)
    print("REGENERATION SUMMARY")
    print("=" * 60)
    for k, ok in results.items():
        status = "OK " if ok else "SKIP"
        print(f"  [{status}] {k}")
    n_ok = sum(1 for v in results.values() if v)
    print(f"\nGenerated {n_ok}/{len(results)} figures into {args.out_dir}/")


if __name__ == "__main__":
    main()
