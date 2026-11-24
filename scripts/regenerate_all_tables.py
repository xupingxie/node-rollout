"""Regenerate the LaTeX body of every table in the paper.

Writes one .tex file per table into the output directory. Each file
contains a `\\begin{tabular}` ... `\\end{tabular}` body that can be
\\input{}'d from the paper.

The Corollary 3 / Proposition 4 / Lemma 1 tables read from
results/{corollary3,proposition4,lemma1}/ produced by the theorem-
verification SLURM jobs (test_corollary3.py / test_proposition4.py /
test_lemma1.py). The training-derived tables read from data/*.json
(L63 ablation, KS pilot, longtime stats, L96 results).

USAGE
-----
    python scripts/regenerate_all_tables.py \\
        --results_dir results \\
        --data_dir data \\
        --out_dir tables

Skips tables whose inputs are missing. Prints a summary at the end.
"""
import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent / "tables"


def run(args, label):
    print(f"\n=== {label} ===")
    proc = subprocess.run([sys.executable] + args, check=False,
                          capture_output=True, text=True)
    ok = proc.returncode == 0
    if proc.stdout.strip():
        print(proc.stdout.strip())
    if not ok and proc.stderr.strip():
        print(proc.stderr.strip())
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results_dir", required=True, type=Path)
    ap.add_argument("--data_dir", required=True, type=Path)
    ap.add_argument("--out_dir", required=True, type=Path)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # Table 1: hetero (CV across systems)
    cor3 = args.results_dir / "corollary3"
    if cor3.exists():
        results["table1_hetero"] = run([
            str(SCRIPT_DIR / "table1_hetero.py"),
            "--corollary3_dir", str(cor3),
            "--out", str(args.out_dir / "table1_hetero.tex"),
        ], "Table 1 (tab:hetero)")
    else:
        results["table1_hetero"] = False
        print(f"\nSKIPPED table 1: {cor3} not found")

    # Table 2: Corollary 3 verification
    if cor3.exists():
        results["table2_cor3"] = run([
            str(SCRIPT_DIR / "table2_cor3.py"),
            "--corollary3_dir", str(cor3),
            "--out", str(args.out_dir / "table2_cor3.tex"),
        ], "Table 2 (tab:cor3)")
    else:
        results["table2_cor3"] = False

    # Table 3: Lemma 1
    lem = args.results_dir / "lemma1"
    if lem.exists() and cor3.exists():
        results["table3_lemma1"] = run([
            str(SCRIPT_DIR / "table3_lemma1.py"),
            "--lemma1_dir", str(lem),
            "--corollary3_dir", str(cor3),
            "--out", str(args.out_dir / "table3_lemma1.tex"),
        ], "Table 3 (tab:lemma1)")
    else:
        results["table3_lemma1"] = False
        print(f"\nSKIPPED table 3: need both {lem} and {cor3}")

    # Table 4: Proposition 4
    prop = args.results_dir / "proposition4"
    if prop.exists():
        results["table4_prop4"] = run([
            str(SCRIPT_DIR / "table4_prop4.py"),
            "--proposition4_dir", str(prop),
            "--out", str(args.out_dir / "table4_prop4.tex"),
        ], "Table 4 (tab:prop4)")
    else:
        results["table4_prop4"] = False
        print(f"\nSKIPPED table 4: {prop} not found")

    # Tables 5, 6, 7, 8, 9: from data/
    if args.data_dir.exists():
        ok = run([
            str(SCRIPT_DIR / "tables_training.py"),
            "--data_dir", str(args.data_dir),
            "--out_dir", str(args.out_dir),
        ], "Tables 5, 6, 7, 8, 9 (training summaries)")
        for k in ["table5_l63_headline", "table6_l63_longtime",
                  "table7_ks_pilot", "table8_l96",
                  "table9_l63_ablation_full"]:
            results[k] = ok
    else:
        for k in ["table5_l63_headline", "table6_l63_longtime",
                  "table7_ks_pilot", "table8_l96",
                  "table9_l63_ablation_full"]:
            results[k] = False
        print(f"\nSKIPPED tables 5-9: {args.data_dir} not found")

    # Summary
    print("\n" + "=" * 60)
    print("TABLE REGENERATION SUMMARY")
    print("=" * 60)
    for k, ok in results.items():
        print(f"  [{'OK ' if ok else 'SKIP'}] {k}")
    n_ok = sum(1 for v in results.values() if v)
    print(f"\nGenerated {n_ok}/{len(results)} tables into {args.out_dir}/")


if __name__ == "__main__":
    main()
