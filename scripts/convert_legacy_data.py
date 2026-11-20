"""Convert your existing training summary files into the schema that
scripts/figures/plot_*.py expects.

You already have these from earlier training work:

    ks_pilot_summary.json         (KS pilot, 7 methods, 5 seeds each)
    longtime_stats_summary.json   (L63 long-time stats, 5 methods)

This script reads them and writes:

    results/ks_pilot/summary.json           (for plot_ks_pilot_bar.py)
    results/longtime_stats/summary.json     (for the long-time stats table,
                                             not currently a figure)

It does NOT regenerate the long-time marginal-density or PSD figures
(plot_longtime_marginals.py / plot_longtime_psd.py): those need the
raw rollout trajectories (K, T, 3) arrays, not the summary statistics.
If you only have the PNG figures from earlier (longtime_marginals.png,
longtime_psd.png), use those directly --- they're already correct.

USAGE
-----
    python scripts/convert_legacy_data.py \\
        --ks_in  /path/to/ks_pilot_summary.json \\
        --lt_in  /path/to/longtime_stats_summary.json \\
        --out_root results

This creates results/ks_pilot/summary.json and
results/longtime_stats/summary.json.

After running, regenerate_all_figures.py will pick up the KS pilot
data and produce fig_ks_pilot_bar.png.
"""
import argparse
import json
from pathlib import Path


def infer_class_ks(record: dict) -> str:
    """Map the legacy `method` field to the plotter's class taxonomy.

    Order matters: prefer the explicit `method` field over substring
    matches on the config name (otherwise 'adaptive_q4_global' picks
    up 'global' instead of 'adaptive').
    """
    m = record.get("method", "").lower()
    if m == "mpnode":
        return "mp_node"
    if m == "single":
        return "single_shoot"
    if m == "global":
        return "global_lyapunov"
    if m == "adaptive":
        return "adaptive"

    # Fall back to substring search on config_name only if method is empty
    name = record.get("config_name", "").lower()
    if "mpnode" in name:
        return "mp_node"
    if "single" in name:
        return "single_shoot"
    if "global" in name and "adaptive" not in name:
        return "global_lyapunov"
    return "adaptive"


def pretty_config_name_ks(record: dict) -> str:
    """Make a human-readable label from config_name."""
    name = record["config_name"]
    # Strip ks_ prefix and _v2 suffix; convert underscores to friendlier form
    name = name.removeprefix("ks_").removesuffix("_v2")
    return name


def convert_ks(in_path: Path, out_path: Path) -> None:
    with open(in_path) as f:
        legacy = json.load(f)

    methods = []
    for r in legacy:
        rec = {
            "config": pretty_config_name_ks(r),
            "class": infer_class_ks(r),
            "valid_time_mean": r["valid_time_over_Tlambda_mean"],
            "valid_time_std": r["valid_time_over_Tlambda_std"],
            "n_seeds": r.get("n_seeds", 5),
        }
        # Optional extras useful for cross-checking
        if "final_grad_norm_mean" in r:
            rec["peak_grad_norm_mean"] = r["final_grad_norm_mean"]
        if "wallclock_mean" in r:
            rec["wallclock_seconds"] = r["wallclock_mean"]
        methods.append(rec)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"methods": methods}, f, indent=2)
    print(f"[convert_ks] wrote {len(methods)} methods to {out_path}")


def convert_longtime(in_path: Path, out_path: Path) -> None:
    """Pass-through copy for downstream consumers; not a figure input here."""
    with open(in_path) as f:
        data = json.load(f)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[convert_longtime] copied {len(data)} records to {out_path}")
    print("  Note: this is summary statistics, not rollout trajectories.")
    print("  plot_longtime_marginals.py / plot_longtime_psd.py need raw "
          "rollouts and CANNOT be regenerated from this file alone.")
    print("  Use your existing PNGs from earlier work instead.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ks_in", type=Path, default=None,
                    help="Path to ks_pilot_summary.json")
    ap.add_argument("--lt_in", type=Path, default=None,
                    help="Path to longtime_stats_summary.json")
    ap.add_argument("--out_root", required=True, type=Path,
                    help="Root of the results/ directory tree")
    args = ap.parse_args()

    if args.ks_in:
        convert_ks(args.ks_in, args.out_root / "ks_pilot" / "summary.json")
    if args.lt_in:
        convert_longtime(args.lt_in,
                         args.out_root / "longtime_stats" / "summary.json")


if __name__ == "__main__":
    main()
