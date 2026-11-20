"""Aggregate runs/<system>/<run>/seed*/summary.json into paper-ready summaries.

Outputs
-------
data/<system>_train_summary.json
    One record per (system, run), mean/std over seeds, in the schema the
    existing table scripts use (config, class, valid_time_mean/std,
    final_loss_mean/std, final_grad_norm_mean/std, n_seeds,
    wallclock_seconds) plus the new fields (valid time at threshold 0.5,
    peak gradient norm, realized Lambda_max, long-time stats).
results/lemma1/training_grad_log.json
    Records for `scripts/test_lemma1.py --mode logs`:
    system, method, W, K, peak_grad_norm, beta_estimate.
results/lemma1/segments_<system>.npz
    All per-segment (Lambda_i, grad_norm_i) pairs pooled over runs, seeds
    and logging steps, with a run label, for the Lemma 1 inequality plot.

    python scripts/aggregate_runs.py --runs_dir runs --out_data data --out_results results
"""
from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

CLASS = {"ss": "single_shoot", "const_g": "const_window_matched_budget",
         "sat": "adaptive", "ramp": "global_lyapunov", "mpnode": "mp_node"}


def ms(vals):
    v = np.array([x for x in vals if x is not None and np.isfinite(x)], float)
    if len(v) == 0:
        return float("nan"), float("nan")
    return float(v.mean()), float(v.std())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs_dir", default="runs", type=Path)
    ap.add_argument("--out_data", default="data", type=Path)
    ap.add_argument("--out_results", default="results", type=Path)
    args = ap.parse_args()

    by = defaultdict(list)
    dataset_ids = {}
    for p in sorted(args.runs_dir.glob("*/*/seed*/summary.json")):
        s = json.load(open(p))
        if not s.get("evaluation_complete") or not s.get("provenance"):
            raise ValueError(f"{p}: incomplete or legacy run; verify the job list before aggregation")
        provenance = s["provenance"]
        if json.loads((p.parent / "manifest.json").read_text()) != provenance:
            raise ValueError(f"{p}: manifest mismatch")
        identity = (provenance["protocol"], provenance["source_sha256"], provenance["dataset_sha256"])
        if s["system"] in dataset_ids and dataset_ids[s["system"]] != identity:
            raise ValueError(f"mixed dataset/protocol identities for {s['system']}")
        dataset_ids[s["system"]] = identity
        by[(s["system"], s["run_name"])].append((p.parent, s))

    per_system = defaultdict(list)
    grad_log = []
    seg_pool = defaultdict(lambda: defaultdict(list))
    for (system, run), items in by.items():
        S = [s for _, s in items]
        rec = {
            "config": run,
            "class": CLASS.get(S[0]["schedule"], S[0]["schedule"]),
            "schedule": S[0]["schedule"],
            "W_nominal": S[0]["W_nominal"],
            "beta": S[0]["beta"],
            "K": S[0]["K"],
            "n_seeds": len(S),
        }
        for key, out in [("valid_time_thr03_mean", "valid_time"),
                         ("valid_time_thr05_mean", "valid_time_thr05"),
                         ("val_valid_time_thr03_mean", "val_valid_time"),
                         ("final_loss", "final_loss"),
                         ("final_mse", "final_mse"),
                         ("final_grad_norm", "final_grad_norm"),
                         ("peak_grad_norm", "peak_grad_norm"),
                         ("final_Lambda_max", "Lambda_max"),
                         ("final_W_mean", "W_mean_final"),
                         ("marginal_mmd_mean", "marginal_mmd"),
                         ("psd_rel_l2_mean", "psd_rel_l2"),
                         ("lambda1_abs_err_mean", "lambda1_error"),
                         ("blowup_frac", "blowup_fraction"),
                         ("blowup_time_mean", "blowup_time"),
                         ("valid_time_thr03_censored_frac", "censored_frac"),
                         ("budget_frac_gt_1p1_after_warmup", "budget_exceed_frac"),
                         ("budget_max_ratio_after_warmup", "budget_max_ratio"),
                         ("stopped_step", "steps"),
                         ("integrated_model_steps", "integrated_steps"),
                         ("wmin_bound_segment_frac", "wmin_bound_frac")]:
            m, sd = ms([s.get(key) for s in S])
            rec[f"{out}_mean"] = m
            rec[f"{out}_std"] = sd
        rec["wallclock_seconds"] = ms([s.get("wallclock_train_s") for s in S])[0]
        rec["stopped_step_mean"] = ms([s.get("stopped_step") for s in S])[0]
        rec["best_step_mean"] = ms([s.get("best_step") for s in S])[0]
        rec["n_hit_ceiling"] = int(sum(1 for s in S if s.get("stopped_early") is False))
        per_system[system].append(rec)

        grad_log.append({
            "system": system, "method": run, "W": S[0]["W_nominal"] * S[0]["T_lambda"]
            if S[0]["schedule"] in ("ss", "ramp", "mpnode") else S[0]["final_W_mean"] * S[0]["T_lambda"]
            if "final_W_mean" in S[0] else float("nan"),
            "K": S[0]["K"],
            "peak_grad_norm": rec["peak_grad_norm_mean"],
            "beta_estimate": rec["Lambda_max_mean"],
        })

        for run_dir, s in items:
            for f in sorted(run_dir.glob("segments_*.npz")):
                z = np.load(f)
                seg_pool[system]["Lambda_i"].append(z["Lambda_i"])
                seg_pool[system]["grad_norm_i"].append(z["grad_norm_i"])
                seg_pool[system]["W_i"].append(z["W_i"])
                seg_pool[system]["run"].append(np.full(len(z["W_i"]), run))
                seg_pool[system]["step"].append(np.full(len(z["W_i"]), int(f.stem.split("_")[1])))

    # paired comparisons: saturating vs matched-budget constant window, per beta, same seeds
    def paired(system, a, b):
        A = {s_["seed"]: s_ for _, s_ in by.get((system, a), [])}
        B = {s_["seed"]: s_ for _, s_ in by.get((system, b), [])}
        seeds = sorted(set(A) & set(B))
        if len(seeds) < 2:
            return None
        d = np.array([A[k]["valid_time_thr03_mean"] - B[k]["valid_time_thr03_mean"] for k in seeds])
        from math import sqrt
        tcrit = {2: 12.71, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571}.get(len(seeds), 2.0)
        se = d.std(ddof=1) / sqrt(len(seeds))
        return {"a": a, "b": b, "n_pairs": len(seeds), "mean_diff": float(d.mean()),
                "ci95": [float(d.mean() - tcrit * se), float(d.mean() + tcrit * se)],
                "diffs": d.round(4).tolist()}
    pairs = {}
    for (system, run) in by:
        if run.startswith("sat_b"):
            beta = run.split("_")[1][1:]
            wmax = run[run.index("_Wmax"):] if "_Wmax" in run else ""     # same admissible window bounds
            pr = paired(system, run, f"const_g_b{beta}{wmax}")
            if pr:
                pairs.setdefault(system, []).append(pr)
    args.out_data.mkdir(parents=True, exist_ok=True)
    for system, prs in pairs.items():
        json.dump(prs, open(args.out_data / f"{system}_paired.json", "w"), indent=2)
        print(f"=== paired differences (valid time, T_lambda), {system}: saturating minus matched constant window")
        for pr in prs:
            print(f"  {pr['a']:34s} - {pr['b']:14s}: {pr['mean_diff']:+.3f}  95% CI [{pr['ci95'][0]:+.3f}, {pr['ci95'][1]:+.3f}]  n={pr['n_pairs']}")
    for system, recs in per_system.items():
        recs.sort(key=lambda r: (r["final_grad_norm_mean"] if np.isfinite(r["final_grad_norm_mean"]) else 1e99))
        out = args.out_data / f"{system}_train_summary.json"
        json.dump(recs, open(out, "w"), indent=2)
        print(f"wrote {out} ({len(recs)} configs)")
        print(f"{'config':34s} {'gn_final':>9s} {'vt03':>14s} {'cens':>5s} {'vt05':>14s} {'Lmax':>6s} {'exc>1.1':>7s} {'hours':>6s} {'best@':>7s} {'ceil':>4s}")
        for r in recs:
            print(f"{r['config']:34s} {r['final_grad_norm_mean']:9.2e} "
                  f"{r['valid_time_mean']:6.3f}±{r['valid_time_std']:<6.3f} {r.get('censored_frac_mean', float('nan')):5.2f} "
                  f"{r['valid_time_thr05_mean']:6.3f}±{r['valid_time_thr05_std']:<6.3f} "
                  f"{r['Lambda_max_mean']:6.2f} {r.get('budget_exceed_frac_mean', float('nan')):7.3f} "
                  f"{r['wallclock_seconds']/3600:6.2f} "
                  f"{r['best_step_mean']:7.0f} {r['n_hit_ceiling']:4d}/{r['n_seeds']}")

    (args.out_results / "lemma1").mkdir(parents=True, exist_ok=True)
    json.dump(grad_log, open(args.out_results / "lemma1" / "training_grad_log.json", "w"), indent=2)
    for system, d in seg_pool.items():
        np.savez_compressed(args.out_results / "lemma1" / f"segments_{system}.npz",
                            **{k: np.concatenate(v) for k, v in d.items()})
        print(f"pooled {sum(len(x) for x in d['Lambda_i'])} segments for {system}")


if __name__ == "__main__":
    main()
