"""Progress of every run of a grid, read from the run directories themselves
(log.jsonl, RUNNING.lock, summary.json), never from launcher console logs.

For running runs: current step, steps/s (from the last ~50 log records),
elapsed time since the process started, and an ETA range: time to
train.min_steps (earliest possible early stop) and to train.n_steps (ceiling).

    python scripts/grid_status.py runs/ks_l22
    python scripts/grid_status.py runs/l63 --only running
    python scripts/grid_status.py runs --all        # every system under runs/
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import yaml


def fmt_h(sec):
    if sec is None or sec != sec:
        return "  ?  "
    return f"{sec/3600:5.1f}h" if sec >= 3600 else f"{sec/60:4.0f}m"


def read_log(log):
    recs = []
    try:
        for line in log.read_text().splitlines()[-400:]:
            try:
                recs.append(json.loads(line))
            except Exception:
                pass
    except FileNotFoundError:
        pass
    return recs


def rate_from(recs):
    """steps/s from log timestamps (field 't'), using only the records after the
    last long pause (a resume after an interruption leaves a gap of hours in
    the timestamps, which would otherwise make the rate look tiny)."""
    ts = [(r["step"], r["t"]) for r in recs if "t" in r]
    if len(ts) < 2:
        return None
    gaps = [b[1] - a[1] for a, b in zip(ts[:-1], ts[1:])]
    typical = sorted(g for g in gaps if g > 0)
    if not typical:
        return None
    med = typical[len(typical) // 2]
    start = 0
    for i, g in enumerate(gaps):
        if g > 20 * med:          # a pause much longer than the logging interval
            start = i + 1
    seg = ts[start:]
    if len(seg) >= 2 and seg[-1][1] > seg[0][1]:
        return (seg[-1][0] - seg[0][0]) / (seg[-1][1] - seg[0][1])
    return None


def one_run(d, stale_s):
    name = f"{d.parent.name}/{d.name}"
    summ, lock, log = d / "summary.json", d / "RUNNING.lock", d / "log.jsonl"
    try:
        cfg = yaml.safe_load(open(d / "config.yaml"))["train"]
        min_steps, n_steps = cfg.get("min_steps", 0), cfg.get("n_steps", 0)
    except Exception:
        min_steps, n_steps = 0, 0
    recs = read_log(log)
    last = recs[-1] if recs else {}
    step = last.get("step", 0)
    lastv = next((r for r in reversed(recs) if "best_val" in r), {})   # last validation record
    if summ.exists():
        s = json.load(open(summ))
        return ("finished", name,
                f"valid {s.get('valid_time_thr03_mean', float('nan')):.3f}  "
                f"stopped {s.get('stopped_step')} ({'early' if s.get('stopped_early') else 'ceiling'})  "
                f"blowup {s.get('blowup_frac', float('nan')):.2f}  "
                f"train {fmt_h(s.get('wallclock_train_s'))}")
    if lock.exists():
        age = time.time() - lock.stat().st_mtime
        # a live run refreshes its lock every 50 steps; a heartbeat older than
        # 5 min almost always means the process died (session ended).
        status = "running" if age < 300 else ("dead?" if age < stale_s else "stale")
        try:
            txt = lock.read_text().split()
            host = txt[0]
            started = datetime.strptime(" ".join(txt[-2:]), "%Y-%m-%d %H:%M:%S").timestamp()
            elapsed = time.time() - started
        except Exception:
            host, elapsed = "?", None
        r = rate_from(recs)
        if r is None and elapsed:
            r = step / elapsed          # fallback: average since start (wrong after a resume)
        eta_min = (max(min_steps - step, 0) / r) if r else None
        eta_max = (max(n_steps - step, 0) / r) if r else None
        return (status, name,
                f"step {step:>6d}/{n_steps}  {r or float('nan'):5.2f} steps/s  "
                f"elapsed {fmt_h(elapsed)}  ETA {fmt_h(eta_min)} to {fmt_h(eta_max)}  "
                f"best {lastv.get('best_val', float('nan')):.2e}@{lastv.get('best_step', '?')} "
                f"bad {lastv.get('bad_vals', '?')}  loss {last.get('loss_train', float('nan')):.2e}  "
                f"heartbeat {age/60:.0f}m ago  {host}")
    if log.exists():
        return ("interrupted", name,
                f"step {step:>6d}/{n_steps}  loss {last.get('loss_train', float('nan')):.2e}  "
                f"(no lock; resumes from checkpoint when launched)")
    return ("unstarted", name, "")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("--only", choices=["running", "dead?", "stale", "finished", "unstarted", "interrupted"])
    ap.add_argument("--stale_s", type=float, default=1800)
    ap.add_argument("--all", action="store_true", help="root contains several systems")
    a = ap.parse_args()
    roots = sorted(p for p in a.root.iterdir() if p.is_dir()) if a.all else [a.root]
    for root in roots:
        rows = [one_run(d, a.stale_s) for d in sorted(p for p in root.glob("*/seed*") if p.is_dir())]
        if a.only:
            rows = [r for r in rows if r[0] == a.only]
        counts = {}
        for st, _, _ in rows:
            counts[st] = counts.get(st, 0) + 1
        print(f"=== {root}")
        for st, name, detail in rows:
            print(f"{st:12s} {name:34s} {detail}")
        print("---", ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())))


if __name__ == "__main__":
    main()
