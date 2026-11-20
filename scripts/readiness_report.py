"""Print the readiness verdict for every ss_W0.5/seed0 run found.

    python scripts/readiness_report.py runs
"""
import json
import sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else "runs")
found = sorted(root.glob("*/ss_W0.5/seed0/summary.json"))
if not found:
    raise SystemExit(f"no */ss_W0.5/seed0/summary.json under {root}; run the readiness check first")
print("readiness uses VALIDATION-window valid time (model selection never touches the test set)")
print(f"{'system':10s} {'val t (0.3)':>14s} {'val t (0.5)':>14s} {'best step':>10s}  verdict")
ready = True
for f in found:
    s = json.loads(f.read_text())
    vt = s.get("val_valid_time_thr03_mean", float("nan"))
    ok = s.get("evaluation_complete", False) and vt > 1.0
    ready &= ok
    print(f"{s['system']:10s} {vt:14.3f} {s.get('val_valid_time_thr05_mean', float('nan')):14.3f} "
          f"{s.get('best_step', -1):10d}  "
          f"{'READY' if ok else 'NOT READY: inspect validation results; any revised config needs a fresh run root'}")
raise SystemExit(0 if ready else 1)
