"""Re-run the evaluation of finished runs from checkpoint_best.pt, no retraining.

Use after changing src/train/evaluate.py (e.g. the blowup-time metrics) so
that existing summaries gain the new fields. Training-side fields in
summary.json are kept; evaluation fields are overwritten.

    python scripts/reevaluate.py runs/ks_l22            # every seed of every run under it
    python scripts/reevaluate.py runs/ks_l22/ss_W0.5/seed0
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
from src.train.config import RunConfig
from src.train.data import Dataset
from src.train.trainer import build_model, pick_device
from src.train.evaluate import valid_time, long_time_stats

root = Path(sys.argv[1] if len(sys.argv) > 1 else "runs")
dataset_dir = Path(sys.argv[2] if len(sys.argv) > 2 else "datasets")
run_dirs = [root] if (root / "summary.json").exists() else sorted(p.parent for p in root.glob("**/summary.json"))
device = pick_device("auto")
ds_cache = {}
for rd in run_dirs:
    cfg = RunConfig.load(rd / "config.yaml")
    ds = ds_cache.setdefault(cfg.data.system, Dataset.load(dataset_dir / f"{cfg.data.system}.npz"))
    ck = rd / "checkpoint_best.pt" if (rd / "checkpoint_best.pt").exists() else rd / "checkpoint_final.pt"
    model = build_model(cfg, ds, device)
    model.load_state_dict(torch.load(ck, map_location=device))
    s = json.load(open(rd / "summary.json"))
    t0 = time.time()
    s.update(valid_time(model, ds, cfg.eval.valid_thresholds))
    s.update(long_time_stats(model, ds, n_long=cfg.eval.n_long_rollouts, lyap_T=cfg.eval.lyap_T,
                             psd_nperseg=cfg.eval.psd_nperseg, mmd_n=cfg.eval.mmd_n,
                             save_rollouts=rd / "rollouts_longtime.npz"))
    s["wallclock_eval_s"] = time.time() - t0
    s["reevaluated"] = True
    json.dump(s, open(rd / "summary.json", "w"), indent=2)
    print(f"{rd}: valid {s['valid_time_thr03_mean']:.3f}  blowup_frac {s['blowup_frac']:.2f}  "
          f"blowup_time {s['blowup_time_mean']:.1f}/{s['T_long']:.0f} T_lambda  "
          f"lambda1 {s['lambda1_model_mean']:.3f}  ({time.time()-t0:.0f}s)")
