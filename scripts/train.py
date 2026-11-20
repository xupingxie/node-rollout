"""Train one (system, schedule, seed) run and evaluate it.

Examples
--------
    # constant window 0.5 T_lambda
    python scripts/train.py --config configs/l63.yaml --schedule ss --W 0.5 --seed 0
    # the theorem's schedule and its matched-budget baseline at beta = 1
    python scripts/train.py --config configs/l63.yaml --schedule sat     --beta 1.0 --seed 0
    python scripts/train.py --config configs/l63.yaml --schedule const_g --beta 1.0 --seed 0
    # global-Lyapunov ramp 0.1 -> 1.0 T_lambda
    python scripts/train.py --config configs/l63.yaml --schedule ramp --W 1.0 --seed 0
    # MP-NODE, K = 3 chunks of a 1.0 T_lambda window
    python scripts/train.py --config configs/l63.yaml --schedule mpnode --W 1.0 --K 3 --seed 0

Any config field can be overridden with --set section.key=value, e.g.
    --set train.n_steps=2000 --set model.width=256 --set train.grad_clip=10

Output: runs/<system>/<run_name>/seed<seed>/
    config.yaml, log.jsonl, segments_*.npz, checkpoint_best/last/final.pt,
    rollouts_longtime.npz, summary.json

Concurrency: a run whose summary.json exists is skipped (--force to redo);
a run whose RUNNING.lock was refreshed within train.lock_stale_s seconds is
being trained by another process (e.g. an interactive session) and is
skipped too, so the same job list can be launched interactively and as a
queued batch job without the two colliding.
"""
from __future__ import annotations
import argparse
import json
import os
import socket
import sys
import time
import threading
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch

from src.train.config import RunConfig
from src.train.data import Dataset
from src.train.trainer import Trainer
from src.train.schedules_train import run_name
from src.train.evaluate import valid_time, long_time_stats
from src.train.provenance import run_identity, ensure_manifest, atomic_json


def parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--schedule", choices=["ss", "const_g", "sat", "ramp", "mpnode"])
    ap.add_argument("--W", type=float, help="nominal horizon in T_lambda (ss, ramp, mpnode)")
    ap.add_argument("--beta", type=float, help="budget in nats (sat, const_g)")
    ap.add_argument("--K", type=int, help="sub-windows (mpnode)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--set", nargs="*", action="append", default=[])
    ap.add_argument("--dataset_dir", default="datasets")
    ap.add_argument("--out_dir", default="runs")
    ap.add_argument("--skip_eval", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="delete the run directory (summary, checkpoints, logs) and train from scratch")
    ap.add_argument("--dataset", default=None,
                    help="dataset file name without .npz (default: the system name); e.g. l63_tau2")
    ap.add_argument("--ignore_lock", action="store_true",
                    help="take over a run whose RUNNING.lock is fresh (use only when its process is known dead)")
    return ap


def config_from_args(args):
    args.set = [x for grp in args.set for x in grp]   # several --set groups accumulate

    cfg = RunConfig.load(args.config, args.set)
    if args.schedule:
        cfg.schedule.name = args.schedule
    if args.W is not None:
        cfg.schedule.W = args.W
    if args.beta is not None:
        cfg.schedule.beta = args.beta
    if args.K is not None:
        cfg.schedule.K = args.K
    cfg.train.seed = args.seed

    return cfg


def main():
    args = parser().parse_args()
    cfg = config_from_args(args)
    ds_path = Path(args.dataset_dir) / f"{args.dataset or cfg.data.system}.npz"
    if not ds_path.exists():
        raise SystemExit(f"dataset {ds_path} missing; run scripts/make_dataset.py first")
    ds = Dataset.load(ds_path)
    # Refuse incompatible datasets: the estimator, FTLE horizon and test horizon
    # must be the ones the configuration asks for.
    problems = []
    if ds.system != cfg.data.system:
        problems.append(f"system: dataset={ds.system} config={cfg.data.system}")
    if ds.ftle_method != cfg.data.ftle_method:
        problems.append(f"ftle_method: dataset={ds.ftle_method} config={cfg.data.ftle_method}")
    if abs(ds.tau - cfg.data.tau_factor * ds.T_lambda) > 1e-6 * ds.T_lambda:
        problems.append(f"tau: dataset={ds.tau:.4f} config={cfg.data.tau_factor * ds.T_lambda:.4f}")
    n_h = ds.test.shape[1] - 1
    if abs(n_h * ds.dt - cfg.data.T_test * ds.T_lambda) > ds.dt:
        problems.append(f"T_test: dataset={n_h * ds.dt / ds.T_lambda:.2f} T_lambda config={cfg.data.T_test}")
    if problems:
        raise SystemExit("dataset/config mismatch, refusing to train:\n  " + "\n  ".join(problems)
                         + "\n  regenerate with scripts/make_dataset.py or adjust the config")
    identity = run_identity(cfg, ds_path)
    run_dir = Path(args.out_dir) / cfg.data.system / run_name(cfg.schedule) / f"seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    lock = run_dir / "RUNNING.lock"
    if not _acquire_lock(lock, cfg.train.lock_stale_s, args.ignore_lock):
        return
    stop_heartbeat = threading.Event()
    def heartbeat():
        while not stop_heartbeat.wait(30):
            lock.touch()
    worker = threading.Thread(target=heartbeat, daemon=True)
    worker.start()
    try:
        if args.force:
            for path in run_dir.iterdir():
                if path.name != "RUNNING.lock":
                    shutil.rmtree(path) if path.is_dir() else path.unlink()
            print(f"--force: cleared {run_dir} under its lock")
        ensure_manifest(run_dir, identity)
        summary_path = run_dir / "summary.json"
        if summary_path.exists():
            old = json.loads(summary_path.read_text())
            if old.get("provenance") != identity:
                raise ValueError("summary provenance mismatch")
            if old.get("evaluation_complete") or (args.skip_eval and old.get("training_complete")):
                print(f"{summary_path}: verified complete; nothing to do")
                return
        cfg.save(run_dir / "config.yaml")
        print(f"run dir: {run_dir}", flush=True)
        _run(cfg, ds, run_dir, args, ds_path, identity, n_h)
    finally:
        stop_heartbeat.set()
        worker.join()
        lock.unlink(missing_ok=True)


def _acquire_lock(lock: Path, stale_s: float, ignore: bool) -> bool:
    """Atomically create RUNNING.lock. Returns False (and prints why) if another
    live process holds it. A stale lock (older than stale_s) or --ignore_lock
    is taken over."""
    text = f"{socket.gethostname()} pid {os.getpid()} started {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    for _ in range(2):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w") as f:
                f.write(text)
            return True
        except FileExistsError:
            try:
                age = time.time() - lock.stat().st_mtime
                holder = lock.read_text().strip()
            except FileNotFoundError:
                continue                      # holder just finished; try again
            if age < stale_s and not ignore:
                print(f"{lock.parent} is being trained by another process "
                      f"({holder}; heartbeat {age:.0f}s ago); skipping")
                return False
            try:
                lock.unlink()                 # stale or explicitly ignored: take over
            except FileNotFoundError:
                pass
    print(f"{lock.parent}: could not acquire lock; skipping")
    return False


def _run(cfg, ds, run_dir, args, ds_path=None, identity=None, n_h=None):
    trainer = Trainer(cfg, ds, run_dir, provenance=identity)
    summary = {
        "system": cfg.data.system,
        "schedule": cfg.schedule.name,
        "run_name": run_name(cfg.schedule),
        "seed": args.seed,
        "W_nominal": cfg.schedule.W,
        "W_max": cfg.schedule.W_max, "loss_norm": cfg.schedule.loss_norm,
        "beta": cfg.schedule.beta if cfg.schedule.name in ("sat", "const_g") else None,
        "K": cfg.schedule.K if cfg.schedule.name == "mpnode" else 1,
        "T_lambda": ds.T_lambda,
        "dt": ds.dt,
        "n_params": sum(p.numel() for p in trainer.model.parameters()),
        "device": str(trainer.device),
        "dataset_file": str(ds_path), "provenance": identity,
        "dataset_ftle_method": ds.ftle_method, "dataset_tau": ds.tau,
        "dataset_test_horizon_T": n_h * ds.dt / ds.T_lambda,
    }
    summary.update(trainer.fit())
    summary["training_complete"] = True
    summary["evaluation_complete"] = False
    atomic_json(run_dir / "summary.json", summary)

    if not args.skip_eval:
        (run_dir / "RUNNING.lock").touch()        # evaluation can take a while; keep the lock fresh
        t0 = time.time()
        summary.update(valid_time(trainer.model, ds, cfg.eval.valid_thresholds))
        summary.update({f"val_{k}": v for k, v in
                        valid_time(trainer.model, ds, cfg.eval.valid_thresholds, split="val").items()
                        if "per_ic" not in k})
        summary.update(long_time_stats(
            trainer.model, ds, n_long=cfg.eval.n_long_rollouts, lyap_T=cfg.eval.lyap_T,
            psd_nperseg=cfg.eval.psd_nperseg, mmd_n=cfg.eval.mmd_n,
            save_rollouts=run_dir / "rollouts_longtime.npz"))
        summary["wallclock_eval_s"] = time.time() - t0

        summary["evaluation_complete"] = True

    atomic_json(run_dir / "summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if not isinstance(v, (list, dict))}, indent=1))


if __name__ == "__main__":
    main()
