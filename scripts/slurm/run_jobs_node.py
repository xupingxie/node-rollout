"""Run one node's job-list shard with a bounded pool of slots per GPU.

Sharding uses original (one-based) line numbers; blank/comment lines are ignored.
A slot remains assigned to its GPU for the entire child lifetime, so short jobs
cannot cause extra processes to accumulate on a GPU running longer jobs.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import os
from pathlib import Path
import queue
import shlex
import socket
import subprocess
import sys
import time


def selected_jobs(jobfile, shard, shards):
    jobs = []
    for line, arguments in enumerate(Path(jobfile).read_text().splitlines(), 1):
        if arguments.strip() and not arguments.lstrip().startswith("#") and (line - 1) % shards == shard:
            jobs.append((line, arguments))
    return jobs


def gpu_slots(n_gpus, per_gpu):
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    devices = visible.split(",") if visible is not None else [str(i) for i in range(n_gpus)]
    if len(devices) < n_gpus or any(not d.strip() or d.strip() == "-1" for d in devices[:n_gpus]):
        raise ValueError(f"requested {n_gpus} GPUs but CUDA_VISIBLE_DEVICES={visible!r}")
    # Interleave GPUs when initially filling the pool.
    return [d.strip() for _ in range(per_gpu) for d in devices[:n_gpus]]


def run_pass(jobs, slots, log_dir, dry_run=False):
    pending = queue.Queue()
    for job in jobs:
        pending.put(job)

    def worker(gpu):
        outcomes = []
        while True:
            try:
                line, arguments = pending.get_nowait()
            except queue.Empty:
                return outcomes
            print(f"[{datetime.now():%H:%M:%S} {socket.gethostname()}] line {line} on GPU {gpu}: {arguments}", flush=True)
            if dry_run:
                outcomes.append((line, arguments, "done"))
                continue
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
            log_path = log_dir / f"line_{line}.out"
            with log_path.open("w") as log:
                child = subprocess.run([sys.executable, "scripts/train.py", *shlex.split(arguments)],
                                       env=env, stdout=log, stderr=subprocess.STDOUT)
            if child.returncode != 0:
                print(f"ERROR: line {line} exit {child.returncode}; see {log_path}", file=sys.stderr, flush=True)
                status = "failed"
            elif "is being trained by another process" in log_path.read_text():
                print(f"line {line}: locked elsewhere; will retry", flush=True)
                status = "locked"
            else:
                status = "done"
            outcomes.append((line, arguments, status))

    with ThreadPoolExecutor(max_workers=len(slots)) as pool:
        futures = [pool.submit(worker, gpu) for gpu in slots]
        return [item for future in futures for item in future.result()]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("jobfile", type=Path)
    ap.add_argument("n_gpus", type=int, nargs="?", default=4)
    ap.add_argument("per_gpu", type=int, nargs="?", default=3)
    ap.add_argument("shard", type=int, nargs="?", default=0)
    ap.add_argument("shards", type=int, nargs="?", default=1)
    args = ap.parse_args()
    if args.n_gpus < 1 or args.per_gpu < 1 or args.shards < 1 or not 0 <= args.shard < args.shards:
        raise ValueError("require positive GPU/process/shard counts and 0 <= shard < shards")
    if os.environ.get("CLEAR_LOCKS", "0") == "1":
        raise ValueError("CLEAR_LOCKS is disabled; wait for stale locks or inspect their owners")
    dry_run = os.environ.get("DRY_RUN", "0") == "1"
    if not dry_run:
        import torch  # fail before launching if the training environment is missing
    if not args.jobfile.is_file() or not args.jobfile.read_text().strip():
        raise ValueError(f"missing/empty job file: {args.jobfile}")
    jobs = selected_jobs(args.jobfile, args.shard, args.shards)
    slots = gpu_slots(args.n_gpus, args.per_gpu)
    max_passes = int(os.environ.get("MAX_PASSES", "48"))
    retry_wait = float(os.environ.get("RETRY_WAIT", "900"))
    if max_passes < 1 or retry_wait < 0:
        raise ValueError("MAX_PASSES must be positive and RETRY_WAIT nonnegative")
    launch = os.environ.get("SLURM_JOB_ID", f"local_{os.getpid()}")
    log_dir = Path("logs") / f"launch_{args.jobfile.stem}_{launch}_shard{args.shard}of{args.shards}"
    if not dry_run:
        log_dir.mkdir(parents=True, exist_ok=True)
    print(f"shard {args.shard}/{args.shards}: {len(jobs)} runs; {args.n_gpus} GPUs x "
          f"{args.per_gpu} slots; logs: {log_dir}", flush=True)
    for attempt in range(max_passes):
        outcomes = run_pass(jobs, slots, log_dir, dry_run)
        if any(status == "failed" for _, _, status in outcomes):
            return 1
        jobs = [(line, arguments) for line, arguments, status in outcomes if status == "locked"]
        if not jobs:
            print(f"shard {args.shard}/{args.shards}: {'dry run complete' if dry_run else 'finished'}", flush=True)
            return 0
        if attempt + 1 < max_passes:
            print(f"{len(jobs)} locked runs; retrying in {retry_wait:g}s", flush=True)
            time.sleep(retry_wait)
    print(f"ERROR: {len(jobs)} runs remain locked after {max_passes} passes", file=sys.stderr)
    return 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
