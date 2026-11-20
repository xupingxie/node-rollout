"""Verify every expected run against a job list, its configuration, code and data.

Returns nonzero if any run is missing, incompatible or lacks evaluation.
"""
import argparse
import json
from pathlib import Path
import shlex
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import parser, config_from_args
from src.train.provenance import run_identity, file_sha256
from src.train.schedules_train import run_name


def check(jobfile):
    seen, hashes = set(), {}
    complete = total = 0
    for line in Path(jobfile).read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        total += 1
        args = parser().parse_args(shlex.split(line))
        cfg = config_from_args(args)
        directory = Path(args.out_dir) / cfg.data.system / run_name(cfg.schedule) / f"seed{args.seed}"
        if directory in seen:
            raise ValueError(f"duplicate run in job list: {directory}")
        seen.add(directory)
        try:
            dataset = Path(args.dataset_dir) / f"{args.dataset or cfg.data.system}.npz"
            if dataset not in hashes:
                hashes[dataset] = file_sha256(dataset)
            identity = run_identity(cfg, dataset, hashes[dataset])
            manifest = json.loads((directory / "manifest.json").read_text())
            summary = json.loads((directory / "summary.json").read_text())
            if manifest != identity or summary.get("provenance") != identity:
                raise ValueError("dataset/config/code identity mismatch")
            if not summary.get("training_complete") or not summary.get("evaluation_complete"):
                raise ValueError("training/evaluation incomplete")
            complete += 1
        except (OSError, ValueError) as exc:
            print(f"[INCOMPLETE] {directory}: {exc}")
    print(f"{jobfile}: {complete}/{total} verified complete")
    return total > 0 and complete == total


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("jobfile", type=Path)
    raise SystemExit(0 if check(ap.parse_args().jobfile) else 1)
