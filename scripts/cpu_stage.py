"""Restartable CPU stage. Completion requires validated, fingerprinted outputs."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.train.provenance import atomic_json, file_sha256

SYSTEMS = ("l63", "rossler", "l96_n40", "ks_l22")
TAUS = ("0.5", "1.0", "2.0")


def numerical_source(source):
    """Launcher-only updates do not change CPU numerical experiments.

    Also projects old v12r1 broad fingerprints onto the same keys, so completed
    datasets/direct experiments survive this operational update. Commands and
    every remaining source hash must still match.
    """
    return {key: value for key, value in source.items()
            if not key.startswith("scripts/slurm/") and key != "scripts/cpu_stage.py"}


def job_identity(identity):
    return dict(identity, source_identity=numerical_source(identity["source_identity"]))


def validate_output(path):
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as z:
            if not z.files:
                raise ValueError(f"empty archive: {path}")
            for key in z.files:
                z[key]  # force decompression and CRC checks on every member
    else:
        records = json.loads(path.read_text())
        if not isinstance(records, list) or not records:
            raise ValueError(f"empty/invalid records: {path}")
        if any(r.get("status", "ok") != "ok" for r in records):
            raise ValueError(f"unsuccessful experiment: {path}")


def run_job(name, command, outputs, state_dir, log_dir, source_identity):
    outputs = [Path(p) for p in outputs]
    stamp = state_dir / f"{name}.json"
    identity = dict(command=command, source_identity=source_identity)
    if stamp.exists():
        saved = json.loads(stamp.read_text())
        if job_identity(saved["identity"]) != job_identity(identity):
            raise ValueError(f"{name}: command/code changed; use a fresh version tag")
        try:
            for p in outputs:
                validate_output(p)
            if saved["outputs"] == {str(p): file_sha256(p) for p in outputs}:
                print(f"[skip] {name}: verified", flush=True)
                return
        except Exception:
            pass  # corrupt/missing output is rerun; the child must succeed
    stamp.unlink(missing_ok=True)
    print(f"[start] {name}", flush=True)
    with open(log_dir / f"cpu_{name}.log", "w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    for p in outputs:
        validate_output(p)
    atomic_json(stamp, dict(identity=identity, outputs={str(p): file_sha256(p) for p in outputs}))
    print(f"[done] {name}: verified", flush=True)


def merge_corollary(root):
    # Never publish a partial system: require exactly the three requested cells.
    merged = {}
    for i, system in enumerate(SYSTEMS):
        records = []
        for tau in TAUS:
            part = root / f"part_{system}_tau{tau}" / "results_task0.json"
            validate_output(part)
            cells = json.loads(part.read_text())
            if len(cells) != 1 or cells[0]["system"] != system or cells[0]["tau_factor"] != float(tau):
                raise ValueError(f"unexpected Corollary 3 cell: {part}")
            records.extend(cells)
        merged[i] = records
    for i, records in merged.items():
        atomic_json(root / f"results_task{i}.json", records)


def main():
    version = sys.argv[1] if len(sys.argv) > 1 else "v12r1"
    quick = os.environ.get("QUICK", "0") == "1"
    if quick:
        version += "_quick"  # cannot contaminate full-data directories
    dataset_dir = Path(f"datasets_{version}")
    root = Path(f"results_{version}")
    state = root / "cpu_completion"
    logs = Path("logs") / version
    for p in (dataset_dir, state, logs):
        p.mkdir(parents=True, exist_ok=True)
    lock = state / "STAGE.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise SystemExit(f"{lock} exists; remove it only after the previous CPU stage has stopped")
    with os.fdopen(fd, "w") as f:
        f.write(f"pid={os.getpid()} slurm_job={os.environ.get('SLURM_JOB_ID', '')}\n")
    try:
        source = {str(p): file_sha256(p) for folder in ("src", "scripts", "configs")
                  for p in sorted(Path(folder).rglob("*")) if p.suffix in (".py", ".yaml", ".sh")}
        source = numerical_source(source)
        stage_identity = dict(quick=quick, source=source)
        protocol = state / "protocol.json"
        if protocol.exists():
            previous = json.loads(protocol.read_text())
            previous["source"] = numerical_source(previous["source"])
            if previous != stage_identity:
                raise ValueError("CPU stage numerical code/settings changed; use a fresh version tag")
        atomic_json(protocol, stage_identity)
        jobs = []
        def add(name, script, args, outputs):
            jobs.append((name, [sys.executable, "-u", f"scripts/{script}"] + args.split(),
                         outputs, state, logs, source))
        dsset = (" --set data.T_train=3 data.T_val=1 data.n_test=2 data.T_test=1 "
                 "data.T_long=2 data.lyap_T=10 data.ftle_stride=40") if quick else ""
        for system in ("l63", "l96_n40", "ks_l22"):
            add(f"ds_{system}", "make_dataset.py", f"--config configs/{system}.yaml --out_dir {dataset_dir}{dsset}",
                [dataset_dir / f"{system}.npz"])
        for system in SYSTEMS:
            for tau in TAUS:
                out = root / "corollary3" / f"part_{system}_tau{tau}"
                add(f"cor3_{system}_tau{tau}", "test_corollary3.py",
                    f"--systems {system} --tau_factors {tau} --n_points {120 if quick else 4000} "
                    f"--seed 0 --task_id 0 --out_dir {out}",
                    [out / "results_task0.json", out / f"ftle_{system}_tau{float(tau):g}.npz"])
        for task in range(4):
            for noise in ("uniform", "gaussian"):
                out = root / "proposition4"
                suffix = "_gaussian" if noise == "gaussian" else ""
                add(f"prop4_{noise}_{task}", "test_proposition4.py",
                    f"--task_id {task} --n_points {100 if quick else 2000} --n_noise_realizations 20 "
                    f"--seed 0 --noise {noise} --out_dir {out}", [out / f"results_task{task}{suffix}.json"])
            out = root / "lemma1"
            add(f"lemma1_{task}", "test_lemma1.py",
                f"--mode tangent --task_id {task} --n_trajectories {3 if quick else 20} "
                f"--W_factors 0.1 0.25 0.5 1.0 1.5 2.0 --seed 0 --out_dir {out}", [out / f"tangent_task{task}.json"])
        failures = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=int(os.environ.get("CPU_JOBS", "27"))) as pool:
            futures = {pool.submit(run_job, *job): job[0] for job in jobs}
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    failures.append(futures[future])
                    print(f"[FAILED] {futures[future]}: {exc}", flush=True)
        if failures:
            raise SystemExit(f"CPU stage incomplete: {failures}; see {logs}")
        merge_corollary(root / "corollary3")
        print(f"CPU stage complete: {dataset_dir}, {root}")
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
