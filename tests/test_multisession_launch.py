"""Test two-allocation ownership and GPU concurrency without SLURM/GPU hardware."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.slurm.run_jobs_node import gpu_slots, run_pass, selected_jobs
from scripts.cpu_stage import job_identity, numerical_source


def test_two_sessions_cover_grid_once(tmp_path):
    jobfile = tmp_path / "grid.txt"
    jobfile.write_text("".join(f"--seed {i}\n" for i in range(65)))
    owned = []
    for session in range(2):
        session_lines = []
        for rank in range(2):
            shard = rank * 2 + session
            lines = [line for line, _ in selected_jobs(jobfile, shard, 4)]
            session_lines.extend(lines)
            assert lines == [line for line, _ in selected_jobs(jobfile, shard, 4)]  # restart ownership
        owned.append(set(session_lines))
    assert not owned[0] & owned[1]
    assert owned[0] | owned[1] == set(range(1, 66))
    assert sorted(map(len, owned)) == [32, 33]


def test_multinode_arguments_and_quoted_paths(tmp_path):
    # Fake only srun/the node worker; execute the real multinode shell logic.
    scripts = tmp_path / "scripts/slurm"
    scripts.mkdir(parents=True)
    capture = tmp_path / "calls.jsonl"
    (scripts / "run_jobs_parallel.sh").write_text(
        '#!/bin/bash\npython - "$@" <<\'PY\'\nimport json, os, sys\n'
        'with open(os.environ["CAPTURE"], "a") as f: f.write(json.dumps(sys.argv[1:])+"\\n")\nPY\n')
    binary = tmp_path / "bin"
    binary.mkdir()
    stub = binary / "srun"
    stub.write_text('#!/bin/bash\nwhile [ "$1" != "bash" ]; do shift; done\n'
                    'for rank in 0 1; do SLURM_PROCID=$rank "$@" || exit $?; done\n')
    stub.chmod(0o755)
    jobfile = tmp_path / "grid with spaces.txt"
    jobfile.write_text("--seed 0\n")
    env = dict(os.environ, PATH=str(binary) + os.pathsep + os.environ["PATH"],
               CAPTURE=str(capture), SLURM_JOB_NUM_NODES="2")
    launcher = str(ROOT / "scripts/slurm/run_jobs_multinode.sh")
    for session in (0, 1):
        result = subprocess.run(["bash", launcher, str(jobfile), "3", str(session), "2"],
                                cwd=tmp_path, env=env, text=True, capture_output=True)
        assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in capture.read_text().splitlines()]
    assert calls == [[str(jobfile), "4", "3", str(shard), "4"] for shard in (0, 2, 1, 3)]
    # The original interface still covers an allocation by itself.
    capture.write_text("")
    subprocess.run(["bash", launcher, str(jobfile), "2"], cwd=tmp_path, env=env, check=True)
    assert [json.loads(line)[3:] for line in capture.read_text().splitlines()] == [["0", "2"], ["1", "2"]]
    invalid = subprocess.run(["bash", launcher, str(jobfile), "3", "2", "2"],
                             cwd=tmp_path, env=env, capture_output=True)
    assert invalid.returncode != 0


def test_gpu_pool_enforces_per_gpu_limit_and_preserves_device_ids(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-first,GPU-second")
    slots = gpu_slots(2, 2)
    active = {d: 0 for d in slots}
    peaks = dict(active)
    calls = []
    guard = threading.Lock()
    def child(command, env, stdout, stderr):
        gpu = env["CUDA_VISIBLE_DEVICES"]
        with guard:
            active[gpu] += 1
            peaks[gpu] = max(peaks[gpu], active[gpu])
            calls.append(command)
        time.sleep(.01 if int(command[-1]) % 2 else .03)
        with guard:
            active[gpu] -= 1
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(subprocess, "run", child)
    jobs = [(i + 1, f'--config "path with spaces.yaml" --seed {i}') for i in range(20)]
    outcomes = run_pass(jobs, slots, tmp_path)
    assert len(outcomes) == len(calls) == 20
    assert all(status == "done" for _, _, status in outcomes)
    assert set(peaks) == {"GPU-first", "GPU-second"}
    assert all(peak == 2 for peak in peaks.values())
    assert all(command[3] == "path with spaces.yaml" for command in calls)
    with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES"):
        gpu_slots(3, 1)


def test_cpu_fingerprint_allows_only_launcher_update():
    old = {"src/ftle/ftle.py": "same", "scripts/test_corollary3.py": "same",
           "scripts/slurm/run_jobs_multinode.sh": "old", "scripts/cpu_stage.py": "old"}
    new = dict(old, **{"scripts/slurm/run_jobs_multinode.sh": "new",
                       "scripts/slurm/run_jobs_node.py": "new", "scripts/cpu_stage.py": "new"})
    assert numerical_source(old) == numerical_source(new)
    assert job_identity(dict(command=["python", "test.py"], source_identity=old)) == job_identity(
        dict(command=["python", "test.py"], source_identity=new))
    changed = dict(new, **{"src/ftle/ftle.py": "different numerics"})
    assert numerical_source(changed) != numerical_source(old)
    assert job_identity(dict(command=["python", "test.py", "--quick"], source_identity=new)) != job_identity(
        dict(command=["python", "test.py"], source_identity=old))
