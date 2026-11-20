"""Regression tests for interrupted jobs and matched comparison generation."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.cpu_stage import run_job, merge_corollary, SYSTEMS, TAUS
from scripts.train import parser, config_from_args
from src.train.config import RunConfig
from src.train.provenance import run_identity, ensure_manifest


def test_manifest_rejects_data_and_config_changes(tmp_path):
    dataset = tmp_path / "data.npz"
    dataset.write_bytes(b"first dataset")
    run = tmp_path / "run"
    run.mkdir()
    cfg = RunConfig()
    identity = run_identity(cfg, dataset)
    ensure_manifest(run, identity)
    ensure_manifest(run, run_identity(cfg, dataset))
    cfg.train.device = "cpu"  # operational override is allowed
    ensure_manifest(run, run_identity(cfg, dataset))
    cfg.train.lr *= 0.5
    with pytest.raises(ValueError, match="changed"):
        ensure_manifest(run, run_identity(cfg, dataset))
    cfg.train.lr *= 2
    dataset.write_bytes(b"changed dataset")
    with pytest.raises(ValueError, match="changed"):
        ensure_manifest(run, run_identity(cfg, dataset))


def test_manifest_rejects_legacy_directory(tmp_path):
    (tmp_path / "checkpoint_last.pt").write_bytes(b"legacy")
    with pytest.raises(ValueError, match="legacy"):
        ensure_manifest(tmp_path, {})


def test_cpu_restart_validates_files_and_propagates_failure(tmp_path):
    output = tmp_path / "part.npz"
    calls = tmp_path / "calls"
    command = [sys.executable, "-c", "import numpy as np; from pathlib import Path; "
               f"np.savez({str(output)!r}, x=[1,2]); "
               f"p=Path({str(calls)!r}); p.write_text(p.read_text()+'x' if p.exists() else 'x')"]
    args = ("part", command, [output], tmp_path, tmp_path, {"source": "v1"})
    run_job(*args)
    run_job(*args)
    assert calls.read_text() == "x"
    output.write_bytes(b"truncated archive")
    run_job(*args)
    assert calls.read_text() == "xx"
    with pytest.raises(subprocess.CalledProcessError):
        run_job("bad", [sys.executable, "-c", "raise SystemExit(2)"],
                [output], tmp_path, tmp_path, {})
    assert not (tmp_path / "bad.json").exists()


def test_corollary_merge_requires_all_cells(tmp_path):
    for system in SYSTEMS:
        for tau in TAUS:
            part = tmp_path / f"part_{system}_tau{tau}"
            part.mkdir()
            (part / "results_task0.json").write_text(json.dumps(
                [dict(system=system, tau_factor=float(tau), status="ok")]))
    missing = tmp_path / "part_ks_l22_tau2.0" / "results_task0.json"
    saved = missing.read_text()
    missing.unlink()
    with pytest.raises(FileNotFoundError):
        merge_corollary(tmp_path)
    assert not list(tmp_path.glob("results_task*.json"))
    missing.write_text(saved)
    merge_corollary(tmp_path)
    assert all(len(json.loads(p.read_text())) == 3 for p in tmp_path.glob("results_task*.json"))


@pytest.mark.parametrize("flag,seeds,count", [("--decisive", 5, 120), ("--lean", 5, 65), ("--readiness", 1, 3)])
def test_generated_grids_match_caps(flag, seeds, count, monkeypatch):
    monkeypatch.chdir(ROOT)
    lines = subprocess.check_output([sys.executable, "scripts/slurm/make_job_list.py", "--system", "ks_l22",
                                     flag, "--seeds", str(seeds)], cwd=ROOT, text=True).splitlines()
    assert len(lines) == count
    import shlex
    configs = [config_from_args(parser().parse_args(shlex.split(line))) for line in lines]
    for cfg in configs:
        if cfg.schedule.name == "sat":
            assert any(c.schedule.name == "const_g" and c.train.seed == cfg.train.seed
                       and c.schedule.beta == cfg.schedule.beta
                       and c.schedule.W_max == cfg.schedule.W_max for c in configs)


@pytest.mark.parametrize("locked", [False, True])
def test_gpu_launcher_propagates_failures(tmp_path, locked):
    import os
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    code = "print('is being trained by another process')" if locked else "raise SystemExit(2)"
    (scripts / "train.py").write_text(code)
    jobfile = tmp_path / "jobs.txt"
    jobfile.write_text("--config fake.yaml\n")
    env = dict(os.environ, MAX_PASSES="1", RETRY_WAIT="0", CLEAR_LOCKS="0", DRY_RUN="0")
    result = subprocess.run(["bash", str(ROOT / "scripts/slurm/run_jobs_parallel.sh"), str(jobfile), "1", "1"],
                            cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert "ERROR:" in result.stderr


def test_tables_include_capped_constant_and_unweighted(tmp_path):
    from scripts.tables.tables_grid import matched_table
    rows = []
    for name, schedule in [("const_g_b1_Wmax0.5", "const_g"),
                           ("sat_b1_q4_inv_W_Wmax0.5", "sat"),
                           ("sat_b1_q4_per_segment_Wmax0.5", "sat")]:
        rows.append(dict(config=name, schedule=schedule, beta=1.0, Lambda_max_mean=1.,
                         final_grad_norm_mean=2., valid_time_mean=3., valid_time_std=.1))
    out = tmp_path / "matched.tex"
    matched_table(rows, "l63", out)
    text = out.read_text()
    assert "W_g" in text and "unweighted" in text and "inverse-window weighted" in text
    assert text.count(r"0.5T_\lambda") == 3


def test_uniform_feasibility_requires_every_segment(tmp_path, monkeypatch):
    from scripts import test_proposition4 as prop
    monkeypatch.setattr(prop, "ftle_table", lambda *a, **kw: dict(
        n_valid=200, valid_mask=np.ones(200, bool), lambda_ftle=np.ones(200), lambda_mean=1.0))
    def one_violation(mu, beta, epsilon):
        W = np.full(len(mu), beta / 2)
        W[0] = beta * 2
        return W
    monkeypatch.setattr(prop, "shrunken_schedule", one_violation)
    records = prop.run_system_robustness_test("l63", 200, [.01], 2., 0, tmp_path,
                                             n_noise_realizations=1, verbose=False)
    assert records[0]["feasibility_fraction_min"] == .995
    assert records[0]["passes_feasibility"] is False


def test_quick_cpu_stage_uses_separate_directories(tmp_path, monkeypatch):
    from scripts import cpu_stage
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("QUICK", "1")
    monkeypatch.setattr(sys, "argv", ["cpu_stage.py", "v12r1"])
    jobs = []
    monkeypatch.setattr(cpu_stage, "run_job", lambda *args: jobs.append(args))
    monkeypatch.setattr(cpu_stage, "merge_corollary", lambda root: None)
    cpu_stage.main()
    assert len(jobs) == 27
    assert (tmp_path / "datasets_v12r1_quick").is_dir()
    assert not (tmp_path / "datasets_v12r1").exists()
    assert not (tmp_path / "results_v12r1_quick/cpu_completion/STAGE.lock").exists()
