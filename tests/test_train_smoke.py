"""Smoke test for the training pipeline (CPU, ~1 minute).

Generates a tiny L63 dataset, trains each schedule family for a handful
of steps, and checks that logs, per-segment dumps, checkpoints and a
summary with finite metrics are written.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
torch = pytest.importorskip("torch")

from src.train.config import RunConfig
from src.train.data import make_dataset, Dataset
from src.train.trainer import Trainer
from src.train.evaluate import valid_time, long_time_stats
from src.train.schedules_train import build_policy


@pytest.fixture(scope="module")
def tiny_dataset(tmp_path_factory):
    out = tmp_path_factory.mktemp("ds") / "l63.npz"
    make_dataset("l63", T_train=6.0, T_val=2.0, n_test=4, T_test=1.0, T_long=4.0,
                 tau_factor=0.5, ftle_stride=20, lyap_T=20.0, seed=0,
                 out_path=out, verbose=False)
    return Dataset.load(out)


def test_dataset_shapes(tiny_dataset):
    ds = tiny_dataset
    assert ds.train.shape[1] == 3
    assert ds.test.shape[0] == 4
    assert np.isfinite(ds.ftle_train).all()
    assert 0.5 < ds.lambda1_true < 1.3


def test_policies_respect_bounds():
    cfg = RunConfig().schedule
    mu = np.array([0.5, 1.0, 2.0, 8.0])
    for name in ("ss", "ramp", "const_g", "sat", "mpnode"):
        cfg.name = name
        pol = build_policy(cfg, T_lambda=1.155, dt=0.01, n_steps=100)
        W, info = pol(50, mu)
        assert W.shape == mu.shape
        assert (W >= pol.W_min - 1e-12).all() and (W <= pol.W_max + 1e-12).all()
    cfg.name = "const_g"
    pol = build_policy(cfg, 1.155, 0.01, 100)
    W, _ = pol(10 ** 6, mu)
    assert np.allclose(W, cfg.beta / mu.max())
    cfg.name = "sat"; cfg.bucketing = "none"
    pol = build_policy(cfg, 1.155, 0.01, 100)
    W, _ = pol(10 ** 6, mu)
    assert np.allclose(W, np.clip(cfg.beta / mu, pol.W_min, pol.W_max))


@pytest.mark.parametrize("schedule", ["ss", "const_g", "sat", "ramp", "mpnode"])
def test_train_few_steps(tiny_dataset, tmp_path, schedule):
    cfg = RunConfig()
    cfg.model.width = 16
    cfg.model.depth = 2
    cfg.schedule.name = schedule
    cfg.schedule.W = 0.2
    cfg.schedule.W_max = 0.3
    cfg.train.n_steps = 6
    cfg.train.batch_size = 8
    cfg.train.log_every = 1
    cfg.train.val_every = 3
    cfg.train.segment_log_every = 3
    cfg.train.segment_log_n = 4
    cfg.train.ckpt_every = 0
    run_dir = tmp_path / schedule
    tr = Trainer(cfg, tiny_dataset, run_dir)
    summ = tr.fit()
    assert np.isfinite(summ["final_loss"]) and summ["final_grad_norm"] > 0
    assert (run_dir / "checkpoint_final.pt").exists()
    lines = [json.loads(l) for l in open(run_dir / "log.jsonl")]
    assert len(lines) == 6 and all("grad_norm_preclip" in l for l in lines)
    segs = list(run_dir.glob("segments_*.npz"))
    assert len(segs) == 2
    z = np.load(segs[0])
    assert z["grad_norm_i"].shape == (4,) and np.isfinite(z["Lambda_i"]).all()


def test_eval_runs(tiny_dataset, tmp_path):
    cfg = RunConfig()
    cfg.model.width = 16; cfg.model.depth = 2
    cfg.train.n_steps = 2; cfg.train.batch_size = 4; cfg.train.ckpt_every = 0
    cfg.schedule.W_max = 0.3
    tr = Trainer(cfg, tiny_dataset, tmp_path / "e")
    tr.fit()
    vt = valid_time(tr.model, tiny_dataset, (0.3, 0.5))
    assert 0.0 <= vt["valid_time_thr03_mean"] <= 1.0 + 1e-9
    lt = long_time_stats(tr.model, tiny_dataset, n_long=2, lyap_T=2.0, psd_nperseg=64, mmd_n=200)
    assert "marginal_mmd_mean" in lt and "lambda1_model_mean" in lt


def test_early_stop_and_resume(tiny_dataset, tmp_path):
    cfg = RunConfig()
    cfg.model.width = 16; cfg.model.depth = 2
    cfg.schedule.W_max = 0.3
    cfg.train.batch_size = 4; cfg.train.log_every = 1
    cfg.train.val_every = 2; cfg.train.segment_log_every = 10 ** 9
    cfg.train.ckpt_every = 4
    # early stop: patience 2 validations, allowed from step 4
    cfg.train.n_steps = 40; cfg.train.min_steps = 4
    cfg.train.early_stop_patience = 2; cfg.train.early_stop_min_delta = 10.0  # never "improves" after first
    run_dir = tmp_path / "es"
    tr = Trainer(cfg, tiny_dataset, run_dir)
    summ = tr.fit()
    assert summ["stopped_early"] and summ["stopped_step"] < 39
    assert (run_dir / "checkpoint_best.pt").exists() and (run_dir / "checkpoint_last.pt").exists()
    # resume: a new trainer on the same dir continues from the saved step
    cfg.train.early_stop_patience = 10 ** 6
    tr2 = Trainer(cfg, tiny_dataset, run_dir)
    assert tr2.start_step == summ["stopped_step"] + 1
    summ2 = tr2.fit()
    assert summ2["stopped_step"] == 39


def test_splits_are_disjoint(tiny_dataset):
    from src.train.data import check_disjoint
    ds = tiny_dataset
    ov = check_disjoint(ds.train, ds.val, ds.test, ds.ref_long)
    assert all(v == 0 for v in ov.values()), ov
    assert ds.ftle_method == "svd"


def small_config():
    cfg = RunConfig()
    cfg.model.width = 8
    cfg.model.depth = 1
    cfg.schedule.W = 0.2
    cfg.schedule.W_max = 0.3
    cfg.train.n_steps = 3
    cfg.train.batch_size = 2
    cfg.train.device = "cpu"
    cfg.train.val_every = 1
    cfg.train.segment_log_n = 1
    cfg.train.segment_log_every = 10000
    return cfg


@pytest.mark.parametrize("early", [False, True])
def test_completed_training_resume_only_evaluates(tiny_dataset, tmp_path, early):
    cfg = small_config()
    if early:
        cfg.train.n_steps = 10
        cfg.train.min_steps = 1
        cfg.train.early_stop_patience = 1
        cfg.train.early_stop_min_delta = 10
    first = Trainer(cfg, tiny_dataset, tmp_path)
    expected = first.fit()
    log_before = (tmp_path / "log.jsonl").read_bytes()
    resumed = Trainer(cfg, tiny_dataset, tmp_path)
    def no_training(*args):
        pytest.fail("completed training must not execute another training step")
    resumed.step = no_training
    actual = resumed.fit()
    assert json.dumps(actual, sort_keys=True) == json.dumps(expected, sort_keys=True)
    assert np.isfinite(actual["final_loss"])
    assert (tmp_path / "log.jsonl").read_bytes() == log_before
    for key, tensor in first.model.state_dict().items():
        assert torch.equal(tensor, resumed.model.state_dict()[key])


@pytest.mark.parametrize("name", ["sat", "const_g"])
def test_wmin_infeasibility_count(tiny_dataset, tmp_path, name):
    cfg = small_config()
    cfg.schedule.name = name
    cfg.schedule.beta = 0.1
    tr = Trainer(cfg, tiny_dataset, tmp_path)
    tr.mu_train[:] = 10.0
    rec, _ = tr.step(0)
    assert rec["Lambda_max_batch"] > rec["beta_k"]
    assert tr.exceed["seg_wmin_bound"] == cfg.train.batch_size
    tr.log_f.close()


def test_mp_actual_forward_work_count(tiny_dataset, tmp_path):
    cfg = small_config()
    cfg.schedule.name = "mpnode"
    cfg.schedule.K = 3
    tr = Trainer(cfg, tiny_dataset, tmp_path)
    executed = []
    rollout = tr.model.rollout
    def measured(x0, n):
        executed.append(len(x0) * n)
        return rollout(x0, n)
    tr.model.rollout = measured
    tr.step(0)
    assert len(executed) == cfg.schedule.K
    assert tr.integrated_steps == sum(executed)
    tr.log_f.close()


def test_checkpoint_rejects_wrong_provenance(tiny_dataset, tmp_path):
    cfg = small_config()
    Trainer(cfg, tiny_dataset, tmp_path, provenance={"data": "one"}).fit()
    with pytest.raises(ValueError, match="provenance"):
        Trainer(cfg, tiny_dataset, tmp_path, provenance={"data": "two"})


def test_cli_skipped_evaluation_resumes_and_grid_checks_identity(tiny_dataset, tmp_path):
    import dataclasses
    import subprocess
    import shlex
    root = Path(__file__).resolve().parents[1]
    np.savez(tmp_path / "l63.npz", **dataclasses.asdict(tiny_dataset))
    cfg = small_config()
    cfg.data.T_test = 1.0
    cfg.eval.n_long_rollouts = 1
    cfg.eval.lyap_T = .2
    cfg.eval.mmd_n = 20
    cfg.eval.psd_nperseg = 64
    config = tmp_path / "config.yaml"
    cfg.save(config)
    args = ["--config", str(config), "--dataset_dir", str(tmp_path), "--out_dir", str(tmp_path / "runs")]
    def train(extra=()):
        return subprocess.run([sys.executable, "scripts/train.py", *args, *extra], cwd=root,
                              text=True, capture_output=True)
    first = train(["--skip_eval"])
    assert first.returncode == 0, first.stderr
    summary_file = next((tmp_path / "runs").glob("*/*/seed*/summary.json"))
    summary = json.loads(summary_file.read_text())
    assert summary["training_complete"] and not summary["evaluation_complete"]
    log = summary_file.parent / "log.jsonl"
    saved_log = log.read_bytes()
    resumed = train()
    assert resumed.returncode == 0, resumed.stderr
    final = json.loads(summary_file.read_text())
    assert final["evaluation_complete"] and np.isfinite(final["final_loss"])
    assert final["final_loss"] == summary["final_loss"]
    assert final["integrated_model_steps"] == summary["integrated_model_steps"]
    assert log.read_bytes() == saved_log
    assert "verified complete" in train().stdout
    jobfile = tmp_path / "jobs.txt"
    jobfile.write_text(shlex.join(args) + "\n")
    checked = subprocess.run([sys.executable, "scripts/check_grid.py", str(jobfile)], cwd=root,
                             capture_output=True, text=True)
    assert checked.returncode == 0, checked.stdout + checked.stderr
    # Exercise the downstream reports with only newly verified outputs.
    for script, extra in [
        ("scripts/aggregate_runs.py", ["--runs_dir", str(tmp_path / "runs"), "--out_data", str(tmp_path / "data"), "--out_results", str(tmp_path / "results")]),
        ("scripts/tables/tables_grid.py", ["--data_dir", str(tmp_path / "data"), "--out_dir", str(tmp_path / "tables")]),
        ("scripts/figures/plot_grid_budget.py", ["--data_dir", str(tmp_path / "data"), "--out", str(tmp_path / "budget.png")]),
        ("scripts/figures/plot_lemma1_segments.py", ["--results_dir", str(tmp_path / "results"), "--data_dir", str(tmp_path / "data"), "--out", str(tmp_path / "segments.png")]),
    ]:
        report = subprocess.run([sys.executable, script, *extra], cwd=root, capture_output=True, text=True)
        assert report.returncode == 0, report.stdout + report.stderr
    assert (tmp_path / "budget.png").stat().st_size > 0
    # A custom dataset filename must NOT bypass the FTLE-horizon check.
    bad = train(["--dataset", "l63", "--set", "data.tau_factor=2"])
    assert bad.returncode != 0 and "tau:" in bad.stderr
    bad = train(["--set", "train.lr=0.002"])
    assert bad.returncode != 0 and "changed" in bad.stderr
    # --force must acquire the lock BEFORE removing the run's files.
    (summary_file.parent / "RUNNING.lock").write_text("test live process")
    saved_summary = summary_file.read_bytes()
    blocked = train(["--force"])
    assert "is being trained by another process" in blocked.stdout
    assert summary_file.read_bytes() == saved_summary
