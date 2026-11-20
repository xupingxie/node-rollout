# Rollout Scheduling for Chaotic Neural ODEs

Code for the paper

> **Rollout Scheduling for Chaotic Neural ODEs: Coverage Gains, Computational Cost, and Predictive Accuracy**
> Ajith Nayananad and Xuping Xie, Department of Mathematics and Statistics, Old Dominion University

## Install

Python 3.10 or newer.

```bash
python -m pip install -r requirements.txt
python -m pytest tests -q
```

Run every command below from the repository root.

## How to run

### 1. Generate a dataset

One dataset per system: `l63` (Lorenz-63), `l96_n40` (Lorenz-96, N = 40),
`ks_l22` (Kuramoto–Sivashinsky, L = 22).

```bash
python scripts/make_dataset.py --config configs/l63.yaml --out_dir datasets
```

This writes `datasets/l63.npz` with the training, validation and test
trajectories and the finite-time Lyapunov exponent (FTLE) profile.

### 2. Train a model

```bash
# constant rollout of 0.5 Lyapunov times
python scripts/train.py --config configs/l63.yaml --schedule ss --W 0.5 --seed 0 \
    --dataset_dir datasets --out_dir runs

# adaptive schedule and the constant schedule at the same instability budget
python scripts/train.py --config configs/l63.yaml --schedule sat     --beta 1.0 --seed 0 --dataset_dir datasets --out_dir runs
python scripts/train.py --config configs/l63.yaml --schedule const_g --beta 1.0 --seed 0 --dataset_dir datasets --out_dir runs
```

Schedules: `ss` (constant rollout length `--W`), `const_g` (constant rollout
at budget `--beta`), `sat` (adaptive rollout at budget `--beta`), `ramp`
(growing rollout up to `--W`), `mpnode` (multiple shooting with `--K` chunks).
Any configuration value can be overridden with `--set section.key=value`,
for example `--set train.n_steps=2000`.

Each run writes `runs/<system>/<run>/seed<seed>/` with its configuration,
training log, checkpoints and a `summary.json` of the evaluation. A finished
run is skipped if launched again; an interrupted run resumes from its last
checkpoint.

### 3. Run a full grid

```bash
# write one line of train.py arguments per run
python scripts/slurm/make_job_list.py --system l63 --seeds 5 --decisive \
    --extra "--dataset_dir datasets --out_dir runs" > jobs_l63.txt

# on a SLURM GPU allocation: run the list, 3 processes per GPU
bash scripts/slurm/run_jobs_multinode.sh jobs_l63.txt 3

# check that every run in the list finished and was evaluated
python scripts/check_grid.py jobs_l63.txt
```

The paper uses `--decisive` for `l63` and `l96_n40` and `--lean` for `ks_l22`.
Without SLURM, run the lines of the job list one at a time with
`python scripts/train.py <line>`.

### 4. Numerical checks of the theory (CPU only, no training)

```bash
python scripts/test_corollary3.py --systems l63 --n_points 1000
python scripts/test_proposition4.py --systems l63 --n_points 500
python scripts/test_lemma1.py --mode tangent --systems l63 --n_trajectories 5
```

### 5. Aggregate results and make tables and figures

```bash
python scripts/aggregate_runs.py --runs_dir runs --out_data data --out_results results
python scripts/tables/tables_grid.py --data_dir data --out_dir tables
python scripts/figures/plot_grid_budget.py --data_dir data --out figures/fig_grid_budget.png
python scripts/figures/plot_lemma1_segments.py --results_dir results --data_dir data --out figures/fig_lemma1_segments.png
```

Run any script with `--help` for its options, except `scripts/cpu_stage.py`,
which takes a version tag as its only argument and starts computing
immediately.

## What is in each folder

### `configs/`

| File | Contents |
|---|---|
| `l63.yaml`, `l96_n40.yaml`, `ks_l22.yaml` | Data, model, schedule and training settings for each system |

### `src/` — library code

| File | Contents |
|---|---|
| `systems/base.py` | Base class for a dynamical system (right-hand side, Jacobian, integrator) |
| `systems/l63.py`, `l96.py`, `rossler.py`, `ks.py` | Lorenz-63, Lorenz-96, Rössler and Kuramoto–Sivashinsky systems |
| `ftle/ftle.py` | FTLE estimation by tangent-linear integration |
| `schedules/schedules.py` | Rollout schedule constructors used by the numerical checks |
| `models/node.py` | Neural ODE with a fixed-step RK4 integrator |
| `train/config.py` | Loading a configuration and applying command-line overrides |
| `train/data.py` | Dataset generation: trajectories, FTLE profile, Lyapunov exponent |
| `train/schedules_train.py` | The five training-time rollout schedules |
| `train/trainer.py` | Training loop, logging and checkpointing |
| `train/evaluate.py` | Evaluation: valid time and long-time statistics |
| `train/provenance.py` | Fingerprints of code, data and configuration for resumable runs |

### `scripts/` — entry points

| File | Contents |
|---|---|
| `make_dataset.py` | Generate the dataset for one system |
| `train.py` | Train and evaluate one run |
| `check_grid.py` | Verify that every run of a job list is complete |
| `grid_status.py` | Show the progress of the runs of a grid |
| `readiness_report.py` | Report whether the constant-rollout pilot run trained well enough |
| `aggregate_runs.py` | Combine run summaries into per-system summaries |
| `reevaluate.py` | Re-run the evaluation of finished runs from their checkpoints |
| `reevaluate_longtime.py` | Re-run only the long-time evaluation, one rollout at a time |
| `test_corollary3.py` | Numerical check of the coverage gain (Corollary 3) |
| `test_proposition4.py` | Numerical check of robustness to FTLE error (Proposition 4) |
| `test_lemma1.py` | Numerical check of the gradient growth rate (Lemma 1) |
| `floor_sweep.py` | Coverage gain as a function of the FTLE floor percentile |
| `extrapolation_check.py` | Diagnostic for extending the FTLE profile beyond its horizon |
| `measure_lambda1.py` | Estimate the largest Lyapunov exponent of each system |
| `cpu_stage.py`, `interactive_cpu_stage.sh` | Run all datasets and numerical checks as one restartable CPU job |
| `aggregate_and_plot.py`, `convert_legacy_data.py`, `regenerate_all_figures.py`, `regenerate_all_tables.py` | Helpers from an earlier version of the study; not needed for the steps above |

### `scripts/tables/` and `scripts/figures/`

| File | Contents |
|---|---|
| `tables/tables_grid.py` | Tables of the training grids |
| `tables/table1_hetero.py` … `table4_prop4.py` | Tables of the numerical checks |
| `tables/tables_training.py` | Tables from an earlier version of the study |
| `figures/plot_corollary3.py` | Coverage gain against FTLE variability |
| `figures/plot_lemma1_tangent.py` | Tangent growth against rollout length |
| `figures/plot_proposition4.py` | Coverage loss against FTLE error |
| `figures/plot_grid_budget.py` | Model accuracy against instability budget |
| `figures/plot_lemma1_segments.py` | Per-segment gradient norm against instability |
| other `figures/plot_*.py` | Figures from an earlier version of the study |

### `scripts/slurm/` — cluster launchers

| File | Contents |
|---|---|
| `make_job_list.py` | Write the job list of a training grid |
| `make_ablation_wmax.py` | Write the job list of the rollout-cap ablation |
| `run_jobs_multinode.sh` | Run a job list across the nodes of an allocation |
| `run_jobs_parallel.sh`, `run_jobs_node.py` | Per-node worker pool used by the launcher |
| `run_*.sbatch` | Batch scripts for datasets, training and the numerical checks |
| `jobs_*.txt` | Example job lists |

The launchers were written for NERSC Perlmutter. Set `--account=YOUR_PROJECT`
and the environment activation for your cluster before submitting.

### `tests/`

| File | Contents |
|---|---|
| `test_smoke.py` | Systems, FTLE estimator and schedule arithmetic |
| `test_train_smoke.py` | A short end-to-end training run on CPU |
| `test_restart_protocol.py` | Resuming interrupted runs |
| `test_multisession_launch.py` | Splitting a job list across allocations |

## Not included

Datasets, model checkpoints and training outputs are not in the repository;
the steps above generate them.

## Acknowledgments

This research used resources of the National Energy Research Scientific
Computing Center, a DOE Office of Science User Facility supported by the
Office of Science of the U.S. Department of Energy under Contract
No. DE-AC02-05CH11231 using NERSC award NERSC DDR-ERCAP0035032.

## Citation

If you use this code, please cite the paper above.
