#!/bin/bash
#SBATCH --job-name=node_train
#SBATCH --account=YOUR_PROJECT          # CHANGE ME (e.g. m1234_g for GPU)
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/train_%A_%a.out
#SBATCH --error=logs/train_%A_%a.err
#
# One training run per array task. The job list file (argument $1) has one
# line of `scripts/train.py` arguments per run; task N runs line N.
#
#   python scripts/make_dataset.py --config configs/l63.yaml          # once per system
#   python scripts/slurm/make_job_list.py --system l63 --seeds 5 > scripts/slurm/jobs_l63.txt
#   sbatch --array=1-$(wc -l < scripts/slurm/jobs_l63.txt) scripts/slurm/run_train.sbatch scripts/slurm/jobs_l63.txt
#
# Runs checkpoint every train.ckpt_every steps and resume automatically
# (train.resume: true) if the same array is resubmitted after a time-out,
# so a KS run that needs more than the wall limit just continues on the
# next submission. Early stopping (train.early_stop_*) ends most runs
# before n_steps. Use --array=1-N%16 to cap concurrency.

set -euo pipefail
JOBFILE=${1:?usage: sbatch run_train.sbatch <jobfile>}

module load pytorch                     # Perlmutter PyTorch module (or activate a Python environment)
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

cd $SLURM_SUBMIT_DIR
mkdir -p logs runs
ARGS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$JOBFILE")
echo "task ${SLURM_ARRAY_TASK_ID}: python scripts/train.py $ARGS"
python scripts/train.py $ARGS
echo "task ${SLURM_ARRAY_TASK_ID} done."
