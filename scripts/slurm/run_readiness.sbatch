#!/bin/bash
#SBATCH --job-name=readiness
#SBATCH --account=YOUR_PROJECT          # CHANGE ME
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --array=0-2
#SBATCH --output=logs/readiness_%A_%a.out
#SBATCH --error=logs/readiness_%A_%a.err
# Readiness check (TRAINING_GUIDE.md section 1): one constant-window
# W = 0.5 T_lambda run per system. Afterwards:
#     python scripts/readiness_report.py runs

set -euo pipefail
module load pytorch                     # or activate your env
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
cd $SLURM_SUBMIT_DIR
mkdir -p logs runs
CONFIGS=(configs/l63.yaml configs/ks_l22.yaml configs/l96_n40.yaml)
python scripts/train.py --config ${CONFIGS[$SLURM_ARRAY_TASK_ID]} --schedule ss --W 0.5 --seed 0
