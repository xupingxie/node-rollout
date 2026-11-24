#!/bin/bash
#SBATCH --job-name=make_datasets
#SBATCH --account=YOUR_PROJECT          # CHANGE ME
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --array=0-2
#SBATCH --output=logs/dataset_%A_%a.out
#SBATCH --error=logs/dataset_%A_%a.err
# Generates datasets/<system>.npz (trajectories + teacher FTLE + true lambda_1).
# KS takes the longest (~30-60 min: 200 T_lambda of ETDRK4 plus FTLE every
# 5th point); L63 and L96 take minutes.

set -euo pipefail
module load python
source $HOME/venv/bin/activate    # CHANGE ME
cd $SLURM_SUBMIT_DIR
mkdir -p logs datasets
CONFIGS=(configs/l63.yaml configs/ks_l22.yaml configs/l96_n40.yaml)
mkdir -p ${OUT_DIR:-datasets}
python scripts/make_dataset.py --config ${CONFIGS[$SLURM_ARRAY_TASK_ID]} --out_dir ${OUT_DIR:-datasets}
