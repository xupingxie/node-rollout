#!/bin/bash
#SBATCH --job-name=corollary3
#SBATCH --account=YOUR_PROJECT          # CHANGE ME
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --array=0-3                     # 4 systems
#SBATCH --output=logs/corollary3_%A_%a.out
#SBATCH --error=logs/corollary3_%A_%a.err

# NERSC Perlmutter SLURM script for Corollary 3 verification.
# Each task processes one system: (l63, rossler, l96_n40, ks_l22).

set -euo pipefail

# --- environment ---
# Set the Python environment for the cluster.
module load python
source $HOME/venv/bin/activate    # CHANGE ME (or use conda activate)

# --- run ---
cd $SLURM_SUBMIT_DIR
mkdir -p logs ${OUT_ROOT:-results}/corollary3

python scripts/test_corollary3.py \
    --task_id $SLURM_ARRAY_TASK_ID \
    --systems l63 rossler l96_n40 ks_l22 \
    --n_points 4000 \
    --tau_factors 0.5 1.0 2.0 \
    --seed 0 \
    --out_dir ${OUT_ROOT:-results}/corollary3

echo "Task $SLURM_ARRAY_TASK_ID done."
