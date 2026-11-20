#!/bin/bash
#SBATCH --job-name=proposition4
#SBATCH --account=YOUR_PROJECT          # CHANGE ME
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --array=0-3
#SBATCH --output=logs/proposition4_%A_%a.out
#SBATCH --error=logs/proposition4_%A_%a.err

set -euo pipefail

module load python
source $HOME/your_venv/bin/activate    # CHANGE ME

cd $SLURM_SUBMIT_DIR
mkdir -p logs ${OUT_ROOT:-results}/proposition4

python scripts/test_proposition4.py \
    --task_id $SLURM_ARRAY_TASK_ID \
    --systems l63 rossler l96_n40 ks_l22 \
    --n_points 2000 \
    --sigma_eps_grid 0.01 0.025 0.05 0.1 0.2 0.4 0.8 \
    --confidence_multiplier 2.0 \
    --n_noise_realizations 20 \
    --seed 0 \
    --noise uniform \
    --out_dir ${OUT_ROOT:-results}/proposition4

# stress test with a 2-sigma gaussian calibration (the v11 setting), separate output files
python scripts/test_proposition4.py \
    --task_id $SLURM_ARRAY_TASK_ID \
    --systems l63 rossler l96_n40 ks_l22 \
    --n_points 2000 \
    --sigma_eps_grid 0.01 0.025 0.05 0.1 0.2 0.4 0.8 \
    --confidence_multiplier 2.0 \
    --n_noise_realizations 20 \
    --seed 0 \
    --noise gaussian \
    --out_dir ${OUT_ROOT:-results}/proposition4

echo "Task $SLURM_ARRAY_TASK_ID done."
