#!/bin/bash
#SBATCH --job-name=lemma1
#SBATCH --account=YOUR_PROJECT          # CHANGE ME
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --array=0-3
#SBATCH --output=logs/lemma1_%A_%a.out
#SBATCH --error=logs/lemma1_%A_%a.err

set -euo pipefail

module load python
source $HOME/venv/bin/activate    # CHANGE ME

cd $SLURM_SUBMIT_DIR
mkdir -p logs ${OUT_ROOT:-results}/lemma1

python scripts/test_lemma1.py \
    --mode tangent \
    --task_id $SLURM_ARRAY_TASK_ID \
    --systems l63 rossler l96_n40 ks_l22 \
    --n_trajectories 20 \
    --W_factors 0.1 0.25 0.5 1.0 1.5 2.0 \
    --seed 0 \
    --out_dir ${OUT_ROOT:-results}/lemma1

echo "Task $SLURM_ARRAY_TASK_ID done."
