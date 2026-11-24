#!/bin/bash
#SBATCH --job-name=aggregate
#SBATCH --account=YOUR_PROJECT          # CHANGE ME
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/aggregate_%j.out
#SBATCH --error=logs/aggregate_%j.err
#SBATCH --dependency=afterok:CORROLARY3_JOBID:PROPOSITION4_JOBID:LEMMA1_JOBID  # CHANGE ME

# Run after the three theorem-verification job arrays complete.
# Use:
#   COR=$(sbatch scripts/slurm/run_corollary3.sbatch | awk '{print $4}')
#   PROP=$(sbatch scripts/slurm/run_proposition4.sbatch | awk '{print $4}')
#   LEM=$(sbatch scripts/slurm/run_lemma1_tangent.sbatch | awk '{print $4}')
#   sbatch --dependency=afterok:$COR:$PROP:$LEM scripts/slurm/run_aggregate.sbatch

set -euo pipefail
module load python
source $HOME/venv/bin/activate    # CHANGE ME

cd $SLURM_SUBMIT_DIR
mkdir -p plots

python scripts/aggregate_and_plot.py \
    --corollary3_dir results/corollary3 \
    --proposition4_dir results/proposition4 \
    --lemma1_dir results/lemma1 \
    --out_dir plots
