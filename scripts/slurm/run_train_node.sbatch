#!/bin/bash
#SBATCH --job-name=node_train_grid
#SBATCH --account=YOUR_PROJECT          # CHANGE ME (GPU allocation, e.g. m1234_g)
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --time=12:00:00
#SBATCH --nodes=1                      # raise to 2-4 nodes to run more concurrently
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --output=logs/grid_%j.out
#SBATCH --error=logs/grid_%j.err
# Whole-node alternative to run_train.sbatch: runs a job list with
# run_jobs_parallel.sh (4 GPUs x 3 processes). Resubmit the same command
# until the log ends with every line finished; completed runs are skipped
# and interrupted runs resume.
#
#   sbatch scripts/slurm/run_train_node.sbatch scripts/slurm/jobs_l63.txt        # 3 procs/GPU
#   sbatch scripts/slurm/run_train_node.sbatch scripts/slurm/jobs_ks_l22.txt 2    # 2 procs/GPU for KS
set -euo pipefail
JOBFILE=${1:?usage: sbatch run_train_node.sbatch <jobfile> [procs_per_gpu=3]}
PER=${2:-3}                             # use 2 for KS (long unrolls need the memory)
module load pytorch
export OMP_NUM_THREADS=4
cd $SLURM_SUBMIT_DIR
bash scripts/slurm/run_jobs_multinode.sh "$JOBFILE" "$PER"
