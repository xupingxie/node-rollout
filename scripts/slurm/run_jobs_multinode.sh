#!/bin/bash
# Run disjoint portions of one job list across nodes AND allocations.
# Usage: run_jobs_multinode.sh <jobfile> [procs_per_gpu=3] [session_id=0] [sessions=1]
# Two independent two-node allocations use identical jobfiles and:
#   session A: bash scripts/slurm/run_jobs_multinode.sh jobs_v12r1/l63.txt 3 0 2
#   session B: bash scripts/slurm/run_jobs_multinode.sh jobs_v12r1/l63.txt 3 1 2
# Resume by requesting a new allocation and repeating the SAME command.
# Defaults preserve the original single-allocation interface.
set -euo pipefail
JOBFILE=${1:?usage: run_jobs_multinode.sh <jobfile> [procs_per_gpu] [session_id] [sessions]}
PER=${2:-3}
SESSION=${3:-0}
SESSIONS=${4:-1}
NN=${SLURM_JOB_NUM_NODES:?run inside a SLURM allocation}
for value in "$PER" "$SESSION" "$SESSIONS" "$NN"; do
  if ! [[ "$value" =~ ^(0|[1-9][0-9]*)$ ]]; then
    echo "ERROR: process, session and node arguments must be integers" >&2
    exit 2
  fi
done
if (( PER < 1 || SESSIONS < 1 || SESSION >= SESSIONS || NN < 1 )); then
  echo "ERROR: require positive process/node/session counts and 0 <= session_id < sessions" >&2
  exit 2
fi
if [ ! -s "$JOBFILE" ]; then
  echo "ERROR: missing or empty job file: $JOBFILE" >&2
  exit 2
fi
echo "Session $SESSION/$SESSIONS: $NN nodes, 4 GPUs/node, $PER processes/GPU"
# First distribute original lines by session; then distribute that session's
# share over its nodes. Session ownership stays disjoint even if node counts differ.
srun -N "$NN" --ntasks="$NN" --ntasks-per-node=1 --gpus-per-node=4 --cpu-bind=none \
  bash -c '
    shard=$(( SLURM_PROCID * $4 + $3 ))
    shards=$(( $5 * $4 ))
    bash scripts/slurm/run_jobs_parallel.sh "$1" 4 "$2" "$shard" "$shards"
  ' _ "$JOBFILE" "$PER" "$SESSION" "$SESSIONS" "$NN"
