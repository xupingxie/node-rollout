#!/bin/bash
# One-node worker pool; run_jobs_multinode.sh handles node/session partitioning.
# Usage: run_jobs_parallel.sh <jobfile> [n_gpus=4] [procs_per_gpu=3] [shard=0] [shards=1]
# Environment: DRY_RUN=1, RETRY_WAIT=900, MAX_PASSES=48.
# Completed runs skip; incomplete runs resume; fresh run locks are retried.
# CLEAR_LOCKS is deliberately disabled. Do not delete another session's locks.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec python "$SCRIPT_DIR/run_jobs_node.py" "$@"
