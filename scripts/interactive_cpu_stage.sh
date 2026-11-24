#!/bin/bash
# Run inside an allocated compute node.
set -euo pipefail
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
exec python -u scripts/cpu_stage.py "${1:-v12r1}"
