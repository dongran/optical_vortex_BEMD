#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-/home/randong/miniconda3/envs/fdtd-memd/bin/python}"
OUTPUT_NAME="${OUTPUT_NAME:-matlab_aligned_fast_preview_560_steps_0600_0897}"
OUTPUT_DIR="${REPO_ROOT}/optical_vortex/optical_vortex_BEMD/python_output/${OUTPUT_NAME}"
LOG_FILE="${OUTPUT_DIR}/run.log"

mkdir -p "$OUTPUT_DIR"
cd "$REPO_ROOT"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

{
  echo "MATLAB-aligned BEMD preview started: $(date --iso-8601=seconds)"
  "$PYTHON" optical_vortex/optical_vortex_BEMD/reprocess_matlab_aligned_npz.py \
    --output-dir "$OUTPUT_DIR" \
    --loam 1 \
    --loam 2 \
    --loam 3 \
    --start-step 600 \
    --end-step 897 \
    --step-stride 3 \
    --nimfs 2 \
    --linear-solver normal \
    --workers 3 \
    --threads-per-worker 1 \
    --continue-on-error

  "$PYTHON" optical_vortex/optical_vortex_BEMD/render_matlab_aligned_comparison.py \
    --input-dir "$OUTPUT_DIR" \
    --output-dir "$OUTPUT_DIR/render" \
    --fps 24 \
    --video-seconds 10
  echo "MATLAB-aligned BEMD preview finished: $(date --iso-8601=seconds)"
} 2>&1 | tee "$LOG_FILE"
