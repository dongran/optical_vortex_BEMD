#!/usr/bin/env bash
# Run MATLAB-aligned Python BEMD for optical_vortex/data/loam*.
#
# Defaults match the legacy MATLAB .mat files:
#   crop=500, nimfs=2, scalar E decomposed, summary uses scalar E residue.
#
# Usage:
#   DATASET=loam3 bash optical_vortex/optical_vortex_BEMD/run_matlab_aligned_bemd.sh
#   DATASET=loam1 WORKERS=16 END_STEP=2000 bash optical_vortex/optical_vortex_BEMD/run_matlab_aligned_bemd.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PIPELINE="${REPO_ROOT}/optical_vortex/optical_vortex_BEMD/bemd_python_loam1_pipeline.py"

DATASET="${DATASET:-loam1}"
START_STEP="${START_STEP:-600}"
# Legacy MATLAB BIMF0..1399 maps to steps 600..1999.
END_STEP="${END_STEP:-1999}"
WORKERS="${WORKERS:-16}"
CROP_SIZE="${CROP_SIZE:-500}"
NIMFS="${NIMFS:-2}"
FPS="${FPS:-12}"
GRIDFIT_LINEAR_SOLVER="${GRIDFIT_LINEAR_SOLVER:-backslash}"

INPUT_DIR="${REPO_ROOT}/optical_vortex/data/${DATASET}"
OUTPUT_DIR="${REPO_ROOT}/optical_vortex/optical_vortex_BEMD/python_output/${DATASET}_matlab_aligned_steps_${START_STEP}_${END_STEP}"
LOG_DIR="${REPO_ROOT}/optical_vortex/optical_vortex_BEMD/python_output/logs"
mkdir -p "$LOG_DIR"

LOG_FILE="${LOG_DIR}/${DATASET}_matlab_aligned_${START_STEP}_${END_STEP}_$(date +%Y%m%d_%H%M%S).log"

ARGS=(
  --input-dir "$INPUT_DIR"
  --output-dir "$OUTPUT_DIR"
  --dataset-label "$DATASET"
  --start-step "$START_STEP"
  --end-step "$END_STEP"
  --crop-size "$CROP_SIZE"
  --nimfs "$NIMFS"
  --decompose-e
  --reconstruction-mode matlab-e-residue
  --gridfit-linear-solver "$GRIDFIT_LINEAR_SOLVER"
  --workers "$WORKERS"
  --threads-per-worker 1
  --fps "$FPS"
)

if [[ -n "${STEP_STRIDE:-}" ]]; then
  ARGS+=(--step-stride "$STEP_STRIDE")
fi
if [[ -n "${LIMIT_STEPS:-}" ]]; then
  ARGS+=(--limit-steps "$LIMIT_STEPS")
fi
if [[ -n "${NO_FRAMES:-}" ]]; then
  ARGS+=(--no-frames --no-video)
fi
if [[ -n "${OVERWRITE:-}" ]]; then
  ARGS+=(--overwrite)
fi
if [[ -n "${OVERWRITE_FRAMES:-}" ]]; then
  ARGS+=(--overwrite-frames)
fi

echo "=== ${DATASET} MATLAB-aligned BEMD batch ==="
echo "repo:    $REPO_ROOT"
echo "input:   $INPUT_DIR"
echo "output:  $OUTPUT_DIR"
echo "steps:   $START_STEP .. $END_STEP"
echo "workers: $WORKERS"
echo "log:     $LOG_FILE"
echo ""
echo "Command:"
echo "python3 $PIPELINE ${ARGS[*]}"
echo ""

cd "$REPO_ROOT"
exec python3 "$PIPELINE" "${ARGS[@]}" 2>&1 | tee "$LOG_FILE"
