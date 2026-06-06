#!/usr/bin/env bash
# Batch BEMD for optical_vortex/data/loam3 (same workflow as loam1).
#
# Usage (inside screen):
#   cd /path/to/Learning_HHT_FDTD_simulation
#   bash optical_vortex/optical_vortex_BEMD/run_loam3_bemd.sh
#
# Optional overrides:
#   WORKERS=16 START_STEP=600 END_STEP=2000 bash optical_vortex/optical_vortex_BEMD/run_loam3_bemd.sh
#   NO_FRAMES=1 bash ...          # skip PNG frames / MP4 (BEMD only)
#   OVERWRITE=1 bash ...          # recompute existing .npz
#   LIMIT_STEPS=3 bash ...        # smoke test (first 3 steps in range)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PIPELINE="${REPO_ROOT}/optical_vortex/optical_vortex_BEMD/bemd_python_loam1_pipeline.py"
INPUT_DIR="${REPO_ROOT}/optical_vortex/data/loam3"
OUTPUT_DIR="${REPO_ROOT}/optical_vortex/optical_vortex_BEMD/python_output/loam3_steps_${START_STEP:-0600}_${END_STEP:-2000}"
LOG_DIR="${REPO_ROOT}/optical_vortex/optical_vortex_BEMD/python_output/logs"
mkdir -p "$LOG_DIR"

START_STEP="${START_STEP:-600}"
END_STEP="${END_STEP:-2000}"
WORKERS="${WORKERS:-16}"
CROP_SIZE="${CROP_SIZE:-560}"
FPS="${FPS:-12}"

LOG_FILE="${LOG_DIR}/loam3_bemd_${START_STEP}_${END_STEP}_$(date +%Y%m%d_%H%M%S).log"

ARGS=(
  --input-dir "$INPUT_DIR"
  --output-dir "$OUTPUT_DIR"
  --start-step "$START_STEP"
  --end-step "$END_STEP"
  --crop-size "$CROP_SIZE"
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

echo "=== loam3 BEMD batch ==="
echo "repo:    $REPO_ROOT"
echo "input:   $INPUT_DIR"
echo "output:  $OUTPUT_DIR"
echo "steps:   $START_STEP .. $END_STEP"
echo "workers: $WORKERS"
echo "log:     $LOG_FILE"
echo ""
echo "Command:"
echo "python $PIPELINE ${ARGS[*]}"
echo ""

cd "$REPO_ROOT"
exec python "$PIPELINE" "${ARGS[@]}" 2>&1 | tee "$LOG_FILE"
