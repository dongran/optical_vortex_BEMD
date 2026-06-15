#!/usr/bin/env bash
# Batch BEMD for loam0..loam3, steps 2000-4000 (same pipeline as 600-2000).
#
# Pipeline:
#   - BEMD on Ex/Ey/Ez (V1/V2/V3) only
#   - vector-denoised: remove IMF1 per component, synthesize |E|
#   - Fixed colorbar across frames / MP4
#   - gridfit linear solver: normal (fast)
#
# Output (does not touch 600-2000 results):
#   python_output/loam{N}_vector_normal_steps_2000_4000/
#
# Usage (inside screen):
#   cd /path/to/Learning_HHT_FDTD_simulation
#   bash optical_vortex/optical_vortex_BEMD/run_all_loam_vector_normal_2000_4000.sh
#
# Optional overrides:
#   WORKERS=16 bash .../run_all_loam_vector_normal_2000_4000.sh
#   DATASETS="loam1 loam3" bash ...
#   NO_FRAMES=1 bash ...              # BEMD only (~114GB total, saves ~10GB)
#   LIMIT_STEPS=3 bash ...            # smoke test
#   CONTINUE_ON_ERROR=1 bash ...      # continue if one dataset fails

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PIPELINE="${REPO_ROOT}/optical_vortex/optical_vortex_BEMD/bemd_python_loam1_pipeline.py"
LOG_DIR="${REPO_ROOT}/optical_vortex/optical_vortex_BEMD/python_output/logs"
mkdir -p "$LOG_DIR"

START_STEP="${START_STEP:-2000}"
END_STEP="${END_STEP:-4000}"
WORKERS="${WORKERS:-16}"
CROP_SIZE="${CROP_SIZE:-560}"
NIMFS="${NIMFS:-3}"
FPS="${FPS:-12}"
OUTPUT_SUFFIX="${OUTPUT_SUFFIX:-vector_normal}"
GRIDFIT_LINEAR_SOLVER="${GRIDFIT_LINEAR_SOLVER:-normal}"
DATASETS="${DATASETS:-loam0 loam1 loam2 loam3}"

MASTER_LOG="${LOG_DIR}/all_loam_${OUTPUT_SUFFIX}_${START_STEP}_${END_STEP}_$(date +%Y%m%d_%H%M%S).log"

run_one_dataset() {
  local dataset="$1"
  local input_dir="${REPO_ROOT}/optical_vortex/data/${dataset}"
  local output_dir="${REPO_ROOT}/optical_vortex/optical_vortex_BEMD/python_output/${dataset}_${OUTPUT_SUFFIX}_steps_${START_STEP}_${END_STEP}"
  local log_file="${LOG_DIR}/${dataset}_${OUTPUT_SUFFIX}_${START_STEP}_${END_STEP}_$(date +%Y%m%d_%H%M%S).log"

  if [[ ! -d "$input_dir" ]]; then
    echo "[${dataset}] SKIP: input dir not found: $input_dir"
    return 0
  fi

  local -a args=(
    --input-dir "$input_dir"
    --output-dir "$output_dir"
    --dataset-label "$dataset"
    --start-step "$START_STEP"
    --end-step "$END_STEP"
    --crop-size "$CROP_SIZE"
    --nimfs "$NIMFS"
    --reconstruction-mode vector-denoised
    --gridfit-linear-solver "$GRIDFIT_LINEAR_SOLVER"
    --workers "$WORKERS"
    --threads-per-worker 1
    --fps "$FPS"
  )

  if [[ -n "${STEP_STRIDE:-}" ]]; then
    args+=(--step-stride "$STEP_STRIDE")
  fi
  if [[ -n "${LIMIT_STEPS:-}" ]]; then
    args+=(--limit-steps "$LIMIT_STEPS")
  fi
  if [[ -n "${NO_FRAMES:-}" ]]; then
    args+=(--no-frames --no-video)
  fi
  if [[ -n "${OVERWRITE:-}" ]]; then
    args+=(--overwrite)
  fi
  if [[ -n "${OVERWRITE_FRAMES:-}" ]]; then
    args+=(--overwrite-frames)
  fi

  echo ""
  echo "================================================================"
  echo "=== ${dataset} vector BEMD steps ${START_STEP}-${END_STEP} ==="
  echo "================================================================"
  echo "input:   $input_dir"
  echo "output:  $output_dir"
  echo "steps:   $START_STEP .. $END_STEP"
  echo "crop:    $CROP_SIZE, nimfs=$NIMFS"
  echo "mode:    vector-denoised"
  echo "gridfit: $GRIDFIT_LINEAR_SOLVER"
  echo "workers: $WORKERS"
  echo "log:     $log_file"
  echo ""
  echo "python3 $PIPELINE ${args[*]}"
  echo ""

  cd "$REPO_ROOT"
  python3 "$PIPELINE" "${args[@]}" 2>&1 | tee "$log_file"
}

{
  echo "=== batch: loam0-3 vector_normal ${START_STEP}-${END_STEP} ==="
  echo "repo:     $REPO_ROOT"
  echo "datasets: $DATASETS"
  echo "suffix:   $OUTPUT_SUFFIX"
  echo "steps:    $START_STEP .. $END_STEP"
  echo "master:   $MASTER_LOG"
  echo ""

  for dataset in $DATASETS; do
    if [[ -n "${CONTINUE_ON_ERROR:-}" ]]; then
      run_one_dataset "$dataset" || echo "[${dataset}] FAILED (continuing)"
    else
      run_one_dataset "$dataset"
    fi
  done

  echo ""
  echo "=== all requested datasets finished ==="
} 2>&1 | tee "$MASTER_LOG"
