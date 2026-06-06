#!/usr/bin/env bash
# Re-run BEMD for loam0..loam3 into NEW output directories (does not touch old runs).
#
# Physical pipeline (canonical):
#   - BEMD on Ex/Ey/Ez (V1/V2/V3) only
#   - Remove IMF1 per component, then synthesize |E|
#   - Fixed colorbar across all frames in each video
#   - gridfit linear solver: backslash (closer to MATLAB A\rhs than normal equations)
#
# Old results stay in:
#   python_output/loam{N}_steps_0600_2000/
# New results go to:
#   python_output/loam{N}_vector_backslash_steps_0600_2000/
#
# Usage (recommended inside screen):
#   cd /path/to/Learning_HHT_FDTD_simulation
#   bash optical_vortex/optical_vortex_BEMD/run_all_loam_vector_backslash.sh
#
# Optional overrides:
#   WORKERS=16 START_STEP=600 END_STEP=2000 bash .../run_all_loam_vector_backslash.sh
#   OUTPUT_SUFFIX=my_tag bash ...          # loam1_my_tag_steps_0600_2000
#   DATASETS="loam1 loam3" bash ...        # subset only
#   NO_FRAMES=1 bash ...                   # skip PNG/MP4 (BEMD only)
#   LIMIT_STEPS=3 bash ...                 # smoke test per dataset
#   CONTINUE_ON_ERROR=1 bash ...           # keep going if one dataset fails

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PIPELINE="${REPO_ROOT}/optical_vortex/optical_vortex_BEMD/bemd_python_loam1_pipeline.py"
LOG_DIR="${REPO_ROOT}/optical_vortex/optical_vortex_BEMD/python_output/logs"
mkdir -p "$LOG_DIR"

START_STEP="${START_STEP:-600}"
END_STEP="${END_STEP:-2000}"
WORKERS="${WORKERS:-16}"
CROP_SIZE="${CROP_SIZE:-560}"
NIMFS="${NIMFS:-3}"
FPS="${FPS:-12}"
OUTPUT_SUFFIX="${OUTPUT_SUFFIX:-vector_backslash}"
GRIDFIT_LINEAR_SOLVER="${GRIDFIT_LINEAR_SOLVER:-backslash}"
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
  echo "=== ${dataset} vector BEMD (new output, no overwrite of old) ==="
  echo "================================================================"
  echo "input:   $input_dir"
  echo "output:  $output_dir"
  echo "steps:   $START_STEP .. $END_STEP"
  echo "crop:    $CROP_SIZE, nimfs=$NIMFS"
  echo "mode:    vector-denoised (Ex/Ey/Ez minus IMF1 -> |E|)"
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
  echo "=== batch: all loam vector backslash ==="
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
