#!/usr/bin/env bash
set -euo pipefail

FINAL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${FINAL_DIR}/.." && pwd)"
VARIANT="mars"
MECHANISMS="actionaware,selfcross,vaeskip"
VAE_PROFILE="safe"

GPU="${GPU:-0}"
ARGS=()
HAS_LOG_PATH=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)
      GPU="$2"
      shift 2
      ;;
    --gpu=*)
      GPU="${1#*=}"
      shift
      ;;
    --log-path)
      HAS_LOG_PATH=true
      ARGS+=("$1" "$2")
      shift 2
      ;;
    --log-path=*)
      HAS_LOG_PATH=true
      ARGS+=("$1")
      shift
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done

mkdir -p "${FINAL_DIR}/logs"
if [[ "${HAS_LOG_PATH}" == false ]]; then
  ARGS+=(--log-path "${FINAL_DIR}/logs/robocasa_${VARIANT}_all_seed195.log")
fi

LOCK_FILE="/tmp/cosmos-policy-gpu-${GPU}.lock"
if [[ "${COSMOS_GPU_LOCK_HELD:-0}" != "1" ]]; then
  exec 9>"${LOCK_FILE}"
  if ! flock -n 9; then
    echo "GPU ${GPU} is already locked by another cosmos-policy eval." >&2
    exit 1
  fi
fi

export GPU
source "${PROJECT_ROOT}/robocasa_scripts/env.sh"

# Match the LIBERO/LIBERO-Pro final runners: reserve otherwise-idle VRAM in
# the evaluation process after the model has loaded. Keeping the reservation
# in-process is required for NVIDIA EXCLUSIVE_PROCESS mode.
export COSMOS_PHYSICAL_GPU="${GPU}"
export COSMOS_SPEED_GUARD_ENABLED="${COSMOS_SPEED_GUARD_ENABLED:-1}"
export COSMOS_SPEED_GUARD_LEAVE_GIB="${COSMOS_SPEED_GUARD_LEAVE_GIB:-10}"
export COSMOS_SPEED_GUARD_MIN_TENSOR_GIB="${COSMOS_SPEED_GUARD_MIN_TENSOR_GIB:-1}"
export COSMOS_SPEED_GUARD_NAME="${COSMOS_SPEED_GUARD_NAME:-测速度}"

export COSMOS_EVAL_VARIANT="${VARIANT}"
export ROBOCASA_FINAL_MECHANISMS="${MECHANISMS}"

if [[ ",${MECHANISMS}," == *",actionaware,"* ]]; then
  export COSMOS_ACTIONAWARE_MODE=route
  export COSMOS_ACTIONAWARE_GATE_CONFIG_JSON='{"episode_latch":false,"logic":"any","name":"robocasa_actionaware_translation_0525","rules":[{"feature":"trans_mag_mean","op":"<=","threshold":0.525}],"type":"rules"}'
  export COSMOS_ACTIONAWARE_TRANS_THRESHOLD=0.525
  export COSMOS_ACTIONAWARE_CHUNK_SIZE=32
  export COSMOS_ACTIONAWARE_FEATURE_STEPS=16
  export COSMOS_ACTIONAWARE_DATASET_STATS="${PROJECT_ROOT}/robocasa_scripts/ckpt/robocasa_dataset_statistics.json"
  export COSMOS_ACTIONAWARE_REPORT_EVERY=200
fi

VAE_SUMMARY_PATH=""
if [[ ",${MECHANISMS}," == *",vaeskip,"* ]]; then
  case "${VAE_PROFILE}" in
    safe)
      # Selected from the four 24x20 exploration runs: this profile had the
      # highest success count (314/480) while still reusing 6.02% of queries.
      DEFAULT_MAIN_IOU_THRESHOLD="0.70"
      DEFAULT_WRIST_IOU_THRESHOLD="0.90"
      ;;
    fast)
      # Higher-reuse alternative retained for eval_standard_fast.sh.
      DEFAULT_MAIN_IOU_THRESHOLD="0.65"
      DEFAULT_WRIST_IOU_THRESHOLD="0.90"
      ;;
    *)
      echo "Missing RoboCasa VAE profile for ${ENTRYPOINT}" >&2
      exit 2
      ;;
  esac
  export COSMOS_SHALLOW_GATE_LAYER="${COSMOS_SHALLOW_GATE_LAYER:-res1}"
  export COSMOS_SHALLOW_GATE_TOPK="${COSMOS_SHALLOW_GATE_TOPK:-0.15}"
  export COSMOS_SHALLOW_GATE_IOU_THRESHOLD="${COSMOS_SHALLOW_GATE_IOU_THRESHOLD:-0.75}"
  export COSMOS_SHALLOW_GATE_WRIST_IOU_THRESHOLD="${COSMOS_SHALLOW_GATE_WRIST_IOU_THRESHOLD:-${DEFAULT_WRIST_IOU_THRESHOLD}}"
  export COSMOS_SHALLOW_GATE_MAIN_IOU_THRESHOLD="${COSMOS_SHALLOW_GATE_MAIN_IOU_THRESHOLD:-${DEFAULT_MAIN_IOU_THRESHOLD}}"
  export COSMOS_SHALLOW_GATE_REUSE_MODE="${COSMOS_SHALLOW_GATE_REUSE_MODE:-hybrid}"
  export COSMOS_SHALLOW_GATE_MAX_CONSECUTIVE_SKIPS="${COSMOS_SHALLOW_GATE_MAX_CONSECUTIVE_SKIPS:-1}"
  export COSMOS_SHALLOW_GATE_QUERY_LAG="${COSMOS_SHALLOW_GATE_QUERY_LAG:-2}"
  export COSMOS_SHALLOW_GATE_WARMUP_QUERIES="${COSMOS_SHALLOW_GATE_WARMUP_QUERIES:-6}"
  export COSMOS_SHALLOW_GATE_CAMERA_MODE="${COSMOS_SHALLOW_GATE_CAMERA_MODE:-main_min}"
  export COSMOS_SHALLOW_GATE_PRIMARY_GUARD_THRESHOLD="${COSMOS_SHALLOW_GATE_PRIMARY_GUARD_THRESHOLD:-0.0}"
  export COSMOS_SHALLOW_GATE_FORCE_FULL_INTERVAL="${COSMOS_SHALLOW_GATE_FORCE_FULL_INTERVAL:-0}"
  export COSMOS_SHALLOW_GATE_FORCE_FULL_AFTER_1STEP="${COSMOS_SHALLOW_GATE_FORCE_FULL_AFTER_1STEP:-False}"
  unset COSMOS_SHALLOW_GATE_TRACE_PATH
  unset COSMOS_SHALLOW_GATE_ANALYSIS_TOPKS
  unset COSMOS_SHALLOW_GATE_DISABLE_REUSE
  unset COSMOS_SHALLOW_GATE_SUITE_LAYER
  unset COSMOS_SHALLOW_GATE_SUITE_TOPK
  unset COSMOS_SHALLOW_GATE_SUITE_IOU_THRESHOLD
  unset COSMOS_SHALLOW_GATE_SUITE_ENABLED
  unset COSMOS_SHALLOW_GATE_SUITE_MAX_CONSECUTIVE_SKIPS
  unset COSMOS_SHALLOW_GATE_SUITE_CAMERA_MODE
  unset COSMOS_SHALLOW_GATE_SUITE_PRIMARY_GUARD_THRESHOLD
  unset COSMOS_SHALLOW_GATE_SUITE_FORCE_FULL_INTERVAL
  unset COSMOS_SHALLOW_GATE_SUITE_FORCE_FULL_AFTER_1STEP
  VAE_SUMMARY_PATH="/tmp/robocasa-final-${VARIANT}-gpu${GPU}-$$-vae-summary.json"
  export COSMOS_SHALLOW_GATE_SUMMARY_PATH="${VAE_SUMMARY_PATH}"
fi

cleanup() {
  if [[ -n "${VAE_SUMMARY_PATH}" && -f "${VAE_SUMMARY_PATH}" ]]; then
    unlink "${VAE_SUMMARY_PATH}"
  fi
}
trap cleanup EXIT

cd "${PROJECT_ROOT}"
python "${FINAL_DIR}/eval_wrapper.py" "${ARGS[@]}"
