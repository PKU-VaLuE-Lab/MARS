#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
MODE="${COSMOS_EVAL_MODE:-mars}"

GPU="${GPU:-0}"
ARGS=()
HAS_LOG_PATH=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    baseline|mars)
      MODE="$1"
      shift
      ;;
    --mode)
      MODE="$2"
      shift 2
      ;;
    --mode=*)
      MODE="${1#*=}"
      shift
      ;;
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

case "${MODE}" in
  baseline)
    VARIANT="baseline"
    MECHANISMS=""
    ;;
  mars)
    VARIANT="mars"
    MECHANISMS="actionaware,selfcross,vaeskip"
    ;;
  *)
    echo "Unknown evaluation mode: ${MODE}. Expected baseline or mars." >&2
    exit 2
    ;;
esac

mkdir -p "${SCRIPT_DIR}/logs"
if [[ "${HAS_LOG_PATH}" == false ]]; then
  ARGS+=(--log-path "${SCRIPT_DIR}/logs/liberopro_${VARIANT}_all_seed195.log")
fi

LOCK_FILE="/tmp/cosmos-policy-gpu-${GPU}.lock"
if [[ "${COSMOS_GPU_LOCK_HELD:-0}" != "1" ]]; then
  exec 9>"${LOCK_FILE}"
  if ! flock -n 9; then
    echo "GPU ${GPU} is already locked by another cosmos-policy eval." >&2
    exit 1
  fi
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate liberocosmos

PYLIB="$(python - <<'PY'
import sysconfig
print(sysconfig.get_path("purelib"))
PY
)"
if [[ -d "${PYLIB}/nvidia" ]]; then
  export CUDA_HOME="${PYLIB}/nvidia"
  export CUDA_PATH="${CUDA_HOME}"
  NVIDIA_LIB_PATHS="$(find "${PYLIB}/nvidia" -maxdepth 3 -type d -name lib | paste -sd:)"
  export LD_LIBRARY_PATH="${NVIDIA_LIB_PATHS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

export PYTHONPATH="${ROOT_DIR}/LIBERO-PRO:${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export COSMOS_PHYSICAL_GPU="${GPU}"
export COSMOS_SPEED_GUARD_ENABLED="${COSMOS_SPEED_GUARD_ENABLED:-1}"
export COSMOS_SPEED_GUARD_LEAVE_GIB="${COSMOS_SPEED_GUARD_LEAVE_GIB:-10}"
export COSMOS_SPEED_GUARD_MIN_TENSOR_GIB="${COSMOS_SPEED_GUARD_MIN_TENSOR_GIB:-1}"
export COSMOS_SPEED_GUARD_NAME="${COSMOS_SPEED_GUARD_NAME:-测速度}"
export LIBERO_CONFIG_PATH="${ROOT_DIR}/.liberopro"
export HF_HOME="${HF_HOME:-/home/zzy/.cache/cosmos-policy/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export WANDB_MODE=disabled
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-${GPU}}"
export EGL_DEVICE_ID="${EGL_DEVICE_ID:-${GPU}}"
export TMPDIR="${SCRIPT_DIR}/cache/tmp/gpu${GPU}"
mkdir -p "${TMPDIR}"
export COSMOS_EVAL_VARIANT="${VARIANT}"
export LIBEROPRO_FINAL_MECHANISMS="${MECHANISMS}"

if [[ ",${MECHANISMS}," == *",actionaware,"* ]]; then
  export COSMOS_ACTIONAWARE_MODE=route
  export COSMOS_ACTIONAWARE_GATE_CONFIG_JSON='{"episode_latch":false,"logic":"any","name":"actionaware_perquery_low_translation_070","rules":[{"feature":"trans_mag_mean","op":"<=","threshold":0.7}],"type":"rules"}'
  export COSMOS_ACTIONAWARE_TRANS_THRESHOLD=0.7
  export COSMOS_ACTIONAWARE_CHUNK_SIZE=16
  export COSMOS_ACTIONAWARE_FEATURE_STEPS=12
  export COSMOS_ACTIONAWARE_REPORT_EVERY=200
fi

VAE_SUMMARY_PATH=""
if [[ ",${MECHANISMS}," == *",vaeskip,"* ]]; then
  # Match LIBERO's profile-selection mechanism. Explicit global variables
  # keep the uniform-override behavior used by exploration/formal runners.
  if [[ -n "${COSMOS_SHALLOW_GATE_PROFILE:-}" ]]; then
    VAE_PROFILE="${COSMOS_SHALLOW_GATE_PROFILE}"
  elif [[ -n "${COSMOS_SHALLOW_GATE_SUITE_LAYER+x}" ||
          -n "${COSMOS_SHALLOW_GATE_SUITE_TOPK+x}" ||
          -n "${COSMOS_SHALLOW_GATE_SUITE_IOU_THRESHOLD+x}" ||
          -n "${COSMOS_SHALLOW_GATE_SUITE_CAMERA_MODE+x}" ||
          -n "${COSMOS_SHALLOW_GATE_SUITE_PRIMARY_GUARD_THRESHOLD+x}" ]]; then
    VAE_PROFILE="custom"
  elif [[ -n "${COSMOS_SHALLOW_GATE_TOPK+x}" ||
          -n "${COSMOS_SHALLOW_GATE_IOU_THRESHOLD+x}" ||
          -n "${COSMOS_SHALLOW_GATE_CAMERA_MODE+x}" ||
          -n "${COSMOS_SHALLOW_GATE_PRIMARY_GUARD_THRESHOLD+x}" ]]; then
    VAE_PROFILE="uniform"
  else
    VAE_PROFILE="liberopro_acc"
  fi
  export COSMOS_SHALLOW_GATE_PROFILE="${VAE_PROFILE}"

  # RGB is the finalized low-overhead VAESkip gate. It bypasses the shallow
  # VAE probe while preserving the same change-map/Top-K-IoU reuse mechanism.
  export COSMOS_SHALLOW_GATE_LAYER="${COSMOS_SHALLOW_GATE_LAYER:-rgb}"
  export COSMOS_SHALLOW_GATE_TOPK="${COSMOS_SHALLOW_GATE_TOPK:-0.15}"
  case "${VAE_PROFILE}" in
    liberopro_acc|acc)
      # Conservative default retained for accuracy-oriented direct launches.
      export COSMOS_SHALLOW_GATE_IOU_THRESHOLD="${COSMOS_SHALLOW_GATE_IOU_THRESHOLD:-0.98}"
      export COSMOS_SHALLOW_GATE_CAMERA_MODE="${COSMOS_SHALLOW_GATE_CAMERA_MODE:-hybrid}"
      export COSMOS_SHALLOW_GATE_PRIMARY_GUARD_THRESHOLD="${COSMOS_SHALLOW_GATE_PRIMARY_GUARD_THRESHOLD:-0.90}"
      export COSMOS_SHALLOW_GATE_PRIMARY_MAX_CONSECUTIVE_SKIPS="${COSMOS_SHALLOW_GATE_PRIMARY_MAX_CONSECUTIVE_SKIPS:-1}"
      export COSMOS_SHALLOW_GATE_WRIST_MAX_CONSECUTIVE_SKIPS="${COSMOS_SHALLOW_GATE_WRIST_MAX_CONSECUTIVE_SKIPS:-1}"
      unset COSMOS_SHALLOW_GATE_SUITE_LAYER
      unset COSMOS_SHALLOW_GATE_SUITE_TOPK
      unset COSMOS_SHALLOW_GATE_SUITE_IOU_THRESHOLD
      unset COSMOS_SHALLOW_GATE_SUITE_ENABLED
      unset COSMOS_SHALLOW_GATE_SUITE_MAX_CONSECUTIVE_SKIPS
      unset COSMOS_SHALLOW_GATE_SUITE_CAMERA_MODE
      unset COSMOS_SHALLOW_GATE_SUITE_PRIMARY_GUARD_THRESHOLD
      unset COSMOS_SHALLOW_GATE_SUITE_FORCE_FULL_INTERVAL
      unset COSMOS_SHALLOW_GATE_SUITE_FORCE_FULL_AFTER_1STEP
      ;;
    liberopro_speed|speed)
      # Uniform PRO SPEED profile selected by the no-Flash seed-195 sweep.
      # The all400 boundary check retained p=.30/w=.45: p=.25/w=.40 gained
      # only 0.93 percentage points of skip but was net -2 successes, while
      # p=.20/w=.35 gained 1.63 points and was net -4. Main-view reuse may
      # continue twice; wrist reuse remains capped at one.
      export COSMOS_SHALLOW_GATE_IOU_THRESHOLD="${COSMOS_SHALLOW_GATE_IOU_THRESHOLD:-0.45}"
      export COSMOS_SHALLOW_GATE_CAMERA_MODE="${COSMOS_SHALLOW_GATE_CAMERA_MODE:-hybrid}"
      export COSMOS_SHALLOW_GATE_PRIMARY_GUARD_THRESHOLD="${COSMOS_SHALLOW_GATE_PRIMARY_GUARD_THRESHOLD:-0.30}"
      export COSMOS_SHALLOW_GATE_PRIMARY_MAX_CONSECUTIVE_SKIPS="${COSMOS_SHALLOW_GATE_PRIMARY_MAX_CONSECUTIVE_SKIPS:-2}"
      export COSMOS_SHALLOW_GATE_WRIST_MAX_CONSECUTIVE_SKIPS="${COSMOS_SHALLOW_GATE_WRIST_MAX_CONSECUTIVE_SKIPS:-1}"
      unset COSMOS_SHALLOW_GATE_SUITE_LAYER
      unset COSMOS_SHALLOW_GATE_SUITE_TOPK
      unset COSMOS_SHALLOW_GATE_SUITE_IOU_THRESHOLD
      unset COSMOS_SHALLOW_GATE_SUITE_ENABLED
      unset COSMOS_SHALLOW_GATE_SUITE_MAX_CONSECUTIVE_SKIPS
      unset COSMOS_SHALLOW_GATE_SUITE_CAMERA_MODE
      unset COSMOS_SHALLOW_GATE_SUITE_PRIMARY_GUARD_THRESHOLD
      unset COSMOS_SHALLOW_GATE_SUITE_FORCE_FULL_INTERVAL
      unset COSMOS_SHALLOW_GATE_SUITE_FORCE_FULL_AFTER_1STEP
      ;;
    uniform)
      export COSMOS_SHALLOW_GATE_IOU_THRESHOLD="${COSMOS_SHALLOW_GATE_IOU_THRESHOLD:-0.98}"
      export COSMOS_SHALLOW_GATE_CAMERA_MODE="${COSMOS_SHALLOW_GATE_CAMERA_MODE:-hybrid}"
      export COSMOS_SHALLOW_GATE_PRIMARY_GUARD_THRESHOLD="${COSMOS_SHALLOW_GATE_PRIMARY_GUARD_THRESHOLD:-0.90}"
      unset COSMOS_SHALLOW_GATE_SUITE_LAYER
      unset COSMOS_SHALLOW_GATE_SUITE_TOPK
      unset COSMOS_SHALLOW_GATE_SUITE_IOU_THRESHOLD
      unset COSMOS_SHALLOW_GATE_SUITE_ENABLED
      unset COSMOS_SHALLOW_GATE_SUITE_MAX_CONSECUTIVE_SKIPS
      unset COSMOS_SHALLOW_GATE_SUITE_CAMERA_MODE
      unset COSMOS_SHALLOW_GATE_SUITE_PRIMARY_GUARD_THRESHOLD
      unset COSMOS_SHALLOW_GATE_SUITE_FORCE_FULL_INTERVAL
      unset COSMOS_SHALLOW_GATE_SUITE_FORCE_FULL_AFTER_1STEP
      ;;
    custom)
      export COSMOS_SHALLOW_GATE_IOU_THRESHOLD="${COSMOS_SHALLOW_GATE_IOU_THRESHOLD:-0.65}"
      export COSMOS_SHALLOW_GATE_CAMERA_MODE="${COSMOS_SHALLOW_GATE_CAMERA_MODE:-min}"
      export COSMOS_SHALLOW_GATE_PRIMARY_GUARD_THRESHOLD="${COSMOS_SHALLOW_GATE_PRIMARY_GUARD_THRESHOLD:-0.0}"
      ;;
    *)
      echo "Unknown COSMOS_SHALLOW_GATE_PROFILE=${VAE_PROFILE}" >&2
      exit 2
      ;;
  esac
  export COSMOS_SHALLOW_GATE_MAX_CONSECUTIVE_SKIPS="${COSMOS_SHALLOW_GATE_MAX_CONSECUTIVE_SKIPS:-1}"
  export COSMOS_SHALLOW_GATE_PRIMARY_MAX_CONSECUTIVE_SKIPS="${COSMOS_SHALLOW_GATE_PRIMARY_MAX_CONSECUTIVE_SKIPS:-${COSMOS_SHALLOW_GATE_MAX_CONSECUTIVE_SKIPS}}"
  export COSMOS_SHALLOW_GATE_WRIST_MAX_CONSECUTIVE_SKIPS="${COSMOS_SHALLOW_GATE_WRIST_MAX_CONSECUTIVE_SKIPS:-${COSMOS_SHALLOW_GATE_MAX_CONSECUTIVE_SKIPS}}"
  export COSMOS_SHALLOW_GATE_QUERY_LAG=1
  export COSMOS_SHALLOW_GATE_FORCE_FULL_INTERVAL=0
  export COSMOS_SHALLOW_GATE_FORCE_FULL_AFTER_1STEP=False
  # Production eval must not inherit probe-only analysis settings: TRACE_PATH
  # intentionally evaluates otherwise guard-blocked queries, while TOPKS adds
  # extra IoU work.  Reuse itself must also remain enabled for final runs.
  unset COSMOS_SHALLOW_GATE_TRACE_PATH
  unset COSMOS_SHALLOW_GATE_ANALYSIS_TOPKS
  unset COSMOS_SHALLOW_GATE_DISABLE_REUSE
  VAE_SUMMARY_PATH="/tmp/liberopro-final-${VARIANT}-gpu${GPU}-$$-vae-summary.json"
  export COSMOS_SHALLOW_GATE_SUMMARY_PATH="${VAE_SUMMARY_PATH}"
fi

cleanup() {
  if [[ -n "${VAE_SUMMARY_PATH}" && -f "${VAE_SUMMARY_PATH}" ]]; then
    unlink "${VAE_SUMMARY_PATH}"
  fi
}
trap cleanup EXIT

cd "${ROOT_DIR}"
python "${SCRIPT_DIR}/eval_wrapper.py" "${ARGS[@]}"
