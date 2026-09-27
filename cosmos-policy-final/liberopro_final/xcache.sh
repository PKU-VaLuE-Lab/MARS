#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

# Shared LIBERO/LIBERO-PRO XCache defaults selected by cross-suite seed-195
# validation. Exploration may override them through the environment.
export COSMOS_XCACHE_NUM_STEPS=5
export COSMOS_XCACHE_FINGERPRINT_TOKENS="${COSMOS_XCACHE_FINGERPRINT_TOKENS:-32}"
export COSMOS_XCACHE_ALPHA="${COSMOS_XCACHE_ALPHA:-0.3}"
export COSMOS_XCACHE_TAU_FLOOR="${COSMOS_XCACHE_TAU_FLOOR:-0.98}"
export COSMOS_XCACHE_MARGIN="${COSMOS_XCACHE_MARGIN:-0.02}"
export COSMOS_XCACHE_DEV_THRESHOLD="${COSMOS_XCACHE_DEV_THRESHOLD:-45.0}"
export COSMOS_XCACHE_FRONT_ANCHOR="${COSMOS_XCACHE_FRONT_ANCHOR:-1}"
export COSMOS_XCACHE_VIEW_GROUPS="${COSMOS_XCACHE_VIEW_GROUPS:-all;2,6;3,7}"
export COSMOS_XCACHE_VERBOSE="${COSMOS_XCACHE_VERBOSE:-True}"
export COSMOS_XCACHE_REPORT_EVERY="${COSMOS_XCACHE_REPORT_EVERY:-200}"
export COSMOS_XCACHE_TRACE_LIMIT="${COSMOS_XCACHE_TRACE_LIMIT:-0}"
unset COSMOS_XCACHE_STATS_PATH

GPU="${GPU:-1}"
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

LOG_DIR="${COSMOS_XCACHE_LOG_DIR:-${SCRIPT_DIR}/xcachelogs}"
mkdir -p "${LOG_DIR}"
if [[ "${HAS_LOG_PATH}" == false ]]; then
  ARGS+=(--log-path "${LOG_DIR}/liberopro_xcache_all2000_seed195.log")
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
export COSMOS_EVAL_VARIANT=xcache
export LIBEROPRO_FINAL_MECHANISMS=""
unset LIBERO_FINAL_MECHANISMS ROBOCASA_FINAL_MECHANISMS
unset COSMOS_ACTIONAWARE_MODE COSMOS_ACTIONAWARE_GATE_CONFIG_JSON
unset COSMOS_STEPSKIP_USE_REL_L1 COSMOS_STEPSKIP_NUM_STEPS
unset COSMOS_DICACHE_NUM_STEPS COSMOS_DICACHE_REL_L1_THRESH
unset COSMOS_C3CACHE_NUM_STEPS COSMOS_C3CACHE_CACHED_STEPS COSMOS_C3CACHE_REFRESH_INTERVAL
unset COSMOS_SHALLOW_GATE_PROFILE COSMOS_SHALLOW_GATE_TRACE_PATH

cd "${ROOT_DIR}"
exec python "${SCRIPT_DIR}/eval_xcache_wrapper.py" "${ARGS[@]}"
