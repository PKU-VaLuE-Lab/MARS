#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

export COSMOS_C3CACHE_NUM_STEPS=5
export COSMOS_C3CACHE_CACHED_STEPS="${COSMOS_C3CACHE_CACHED_STEPS:-0,1,2,3}"
export COSMOS_C3CACHE_REFRESH_INTERVAL="${COSMOS_C3CACHE_REFRESH_INTERVAL:-8}"
export COSMOS_C3CACHE_VERBOSE="${COSMOS_C3CACHE_VERBOSE:-True}"
export COSMOS_C3CACHE_REPORT_EVERY="${COSMOS_C3CACHE_REPORT_EVERY:-200}"
unset COSMOS_C3CACHE_STATS_PATH

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

LOG_DIR="${COSMOS_C3CACHE_LOG_DIR:-${SCRIPT_DIR}/c3cachelogs}"
mkdir -p "${LOG_DIR}"
if [[ "${HAS_LOG_PATH}" == false ]]; then
  STEP_TAG="${COSMOS_C3CACHE_CACHED_STEPS//,/}"
  ARGS+=(--log-path "${LOG_DIR}/liberopro_c3cache_s${STEP_TAG}_tau${COSMOS_C3CACHE_REFRESH_INTERVAL}_all_seed195.log")
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
export COSMOS_EVAL_VARIANT=c3cache
export LIBEROPRO_FINAL_MECHANISMS=""
unset LIBERO_FINAL_MECHANISMS ROBOCASA_FINAL_MECHANISMS
unset COSMOS_ACTIONAWARE_MODE COSMOS_ACTIONAWARE_GATE_CONFIG_JSON
unset COSMOS_STEPSKIP_USE_REL_L1 COSMOS_STEPSKIP_NUM_STEPS
unset COSMOS_DICACHE_NUM_STEPS COSMOS_DICACHE_REL_L1_THRESH
unset COSMOS_SHALLOW_GATE_PROFILE COSMOS_SHALLOW_GATE_TRACE_PATH

cd "${ROOT_DIR}"
python "${SCRIPT_DIR}/eval_c3cache_wrapper.py" "${ARGS[@]}"
