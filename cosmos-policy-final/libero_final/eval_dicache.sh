#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

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

mkdir -p "${SCRIPT_DIR}/logs"
if [[ "${HAS_LOG_PATH}" == false ]]; then
  ARGS+=(--log-path "${SCRIPT_DIR}/logs/libero_dicache_all_seed195.log")
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

export CUDA_VISIBLE_DEVICES="${GPU}"
export COSMOS_PHYSICAL_GPU="${GPU}"
export COSMOS_SPEED_GUARD_ENABLED="${COSMOS_SPEED_GUARD_ENABLED:-1}"
export COSMOS_SPEED_GUARD_LEAVE_GIB="${COSMOS_SPEED_GUARD_LEAVE_GIB:-10}"
export COSMOS_SPEED_GUARD_MIN_TENSOR_GIB="${COSMOS_SPEED_GUARD_MIN_TENSOR_GIB:-1}"
export COSMOS_SPEED_GUARD_NAME="${COSMOS_SPEED_GUARD_NAME:-测速度}"
export LIBERO_CONFIG_PATH="${ROOT_DIR}/.libero"
export HF_HOME="${HF_HOME:-/home/zzy/.cache/cosmos-policy/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export WANDB_MODE=disabled
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-${GPU}}"
export EGL_DEVICE_ID="${EGL_DEVICE_ID:-${GPU}}"
export COSMOS_EVAL_VARIANT=dicache

# Five denoising steps with the official retention ratio: step 1 is a full
# pass; steps 2-5 probe one block and reuse the cache only while the gate passes.
export COSMOS_DICACHE_NUM_STEPS=5
export COSMOS_DICACHE_RET_RATIO=0.2
unset COSMOS_DICACHE_WARMUP_STEPS
export COSMOS_DICACHE_PROBE_DEPTH=1
export COSMOS_DICACHE_REL_L1_THRESH=0.08
export COSMOS_DICACHE_GAMMA_MIN=1.0
export COSMOS_DICACHE_GAMMA_MAX=2.0
export COSMOS_DICACHE_VERBOSE=True
export COSMOS_DICACHE_REPORT_EVERY="${COSMOS_DICACHE_REPORT_EVERY:-200}"
unset COSMOS_DICACHE_STATS_PATH

cd "${ROOT_DIR}"
python "${SCRIPT_DIR}/eval_dicache_wrapper.py" "${ARGS[@]}"
