#!/usr/bin/env bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
GPU="${GPU:-0}"

source /home/zzy/anaconda3/etc/profile.d/conda.sh
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

ROBOCASA_SHARED_ROOT="/data/Albus/zhangyongchang"
ROBOCASA_SHARED_SITE="${ROBOCASA_SHARED_ROOT}/github_cache/cosmos-policy/.venv/lib/python3.10/site-packages"
export PYTHONPATH="${ROBOCASA_SHARED_ROOT}/benchmark_workspace/python_targets/mujoco326_nodeps:${SCRIPT_DIR}/vendor:${ROBOCASA_SHARED_ROOT}/robocasa_workspace/robosuite:${ROBOCASA_SHARED_SITE}:${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

export CUDA_VISIBLE_DEVICES="${GPU}"
export WANDB_MODE=disabled
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID="${GPU}"
export EGL_DEVICE_ID="${GPU}"
export COSMOS_ROBOCASA_XML_CACHE="${COSMOS_ROBOCASA_XML_CACHE:-/tmp/cosmos-policy-robocasa-cache/gpu${GPU}/xml}"
export TMPDIR="${COSMOS_ROBOCASA_TMPDIR:-/tmp/cosmos-policy-robocasa-cache/gpu${GPU}/tmp}"
export HF_HOME="${HF_HOME:-${ROOT_DIR}/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
mkdir -p "${COSMOS_ROBOCASA_XML_CACHE}" "${TMPDIR}" "${SCRIPT_DIR}/logsfinal" "${SCRIPT_DIR}/logsexplore"
