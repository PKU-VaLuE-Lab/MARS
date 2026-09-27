#!/usr/bin/env bash
set -euo pipefail

FINAL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${FINAL_DIR}/.." && pwd)"
PYTHON_BIN="${ROOT_DIR}/.conda/bin/python"

VARIANT="mars"
GPU="${GPU:-0}"
TRIALS="${TRIALS:-50}"
MAX_TASKS="${MAX_TASKS:-0}"
SEED="${SEED:-195}"
OUTPUT_DIR="${OUTPUT_DIR:-${FINAL_DIR}/results/${VARIANT}_all2000_seed${SEED}}"
LOG_PATH="${LOG_PATH:-${FINAL_DIR}/logs/fastwam_${VARIANT}_all2000_seed${SEED}.log}"

export FASTWAM_ACTIONAWARE_ENABLED=1
export FASTWAM_ACTIONAWARE_MODE=route
export FASTWAM_ACTIONAWARE_DRAFT_STEPS="${FASTWAM_ACTIONAWARE_DRAFT_STEPS:-1}"
export FASTWAM_ACTIONAWARE_TRANS_THRESHOLD="${FASTWAM_ACTIONAWARE_TRANS_THRESHOLD:-0.30}"
export FASTWAM_ACTION_CACHE_ENABLED="${FASTWAM_ACTION_CACHE_ENABLED:-1}"
export FASTWAM_ACTION_CACHE_MODE="${FASTWAM_ACTION_CACHE_MODE:-input_accum_residual}"
export FASTWAM_ACTION_CACHE_RELATIVE_L1_THRESHOLD="${FASTWAM_ACTION_CACHE_RELATIVE_L1_THRESHOLD:-0.20}"
export FASTWAM_ACTION_CACHE_RELATIVE_MAX_THRESHOLD="${FASTWAM_ACTION_CACHE_RELATIVE_MAX_THRESHOLD:-inf}"
export FASTWAM_ACTION_CACHE_MAX_CONSECUTIVE_REUSE="${FASTWAM_ACTION_CACHE_MAX_CONSECUTIVE_REUSE:-3}"
export FASTWAM_ACTION_CACHE_START_BLOCK="${FASTWAM_ACTION_CACHE_START_BLOCK:-1}"

mkdir -p "${FINAL_DIR}/logs" "${OUTPUT_DIR}" \
  "${ROOT_DIR}/.home" \
  "${ROOT_DIR}/.cache/pip" "${ROOT_DIR}/.cache/huggingface" \
  "${ROOT_DIR}/.cache/modelscope" "${ROOT_DIR}/.cache/torch" \
  "${ROOT_DIR}/.cache/datasets" "${ROOT_DIR}/.cache/matplotlib" \
  "${ROOT_DIR}/.cache/numba" "${ROOT_DIR}/.cache/wandb" \
  "${ROOT_DIR}/.cache/nv" "${ROOT_DIR}/.cache/triton" \
  "${ROOT_DIR}/.cache/torchinductor" "${ROOT_DIR}/.cache/tmp"

export CUDA_VISIBLE_DEVICES="${GPU}"
export FASTWAM_PHYSICAL_GPU="${GPU}"
export FASTWAM_ZSPEED_RESERVE="${FASTWAM_ZSPEED_RESERVE:-1}"
export FASTWAM_ZSPEED_LEAVE_GIB="${FASTWAM_ZSPEED_LEAVE_GIB:-12}"
export FASTWAM_ACTIONAWARE_DATASET_STATS="${ROOT_DIR}/checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json"
export FASTWAM_ACTIONAWARE_FEATURE_STEPS="${FASTWAM_ACTIONAWARE_FEATURE_STEPS:-10}"
export FASTWAM_ACTIONAWARE_REPORT_EVERY="${FASTWAM_ACTIONAWARE_REPORT_EVERY:-100}"
export DIFFSYNTH_MODEL_BASE_PATH="${ROOT_DIR}/checkpoints"
export DIFFSYNTH_DOWNLOAD_SOURCE=huggingface
export LIBERO_CONFIG_PATH="${ROOT_DIR}/.libero"
export HOME="${ROOT_DIR}/.home"
export HF_HOME="${ROOT_DIR}/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="${ROOT_DIR}/.cache/huggingface/hub"
export HF_DATASETS_CACHE="${ROOT_DIR}/.cache/datasets"
export TRANSFORMERS_CACHE="${ROOT_DIR}/.cache/huggingface/transformers"
export MODELSCOPE_CACHE="${ROOT_DIR}/.cache/modelscope"
export MODELSCOPE_CACHE_DIR="${ROOT_DIR}/.cache/modelscope"
export XDG_CACHE_HOME="${ROOT_DIR}/.cache"
export PIP_CACHE_DIR="${ROOT_DIR}/.cache/pip"
export TORCH_HOME="${ROOT_DIR}/.cache/torch"
export CUDA_CACHE_PATH="${ROOT_DIR}/.cache/nv"
export TRITON_CACHE_DIR="${ROOT_DIR}/.cache/triton"
export TORCHINDUCTOR_CACHE_DIR="${ROOT_DIR}/.cache/torchinductor"
export MPLCONFIGDIR="${ROOT_DIR}/.cache/matplotlib"
export NUMBA_CACHE_DIR="${ROOT_DIR}/.cache/numba"
export WANDB_DIR="${ROOT_DIR}/.cache/wandb"
export TMPDIR="${ROOT_DIR}/.cache/tmp"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID="${GPU}"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled

GPU_MODE="$(nvidia-smi -i "${GPU}" --query-gpu=compute_mode --format=csv,noheader,nounits | tr -d '_' | tr '[:upper:]' '[:lower:]')"
if [[ "${GPU_MODE}" != "exclusiveprocess" ]]; then
  echo "GPU ${GPU} must be in EXCLUSIVE_PROCESS mode; got ${GPU_MODE}" >&2
  exit 1
fi

exec > >(tee -a "${LOG_PATH}") 2>&1

exec flock "/tmp/cosmos-policy-gpu-${GPU}.lock" \
  "${PYTHON_BIN}" "${ROOT_DIR}/experiments/libero/eval_libero_all.py" \
  task=libero_uncond_2cam224_1e-4 \
  "ckpt=${ROOT_DIR}/checkpoints/fastwam_release/libero_uncond_2cam224.pt" \
  model.redirect_common_files=false \
  "EVALUATION.dataset_stats_path=${ROOT_DIR}/checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json" \
  "EVALUATION.num_trials=${TRIALS}" \
  "EVALUATION.output_dir=${OUTPUT_DIR}" \
  +EVALUATION.save_rollout_video=false \
  "+EVALUATION.max_tasks=${MAX_TASKS}" \
  +EVALUATION.resume=true \
  'MULTIRUN.task_suite_names=[libero_goal,libero_object,libero_spatial,libero_10]' \
  "seed=${SEED}"
