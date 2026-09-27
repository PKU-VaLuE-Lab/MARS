#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 6 ]]; then
  echo "Usage: $0 <gpu> <seed> [episodes-per-task=50] [suite] [task-start] [task-end]" >&2
  exit 2
fi

gpu="$1"
seed="$2"
test_num="${3:-50}"
suite="${4:-libero_10}"
task_start="${5:-0}"
task_end="${6:-10}"
cache_mode="route"
action_mode="route"

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python}"
checkpoint="${LINGBOT_MODEL_PATH:-${project_root}/checkpoints/lingbot-va-posttrain-libero-long}"
run_name="mars_${suite}_seed${seed}_n${test_num}_tasks${task_start}-${task_end}"
out_dir="${LINGBOT_OUT_DIR:-${project_root}/results/${run_name}}"
log_path="${LINGBOT_LOG_PATH:-${project_root}/logs/${run_name}.log}"

if [[ ! -d "${checkpoint}" ]]; then
  echo "LingBot-VA checkpoint not found: ${checkpoint}" >&2
  exit 1
fi
if [[ ! -d "${project_root}/third_party/LIBERO/libero" ]]; then
  echo "LIBERO not found under ${project_root}/third_party/LIBERO" >&2
  exit 1
fi

mkdir -p "${out_dir}" "$(dirname "${log_path}")" "${project_root}/.cache"

export CUDA_VISIBLE_DEVICES="${gpu}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export PYTHONPATH="${project_root}:${project_root}/third_party/LIBERO${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${HF_HOME:-${project_root}/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-${project_root}/.cache/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${project_root}/.cache/xdg}"
export WANDB_MODE="disabled"
export PYTHONUNBUFFERED="1"
export LINGBOT_MODEL_PATH="${checkpoint}"
export LINGBOT_SAVE_DEBUG="0"
export LINGBOT_SHOW_PROGRESS="0"
export LINGBOT_ENABLE_OFFLOAD="${LINGBOT_ENABLE_OFFLOAD:-1}"
export LINGBOT_VIDEO_STEPS="20"
export LINGBOT_ACTION_STEPS="50"
export LINGBOT_VIDEO_CACHE_MODE="${cache_mode}"
export LINGBOT_VIDEO_RELATIVE_L1_THRESHOLD="${LINGBOT_VIDEO_RELATIVE_L1_THRESHOLD:-0.15}"
export LINGBOT_ACTION_CACHE_MODE="${cache_mode}"
export LINGBOT_ACTION_RELATIVE_L1_THRESHOLD="${LINGBOT_ACTION_RELATIVE_L1_THRESHOLD:-0.25}"
export LINGBOT_ACTIONAWARE_MODE="${action_mode}"
export LINGBOT_ACTIONAWARE_DRAFT_STEPS="${LINGBOT_ACTIONAWARE_DRAFT_STEPS:-1}"
export LINGBOT_ACTIONAWARE_FEATURE_STEPS="${LINGBOT_ACTIONAWARE_FEATURE_STEPS:-10}"
export LINGBOT_ACTIONAWARE_TRANS_THRESHOLD="${LINGBOT_ACTIONAWARE_TRANS_THRESHOLD:-0.25}"

"${python_bin}" "${project_root}/lingbotva_final/eval_local.py" \
  --suite "${suite}" \
  --task-range "${task_start}" "${task_end}" \
  --test-num "${test_num}" \
  --seed "${seed}" \
  --out-dir "${out_dir}" \
  --log-path "${log_path}" 2>&1 | tee -a "${log_path}"
