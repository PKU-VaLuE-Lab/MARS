#!/usr/bin/env bash
set -euo pipefail

FINAL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OPENPI_ROOT="$(cd -- "${FINAL_DIR}/.." && pwd)"
export OPENPI_ROOT
export CONDA_PKGS_DIRS="${OPENPI_ROOT}/.cache/conda/pkgs"
export CONDA_ENVS_PATH="${OPENPI_ROOT}/.cache/conda/envs"
export PIP_CACHE_DIR="${OPENPI_ROOT}/.cache/pip"
export UV_CACHE_DIR="${OPENPI_ROOT}/.cache/uv"
export XDG_CACHE_HOME="${OPENPI_ROOT}/.cache"
export HF_HOME="${OPENPI_ROOT}/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="${OPENPI_ROOT}/.cache/huggingface/hub"
export TRANSFORMERS_CACHE="${OPENPI_ROOT}/.cache/huggingface/transformers"
export OPENPI_DATA_HOME="${OPENPI_ROOT}/.cache/openpi"
export LIBERO_CONFIG_PATH="${OPENPI_ROOT}/.cache/libero"
export TMPDIR="${OPENPI_ROOT}/.cache/tmp"
export PYTHONPATH="${OPENPI_ROOT}/third_party/libero${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json

mkdir -p \
  "${CONDA_PKGS_DIRS}" "${PIP_CACHE_DIR}" "${UV_CACHE_DIR}" \
  "${HUGGINGFACE_HUB_CACHE}" "${TRANSFORMERS_CACHE}" \
  "${OPENPI_DATA_HOME}" "${LIBERO_CONFIG_PATH}" "${TMPDIR}" \
  "${FINAL_DIR}/logs"

if [[ ! -f "${LIBERO_CONFIG_PATH}/config.yaml" ]]; then
  benchmark_root="${OPENPI_ROOT}/third_party/libero/libero/libero"
  printf 'assets: %s\nbddl_files: %s\nbenchmark_root: %s\ndatasets: %s\ninit_states: %s\n' \
    "${benchmark_root}/assets" \
    "${benchmark_root}/bddl_files" \
    "${benchmark_root}" \
    "${OPENPI_ROOT}/third_party/libero/libero/datasets" \
    "${benchmark_root}/init_files" \
    >"${LIBERO_CONFIG_PATH}/config.yaml"
fi

GPU="${GPU:-2}"
PORT="${PORT:-$((8200 + GPU))}"
TRIALS_PER_TASK="${TRIALS_PER_TASK:-50}"
MAX_TASKS="${MAX_TASKS:-0}"
TASK_START_INDEX="${TASK_START_INDEX:-0}"
FIRST_TASK_TRIAL_START="${FIRST_TASK_TRIAL_START:-0}"
SEED="${SEED:-195}"
SUITES="${SUITES:-libero_spatial,libero_object,libero_goal,libero_10}"
TIMING_WARMUP_QUERIES="${TIMING_WARMUP_QUERIES:-1}"
NUM_STEPS="${OPENPI_NUM_STEPS:-10}"
AA_SHORT_STEPS="${OPENPI_AA_SHORT_STEPS:-1}"
AA_THRESHOLD="${OPENPI_AA_THRESHOLD:-0.7}"
AA_CACHE_REL_L1_THRESHOLD="${OPENPI_AA_CACHE_REL_L1_THRESHOLD:-0.4}"
AA_MAX_CACHED_STEPS="${OPENPI_AA_MAX_CACHED_STEPS:-4}"

if [[ "${GPU}" != "0" && "${GPU}" != "1" && "${GPU}" != "2" && "${GPU}" != "3" && "${GPU}" != "4" ]]; then
  echo "OpenPI evaluations are restricted to GPU0-GPU4; got GPU=${GPU}" >&2
  exit 2
fi

VARIANT=mars
RESULT_LOG="${LOG_PATH:-${FINAL_DIR}/logs/pi05_libero_${VARIANT}_all2000_seed${SEED}.log}"
SERVER_LOG="${RESULT_LOG%.log}_server.log"
if [[ -e "${RESULT_LOG}" || -e "${SERVER_LOG}" ]]; then
  echo "Refusing to overwrite an existing OpenPI log: ${RESULT_LOG} or ${SERVER_LOG}" >&2
  exit 1
fi

LOCK_PATH="/tmp/cosmos-policy-gpu-${GPU}.lock"
exec 9>"${LOCK_PATH}"
if ! flock -n 9; then
  echo "GPU${GPU} is already locked" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${GPU}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID="${GPU}"

SERVER_ARGS=(
  --env LIBERO
  --port "${PORT}"
  --num-steps "${NUM_STEPS}"
)
SERVER_ARGS+=(
  --actionaware
  --actionaware-short-steps "${AA_SHORT_STEPS}"
  --actionaware-threshold "${AA_THRESHOLD}"
  --actionaware-cache-rel-l1-threshold "${AA_CACHE_REL_L1_THRESHOLD}"
  --actionaware-max-cached-steps "${AA_MAX_CACHED_STEPS}"
)

{
  echo "[OpenPI final] variant=${VARIANT} gpu=${GPU} seed=${SEED} suites=${SUITES} trials_per_task=${TRIALS_PER_TASK} max_tasks=${MAX_TASKS}"
  echo "[OpenPI sampler] num_steps=${NUM_STEPS} short_steps=${AA_SHORT_STEPS} action_threshold=${AA_THRESHOLD} cache_rel_l1_threshold=${AA_CACHE_REL_L1_THRESHOLD} max_cached_steps=${AA_MAX_CACHED_STEPS}"
} | tee "${RESULT_LOG}"

cd "${OPENPI_ROOT}"
"${OPENPI_ROOT}/.conda/bin/python" scripts/serve_policy.py "${SERVER_ARGS[@]}" \
  >"${SERVER_LOG}" 2>&1 &
server_pid=$!
cleanup() {
  if kill -0 "${server_pid}" 2>/dev/null; then
    kill -TERM "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

for _ in {1..1800}; do
  if curl --silent --fail "http://127.0.0.1:${PORT}/healthz" >/dev/null; then
    break
  fi
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    echo "OpenPI server exited before becoming ready; see ${SERVER_LOG}" >&2
    exit 1
  fi
  sleep 1
done
if ! curl --silent --fail "http://127.0.0.1:${PORT}/healthz" >/dev/null; then
  echo "Timed out waiting for OpenPI server; see ${SERVER_LOG}" >&2
  exit 1
fi

"${OPENPI_ROOT}/.conda-libero/bin/python" examples/libero/eval_all.py \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --num-trials-per-task "${TRIALS_PER_TASK}" \
  --max-tasks "${MAX_TASKS}" \
  --task-start-index "${TASK_START_INDEX}" \
  --first-task-trial-start "${FIRST_TASK_TRIAL_START}" \
  --suites "${SUITES}" \
  --seed "${SEED}" \
  --timing-warmup-queries "${TIMING_WARMUP_QUERIES}" \
  2>&1 | tee -a "${RESULT_LOG}"
