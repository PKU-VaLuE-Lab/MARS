#!/usr/bin/env bash
set -euo pipefail

FINAL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${FINAL_DIR}/env.sh"

GPU="${GPU:-2}"
PORT="${PORT:-$((8200 + GPU))}"
TRIALS_PER_TASK="${TRIALS_PER_TASK:-50}"
MAX_TASKS="${MAX_TASKS:-0}"
TASK_START_INDEX="${TASK_START_INDEX:-0}"
FIRST_TASK_TRIAL_START="${FIRST_TASK_TRIAL_START:-0}"
SEED="${SEED:-195}"
SUITES="${SUITES:-libero_spatial,libero_object,libero_goal,libero_10}"
TIMING_WARMUP_QUERIES="${TIMING_WARMUP_QUERIES:-1}"
ACTIONAWARE="${OPENPI_ACTIONAWARE:-0}"
NUM_STEPS="${OPENPI_NUM_STEPS:-10}"
AA_SHORT_STEPS="${OPENPI_AA_SHORT_STEPS:-5}"
AA_THRESHOLD="${OPENPI_AA_THRESHOLD:-0.35}"
AA_CACHE_REL_L1_THRESHOLD="${OPENPI_AA_CACHE_REL_L1_THRESHOLD:-0.0}"
AA_MAX_CACHED_STEPS="${OPENPI_AA_MAX_CACHED_STEPS:-0}"

if [[ "${GPU}" != "0" && "${GPU}" != "1" && "${GPU}" != "2" && "${GPU}" != "3" && "${GPU}" != "4" ]]; then
  echo "OpenPI evaluations are restricted to GPU0-GPU4; got GPU=${GPU}" >&2
  exit 2
fi

VARIANT=baseline
if [[ "${ACTIONAWARE}" == "1" ]]; then
  VARIANT=actionaware
fi
RESULT_LOG="${LOG_PATH:-${FINAL_DIR}/logs/pi05_libero_${VARIANT}_all2000_seed${SEED}.log}"
SERVER_LOG="${RESULT_LOG%.log}_server.log"
if [[ -e "${RESULT_LOG}" || -e "${SERVER_LOG}" ]]; then
  echo "Refusing to overwrite an existing OpenPI log: ${RESULT_LOG} or ${SERVER_LOG}" >&2
  exit 1
fi

# Enter the signal-protected label before taking the GPU lock.  Children
# inherit the label, so an unrelated shell cannot terminate the zspeed job.
if [[ "${OPENPI_ZSPEED_REEXECED:-0}" != "1" ]] && command -v aa-exec >/dev/null 2>&1; then
  export OPENPI_ZSPEED_REEXECED=1
  exec /usr/bin/aa-exec -p cosmos-zspeed-protect -- "$0" "$@"
fi

LOCK_PATH="/tmp/cosmos-policy-gpu-${GPU}.lock"
exec 9>"${LOCK_PATH}"
if ! flock -n 9; then
  echo "GPU${GPU} is already locked" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${GPU}"
export OPENPI_PHYSICAL_GPU="${GPU}"
export OPENPI_ZSPEED_ENABLED=1
export OPENPI_ZSPEED_PROCESS_NAME=zspeed
export OPENPI_ZSPEED_LEAVE_GIB="${OPENPI_ZSPEED_LEAVE_GIB:-10}"
export OPENPI_ZSPEED_MINIMUM_GIB="${OPENPI_ZSPEED_MINIMUM_GIB:-1}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID="${GPU}"

SERVER_ARGS=(
  --env LIBERO
  --port "${PORT}"
  --num-steps "${NUM_STEPS}"
)
if [[ "${ACTIONAWARE}" == "1" ]]; then
  SERVER_ARGS+=(
    --actionaware
    --actionaware-short-steps "${AA_SHORT_STEPS}"
    --actionaware-threshold "${AA_THRESHOLD}"
    --actionaware-cache-rel-l1-threshold "${AA_CACHE_REL_L1_THRESHOLD}"
    --actionaware-max-cached-steps "${AA_MAX_CACHED_STEPS}"
  )
fi

{
  echo "[OpenPI final] variant=${VARIANT} gpu=${GPU} seed=${SEED} suites=${SUITES} trials_per_task=${TRIALS_PER_TASK} max_tasks=${MAX_TASKS}"
  echo "[OpenPI sampler] num_steps=${NUM_STEPS} short_steps=${AA_SHORT_STEPS} action_threshold=${AA_THRESHOLD} cache_rel_l1_threshold=${AA_CACHE_REL_L1_THRESHOLD} max_cached_steps=${AA_MAX_CACHED_STEPS}"
} | tee "${RESULT_LOG}"

cd "${OPENPI_ROOT}"
/bin/bash -c 'process_name=$1; shift; exec -a "${process_name}" "$@"' \
  _ "${OPENPI_ZSPEED_PROCESS_NAME}" \
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
