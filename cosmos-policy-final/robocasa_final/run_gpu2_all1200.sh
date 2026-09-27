#!/usr/bin/env bash
set -euo pipefail

FINAL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${FINAL_DIR}/.." && pwd)"
GPU=2
SEED=195
TRIALS_PER_TASK=50
QUEUE_LOG="${FINAL_DIR}/logs/robocasa_gpu2_selected_all1200_queue.log"

mkdir -p "${FINAL_DIR}/logs"

compute_mode="$({
  nvidia-smi -i "${GPU}" \
    --query-gpu=compute_mode \
    --format=csv,noheader,nounits
} | tr -d '[:space:]_')"
if [[ "${compute_mode,,}" != "exclusiveprocess" ]]; then
  echo "GPU ${GPU} must be in EXCLUSIVE_PROCESS mode; current mode=${compute_mode}" >&2
  exit 1
fi

exec 9>"/tmp/cosmos-policy-gpu-${GPU}.lock"
flock 9
export COSMOS_GPU_LOCK_HELD=1

scripts=(
  eval_standard.sh
  eval_actionaware_vaeskip.sh
  eval_selfcross_vaeskip.sh
)
variants=(
  standard
  actionaware_vaeskip
  selfcross_vaeskip
)

echo "QUEUE_START time=$(date --iso-8601=seconds) gpu=${GPU} trials_per_task=${TRIALS_PER_TASK} seed=${SEED}" >>"${QUEUE_LOG}"
for index in "${!scripts[@]}"; do
  script="${scripts[index]}"
  variant="${variants[index]}"
  result_log="${FINAL_DIR}/logs/robocasa_${variant}_selected_all1200_seed${SEED}.log"
  launch_log="${FINAL_DIR}/logs/robocasa_${variant}_selected_all1200_seed${SEED}_launch.log"

  echo "RUN_START time=$(date --iso-8601=seconds) variant=${variant} result_log=${result_log}" >>"${QUEUE_LOG}"
  rc=0
  "${FINAL_DIR}/${script}" all \
    --gpu "${GPU}" \
    --num-trials-per-task "${TRIALS_PER_TASK}" \
    --seed "${SEED}" \
    --log-path "${result_log}" \
    >"${launch_log}" 2>&1 || rc=$?
  echo "RUN_DONE time=$(date --iso-8601=seconds) variant=${variant} rc=${rc}" >>"${QUEUE_LOG}"
  if (( rc != 0 )); then
    exit "${rc}"
  fi
done

echo "QUEUE_DONE time=$(date --iso-8601=seconds)" >>"${QUEUE_LOG}"

# Keep GPU2 protected after all three evaluations, matching the LIBERO queues.
source /home/zzy/anaconda3/etc/profile.d/conda.sh
conda activate liberocosmos
export CUDA_VISIBLE_DEVICES="${GPU}"
export COSMOS_PHYSICAL_GPU="${GPU}"
export COSMOS_SPEED_GUARD_ENABLED=1
export COSMOS_SPEED_GUARD_LEAVE_GIB="${COSMOS_SPEED_GUARD_LEAVE_GIB:-10}"
export COSMOS_SPEED_GUARD_MIN_TENSOR_GIB="${COSMOS_SPEED_GUARD_MIN_TENSOR_GIB:-1}"
export COSMOS_SPEED_GUARD_NAME="${COSMOS_SPEED_GUARD_NAME:-测速度}"
exec python "${PROJECT_ROOT}/gpu_postrun_holder.py" >>"${QUEUE_LOG}" 2>&1
