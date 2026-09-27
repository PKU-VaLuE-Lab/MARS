#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
source "${ROOT_DIR}/gpu_queue_guard.sh"
GPU="${GPU:-2}"
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
    *)
      echo "Unsupported argument: $1" >&2
      exit 2
      ;;
  esac
done

mkdir -p "${SCRIPT_DIR}/logs"
QUEUE_LOG="${SCRIPT_DIR}/logs/libero_queue_all_seed195.log"
exec >>"${QUEUE_LOG}" 2>&1

LOCK_FILE="/tmp/cosmos-policy-gpu-${GPU}.lock"
exec 8>"${LOCK_FILE}"
if ! flock -n 8; then
  echo "[$(date --iso-8601=seconds)] GPU ${GPU} is already locked; queue not started."
  exit 1
fi
export COSMOS_GPU_LOCK_HELD=1

run_eval() {
  local name="$1"
  local entrypoint="$2"
  cosmos_verify_gpu_protection "${GPU}"
  echo "[$(date --iso-8601=seconds)] START ${name} on GPU ${GPU}"
  "${entrypoint}" --gpu "${GPU}" all
  cosmos_verify_gpu_protection "${GPU}"
  echo "[$(date --iso-8601=seconds)] DONE ${name} on GPU ${GPU}"
}

echo "[$(date --iso-8601=seconds)] Acquired ${LOCK_FILE}"
cosmos_verify_gpu_protection "${GPU}"
run_eval baseline "${SCRIPT_DIR}/eval.sh"
run_eval standard "${SCRIPT_DIR}/eval_standard.sh"
run_eval dicache "${SCRIPT_DIR}/eval_dicache.sh"
echo "[$(date --iso-8601=seconds)] LIBERO queue complete on GPU ${GPU}"
