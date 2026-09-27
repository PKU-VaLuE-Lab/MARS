#!/usr/bin/env bash
set -euo pipefail

ABLATION_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${ABLATION_DIR}/.." && pwd)"
LOG_DIR="${ABLATION_DIR}/logs"
COMPLETION_MARKER="LIBERO-PRO ALL final:"
LAYERS=(rgb conv1 stage1_res0)
CLAIMED_GPU=""
CURRENT_UNIT=""
CLAIM_DIR=""
CLAIM_RECORD="${LOG_DIR}/liberopro_seed195_claimed_gpu.txt"

if (( EUID != 0 )); then
  echo "run_layer_ablation_all2000_when_free.sh must run as root" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"
chown zzy:zzy "${LOG_DIR}"
chmod 0775 "${LOG_DIR}"

result_path() {
  local layer="$1"
  local tag="${layer//_/-}"
  printf '%s/liberopro_standard_speed_layer_%s_t015_w045_p030_pm2_wm1_all2000_seed195.log' \
    "${LOG_DIR}" "${tag}"
}

unit_name() {
  local layer="$1"
  local gpu="$2"
  local tag="${layer//_/-}"
  printf 'z-pro-lay-%s-g%s' "${tag}" "${gpu}"
}

unit_exists() {
  local unit="$1"
  local state
  state="$(systemctl show "${unit}" --property=LoadState --value 2>/dev/null || true)"
  [[ -n "${state}" && "${state}" != "not-found" ]]
}

gpu_is_truly_free() {
  local gpu="$1"
  local used mode pids
  used="$(nvidia-smi -i "${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
  mode="$(nvidia-smi -i "${gpu}" --query-gpu=compute_mode --format=csv,noheader,nounits | tr -d ' _' | tr '[:upper:]' '[:lower:]')"
  pids="$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -cd '0-9\n')"
  [[ "${used}" =~ ^[0-9]+$ ]] || return 1
  [[ "${used}" -le 100 && "${mode}" == "default" && -z "${pids}" ]]
}

wait_for_free_gpu() {
  local gpu now next_report=0 candidate_claim used pids
  while true; do
    for gpu in 0 1 2 3 4 5; do
      candidate_claim="/tmp/cosmos-policy-free-gpu-claim-${gpu}"
      mkdir "${candidate_claim}" 2>/dev/null || continue
      if ! gpu_is_truly_free "${gpu}"; then
        rmdir "${candidate_claim}"
        continue
      fi
      # Require a stable idle window so a GPU between two external jobs is not
      # mistaken for an actually available device.
      sleep 15
      if ! gpu_is_truly_free "${gpu}"; then
        echo "[liberopro-layer] GPU${gpu} was only transiently idle; skipping it"
        rmdir "${candidate_claim}"
        continue
      fi
      echo "[liberopro-layer] GPU${gpu} stayed idle for 15s; claiming it"
      nvidia-smi -i "${gpu}" -c EXCLUSIVE_PROCESS
      used="$(nvidia-smi -i "${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
      pids="$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -cd '0-9\n')"
      if [[ "${used}" -le 100 && -z "${pids}" ]]; then
        CLAIMED_GPU="${gpu}"
        CLAIM_DIR="${candidate_claim}"
        printf '%s\n' "${gpu}" > "${CLAIM_RECORD}"
        chown zzy:zzy "${CLAIM_RECORD}"
        return 0
      fi
      echo "[liberopro-layer] GPU${gpu} became busy during final claim; leaving external process untouched"
      rmdir "${candidate_claim}"
    done
    now="$(date +%s)"
    if (( now >= next_report )); then
      echo "[liberopro-layer] waiting: no Default GPU with <=100 MiB and zero compute PIDs"
      nvidia-smi --query-gpu=index,memory.used,utilization.gpu,compute_mode --format=csv,noheader
      next_report=$((now + 300))
    fi
    sleep 30
  done
}

preflight() {
  local layer gpu unit result candidate
  for layer in "${LAYERS[@]}"; do
    result="$(result_path "${layer}")"
    for candidate in "${result}" "${result%.log}_launch.log" "${result%.log}_queue.log"; do
      if [[ -e "${candidate}" ]]; then
        echo "refusing to overwrite ${candidate}" >&2
        exit 1
      fi
    done
    for gpu in 0 1 2 3 4 5; do
      unit="$(unit_name "${layer}" "${gpu}")"
      if unit_exists "${unit}"; then
        echo "refusing existing unit ${unit}" >&2
        exit 1
      fi
    done
  done
}

launch_layer() {
  local layer="$1"
  local result
  result="$(result_path "${layer}")"
  CURRENT_UNIT="$(unit_name "${layer}" "${CLAIMED_GPU}")"
  /usr/bin/python3 "${ROOT_DIR}/launch_protected_eval.py" \
    --unit "${CURRENT_UNIT}" \
    --benchmark liberopro \
    --mechanism standard \
    --gpu "${CLAIMED_GPU}" \
    --trials-per-task 10 \
    --seed 195 \
    --topk 0.15 \
    --wrist-iou 0.45 \
    --primary-iou 0.30 \
    --primary-maxskip 2 \
    --wrist-maxskip 1 \
    --shallow-layer "${layer}" \
    --selection all \
    --max-tasks 0 \
    --result-log "${result}"
}

wait_for_layer() {
  local layer="$1"
  local result queue state restarts
  result="$(result_path "${layer}")"
  queue="${result%.log}_queue.log"
  while true; do
    if [[ -f "${result}" && -f "${queue}" ]] && \
       grep -qF "${COMPLETION_MARKER}" "${result}" && \
       grep -q '"event": "eval_done".*"rc": 0' "${queue}"; then
      echo "[liberopro-layer] complete gpu=${CLAIMED_GPU} layer=${layer} result=${result}"
      return 0
    fi
    state="$(systemctl show "${CURRENT_UNIT}" --property=ActiveState --value 2>/dev/null || true)"
    restarts="$(systemctl show "${CURRENT_UNIT}" --property=NRestarts --value 2>/dev/null || true)"
    if [[ "${state}" != "active" && "${state}" != "activating" ]]; then
      echo "evaluation stopped unexpectedly: unit=${CURRENT_UNIT} state=${state} restarts=${restarts}" >&2
      return 1
    fi
    sleep 30
  done
}

stop_current_holder() {
  if [[ -n "${CURRENT_UNIT}" ]] && unit_exists "${CURRENT_UNIT}"; then
    if ! /usr/bin/aa-exec -p cosmos-zspeed-controller -- \
      /usr/bin/python3 "${ROOT_DIR}/protected_service_stop.py" "${CURRENT_UNIT}"; then
      unit_exists "${CURRENT_UNIT}" && return 1
    fi
  fi
  CURRENT_UNIT=""
}

release_gpu() {
  local used
  [[ -n "${CLAIMED_GPU}" ]] || return 0
  stop_current_holder
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    used="$(nvidia-smi -i "${CLAIMED_GPU}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
    [[ "${used}" -le 100 ]] && break
    sleep 1
  done
  nvidia-smi -i "${CLAIMED_GPU}" -c DEFAULT
  echo "[liberopro-layer] released GPU${CLAIMED_GPU} and restored DEFAULT mode"
  [[ -z "${CLAIM_DIR}" ]] || rmdir "${CLAIM_DIR}" 2>/dev/null || true
  CLAIM_DIR=""
  CLAIMED_GPU=""
}

trap release_gpu EXIT
trap 'release_gpu; exit 130' INT TERM

preflight
wait_for_free_gpu

for index in "${!LAYERS[@]}"; do
  layer="${LAYERS[${index}]}"
  echo "[liberopro-layer] launch gpu=${CLAIMED_GPU} layer=${layer}"
  launch_layer "${layer}"
  wait_for_layer "${layer}"
  stop_current_holder
done

release_gpu
echo "[liberopro-layer] all three all-2000 runs complete"
