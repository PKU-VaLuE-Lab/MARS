#!/usr/bin/env bash
set -euo pipefail

FINAL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${FINAL_DIR}/.." && pwd)"
STATE_DIR="${FINAL_DIR}/runstate10000/baseline_seed195"
FORMAL_LOG="${FINAL_DIR}/logs10000/liberopro_baseline_all10000_seed195.log"
MERGE_STATE="${STATE_DIR}/merge_state.json"

GPU3_UNIT="z-pro-base10k-g3"
GPU5_UNIT="z-pro-base10k-g5"
MERGE_UNIT="z-pro-base10k-merge"
GPU3_OLD_UNIT="z-fastwam-calib-speed-g3"
GPU5_OLD_UNIT="z-fastwam-baseline-all2000-g5"
GPU3_LOG="${STATE_DIR}/gpu3_spatial_object/liberopro_baseline_spatial_object_all5000_seed195.log"
GPU5_LOG="${STATE_DIR}/gpu5_goal_long/liberopro_baseline_goal_long_all5000_seed195.log"

if (( EUID != 0 )); then
  echo "launch_baseline_all10000_split.sh must run as root" >&2
  exit 1
fi

unit_must_be_absent() {
  local unit="$1"
  local state
  state="$(systemctl show "${unit}" --property=LoadState --value 2>/dev/null || true)"
  if [[ -n "${state}" && "${state}" != "not-found" ]]; then
    echo "refusing existing unit ${unit}: ${state}" >&2
    exit 1
  fi
}

wait_gpu_empty() {
  local gpu="$1"
  local deadline=$((SECONDS + 45))
  while (( SECONDS < deadline )); do
    if [[ -z "$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits)" ]]; then
      return 0
    fi
    sleep 0.25
  done
  echo "GPU ${gpu} still has a compute process after protected stop" >&2
  return 1
}

for unit in "${GPU3_UNIT}" "${GPU5_UNIT}" "${MERGE_UNIT}"; do
  unit_must_be_absent "${unit}"
done

if find "${FINAL_DIR}/logs10000" -mindepth 1 -maxdepth 1 -type f -print -quit | grep -q .; then
  echo "refusing non-empty logs10000; expected a fresh formal output directory" >&2
  exit 1
fi
for result in "${GPU3_LOG}" "${GPU5_LOG}"; do
  for path in "${result}" "${result%.log}_launch.log" "${result%.log}_queue.log"; do
    if [[ -e "${path}" ]]; then
      echo "refusing to overwrite shard output: ${path}" >&2
      exit 1
    fi
  done
done
if [[ -e "${FORMAL_LOG}" || -e "${MERGE_STATE}" ]]; then
  echo "refusing to overwrite all10000 merge output/state" >&2
  exit 1
fi

for gpu in 3 5; do
  mode="$(nvidia-smi -i "${gpu}" --query-gpu=compute_mode --format=csv,noheader,nounits | tr -d '[:space:]_' | tr '[:upper:]' '[:lower:]')"
  if [[ "${mode}" != "exclusiveprocess" ]]; then
    echo "GPU ${gpu} is not EXCLUSIVE_PROCESS: ${mode}" >&2
    exit 1
  fi
done

/home/zzy/anaconda3/envs/liberocosmos/bin/python -m py_compile \
  "${ROOT_DIR}/launch_protected_eval.py" \
  "${FINAL_DIR}/merge_all10000_logs.py" \
  "${FINAL_DIR}/eval_wrapper.py" \
  "${FINAL_DIR}/suite_config.py"

install -d -o zzy -g zzy -m 775 \
  "${STATE_DIR}" \
  "$(dirname -- "${GPU3_LOG}")" \
  "$(dirname -- "${GPU5_LOG}")"

launch_shard() {
  local unit="$1"
  local gpu="$2"
  local selection="$3"
  local result_log="$4"
  /usr/bin/python3 "${ROOT_DIR}/launch_protected_eval.py" \
    --unit "${unit}" --benchmark liberopro --mechanism baseline --gpu "${gpu}" \
    --trials-per-task 50 --seed 195 --selection "${selection}" \
    --topk 0.15 --wrist-iou 0.98 --primary-iou 0.90 \
    --primary-maxskip 1 --wrist-maxskip 1 \
    --result-log "${result_log}"
}

# Minimize the unreserved transition: stop and immediately replace one completed
# FastWAM holder at a time. GPU4 is deliberately untouched.
/usr/bin/aa-exec -p cosmos-zspeed-controller -- \
  /usr/bin/python3 "${ROOT_DIR}/protected_service_stop.py" "${GPU3_OLD_UNIT}"
wait_gpu_empty 3
launch_shard "${GPU3_UNIT}" 3 "spatial,object" "${GPU3_LOG}"

/usr/bin/aa-exec -p cosmos-zspeed-controller -- \
  /usr/bin/python3 "${ROOT_DIR}/protected_service_stop.py" "${GPU5_OLD_UNIT}"
wait_gpu_empty 5
launch_shard "${GPU5_UNIT}" 5 "goal,10" "${GPU5_LOG}"

systemd-run \
  --unit="${MERGE_UNIT}" --uid=zzy --gid=zzy \
  --property=Restart=on-failure --property=RestartSec=2s \
  --property=StartLimitIntervalSec=0 \
  /usr/bin/python3 "${FINAL_DIR}/merge_all10000_logs.py" \
  --output "${FORMAL_LOG}" --state "${MERGE_STATE}" \
  --source "GPU3 spatial+object=${GPU3_LOG}" \
  --source "GPU5 goal+long=${GPU5_LOG}" \
  --expected-episodes 10000

systemctl show "${GPU3_UNIT}" "${GPU5_UNIT}" "${MERGE_UNIT}" \
  --property=Id --property=ActiveState --property=MainPID \
  --property=NRestarts --no-pager
