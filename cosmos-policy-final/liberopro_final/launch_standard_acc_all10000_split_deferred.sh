#!/usr/bin/env bash
set -euo pipefail

FINAL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${FINAL_DIR}/.." && pwd)"
STATE_DIR="${FINAL_DIR}/runstate10000/standard_acc_seed195"
FORMAL_LOG="${FINAL_DIR}/logs10000/liberopro_standard_acc_all10000_seed195.log"
MERGE_STATE="${STATE_DIR}/merge_state.json"

GPU2_EVAL_UNIT="z-pro-stdacc10k-g2"
GPU4_EVAL_UNIT="z-pro-stdacc10k-g4"
GPU2_QUEUE_UNIT="z-pro-stdacc10k-queue-g2"
GPU4_QUEUE_UNIT="z-pro-stdacc10k-queue-g4"
MERGE_UNIT="z-pro-stdacc10k-merge"
GPU4_OLD_UNIT="z-fastwam-acc-all2000-g4"
GPU2_LOG="${STATE_DIR}/gpu2_goal_long/liberopro_standard_acc_goal_long_all5000_seed195.log"
GPU4_LOG="${STATE_DIR}/gpu4_spatial_object/liberopro_standard_acc_spatial_object_all5000_seed195.log"
FASTWAM_ACC_SUMMARY="/data/Albus/zzy/WAM_ICLR27/FastWAM/fastwam_final/results/actionaware_acc_all2000_seed195/summary.json"

if (( EUID != 0 )); then
  echo "launch_standard_acc_all10000_split_deferred.sh must run as root" >&2
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

for unit in \
  "${GPU2_EVAL_UNIT}" "${GPU4_EVAL_UNIT}" \
  "${GPU2_QUEUE_UNIT}" "${GPU4_QUEUE_UNIT}" "${MERGE_UNIT}"; do
  unit_must_be_absent "${unit}"
done
for result in "${GPU2_LOG}" "${GPU4_LOG}"; do
  for path in "${result}" "${result%.log}_launch.log" "${result%.log}_queue.log"; do
    if [[ -e "${path}" ]]; then
      echo "refusing to overwrite shard output: ${path}" >&2
      exit 1
    fi
  done
done
if [[ -e "${FORMAL_LOG}" || -e "${MERGE_STATE}" ]]; then
  echo "refusing to overwrite Standard Acc all10000 output/state" >&2
  exit 1
fi

/home/zzy/anaconda3/envs/liberocosmos/bin/python -m py_compile \
  "${ROOT_DIR}/launch_protected_eval.py" \
  "${FINAL_DIR}/deferred_standard_acc_shard.py" \
  "${FINAL_DIR}/merge_all10000_logs.py" \
  "${FINAL_DIR}/eval_wrapper.py" \
  "${FINAL_DIR}/suite_config.py"

install -d -o zzy -g zzy -m 775 \
  "${STATE_DIR}" \
  "$(dirname -- "${GPU2_LOG}")" \
  "$(dirname -- "${GPU4_LOG}")"

# GPU2 is currently allowed to belong to another job. This root queue only
# observes it and does not signal anything; it launches immediately once empty.
systemd-run \
  --unit="${GPU2_QUEUE_UNIT}" \
  --property=Restart=on-failure --property=RestartSec=2s \
  --property=StartLimitIntervalSec=0 \
  /usr/bin/python3 "${FINAL_DIR}/deferred_standard_acc_shard.py" \
  --unit "${GPU2_EVAL_UNIT}" --gpu 2 --selection "goal,10" \
  --result-log "${GPU2_LOG}"

# GPU4 remains with FastWAM until its authoritative all2000 summary exists.
systemd-run \
  --unit="${GPU4_QUEUE_UNIT}" \
  --property=Restart=on-failure --property=RestartSec=2s \
  --property=StartLimitIntervalSec=0 \
  /usr/bin/python3 "${FINAL_DIR}/deferred_standard_acc_shard.py" \
  --unit "${GPU4_EVAL_UNIT}" --gpu 4 --selection "spatial,object" \
  --wait-for-json "${FASTWAM_ACC_SUMMARY}" --expected-episodes 2000 \
  --stop-unit "${GPU4_OLD_UNIT}" \
  --result-log "${GPU4_LOG}"

systemd-run \
  --unit="${MERGE_UNIT}" --uid=zzy --gid=zzy \
  --property=Restart=on-failure --property=RestartSec=2s \
  --property=StartLimitIntervalSec=0 \
  /usr/bin/python3 "${FINAL_DIR}/merge_all10000_logs.py" \
  --output "${FORMAL_LOG}" --state "${MERGE_STATE}" \
  --source "GPU2 goal+long=${GPU2_LOG}" \
  --source "GPU4 spatial+object=${GPU4_LOG}" \
  --expected-episodes 10000

systemctl show \
  "${GPU2_QUEUE_UNIT}" "${GPU4_QUEUE_UNIT}" "${MERGE_UNIT}" \
  --property=Id --property=ActiveState --property=MainPID \
  --property=NRestarts --no-pager
