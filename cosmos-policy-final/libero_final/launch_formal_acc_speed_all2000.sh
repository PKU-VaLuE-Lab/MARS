#!/usr/bin/env bash
set -euo pipefail

FINAL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${FINAL_DIR}/.." && pwd)"

if (( EUID != 0 )); then
  echo "launch_formal_acc_speed_all2000.sh must run as root" >&2
  exit 1
fi

ACC_TOPK=0.15
ACC_WRIST=0.50
ACC_PRIMARY=0.86
ACC_PRIMARY_MAXSKIP=2
ACC_WRIST_MAXSKIP=1
SPEED_TOPK=0.15
SPEED_WRIST=0.65
SPEED_PRIMARY=0.50
SPEED_PRIMARY_MAXSKIP=2
SPEED_WRIST_MAXSKIP=1

declare -a outputs=(
  "${FINAL_DIR}/logs/libero_standard_acc_t015_w050_p086_pm2_wm1_all2000_seed195.log"
  "${FINAL_DIR}/logs/libero_actionaware_vaeskip_acc_t015_w050_p086_pm2_wm1_all2000_seed195.log"
  "${FINAL_DIR}/logs/libero_selfcross_vaeskip_acc_t015_w050_p086_pm2_wm1_all2000_seed195.log"
  "${FINAL_DIR}/logs/libero_standard_speed_t015_w065_p050_pm2_wm1_all2000_seed195_rerun.log"
)
declare -a units=(
  z-lib-accstd-g0
  z-lib-accaa-g3
  z-lib-accsc-g5
  z-lib-speedstd-g4
)

for output in "${outputs[@]}"; do
  for candidate in \
    "${output}" \
    "${output%.log}_launch.log" \
    "${output%.log}_queue.log"; do
    if [[ -e "${candidate}" ]]; then
      echo "refusing to overwrite formal output: ${candidate}" >&2
      exit 1
    fi
  done
done
for unit in "${units[@]}"; do
  load_state="$(systemctl show "${unit}" --property=LoadState --value 2>/dev/null || true)"
  if [[ -n "${load_state}" && "${load_state}" != "not-found" ]]; then
    echo "refusing existing unit ${unit}: ${load_state}" >&2
    exit 1
  fi
done

/usr/bin/aa-exec -p cosmos-zspeed-controller -- \
  /usr/bin/python3 "${ROOT_DIR}/protected_service_stop.py" \
  z-e79-g0 z-e82-g3 z-e83-g5 z-e84-g4

/usr/bin/python3 "${ROOT_DIR}/launch_protected_eval.py" \
  --unit z-lib-accstd-g0 --benchmark libero --mechanism standard --gpu 0 \
  --trials-per-task 50 --seed 195 \
  --shallow-layer rgb \
  --topk "${ACC_TOPK}" --wrist-iou "${ACC_WRIST}" --primary-iou "${ACC_PRIMARY}" \
  --primary-maxskip "${ACC_PRIMARY_MAXSKIP}" --wrist-maxskip "${ACC_WRIST_MAXSKIP}" \
  --result-log "${outputs[0]}"

/usr/bin/python3 "${ROOT_DIR}/launch_protected_eval.py" \
  --unit z-lib-accaa-g3 --benchmark libero --mechanism actionaware_vaeskip --gpu 3 \
  --trials-per-task 50 --seed 195 \
  --shallow-layer rgb \
  --topk "${ACC_TOPK}" --wrist-iou "${ACC_WRIST}" --primary-iou "${ACC_PRIMARY}" \
  --primary-maxskip "${ACC_PRIMARY_MAXSKIP}" --wrist-maxskip "${ACC_WRIST_MAXSKIP}" \
  --result-log "${outputs[1]}"

/usr/bin/python3 "${ROOT_DIR}/launch_protected_eval.py" \
  --unit z-lib-accsc-g5 --benchmark libero --mechanism selfcross_vaeskip --gpu 5 \
  --trials-per-task 50 --seed 195 \
  --shallow-layer rgb \
  --topk "${ACC_TOPK}" --wrist-iou "${ACC_WRIST}" --primary-iou "${ACC_PRIMARY}" \
  --primary-maxskip "${ACC_PRIMARY_MAXSKIP}" --wrist-maxskip "${ACC_WRIST_MAXSKIP}" \
  --result-log "${outputs[2]}"

/usr/bin/python3 "${ROOT_DIR}/launch_protected_eval.py" \
  --unit z-lib-speedstd-g4 --benchmark libero --mechanism standard --gpu 4 \
  --trials-per-task 50 --seed 195 \
  --shallow-layer rgb \
  --topk "${SPEED_TOPK}" --wrist-iou "${SPEED_WRIST}" --primary-iou "${SPEED_PRIMARY}" \
  --primary-maxskip "${SPEED_PRIMARY_MAXSKIP}" --wrist-maxskip "${SPEED_WRIST_MAXSKIP}" \
  --result-log "${outputs[3]}"

systemctl show "${units[@]}" \
  --property=Id --property=ActiveState --property=MainPID \
  --property=NRestarts --no-pager
