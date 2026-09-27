#!/usr/bin/env bash
set -euo pipefail

FINAL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${FINAL_DIR}/.." && pwd)"

OLD_UNITS=(
  "z-lib-accaa-g3"
  "z-lib-speedstd-g4"
  "z-lib-accsc-g5"
)
NEW_UNITS=(
  "z-pro-speed-wovae-g3"
  "z-pro-speed-wosc-g4"
  "z-pro-speed-woaa-g5"
)
MECHANISMS=(
  "actionaware_selfcross"
  "actionaware_vaeskip"
  "selfcross_vaeskip"
)
GPUS=(3 4 5)
RESULT_LOGS=(
  "${FINAL_DIR}/logs/liberopro_actionaware_selfcross_speed_t015_w045_p030_pm2_wm1_all2000_seed195.log"
  "${FINAL_DIR}/logs/liberopro_actionaware_vaeskip_speed_t015_w045_p030_pm2_wm1_all2000_seed195.log"
  "${FINAL_DIR}/logs/liberopro_selfcross_vaeskip_speed_t015_w045_p030_pm2_wm1_all2000_seed195.log"
)

if (( EUID != 0 )); then
  echo "launch_formal_speed_ablations_all2000.sh must run as root" >&2
  exit 1
fi

# Complete every destructive preflight before releasing any completed holder.
for index in "${!NEW_UNITS[@]}"; do
  unit="${NEW_UNITS[index]}"
  result_log="${RESULT_LOGS[index]}"
  for candidate in \
    "${result_log}" \
    "${result_log%.log}_launch.log" \
    "${result_log%.log}_queue.log"; do
    if [[ -e "${candidate}" ]]; then
      echo "refusing to overwrite formal output: ${candidate}" >&2
      exit 1
    fi
  done
  load_state="$(systemctl show "${unit}" --property=LoadState --value 2>/dev/null || true)"
  if [[ -n "${load_state}" && "${load_state}" != "not-found" ]]; then
    echo "refusing existing unit ${unit}: ${load_state}" >&2
    exit 1
  fi
done

# Replace each completed LIBERO holder immediately with its corresponding
# LIBERO-PRO ablation. GPU2 and GPUs0-1 are deliberately untouched.
for index in "${!NEW_UNITS[@]}"; do
  old_unit="${OLD_UNITS[index]}"
  new_unit="${NEW_UNITS[index]}"
  mechanism="${MECHANISMS[index]}"
  gpu="${GPUS[index]}"
  result_log="${RESULT_LOGS[index]}"

  /usr/bin/aa-exec -p cosmos-zspeed-controller -- \
    /usr/bin/python3 "${ROOT_DIR}/protected_service_stop.py" "${old_unit}"

  /usr/bin/python3 "${ROOT_DIR}/launch_protected_eval.py" \
    --unit "${new_unit}" --benchmark liberopro \
    --mechanism "${mechanism}" --gpu "${gpu}" \
    --trials-per-task 10 --seed 195 \
    --topk 0.15 --wrist-iou 0.45 --primary-iou 0.30 \
    --primary-maxskip 2 --wrist-maxskip 1 \
    --shallow-layer rgb \
    --result-log "${result_log}"
done

systemctl show "${NEW_UNITS[@]}" \
  --property=Id --property=ActiveState --property=MainPID \
  --property=NRestarts --no-pager
