#!/usr/bin/env bash
set -euo pipefail

FINAL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${FINAL_DIR}/.." && pwd)"
RESULT_LOG="${FINAL_DIR}/logs/liberopro_standard_speed_rgbopt_t015_w045_p030_pm2_wm1_all2000_seed195.log"
UNIT="z-pro-rgb-speedstd-g1"

if (( EUID != 0 )); then
  echo "launch_formal_speed_all2000.sh must run as root" >&2
  exit 1
fi

for candidate in \
  "${RESULT_LOG}" \
  "${RESULT_LOG%.log}_launch.log" \
  "${RESULT_LOG%.log}_queue.log"; do
  if [[ -e "${candidate}" ]]; then
    echo "refusing to overwrite formal output: ${candidate}" >&2
    exit 1
  fi
done
load_state="$(systemctl show "${UNIT}" --property=LoadState --value 2>/dev/null || true)"
if [[ -n "${load_state}" && "${load_state}" != "not-found" ]]; then
  echo "refusing existing unit ${UNIT}: ${load_state}" >&2
  exit 1
fi

# Stop both completed exploration holders first.  Only GPU1 is relaunched;
# GPU2 deliberately remains empty while retaining EXCLUSIVE_PROCESS mode.
/usr/bin/aa-exec -p cosmos-zspeed-controller -- \
  /usr/bin/python3 "${ROOT_DIR}/protected_service_stop.py" \
  z-pro5b-g1 z-pro6b-g2

/usr/bin/python3 "${ROOT_DIR}/launch_protected_eval.py" \
  --unit "${UNIT}" --benchmark liberopro --mechanism standard --gpu 1 \
  --trials-per-task 10 --seed 195 \
  --topk 0.15 --wrist-iou 0.45 --primary-iou 0.30 \
  --primary-maxskip 2 --wrist-maxskip 1 \
  --shallow-layer rgb \
  --result-log "${RESULT_LOG}"

systemctl show "${UNIT}" \
  --property=Id --property=ActiveState --property=MainPID \
  --property=NRestarts --no-pager
