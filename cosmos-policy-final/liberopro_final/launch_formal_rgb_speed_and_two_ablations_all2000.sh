#!/usr/bin/env bash
set -euo pipefail

FINAL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${FINAL_DIR}/.." && pwd)"

if (( EUID != 0 )); then
  echo "launch_formal_rgb_speed_and_two_ablations_all2000.sh must run as root" >&2
  exit 1
fi

readonly TOPK=0.15
readonly WRIST_IOU=0.45
readonly PRIMARY_IOU=0.30
readonly PRIMARY_MAXSKIP=2
readonly WRIST_MAXSKIP=1
readonly TRIALS_PER_TASK=10
readonly SEED=195

declare -a units=(
  z-pro-rgb-speedstd-g1
  z-pro-rgb-woaa-g4
  z-pro-rgb-woattn-g5
)
declare -a mechanisms=(
  standard
  selfcross_vaeskip
  actionaware_vaeskip
)
declare -a gpus=(1 4 5)
declare -a outputs=(
  "${FINAL_DIR}/logs/liberopro_standard_speed_rgbopt_t015_w045_p030_pm2_wm1_all2000_seed195.log"
  "${FINAL_DIR}/logs/liberopro_wo_actionaware_speed_rgbopt_t015_w045_p030_pm2_wm1_all2000_seed195.log"
  "${FINAL_DIR}/logs/liberopro_wo_attention_speed_rgbopt_t015_w045_p030_pm2_wm1_all2000_seed195.log"
)

# Finish every non-mutating preflight before launching any member of the trio.
for index in "${!units[@]}"; do
  unit="${units[index]}"
  output="${outputs[index]}"
  for candidate in \
    "${output}" \
    "${output%.log}_launch.log" \
    "${output%.log}_queue.log"; do
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
  gpu="${gpus[index]}"
  mode="$(nvidia-smi -i "${gpu}" --query-gpu=compute_mode --format=csv,noheader,nounits | tr -d '[:space:]_')"
  if [[ "${mode,,}" != "exclusiveprocess" ]]; then
    echo "GPU ${gpu} is not EXCLUSIVE_PROCESS: ${mode}" >&2
    exit 1
  fi
done

for index in "${!units[@]}"; do
  /usr/bin/python3 "${ROOT_DIR}/launch_protected_eval.py" \
    --unit "${units[index]}" \
    --benchmark liberopro \
    --mechanism "${mechanisms[index]}" \
    --gpu "${gpus[index]}" \
    --trials-per-task "${TRIALS_PER_TASK}" \
    --seed "${SEED}" \
    --topk "${TOPK}" \
    --wrist-iou "${WRIST_IOU}" \
    --primary-iou "${PRIMARY_IOU}" \
    --primary-maxskip "${PRIMARY_MAXSKIP}" \
    --wrist-maxskip "${WRIST_MAXSKIP}" \
    --shallow-layer rgb \
    --result-log "${outputs[index]}"
done

systemctl show "${units[@]}" \
  --property=Id --property=ActiveState --property=MainPID \
  --property=NRestarts --no-pager
