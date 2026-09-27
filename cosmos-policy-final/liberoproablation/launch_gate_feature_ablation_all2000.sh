#!/usr/bin/env bash
set -euo pipefail

ABLATION_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${ABLATION_DIR}/.." && pwd)"
LOG_DIR="${ABLATION_DIR}/logs"

if (( EUID != 0 )); then
  echo "launch_gate_feature_ablation_all2000.sh must run as root" >&2
  exit 1
fi

readonly TOPK=0.15
readonly WRIST_IOU=0.45
readonly PRIMARY_IOU=0.30
readonly PRIMARY_MAXSKIP=2
readonly WRIST_MAXSKIP=1
readonly TRIALS_PER_TASK=10
readonly SEED=195

# Increasingly deep gates. Shallow layers probe only the two current T=1
# camera images; they never run or retain a full-sequence encoder prefix.
declare -ar LAYERS=(rgb conv1 res1 down0 stage1_res0 stage1_res1)
declare -ar GPUS=(0 1 2 3 4 5)
declare -ar UNITS=(
  z-pro-gate-rgb-g0
  z-pro-gate-conv1-g1
  z-pro-gate-res1-g2
  z-pro-gate-down0-g3
  z-pro-gate-s1r0-g4
  z-pro-gate-s1r1-g5
)

mkdir -p "${LOG_DIR}"
chown zzy:zzy "${LOG_DIR}"
chmod 0775 "${LOG_DIR}"

declare -a OUTPUTS=()
for index in "${!LAYERS[@]}"; do
  layer="${LAYERS[index]}"
  output="${LOG_DIR}/liberopro_standard_speed_gate_${layer}_t015_w045_p030_pm2_wm1_all2000_seed195.log"
  OUTPUTS+=("${output}")
  for candidate in \
    "${output}" \
    "${output%.log}_launch.log" \
    "${output%.log}_queue.log" \
    "${output%.log}.flops.json"; do
    if [[ -e "${candidate}" ]]; then
      echo "refusing to overwrite formal output: ${candidate}" >&2
      exit 1
    fi
  done
  unit="${UNITS[index]}"
  load_state="$(systemctl show "${unit}" --property=LoadState --value 2>/dev/null || true)"
  if [[ -n "${load_state}" && "${load_state}" != "not-found" ]]; then
    active_state="$(systemctl show "${unit}" --property=ActiveState --value 2>/dev/null || true)"
    if [[ "${active_state}" == "inactive" || "${active_state}" == "failed" ]]; then
      systemctl reset-failed "${unit}" 2>/dev/null || true
    else
      echo "refusing existing unit ${unit}: ${load_state}/${active_state}" >&2
      exit 1
    fi
  fi
  gpu="${GPUS[index]}"
  mode="$(nvidia-smi -i "${gpu}" --query-gpu=compute_mode --format=csv,noheader,nounits | tr -d '[:space:]_' | tr '[:upper:]' '[:lower:]')"
  if [[ "${mode}" != "exclusiveprocess" ]]; then
    echo "GPU ${gpu} is not EXCLUSIVE_PROCESS: ${mode}" >&2
    exit 1
  fi
done

launched=()
cleanup_partial_launch() {
  if (( ${#launched[@]} > 0 )); then
    /usr/bin/aa-exec -p cosmos-zspeed-controller -- \
      /usr/bin/python3 "${ROOT_DIR}/protected_service_stop.py" "${launched[@]}" || true
  fi
}
trap cleanup_partial_launch ERR INT TERM

for index in "${!LAYERS[@]}"; do
  # Hand each GPU directly from its idle zspeed tensor to the protected eval.
  # Per-card handoff avoids leaving all six cards empty during sequential
  # launcher setup; EXCLUSIVE_PROCESS remains enabled throughout.
  systemctl stop "zspeed-reserve-gpu${GPUS[index]}" 2>/dev/null || true
  /usr/bin/python3 "${ROOT_DIR}/launch_protected_eval.py" \
    --unit "${UNITS[index]}" \
    --benchmark liberopro \
    --mechanism standard \
    --gpu "${GPUS[index]}" \
    --trials-per-task "${TRIALS_PER_TASK}" \
    --seed "${SEED}" \
    --topk "${TOPK}" \
    --wrist-iou "${WRIST_IOU}" \
    --primary-iou "${PRIMARY_IOU}" \
    --primary-maxskip "${PRIMARY_MAXSKIP}" \
    --wrist-maxskip "${WRIST_MAXSKIP}" \
    --shallow-layer "${LAYERS[index]}" \
    --selection all \
    --max-tasks 0 \
    --result-log "${OUTPUTS[index]}"
  launched+=("${UNITS[index]}")
done
trap - ERR INT TERM

systemctl show "${UNITS[@]}" \
  --property=Id --property=ActiveState --property=MainPID \
  --property=NRestarts --no-pager
