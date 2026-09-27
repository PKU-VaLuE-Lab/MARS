#!/usr/bin/env bash

cosmos_verify_gpu_protection() {
  local gpu="$1"
  local heartbeat="${COSMOS_GPU_MONITOR_HEARTBEAT:-/tmp/cosmos-policy-gpu-${gpu}.monitor.json}"
  local current_sid
  local compute_mode

  compute_mode="$(nvidia-smi -i "${gpu}" --query-gpu=compute_mode --format=csv,noheader,nounits | tr -d '[:space:]_')"
  if [[ "${compute_mode,,}" != "exclusiveprocess" ]]; then
    echo "GPU ${gpu} is not in EXCLUSIVE_PROCESS mode: ${compute_mode}" >&2
    return 1
  fi
  if [[ ! -f "${heartbeat}" ]]; then
    echo "GPU ${gpu} root monitor heartbeat is missing: ${heartbeat}" >&2
    return 1
  fi

  current_sid="$(ps -o sid= -p "$$" | tr -d '[:space:]')"
  /usr/bin/python3 - "${heartbeat}" "${gpu}" "${current_sid}" <<'PY'
import json
import os
import signal
import sys
import time
from pathlib import Path

path = Path(sys.argv[1])
expected_gpu = int(sys.argv[2])
expected_sid = int(sys.argv[3])
data = json.loads(path.read_text(encoding="utf-8"))
age = time.time() - path.stat().st_mtime
if data.get("gpu") != expected_gpu:
    raise SystemExit(f"monitor GPU mismatch: {data.get('gpu')} != {expected_gpu}")
if data.get("allowed_sid") != expected_sid:
    raise SystemExit(f"monitor SID mismatch: {data.get('allowed_sid')} != {expected_sid}")
if age > 5:
    raise SystemExit(f"monitor heartbeat is stale: {age:.1f}s")
pid = int(data["monitor_pid"])
try:
    os.kill(pid, 0)
except ProcessLookupError:
    raise SystemExit(f"monitor PID {pid} is not alive")
except PermissionError:
    # A non-root evaluation user cannot signal the root watchdog; EPERM still
    # proves that the PID exists.
    pass
PY
}
