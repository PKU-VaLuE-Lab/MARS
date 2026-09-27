#!/usr/bin/env bash
set -euo pipefail

FINAL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${FINAL_DIR}/env.sh"

SEED="${SEED:-195}"
TRIALS_PER_TASK="${TRIALS_PER_TASK:-50}"
MERGE_ONLY="${MERGE_ONLY:-0}"
FINAL_LOG="${FINAL_DIR}/logs/pi05_libero_baseline_all2000_seed${SEED}.log"
WORK_DIR="${FINAL_DIR}/logs/.baseline_split_01_seed${SEED}"
GPU0_LOG="${WORK_DIR}/gpu0_spatial_object.log"
GPU1_LOG="${WORK_DIR}/gpu1_goal_long.log"

if [[ -e "${FINAL_LOG}" ]]; then
  echo "Refusing to overwrite existing final log: ${FINAL_LOG}" >&2
  exit 1
fi
if [[ "${MERGE_ONLY}" != "1" ]]; then
  if [[ -e "${WORK_DIR}" ]]; then
    echo "Refusing to reuse existing split work directory: ${WORK_DIR}" >&2
    exit 1
  fi
  mkdir -p "${WORK_DIR}"
else
  [[ -f "${GPU0_LOG}" ]] || { echo "Missing GPU0 shard: ${GPU0_LOG}" >&2; exit 1; }
  [[ -f "${GPU1_LOG}" ]] || { echo "Missing GPU1 shard: ${GPU1_LOG}" >&2; exit 1; }
fi

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  for pid in "${pid0:-}" "${pid1:-}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill -TERM "${pid}" 2>/dev/null || true
    fi
  done
  wait "${pid0:-}" 2>/dev/null || true
  wait "${pid1:-}" 2>/dev/null || true
  exit "${status}"
}

wait_for_gpu_idle() {
  local gpu="$1"
  local memory_used
  while true; do
    memory_used="$(nvidia-smi --id="${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
    if flock -n "/tmp/cosmos-policy-gpu-${gpu}.lock" -c true && [[ "${memory_used}" -lt 1024 ]]; then
      echo "[baseline split] GPU${gpu} is idle (${memory_used} MiB); launching shard"
      return 0
    fi
    echo "[baseline split] waiting for GPU${gpu}: memory_used=${memory_used} MiB"
    sleep 10
  done
}

if [[ "${MERGE_ONLY}" != "1" ]]; then
  trap cleanup EXIT INT TERM
  echo "[baseline split] starting GPU0=spatial,object GPU1=goal,long seed=${SEED} trials_per_task=${TRIALS_PER_TASK}"
  (
    wait_for_gpu_idle 0
    GPU=0 PORT=8200 SEED="${SEED}" TRIALS_PER_TASK="${TRIALS_PER_TASK}" \
      SUITES=libero_spatial,libero_object LOG_PATH="${GPU0_LOG}" \
      bash "${FINAL_DIR}/eval.sh"
  ) &
  pid0=$!
  (
    wait_for_gpu_idle 1
    GPU=1 PORT=8201 SEED="${SEED}" TRIALS_PER_TASK="${TRIALS_PER_TASK}" \
      SUITES=libero_goal,libero_10 LOG_PATH="${GPU1_LOG}" \
      bash "${FINAL_DIR}/eval.sh"
  ) &
  pid1=$!

  status0=0
  status1=0
  wait "${pid0}" || status0=$?
  wait "${pid1}" || status1=$?
  if [[ "${status0}" -ne 0 || "${status1}" -ne 0 ]]; then
    echo "Split baseline failed: gpu0_status=${status0} gpu1_status=${status1}" >&2
    exit 1
  fi
fi

"${OPENPI_ROOT}/.conda/bin/python" - "${GPU0_LOG}" "${GPU1_LOG}" "${FINAL_LOG}" "${SEED}" "${TRIALS_PER_TASK}" <<'PY'
from __future__ import annotations

import re
import sys
from collections import OrderedDict
from pathlib import Path

shards = [Path(sys.argv[1]), Path(sys.argv[2])]
destination = Path(sys.argv[3])
seed = int(sys.argv[4])
trials = int(sys.argv[5])
expected_suites = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]

suite_results: OrderedDict[str, list[int]] = OrderedDict()
body: list[str] = []
timing_count = 0
timing_query_sum = 0.0
timing_policy_sum = 0.0
timing_roundtrip_sum = 0.0

for shard in shards:
    lines = shard.read_text(encoding="utf-8", errors="replace").splitlines()
    current_suite = None
    copying = False
    shard_complete = False
    for line in lines:
        if "OPENPI LIBERO suite start:" in line:
            current_suite = line.rsplit(":", 1)[-1].strip()
            if current_suite not in expected_suites:
                raise RuntimeError(f"unexpected suite in {shard}: {current_suite}")
            if current_suite in suite_results:
                raise RuntimeError(f"duplicate suite in split logs: {current_suite}")
            suite_results[current_suite] = [0, 0]
            copying = True
        if line.startswith("INFO:root:OPENPI LIBERO suite rates:"):
            copying = False
        if copying:
            body.append(line)
            if line.startswith("INFO:root:Success:"):
                success = line.endswith("True")
                suite_results[current_suite][0 if success else 1] += 1
        match = re.search(
            r"OPENPI query timing final: count=(\d+) mean_querytime=([0-9.]+) sec "
            r"mean_policy=([0-9.]+) sec mean_roundtrip=([0-9.]+) sec",
            line,
        )
        if match:
            count = int(match.group(1))
            timing_count += count
            timing_query_sum += count * float(match.group(2))
            timing_policy_sum += count * float(match.group(3))
            timing_roundtrip_sum += count * float(match.group(4))
            shard_complete = True
    if not shard_complete:
        raise RuntimeError(f"split log is incomplete: {shard}")

if list(suite_results) != expected_suites:
    raise RuntimeError(f"suite order mismatch: {list(suite_results)}")
expected_episodes = len(expected_suites) * 10 * trials
episodes = sum(sum(value) for value in suite_results.values())
successes = sum(value[0] for value in suite_results.values())
if episodes != expected_episodes:
    raise RuntimeError(f"expected {expected_episodes} episodes, found {episodes}")
if timing_count <= 0:
    raise RuntimeError("no timed queries found")

rates = {suite: value[0] / sum(value) for suite, value in suite_results.items()}
output = [
    f"[OpenPI final] variant=baseline gpu=0,1 seed={seed} suites={','.join(expected_suites)} "
    f"trials_per_task={trials} max_tasks=0",
    "[OpenPI sampler] num_steps=10 short_steps=5 action_threshold=0.35 "
    "cache_rel_l1_threshold=0.0 max_cached_steps=0",
    *body,
    f"INFO:root:OPENPI LIBERO suite rates: {rates}",
    f"INFO:root:OPENPI LIBERO ALL final: {successes}/{episodes} = {successes / episodes:.4f}",
    "INFO:root:OPENPI query timing final: "
    f"count={timing_count} mean_querytime={timing_query_sum / timing_count:.6f} sec "
    f"mean_policy={timing_policy_sum / timing_count:.6f} sec "
    f"mean_roundtrip={timing_roundtrip_sum / timing_count:.6f} sec",
]
destination.write_text("\n".join(output) + "\n", encoding="utf-8")
print(f"[baseline split] merged {successes}/{episodes} into {destination}")
PY

echo "[baseline split] complete: ${FINAL_LOG}"
trap - EXIT INT TERM
