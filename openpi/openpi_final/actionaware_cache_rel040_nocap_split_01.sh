#!/usr/bin/env bash
set -euo pipefail

FINAL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${FINAL_DIR}/env.sh"

SEED="${SEED:-195}"
TRIALS_PER_TASK="${TRIALS_PER_TASK:-50}"
MERGE_ONLY="${MERGE_ONLY:-0}"
FINAL_LOG="${FINAL_DIR}/logs/pi05_libero_actionaware_cache_rel040_nocap_all2000_seed${SEED}.log"
WORK_DIR="${FINAL_DIR}/logs/.actionaware_cache_rel040_nocap_split_01_seed${SEED}"
GPU0_LOG="${WORK_DIR}/gpu0_goal_long.log"
GPU1_LOG="${WORK_DIR}/gpu1_object_spatial.log"

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
  for pid in "${pid0:-}" "${pid1:-}"; do
    [[ -n "${pid}" ]] && wait "${pid}" 2>/dev/null || true
  done
  exit "${status}"
}

wait_for_gpu_idle() {
  local gpu="$1"
  local memory_used
  while true; do
    memory_used="$(nvidia-smi --id="${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
    if flock -n "/tmp/cosmos-policy-gpu-${gpu}.lock" -c true && [[ "${memory_used}" -lt 1024 ]]; then
      echo "[Relative-L1 0.4 no-cap] GPU${gpu} is idle (${memory_used} MiB); launching shard"
      return 0
    fi
    echo "[Relative-L1 0.4 no-cap] waiting for GPU${gpu}: memory_used=${memory_used} MiB"
    sleep 10
  done
}

launch_shard() {
  local gpu="$1"
  local port="$2"
  local suites="$3"
  local log_path="$4"
  wait_for_gpu_idle "${gpu}"
  exec env \
    GPU="${gpu}" PORT="${port}" SEED="${SEED}" TRIALS_PER_TASK="${TRIALS_PER_TASK}" \
    TIMING_WARMUP_QUERIES=1 SUITES="${suites}" LOG_PATH="${log_path}" \
    bash "${FINAL_DIR}/actionaware_cache_rel040_nocap.sh"
}

if [[ "${MERGE_ONLY}" != "1" ]]; then
  trap cleanup EXIT INT TERM
  echo "[Relative-L1 0.4 no-cap] GPU0=goal,long GPU1=object,spatial seed=${SEED} trials_per_task=${TRIALS_PER_TASK}"
  (launch_shard 0 8200 libero_goal,libero_10 "${GPU0_LOG}") >"${WORK_DIR}/gpu0_launch.log" 2>&1 &
  pid0=$!
  (launch_shard 1 8201 libero_object,libero_spatial "${GPU1_LOG}") >"${WORK_DIR}/gpu1_launch.log" 2>&1 &
  pid1=$!

  status0=0
  status1=0
  wait "${pid0}" || status0=$?
  wait "${pid1}" || status1=$?
  if [[ "${status0}" -ne 0 || "${status1}" -ne 0 ]]; then
    echo "Relative-L1 0.4 no-cap evaluation failed: gpu0=${status0} gpu1=${status1}" >&2
    exit 1
  fi
fi

"${OPENPI_ROOT}/.conda/bin/python" - \
  "${GPU0_LOG}" "${GPU1_LOG}" "${FINAL_LOG}" "${SEED}" "${TRIALS_PER_TASK}" <<'PY'
from __future__ import annotations

import re
import sys
from collections import OrderedDict
from pathlib import Path

shards = [Path(sys.argv[1]), Path(sys.argv[2])]
destination = Path(sys.argv[3])
seed = int(sys.argv[4])
trials = int(sys.argv[5])
expected_suites = ["libero_goal", "libero_object", "libero_spatial", "libero_10"]

suite_results: dict[str, list[int]] = {}
suite_bodies: dict[str, list[str]] = {}
timing_count = 0
timing_query_sum = 0.0
timing_policy_sum = 0.0
timing_roundtrip_sum = 0.0
aa_queries = 0
aa_route_steps = 0
aa_full_forwards = 0
aa_reused_forwards = 0
aa_score_sum = 0.0

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
            suite_bodies[current_suite] = []
            copying = True
        if line.startswith("INFO:root:OPENPI LIBERO suite rates:"):
            copying = False
        if copying:
            suite_bodies[current_suite].append(line)
            if line.startswith("INFO:root:Success:"):
                suite_results[current_suite][0 if line.endswith("True") else 1] += 1
            match = re.search(
                r"ActionAware query \d+: score=([0-9.eE+-]+) route_steps=(\d+) full=(\d+) reused=(\d+)",
                line,
            )
            if match:
                aa_queries += 1
                aa_score_sum += float(match.group(1))
                aa_route_steps += int(match.group(2))
                aa_full_forwards += int(match.group(3))
                aa_reused_forwards += int(match.group(4))
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

if set(suite_results) != set(expected_suites):
    raise RuntimeError(f"suite mismatch: {sorted(suite_results)}")
expected_episodes = len(expected_suites) * 10 * trials
episodes = sum(sum(value) for value in suite_results.values())
successes = sum(value[0] for value in suite_results.values())
if episodes != expected_episodes:
    raise RuntimeError(f"expected {expected_episodes} episodes, found {episodes}")
if timing_count <= 0 or aa_queries <= 0:
    raise RuntimeError("missing timing or ActionAware diagnostics")

rates = OrderedDict(
    (suite, suite_results[suite][0] / sum(suite_results[suite])) for suite in expected_suites
)
body = [line for suite in expected_suites for line in suite_bodies[suite]]
all_forwards = aa_full_forwards + aa_reused_forwards
output = [
    f"[OpenPI final] variant=actionaware_cache_rel040_nocap gpu=0,1 seed={seed} "
    f"suites={','.join(expected_suites)} trials_per_task={trials} max_tasks=0",
    "[OpenPI sampler] num_steps=10 short_steps=1 action_threshold=0.7 "
    "cache_rel_l1_threshold=0.4 max_cached_steps=-1 (unlimited)",
    *body,
    f"INFO:root:OPENPI LIBERO suite rates: {dict(rates)}",
    f"INFO:root:OPENPI LIBERO ALL final: {successes}/{episodes} = {successes / episodes:.4f}",
    "INFO:root:OPENPI query timing final: "
    f"count={timing_count} mean_querytime={timing_query_sum / timing_count:.6f} sec "
    f"mean_policy={timing_policy_sum / timing_count:.6f} sec "
    f"mean_roundtrip={timing_roundtrip_sum / timing_count:.6f} sec",
    "INFO:root:OPENPI ActionAware final: "
    f"queries={aa_queries} mean_route_steps={aa_route_steps / aa_queries:.4f} "
    f"full_forwards={aa_full_forwards} reused_forwards={aa_reused_forwards} "
    f"reuse_ratio={aa_reused_forwards / all_forwards:.4f} mean_score={aa_score_sum / aa_queries:.6f}",
]
destination.write_text("\n".join(output) + "\n", encoding="utf-8")
print(f"[Relative-L1 0.4 no-cap] merged {successes}/{episodes} into {destination}")
PY

echo "[Relative-L1 0.4 no-cap] complete: ${FINAL_LOG}"
trap - EXIT INT TERM
