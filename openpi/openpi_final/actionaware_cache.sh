#!/usr/bin/env bash
set -euo pipefail

FINAL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Selected on seed-195, 200-episode exploration.  High threshold plus a
# one-step cap implements: evaluate once, reuse once, then refresh.
export OPENPI_ACTIONAWARE="${OPENPI_ACTIONAWARE:-1}"
export OPENPI_NUM_STEPS="${OPENPI_NUM_STEPS:-10}"
export OPENPI_AA_SHORT_STEPS="${OPENPI_AA_SHORT_STEPS:-1}"
export OPENPI_AA_THRESHOLD="${OPENPI_AA_THRESHOLD:-0.7}"
export OPENPI_AA_CACHE_REL_L1_THRESHOLD="${OPENPI_AA_CACHE_REL_L1_THRESHOLD:-1000000}"
export OPENPI_AA_MAX_CACHED_STEPS="${OPENPI_AA_MAX_CACHED_STEPS:-1}"

exec "${FINAL_DIR}/eval.sh" "$@"
