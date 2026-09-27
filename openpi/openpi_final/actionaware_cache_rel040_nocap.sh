#!/usr/bin/env bash
set -euo pipefail

FINAL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Threshold-only Relative-L1 cache.  A negative max_cached_steps explicitly
# means there is no forced consecutive-reuse refresh; accumulated Relative-L1
# above 0.4 is the only cache-refresh condition.
export OPENPI_ACTIONAWARE=1
export OPENPI_NUM_STEPS=10
export OPENPI_AA_SHORT_STEPS=1
export OPENPI_AA_THRESHOLD=0.7
export OPENPI_AA_CACHE_REL_L1_THRESHOLD=0.4
export OPENPI_AA_MAX_CACHED_STEPS=-1

exec "${FINAL_DIR}/eval.sh" "$@"
