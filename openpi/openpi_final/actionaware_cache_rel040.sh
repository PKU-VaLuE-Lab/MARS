#!/usr/bin/env bash
set -euo pipefail

FINAL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Relative-L1 cache selected for the full seed-195 validation.  Within each
# query, reuse while accumulated relative L1 remains <= 0.4, with a hard cap
# of four consecutive cached diffusion steps before a mandatory refresh.
export OPENPI_ACTIONAWARE=1
export OPENPI_NUM_STEPS=10
export OPENPI_AA_SHORT_STEPS=1
export OPENPI_AA_THRESHOLD=0.7
export OPENPI_AA_CACHE_REL_L1_THRESHOLD=0.4
export OPENPI_AA_MAX_CACHED_STEPS=4

exec "${FINAL_DIR}/eval.sh" "$@"
