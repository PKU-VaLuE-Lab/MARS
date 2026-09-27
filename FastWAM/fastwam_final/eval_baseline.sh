#!/usr/bin/env bash
set -euo pipefail
export FASTWAM_VARIANT=baseline
exec "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/eval.sh" "$@"
