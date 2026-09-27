#!/usr/bin/env python3
"""Run the finalized RoboCasa DiCache comparison without FlashAttention."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_DIR = ROOT / "robocasa_scripts" / "wrapperfinal"
CANONICAL_PATH = CANONICAL_DIR / "eval_dicache_wrapper.py"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CANONICAL_DIR))

from disable_flash_attention import install as install_no_flash

install_no_flash()

spec = importlib.util.spec_from_file_location(
    "_robocasa_canonical_dicache", CANONICAL_PATH
)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load DiCache wrapper: {CANONICAL_PATH}")
dicache = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = dicache
spec.loader.exec_module(dicache)

from cosmos_policy.experiments.robot import cosmos_utils
from gpu_speed_guard import install_on_cosmos_utils

# Wrap the already-patched DiCache loader so the reservation is made only
# after the policy is fully loaded and the DiCache runtime patch is installed.
install_on_cosmos_utils(cosmos_utils)

previous_patch_runtime = dicache.eval_wrapper.patch_runtime


def patch_runtime_with_final_marker(module, *args, **kwargs):
    previous_patch_runtime(module, *args, **kwargs)
    print("[RoboCasa final] enabled mechanisms=dicache")
    print(
        "[RoboCasa DiCache config] "
        f"probe_depth={dicache.DICACHE_PROBE_DEPTH} "
        f"window_size={dicache.DICACHE_NUM_STEPS} "
        f"ret_ratio={dicache.DICACHE_RET_RATIO} "
        f"rel_l1_thresh={dicache.DICACHE_REL_L1_THRESH} "
        f"gamma=[{dicache.DICACHE_GAMMA_MIN},{dicache.DICACHE_GAMMA_MAX}]"
    )


dicache.eval_wrapper.patch_runtime = patch_runtime_with_final_marker


def main() -> None:
    dicache.eval_wrapper.main()


if __name__ == "__main__":
    main()
