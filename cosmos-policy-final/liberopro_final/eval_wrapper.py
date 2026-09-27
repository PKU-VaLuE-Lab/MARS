#!/usr/bin/env python3
"""Run the finalized LIBERO-Pro baseline or any validated accelerator subset."""

from __future__ import annotations

import os
import sys
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FINAL_DIR = Path(__file__).resolve().parent
WRAPPER_DIR = ROOT / "liberopro_scripts" / "wrapperfinal"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(FINAL_DIR))
sys.path.insert(0, str(WRAPPER_DIR))
os.chdir(ROOT)

from disable_flash_attention import install as install_no_flash

install_no_flash()

KNOWN_MECHANISMS = {"actionaware", "selfcross", "vaeskip"}
MECHANISMS = {
    item.strip().lower()
    for item in os.environ.get("LIBEROPRO_FINAL_MECHANISMS", "").split(",")
    if item.strip()
}
unknown = MECHANISMS - KNOWN_MECHANISMS
if unknown:
    raise ValueError(f"Unknown LIBERO-Pro final mechanisms: {sorted(unknown)}")

# Conservative Bottom-20 relative-L1 selection validated for the seed-195
# LIBERO-Pro standard-safe configuration.
SELF_HEADS_BY_BLOCK = {
    2: [2, 13],
    3: [1, 3, 8, 9, 10, 12, 15],
    4: [1, 5, 9, 12],
    5: [11, 12],
    6: [12],
    7: [0],
    8: [11],
    9: [12],
    10: [0],
}
CROSS_BLOCKS = list(range(2, 11))

shallowvaeskip = None
actionaware = None

# Import order builds one model-loading chain:
# baseline -> VAE reuse -> ActionAware -> Self/Cross pruning.
if "vaeskip" in MECHANISMS:
    import eval_shallowvaeskip_wrapper as shallowvaeskip

if "actionaware" in MECHANISMS:
    import eval_actionaware_wrapper as actionaware

from cosmos_policy.experiments.robot import cosmos_utils

if "selfcross" in MECHANISMS:
    from eval_crossskip_wrapper import install_crossskip
    from selfhead_pruning import install_selfhead_pruning_from_mapping

    previous_get_model = cosmos_utils.get_model

    def get_model_with_selfcross(cfg):
        model, cosmos_config = previous_get_model(cfg)
        install_selfhead_pruning_from_mapping(
            model,
            SELF_HEADS_BY_BLOCK,
            expected_num_blocks=28,
            expected_num_heads=16,
            log_fn=print,
        )
        install_crossskip(model, CROSS_BLOCKS)
        return model, cosmos_config

    cosmos_utils.get_model = get_model_with_selfcross

from gpu_speed_guard import install_on_cosmos_utils

install_on_cosmos_utils(cosmos_utils)

import eval_wrapper as base_eval
from suite_config import ensure_default_trial_count, install_on_base_eval

previous_patch_runtime = base_eval.patch_runtime
install_on_base_eval(base_eval)


def patch_runtime_with_final_marker(module, *args, **kwargs):
    previous_patch_runtime(module, *args, **kwargs)
    if shallowvaeskip is not None:
        shallowvaeskip._set_flop_profile_callable(module.get_action)
    enabled = ",".join(sorted(MECHANISMS)) if MECHANISMS else "baseline"
    print(f"[LIBERO-Pro final] enabled mechanisms={enabled}")


base_eval.patch_runtime = patch_runtime_with_final_marker


def _explicit_log_path() -> Path | None:
    for index, arg in enumerate(sys.argv[1:]):
        if arg == "--log-path" and index + 2 <= len(sys.argv[1:]):
            return Path(sys.argv[index + 2]).expanduser().resolve()
        if arg.startswith("--log-path="):
            return Path(arg.split("=", 1)[1]).expanduser().resolve()
    return None


def main() -> None:
    ensure_default_trial_count()
    base_eval.main()
    final_lines = StringIO()
    with redirect_stdout(final_lines):
        if actionaware is not None:
            actionaware._finalize_actionaware_stats()
        if shallowvaeskip is not None:
            suite_overrides = any(
                (
                    shallowvaeskip.SUITE_LAYERS,
                    shallowvaeskip.SUITE_TOPK,
                    shallowvaeskip.SUITE_IOU_THRESHOLDS,
                    shallowvaeskip.SUITE_CAMERA_MODES,
                    shallowvaeskip.SUITE_PRIMARY_GUARD_THRESHOLDS,
                    shallowvaeskip.SUITE_MAX_CONSECUTIVE_SKIPS,
                    shallowvaeskip.SUITE_ENABLED,
                )
            )
            print(
                "[ShallowVAESkip config] "
                f"layer={shallowvaeskip.LAYER} "
                f"topk={shallowvaeskip.TOPK_FRACTION} "
                f"iou={shallowvaeskip.IOU_THRESHOLD} "
                f"camera={shallowvaeskip.CAMERA_MODE} "
                f"primary_iou={shallowvaeskip.PRIMARY_GUARD_THRESHOLD} "
                f"maxskip={shallowvaeskip.MAX_CONSECUTIVE_SKIPS} "
                f"primary_maxskip={shallowvaeskip.PRIMARY_MAX_CONSECUTIVE_SKIPS} "
                f"wrist_maxskip={shallowvaeskip.WRIST_MAX_CONSECUTIVE_SKIPS} "
                f"query_lag={shallowvaeskip.QUERY_LAG} "
                f"uniform={str(not suite_overrides).lower()} "
                f"suite_iou={shallowvaeskip.SUITE_IOU_THRESHOLDS} "
                f"suite_camera={shallowvaeskip.SUITE_CAMERA_MODES} "
                f"suite_primary={shallowvaeskip.SUITE_PRIMARY_GUARD_THRESHOLDS}"
            )
            shallowvaeskip._write_summary()
    text = final_lines.getvalue()
    if text:
        print(text, end="")
        log_path = _explicit_log_path()
        if log_path is not None:
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(text)
    if shallowvaeskip is not None:
        flop_text = shallowvaeskip._profile_flops_outside_querytime()
        if flop_text:
            print(flop_text, end="")
            log_path = _explicit_log_path()
            if log_path is not None:
                with log_path.open("a", encoding="utf-8") as stream:
                    stream.write(flop_text)


if __name__ == "__main__":
    main()
