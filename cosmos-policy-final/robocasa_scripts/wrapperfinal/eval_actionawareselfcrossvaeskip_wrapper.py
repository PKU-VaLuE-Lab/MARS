#!/usr/bin/env python3
"""Compose ActionAware routing, H40 SelfCross, and shallow-VAE reuse."""

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WRAPPER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(WRAPPER_DIR))
os.chdir(ROOT)


def _strip_args(argv: list[str]) -> tuple[list[str], str | None, str | None]:
    stripped = [argv[0]]
    config_path = os.environ.get("COSMOS_SELFHEADSKIP_CONFIG")
    skip_cross_blocks = os.environ.get("COSMOS_CROSSSKIP_BLOCKS")
    idx = 1
    while idx < len(argv):
        arg = argv[idx]
        if arg == "--self-head-config":
            if idx + 1 >= len(argv):
                raise SystemExit("--self-head-config requires a path")
            config_path = argv[idx + 1]
            idx += 2
            continue
        if arg.startswith("--self-head-config="):
            config_path = arg.split("=", 1)[1]
            idx += 1
            continue
        if arg == "--skip-cross-blocks":
            if idx + 1 >= len(argv):
                raise SystemExit("--skip-cross-blocks requires a block spec")
            skip_cross_blocks = argv[idx + 1]
            idx += 2
            continue
        if arg.startswith("--skip-cross-blocks="):
            skip_cross_blocks = arg.split("=", 1)[1]
            idx += 1
            continue
        stripped.append(arg)
        idx += 1
    return stripped, config_path, skip_cross_blocks


def main() -> None:
    sys.argv, config_path, skip_cross_blocks = _strip_args(sys.argv)
    if not config_path:
        raise SystemExit("Missing --self-head-config or COSMOS_SELFHEADSKIP_CONFIG")
    if not skip_cross_blocks:
        raise SystemExit("Missing --skip-cross-blocks or COSMOS_CROSSSKIP_BLOCKS")

    config_path = str(Path(config_path).expanduser().resolve())
    os.environ["COSMOS_SELFHEADSKIP_CONFIG"] = config_path
    os.environ["COSMOS_CROSSSKIP_BLOCKS"] = skip_cross_blocks
    os.environ["COSMOS_STEPSKIP_REUSE_MODE"] = "residual"
    os.environ["COSMOS_STEPSKIP_USE_REL_L1"] = "0"
    os.environ["COSMOS_STEPSKIP_NUM_STEPS"] = "5"

    import eval_shallowvaeskip_wrapper as shallowvaeskip
    from cosmos_policy.experiments.robot import cosmos_utils
    import eval_stepskip_wrapper as stepskip

    get_model_with_shallowvaeskip = cosmos_utils.get_model

    def get_model_with_stepskip(cfg):
        model, cosmos_config = get_model_with_shallowvaeskip(cfg)
        stepskip._install_stepskip(model)
        return model, cosmos_config

    cosmos_utils.get_model = get_model_with_stepskip
    sys.modules["eval_stepskip_wrapper"] = stepskip

    import eval_actionaware_wrapper as actionaware
    from eval_crossskip_wrapper import install_crossskip, parse_block_indices
    from selfhead_pruning import install_selfhead_pruning

    get_model_with_actionaware = cosmos_utils.get_model
    patch_runtime_with_all_gates = actionaware.eval_wrapper.patch_runtime
    cross_blocks = parse_block_indices(skip_cross_blocks)

    def get_model_with_all_accelerators(cfg):
        model, cosmos_config = get_model_with_actionaware(cfg)
        install_selfhead_pruning(model, config_path, log_fn=print)
        install_crossskip(model, cross_blocks)
        return model, cosmos_config

    def patch_runtime_without_duplicate_logs(eval_module, run_vae_decoder, rollout_videos):
        patch_runtime_with_all_gates(eval_module, run_vae_decoder, rollout_videos)
        eval_module.log_message = lambda message, log_file=None: print(message)

    cosmos_utils.get_model = get_model_with_all_accelerators
    actionaware.eval_wrapper.patch_runtime = patch_runtime_without_duplicate_logs
    actionaware.eval_wrapper.main()
    actionaware._finalize_actionaware_stats()
    shallowvaeskip._write_summary()


if __name__ == "__main__":
    main()
