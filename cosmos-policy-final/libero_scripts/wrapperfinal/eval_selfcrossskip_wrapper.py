#!/usr/bin/env python3
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

    config_path = str(Path(config_path).resolve())
    os.environ["COSMOS_SELFHEADSKIP_CONFIG"] = config_path
    os.environ["COSMOS_CROSSSKIP_BLOCKS"] = skip_cross_blocks
    os.environ.setdefault("COSMOS_EVAL_VARIANT", "selfcrossskip")

    from cosmos_policy.experiments.robot.libero import run_libero_eval as module
    import eval_wrapper
    from eval_crossskip_wrapper import install_crossskip, parse_block_indices
    from selfhead_pruning import install_selfhead_pruning

    original_get_model = module.get_model
    original_patch_runtime = eval_wrapper.patch_runtime
    cross_blocks = parse_block_indices(skip_cross_blocks)

    def patch_runtime_without_duplicate_logs(eval_module, run_vae_decoder, rollout_videos):
        original_patch_runtime(eval_module, run_vae_decoder, rollout_videos)

        def single_log_message(message, log_file=None):
            print(message)

        eval_module.log_message = single_log_message

    def get_model_with_selfcrossskip(cfg):
        model, cosmos_config = original_get_model(cfg)
        install_selfhead_pruning(model, config_path, log_fn=print)
        install_crossskip(model, cross_blocks)
        return model, cosmos_config

    module.get_model = get_model_with_selfcrossskip
    eval_wrapper.patch_runtime = patch_runtime_without_duplicate_logs
    eval_wrapper.main()


if __name__ == "__main__":
    main()
