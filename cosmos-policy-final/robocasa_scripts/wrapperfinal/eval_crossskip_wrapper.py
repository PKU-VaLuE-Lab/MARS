#!/usr/bin/env python3
import argparse
import os
import sys
import time
import types
from pathlib import Path
from typing import Optional

import torch
import torch.amp as amp
from einops import rearrange


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

DEFAULT_CROSSSKIP_BLOCK_SPEC = "2-11"

from cosmos_policy._src.predict2.networks.minimal_v4_dit import VideoSize

os.environ.setdefault("LIBERO_CONFIG_PATH", str(ROOT / ".libero"))
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(ROOT / ".cache" / "huggingface" / "transformers"))
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def parse_bool(value):
    if isinstance(value, bool):
        return value
    normalized = value.lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def selected_suites(selection):
    aliases = {
        "all": ["libero_spatial", "libero_goal", "libero_object", "libero_10"],
        "spatial": ["libero_spatial"],
        "goal": ["libero_goal"],
        "object": ["libero_object"],
        "10": ["libero_10"],
        "libero_spatial": ["libero_spatial"],
        "libero_goal": ["libero_goal"],
        "libero_object": ["libero_object"],
        "libero_10": ["libero_10"],
    }
    result = []
    for item in selection.split(","):
        key = item.strip()
        if not key:
            continue
        if key not in aliases:
            raise ValueError(f"Unknown LIBERO suite selection: {key}")
        result.extend(aliases[key])
    return result or aliases["all"]


def patch_runtime(module, run_vae_decoder, rollout_videos):
    if not rollout_videos:
        module.save_rollout_video = lambda *args, **kwargs: None
        module.save_rollout_video_with_future_image_predictions = lambda *args, **kwargs: None

    if not run_vae_decoder:
        from cosmos_policy.experiments.robot import cosmos_utils

        empty_predictions = {
            "future_image": None,
            "future_image2": None,
            "future_wrist_image": None,
            "future_wrist_image2": None,
        }
        cosmos_utils.get_future_images_from_generated_samples = lambda *args, **kwargs: dict(empty_predictions)

        original_get_action = module.get_action

        def get_action_without_decoded_future_images(*args, **kwargs):
            result = original_get_action(*args, **kwargs)
            predictions = result.get("future_image_predictions")
            if isinstance(predictions, dict):
                for key, value in empty_predictions.items():
                    predictions.setdefault(key, value)
            return result

        module.get_action = get_action_without_decoded_future_images

    def single_log_setup(cfg, task_identifier, log_dir, run_id_note=None, use_wandb=False, **kwargs):
        run_id = f"ENV_EVAL-{task_identifier}-cosmos-single-log"
        if run_id_note:
            run_id += f"--{run_id_note}"
        return None, str(cfg.master_log_path), run_id

    module.setup_logging = single_log_setup


def patch_config_checkpoint_resolver(local_policy_ckpt):
    from cosmos_policy._src.imaginaire.utils import checkpoint_db

    original_get_checkpoint_path = checkpoint_db.get_checkpoint_path

    def local_get_checkpoint_path(checkpoint_uri):
        if isinstance(checkpoint_uri, str) and checkpoint_uri.startswith("hf://"):
            if checkpoint_uri.endswith("/tokenizer/tokenizer.pth"):
                return str(Path(local_policy_ckpt).with_name("tokenizer.pth"))
            if checkpoint_uri.endswith("/model-480p-16fps.pt"):
                return str(Path(local_policy_ckpt).with_name("model-480p-16fps.pt"))
            if "Cosmos-Policy-ALOHA-Predict2-2B" in checkpoint_uri:
                return str(local_policy_ckpt)
        return original_get_checkpoint_path(checkpoint_uri)

    checkpoint_db.get_checkpoint_path = local_get_checkpoint_path


def patch_libero_assets_path():
    assets_path = ROOT / ".cache" / "libero" / "assets"
    if not assets_path.exists():
        return

    import libero.libero as libero_pkg

    def project_assets_path():
        return str(assets_path)

    libero_pkg._assets_path_cache = str(assets_path)
    libero_pkg.get_assets_path = project_assets_path
    for module_name, loaded_module in list(sys.modules.items()):
        if module_name.startswith("libero.") and hasattr(loaded_module, "get_assets_path"):
            setattr(loaded_module, "get_assets_path", project_assets_path)


def parse_block_indices(spec):
    blocks = set()
    for item in spec.split(","):
        part = item.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            if end < start:
                raise ValueError(f"Invalid block range: {part}")
            blocks.update(range(start, end + 1))
        else:
            blocks.add(int(part))
    return blocks


def block_spec_tag(spec):
    tag = spec.strip().replace(" ", "")
    tag = tag.replace(",", "_").replace("-", "to")
    return tag or "none"


def _crossskip_block_forward(
    self,
    x_B_T_H_W_D: torch.Tensor,
    emb_B_T_D: torch.Tensor,
    crossattn_emb: torch.Tensor,
    rope_emb_L_1_1_D: Optional[torch.Tensor] = None,
    adaln_lora_B_T_3D: Optional[torch.Tensor] = None,
    extra_per_block_pos_emb: Optional[torch.Tensor] = None,
    kv_cache_cfg=None,
) -> torch.Tensor:
    if extra_per_block_pos_emb is not None:
        x_B_T_H_W_D = x_B_T_H_W_D + extra_per_block_pos_emb

    with amp.autocast("cuda", enabled=self.use_wan_fp32_strategy, dtype=torch.float32):
        if self.use_adaln_lora:
            shift_self_attn_B_T_D, scale_self_attn_B_T_D, gate_self_attn_B_T_D = (
                self.adaln_modulation_self_attn(emb_B_T_D) + adaln_lora_B_T_3D
            ).chunk(3, dim=-1)
            shift_mlp_B_T_D, scale_mlp_B_T_D, gate_mlp_B_T_D = (
                self.adaln_modulation_mlp(emb_B_T_D) + adaln_lora_B_T_3D
            ).chunk(3, dim=-1)
        else:
            shift_self_attn_B_T_D, scale_self_attn_B_T_D, gate_self_attn_B_T_D = self.adaln_modulation_self_attn(
                emb_B_T_D
            ).chunk(3, dim=-1)
            shift_mlp_B_T_D, scale_mlp_B_T_D, gate_mlp_B_T_D = self.adaln_modulation_mlp(emb_B_T_D).chunk(3, dim=-1)

    shift_self_attn_B_T_1_1_D = rearrange(shift_self_attn_B_T_D, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)
    scale_self_attn_B_T_1_1_D = rearrange(scale_self_attn_B_T_D, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)
    gate_self_attn_B_T_1_1_D = rearrange(gate_self_attn_B_T_D, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)
    shift_mlp_B_T_1_1_D = rearrange(shift_mlp_B_T_D, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)
    scale_mlp_B_T_1_1_D = rearrange(scale_mlp_B_T_D, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)
    gate_mlp_B_T_1_1_D = rearrange(gate_mlp_B_T_D, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)

    _, T, H, W, _ = x_B_T_H_W_D.shape

    def _fn(_x_B_T_H_W_D, _norm_layer, _scale_B_T_1_1_D, _shift_B_T_1_1_D):
        return _norm_layer(_x_B_T_H_W_D) * (1 + _scale_B_T_1_1_D) + _shift_B_T_1_1_D

    normalized_x_B_T_H_W_D = _fn(
        x_B_T_H_W_D,
        self.layer_norm_self_attn,
        scale_self_attn_B_T_1_1_D,
        shift_self_attn_B_T_1_1_D,
    )
    video_size = VideoSize(T=T, H=H, W=W)
    if self.cp_size is not None and self.cp_size > 1:
        video_size = VideoSize(T=T * self.cp_size, H=H, W=W)

    result_B_T_H_W_D = rearrange(
        self.self_attn(
            rearrange(normalized_x_B_T_H_W_D, "b t h w d -> b (t h w) d"),
            None,
            rope_emb=rope_emb_L_1_1_D,
            video_size=video_size,
            kv_cache_cfg=kv_cache_cfg,
        ),
        "b (t h w) d -> b t h w d",
        t=T,
        h=H,
        w=W,
    )
    x_B_T_H_W_D = x_B_T_H_W_D + gate_self_attn_B_T_1_1_D * result_B_T_H_W_D

    normalized_x_B_T_H_W_D = _fn(
        x_B_T_H_W_D,
        self.layer_norm_mlp,
        scale_mlp_B_T_1_1_D,
        shift_mlp_B_T_1_1_D,
    )
    result_B_T_H_W_D = self.mlp(normalized_x_B_T_H_W_D)
    x_B_T_H_W_D = x_B_T_H_W_D + gate_mlp_B_T_1_1_D * result_B_T_H_W_D
    return x_B_T_H_W_D


def install_crossskip(model, block_indices):
    net = getattr(model, "net", None)
    if net is None or not hasattr(net, "blocks"):
        raise RuntimeError("CrossSkip expected model.net.blocks.")
    num_blocks = len(net.blocks)
    invalid = sorted(idx for idx in block_indices if idx < 0 or idx >= num_blocks)
    if invalid:
        raise ValueError(f"CrossSkip block index out of range 0-{num_blocks - 1}: {invalid}")
    patched_blocks = []
    for block_idx in sorted(block_indices):
        block = net.blocks[block_idx]
        if not hasattr(block, "cross_attn"):
            continue
        if not hasattr(block, "_crossskip_original_forward"):
            block._crossskip_original_forward = block.forward
            block.forward = types.MethodType(_crossskip_block_forward, block)
        patched_blocks.append(block_idx)
    print(
        f"[CrossSkip] installed on {len(patched_blocks)}/{num_blocks} blocks; "
        f"skipping cross-attention blocks={patched_blocks}"
    )


def build_cfg(module, args, suite_name, log_path):
    cfg = module.PolicyEvalConfig(
        config="cosmos_predict2_2b_480p_libero__inference_only",
        ckpt_path=str(ROOT / "ckpt" / "Cosmos-Policy-LIBERO-Predict2-2B.pt"),
        config_file="cosmos_policy/config/config.py",
        use_wrist_image=True,
        use_proprio=True,
        normalize_proprio=True,
        unnormalize_actions=True,
        dataset_stats_path=str(ROOT / "ckpt" / "libero_dataset_statistics.json"),
        t5_text_embeddings_path=str(ROOT / "ckpt" / "libero_t5_embeddings.pkl"),
        trained_with_image_aug=True,
        chunk_size=16,
        num_open_loop_steps=16,
        task_suite_name=suite_name,
        num_trials_per_task=args.num_trials_per_task,
        local_log_dir=str(ROOT / "libero_scripts" / "logsfinal"),
        randomize_seed=False,
        data_collection=False,
        available_gpus="0",
        seed=args.seed,
        use_variance_scale=False,
        deterministic=True,
        run_id_note=args.run_id_note,
        ar_future_prediction=False,
        ar_value_prediction=False,
        use_jpeg_compression=True,
        flip_images=True,
        num_denoising_steps_action=args.num_denoising_steps_action,
        num_denoising_steps_future_state=args.num_denoising_steps_future_state,
        num_denoising_steps_value=args.num_denoising_steps_value,
        use_wandb=False,
    )
    cfg.master_log_path = log_path
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("selection", nargs="?", default=None, help="all, spatial, goal, object, 10, or comma list")
    parser.add_argument("--task-set", default="all", help="Alias for positional selection")
    parser.add_argument("--num-trials-per-task", type=int, default=50)
    parser.add_argument("--max-tasks", type=int, default=0, help="Debug helper; 0 means all tasks")
    parser.add_argument("--seed", type=int, default=195)
    parser.add_argument("--num-denoising-steps-action", type=int, default=5)
    parser.add_argument("--num-denoising-steps-future-state", type=int, default=1)
    parser.add_argument("--num-denoising-steps-value", type=int, default=1)
    parser.add_argument("--rollout-videos", type=parse_bool, default=False)
    parser.add_argument("--run-vae-decoder", type=parse_bool, default=False)
    parser.add_argument("--run-id-note", default="chkpt45000--5stepAct--seed195--deterministic")
    parser.add_argument("--log-path", default=None)
    parser.add_argument(
        "--skip-cross-blocks",
        default=DEFAULT_CROSSSKIP_BLOCK_SPEC,
        help=f"Comma list or inclusive ranges, e.g. 0-10 or 0,2-10 (default: {DEFAULT_CROSSSKIP_BLOCK_SPEC})",
    )
    args = parser.parse_args()

    selection = args.selection or args.task_set
    suites = selected_suites(selection)
    skip_cross_blocks = parse_block_indices(args.skip_cross_blocks)
    log_dir = ROOT / "libero_scripts" / "logsfinal"
    log_dir.mkdir(parents=True, exist_ok=True)
    eval_variant = os.environ.get("COSMOS_EVAL_VARIANT", "").strip()
    crossskip_tag = f"crossskip_b{block_spec_tag(args.skip_cross_blocks)}"
    variant_part = f"_{eval_variant}" if eval_variant else f"_{crossskip_tag}"
    log_path = (
        Path(args.log_path)
        if args.log_path
        else log_dir / f"libero_eval{variant_part}_{time.strftime('%Y%m%d')}_seed{args.seed}.log"
    )

    with open(log_path, "a", buffering=1) as log_file:
        sys.stdout = Tee(sys.__stdout__, log_file)
        sys.stderr = Tee(sys.__stderr__, log_file)

        from cosmos_policy.experiments.robot.libero import run_libero_eval as module

        patch_runtime(module, args.run_vae_decoder, args.rollout_videos)
        patch_libero_assets_path()

        base_cfg = build_cfg(module, args, suites[0], log_path)
        if base_cfg.deterministic:
            os.environ["DETERMINISTIC"] = "True"
        module.validate_config(base_cfg)
        module.set_seed_everywhere(base_cfg.seed)
        module.init_t5_text_embeddings_cache(base_cfg.t5_text_embeddings_path)
        dataset_stats = module.load_dataset_stats(base_cfg.dataset_stats_path)

        patch_config_checkpoint_resolver(Path(base_cfg.ckpt_path))
        model, cosmos_config = module.get_model(base_cfg)
        install_crossskip(model, skip_cross_blocks)
        assert base_cfg.chunk_size == cosmos_config.dataloader_train.dataset.chunk_size, (
            f"Mismatch found between train and test chunk sizes! Train: "
            f"{cosmos_config.dataloader_train.dataset.chunk_size}, Test: {base_cfg.chunk_size}"
        )
        planning_model = None
        resize_size = module.get_image_resize_size(base_cfg.model_family)
        benchmark_dict = module.benchmark.get_benchmark_dict()

        print(f"Master log: {log_path}")
        print(f"LIBERO suites: {', '.join(suites)}")
        print(f"rollout_videos={args.rollout_videos}, run_vae_decoder={args.run_vae_decoder}")
        print(f"CrossSkip skip_cross_blocks={sorted(skip_cross_blocks)}")

        grand_episodes = 0
        grand_successes = 0
        for suite_name in suites:
            cfg = build_cfg(module, args, suite_name, log_path)
            module.validate_config(cfg)
            task_suite = benchmark_dict[cfg.task_suite_name]()
            task_count = task_suite.n_tasks if args.max_tasks <= 0 else min(args.max_tasks, task_suite.n_tasks)
            total_episodes = 0
            total_successes = 0
            module.log_message(f"\n===== LIBERO suite: {suite_name} ({task_count}/{task_suite.n_tasks} tasks) =====")
            for task_id in module.tqdm.tqdm(range(task_count)):
                total_episodes, total_successes = module.run_task(
                    cfg,
                    task_suite,
                    task_id,
                    model,
                    planning_model,
                    dataset_stats,
                    None,
                    resize_size,
                    total_episodes,
                    total_successes,
                    None,
                )
            rate = float(total_successes) / float(total_episodes) if total_episodes else 0.0
            grand_episodes += total_episodes
            grand_successes += total_successes
            module.log_message(
                f"LIBERO suite {suite_name} final: {total_successes}/{total_episodes} = {rate:.4f}"
            )

        grand_rate = float(grand_successes) / float(grand_episodes) if grand_episodes else 0.0
        module.log_message(
            f"\nLIBERO ALL final: {grand_successes}/{grand_episodes} = {grand_rate:.4f}"
        )
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__


if __name__ == "__main__":
    main()
