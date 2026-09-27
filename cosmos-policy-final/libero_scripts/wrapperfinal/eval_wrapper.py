#!/usr/bin/env python3
import argparse
import os
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

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
    parser.add_argument(
        "--trial-indices",
        default="",
        help="Optional comma-separated DEFAULT initial-state indices; overrides num trials",
    )
    parser.add_argument("--max-tasks", type=int, default=0, help="Debug helper; 0 means all tasks")
    parser.add_argument("--seed", type=int, default=195)
    parser.add_argument("--num-denoising-steps-action", type=int, default=5)
    parser.add_argument("--num-denoising-steps-future-state", type=int, default=1)
    parser.add_argument("--num-denoising-steps-value", type=int, default=1)
    parser.add_argument("--rollout-videos", type=parse_bool, default=False)
    parser.add_argument("--run-vae-decoder", type=parse_bool, default=False)
    parser.add_argument("--run-id-note", default="chkpt45000--5stepAct--seed195--deterministic")
    parser.add_argument("--log-path", default=None)
    args = parser.parse_args()

    trial_indices = []
    if args.trial_indices:
        trial_indices = [int(item.strip()) for item in args.trial_indices.split(",") if item.strip()]
        if not trial_indices or any(index < 0 for index in trial_indices):
            parser.error("--trial-indices must contain non-negative comma-separated integers")
        if len(set(trial_indices)) != len(trial_indices):
            parser.error("--trial-indices must not contain duplicates")
        args.num_trials_per_task = len(trial_indices)

    selection = args.selection or args.task_set
    suites = selected_suites(selection)
    log_dir = ROOT / "libero_scripts" / "logsfinal"
    log_dir.mkdir(parents=True, exist_ok=True)
    eval_variant = os.environ.get("COSMOS_EVAL_VARIANT", "").strip()
    variant_part = f"_{eval_variant}" if eval_variant else ""
    log_path = (
        Path(args.log_path)
        if args.log_path
        else log_dir / f"libero_eval{variant_part}_{time.strftime('%Y%m%d')}_seed{args.seed}.log"
    )

    with open(log_path, "a", buffering=1) as log_file:
        sys.stdout = Tee(sys.__stdout__, log_file)
        sys.stderr = Tee(sys.__stderr__, log_file)

        from cosmos_policy.experiments.robot.libero import run_libero_eval as module

        if trial_indices:
            original_load_initial_states = module.load_initial_states

            def load_selected_initial_states(cfg, task_suite, task_id, log_file=None):
                initial_states, all_initial_states = original_load_initial_states(
                    cfg, task_suite, task_id, log_file
                )
                if all_initial_states is not None:
                    raise ValueError("--trial-indices currently requires initial_states_path=DEFAULT")
                if max(trial_indices) >= len(initial_states):
                    raise IndexError(
                        f"Trial index {max(trial_indices)} exceeds {len(initial_states)} available states"
                    )
                return [initial_states[index] for index in trial_indices], all_initial_states

            module.load_initial_states = load_selected_initial_states

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
        if trial_indices:
            print(f"trial_indices={trial_indices}")

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
