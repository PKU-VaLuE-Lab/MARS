#!/usr/bin/env python3
"""Run the official RoboCasa 24-task evaluation into one aggregate log."""

import argparse
import os
import re
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(ROOT / ".cache" / "huggingface" / "transformers"))
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

# The paper/README evaluates the 24 manipulation tasks. NavigateKitchen is
# registered by RoboCasa but is not part of this 24-task checkpoint protocol.
OFFICIAL_TASKS = (
    "PnPCounterToCab",
    "PnPCabToCounter",
    "PnPCounterToSink",
    "PnPSinkToCounter",
    "PnPCounterToMicrowave",
    "PnPMicrowaveToCounter",
    "PnPCounterToStove",
    "PnPStoveToCounter",
    "OpenSingleDoor",
    "CloseSingleDoor",
    "OpenDoubleDoor",
    "CloseDoubleDoor",
    "OpenDrawer",
    "CloseDrawer",
    "TurnOnSinkFaucet",
    "TurnOffSinkFaucet",
    "TurnSinkSpout",
    "TurnOnStove",
    "TurnOffStove",
    "CoffeeSetupMug",
    "CoffeeServeMug",
    "CoffeePressButton",
    "TurnOnMicrowave",
    "TurnOffMicrowave",
)

TASK_HEADER_RE = re.compile(r"^===== ROBOCASA task: ([A-Za-z0-9_]+) \(")
EPISODE_RE = re.compile(
    r"^  Episode ([0-9]+): (SUCCESS|FAILURE) \(length: ([0-9]+)\)"
)
TASK_FINAL_RE = re.compile(
    r"^ROBOCASA task ([A-Za-z0-9_]+) final: "
    r"([0-9]+)/([0-9]+) = ([0-9.]+);"
)


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


def selected_tasks(selection: str) -> list[str]:
    if selection.strip().lower() == "all":
        return list(OFFICIAL_TASKS)
    lookup = {task.lower(): task for task in OFFICIAL_TASKS}
    tasks = []
    for item in selection.split(","):
        key = item.strip().lower()
        if not key:
            continue
        if key not in lookup:
            raise ValueError(
                f"Unknown RoboCasa task {item!r}; use all or one of {list(OFFICIAL_TASKS)}"
            )
        tasks.append(lookup[key])
    return tasks or list(OFFICIAL_TASKS)


def parse_resume_log(
    log_path: Path,
    selected: list[str],
) -> tuple[set[str], dict[str, dict[int, tuple[bool, int]]], dict[str, float]]:
    """Recover completed tasks and unique episode results from an aggregate log."""

    selected_set = set(selected)
    completed: set[str] = set()
    episodes = {task: {} for task in selected}
    task_rates: dict[str, float] = {}
    active_task = None
    with log_path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            header = TASK_HEADER_RE.match(line)
            if header:
                candidate = header.group(1)
                active_task = candidate if candidate in selected_set else None
                continue
            episode = EPISODE_RE.match(line)
            if episode and active_task is not None:
                episode_idx = int(episode.group(1))
                episodes[active_task][episode_idx] = (
                    episode.group(2) == "SUCCESS",
                    int(episode.group(3)),
                )
                continue
            final = TASK_FINAL_RE.match(line)
            if final:
                task_name = final.group(1)
                if task_name in selected_set:
                    completed.add(task_name)
                    task_rates[task_name] = float(final.group(4))
                active_task = None
    return completed, episodes, task_rates


def contiguous_episode_count(
    records: dict[int, tuple[bool, int]],
    expected_total: int,
) -> int:
    count = 0
    while count in records:
        count += 1
    unexpected = sorted(index for index in records if index >= count)
    if unexpected:
        raise RuntimeError(
            "Resume log has non-contiguous episode indices after "
            f"{count - 1}: {unexpected}"
        )
    if count > expected_total:
        raise RuntimeError(
            f"Resume log has {count} episodes but expected at most {expected_total}"
        )
    return count


def patch_runtime(module, run_vae_decoder, rollout_videos):
    # Video emission is an output-only side effect and is disabled for large
    # aggregate runs unless explicitly requested.
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
        cosmos_utils.get_future_images_from_generated_samples = (
            lambda *args, **kwargs: dict(empty_predictions)
        )
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
            # The config registry imports every robot experiment before
            # selecting RoboCasa. Prevent that import-only ALOHA entry from
            # triggering an unrelated 3.91 GB download.
            if "Cosmos-Policy-ALOHA-Predict2-2B" in checkpoint_uri:
                return str(local_policy_ckpt)
        return original_get_checkpoint_path(checkpoint_uri)

    checkpoint_db.get_checkpoint_path = local_get_checkpoint_path


def build_cfg(module, args, task_name, log_path):
    ckpt_dir = ROOT / "robocasa_scripts" / "ckpt"
    cfg = module.PolicyEvalConfig(
        suite="robocasa",
        config="cosmos_predict2_2b_480p_robocasa_50_demos_per_task__inference",
        ckpt_path=str(ckpt_dir / "Cosmos-Policy-RoboCasa-Predict2-2B.pt"),
        config_file="cosmos_policy/config/config.py",
        use_third_person_image=True,
        num_third_person_images=2,
        use_wrist_image=True,
        num_wrist_images=1,
        use_proprio=True,
        normalize_proprio=True,
        unnormalize_actions=True,
        dataset_stats_path=str(ckpt_dir / "robocasa_dataset_statistics.json"),
        t5_text_embeddings_path=str(ckpt_dir / "robocasa_t5_embeddings.pkl"),
        trained_with_image_aug=True,
        chunk_size=32,
        num_open_loop_steps=16,
        task_name=task_name,
        num_trials_per_task=args.num_trials_per_task,
        local_log_dir=str(ROOT / "robocasa_scripts" / "logsfinal"),
        seed=args.seed,
        randomize_seed=False,
        deterministic=True,
        use_variance_scale=False,
        use_jpeg_compression=True,
        flip_images=True,
        num_denoising_steps_action=args.num_denoising_steps_action,
        num_denoising_steps_future_state=args.num_denoising_steps_future_state,
        num_denoising_steps_value=args.num_denoising_steps_value,
        data_collection=False,
        use_parallel_inference=False,
        num_queries_best_of_n=1,
        use_wandb=False,
        run_id_note=args.run_id_note,
    )
    cfg.master_log_path = log_path
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("selection", nargs="?", default=None, help="all or a comma-separated task list")
    parser.add_argument("--task-set", default="all", help="Alias for positional selection")
    parser.add_argument("--num-trials-per-task", type=int, default=50)
    parser.add_argument("--max-tasks", type=int, default=0, help="Debug helper; 0 means all selected tasks")
    parser.add_argument("--seed", type=int, default=195)
    parser.add_argument("--num-denoising-steps-action", type=int, default=5)
    parser.add_argument("--num-denoising-steps-future-state", type=int, default=1)
    parser.add_argument("--num-denoising-steps-value", type=int, default=1)
    parser.add_argument("--rollout-videos", type=parse_bool, default=False)
    parser.add_argument("--run-vae-decoder", type=parse_bool, default=True)
    parser.add_argument("--run-id-note", default="chkpt45000--5stepAct--seed195--deterministic")
    parser.add_argument("--log-path", default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume the first incomplete task/episode found in --log-path",
    )
    args = parser.parse_args()

    tasks = selected_tasks(args.selection or args.task_set)
    if args.max_tasks > 0:
        tasks = tasks[: args.max_tasks]
    log_dir = ROOT / "robocasa_scripts" / "logsfinal"
    log_dir.mkdir(parents=True, exist_ok=True)
    eval_variant = os.environ.get("COSMOS_EVAL_VARIANT", "").strip()
    variant_part = f"_{eval_variant}" if eval_variant else ""
    log_path = (
        Path(args.log_path)
        if args.log_path
        else log_dir / f"robocasa_eval{variant_part}_{time.strftime('%Y%m%d')}_seed{args.seed}.log"
    )
    completed_tasks: set[str] = set()
    existing_episodes = {task: {} for task in tasks}
    existing_task_rates: dict[str, float] = {}
    if args.resume:
        if not log_path.is_file():
            raise FileNotFoundError(f"Cannot resume missing log: {log_path}")
        completed_tasks, existing_episodes, existing_task_rates = parse_resume_log(
            log_path,
            tasks,
        )
    tasks_to_run = [task for task in tasks if task not in completed_tasks]
    if not tasks_to_run:
        raise RuntimeError(f"All selected tasks are already complete in {log_path}")

    log_mode = "a" if args.resume else "w"
    with open(log_path, log_mode, buffering=1) as log_file:
        sys.stdout = Tee(sys.__stdout__, log_file)
        sys.stderr = Tee(sys.__stderr__, log_file)

        from cosmos_policy.experiments.robot.robocasa import run_robocasa_eval as module

        patch_runtime(module, args.run_vae_decoder, args.rollout_videos)
        base_cfg = build_cfg(module, args, tasks_to_run[0], log_path)
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

        print(f"Master log: {log_path}")
        print(f"RoboCasa official tasks: {len(tasks)}/{len(OFFICIAL_TASKS)}")
        print(f"Tasks: {', '.join(tasks)}")
        if args.resume:
            existing_count = sum(len(records) for records in existing_episodes.values())
            existing_successes = sum(
                int(success)
                for records in existing_episodes.values()
                for success, _ in records.values()
            )
            print(
                f"Resume: completed_tasks={len(completed_tasks)} "
                f"existing={existing_successes}/{existing_count} "
                f"remaining_tasks={len(tasks_to_run)}"
            )
        print(
            "Official protocol: chunk=32 open_loop=16 "
            f"trials={args.num_trials_per_task} seed={args.seed} "
            "deterministic=True denoise="
            f"{args.num_denoising_steps_action}/"
            f"{args.num_denoising_steps_future_state}/"
            f"{args.num_denoising_steps_value} jpeg=True flip=True"
        )
        print(f"rollout_videos={args.rollout_videos}, run_vae_decoder={args.run_vae_decoder}")

        grand_episodes = sum(len(records) for records in existing_episodes.values())
        grand_successes = sum(
            int(success)
            for records in existing_episodes.values()
            for success, _ in records.values()
        )
        task_rates = dict(existing_task_rates)
        for task_name in tasks_to_run:
            cfg = build_cfg(module, args, task_name, log_path)
            prefix_records = existing_episodes.get(task_name, {})
            start_episode_idx = contiguous_episode_count(
                prefix_records,
                cfg.num_trials_per_task,
            )
            cfg.episode_start_index = start_episode_idx
            module.validate_config(cfg)
            module.log_message(
                f"\n===== ROBOCASA task: {task_name} ({cfg.num_trials_per_task} trials) ====="
            )
            if start_episode_idx:
                module.log_message(
                    f"ROBOCASA resume: {task_name} starts at episode "
                    f"{start_episode_idx + 1}/{cfg.num_trials_per_task}"
                )
            rate, avg_length, successes = module.run_task(
                cfg,
                task_name,
                model,
                None,
                dataset_stats,
                None,
                None,
            )
            new_episodes = len(successes)
            new_successful = int(sum(successes))
            prefix_episodes = len(prefix_records)
            prefix_successful = sum(int(success) for success, _ in prefix_records.values())
            prefix_length_sum = sum(length for _, length in prefix_records.values())
            episodes = prefix_episodes + new_episodes
            successful = prefix_successful + new_successful
            combined_rate = successful / episodes if episodes else 0.0
            combined_avg_length = (
                prefix_length_sum + float(avg_length) * new_episodes
            ) / episodes if episodes else 0.0
            grand_episodes += new_episodes
            grand_successes += new_successful
            task_rates[task_name] = combined_rate
            module.log_message(
                f"ROBOCASA task {task_name} final: {successful}/{episodes} = "
                f"{combined_rate:.4f}; avg_length={combined_avg_length:.1f}"
            )

        grand_rate = grand_successes / grand_episodes if grand_episodes else 0.0
        module.log_message(f"\nROBOCASA task rates: {task_rates}")
        module.log_message(
            f"ROBOCASA ALL final: {grand_successes}/{grand_episodes} = {grand_rate:.4f}"
        )
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__


if __name__ == "__main__":
    main()
