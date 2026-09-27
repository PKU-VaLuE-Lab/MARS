#!/usr/bin/env python3
"""Single-model, resumable evaluation over all four LIBERO suites."""

from __future__ import annotations

import json
import hashlib
import logging
import os
import sys
import time
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.libero.eval_libero_single import (  # noqa: E402
    NumpyEncoder,
    _load_model_checkpoint,
    _mixed_precision_to_model_dtype,
    _resolve_dataset_stats_path,
    _resolve_eval_device,
    get_query_timing,
    reset_query_timing,
    run_single_task,
)
from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor  # noqa: E402
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json  # noqa: E402
from fastwam.models.wan22.actionaware import get_actionaware_router  # noqa: E402
from fastwam.models.wan22.adaptive_action_cache import (  # noqa: E402
    get_action_block_cache_stats,
)
from fastwam.utils.pytorch_utils import set_global_seed  # noqa: E402
from libero.libero import benchmark  # noqa: E402


OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("max", lambda x: max(x), replace=True)
OmegaConf.register_new_resolver("split", lambda s, idx: s.split("/")[int(idx)], replace=True)


ACCELERATION_ENV_NAMES = (
    "FASTWAM_ACTIONAWARE_ENABLED",
    "FASTWAM_ACTIONAWARE_MODE",
    "FASTWAM_ACTIONAWARE_DRAFT_STEPS",
    "FASTWAM_ACTIONAWARE_TRANS_THRESHOLD",
    "FASTWAM_ACTIONAWARE_FEATURE_STEPS",
    "FASTWAM_ACTION_CACHE_ENABLED",
    "FASTWAM_ACTION_CACHE_MODE",
    "FASTWAM_ACTION_CACHE_RELATIVE_L1_THRESHOLD",
    "FASTWAM_ACTION_CACHE_RELATIVE_MAX_THRESHOLD",
    "FASTWAM_ACTION_CACHE_MAX_CONSECUTIVE_REUSE",
    "FASTWAM_ACTION_CACHE_START_BLOCK",
)


def _acceleration_config() -> dict[str, str | None]:
    return {name: os.environ.get(name) for name in ACCELERATION_ENV_NAMES}


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, cls=NumpyEncoder) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _task_result_path(output_dir: Path, suite_name: str, task_id: int) -> Path:
    return output_dir / "task_results" / f"{suite_name}_task{task_id:02d}.json"


def _completed_result(
    path: Path,
    num_trials: int,
    seed: int | None,
    run_signature: str,
) -> dict | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if int(payload.get("total_episodes", -1)) != num_trials:
        return None
    if payload.get("seed") != seed:
        return None
    if payload.get("run_signature") != run_signature:
        return None
    if int(payload.get("successes", -1)) < 0:
        return None
    return payload


def _run_signature(cfg: DictConfig, suite_names: list[str]) -> str:
    payload = {
        "ckpt": str(Path(str(cfg.ckpt)).resolve()),
        "seed": None if cfg.get("seed") is None else int(cfg.seed),
        "num_trials": int(cfg.EVALUATION.num_trials),
        "num_inference_steps": int(cfg.EVALUATION.num_inference_steps),
        "replan_steps": int(cfg.EVALUATION.replan_steps),
        "num_steps_wait": int(cfg.EVALUATION.num_steps_wait),
        "suite_names": suite_names,
        "model": OmegaConf.to_container(cfg.model, resolve=True),
        "actionaware": _acceleration_config(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _reserve_remaining_vram() -> None:
    if os.environ.get("FASTWAM_ZSPEED_RESERVE", "0").strip().lower() not in {"1", "true", "yes", "on"}:
        return
    physical_gpu = int(os.environ["FASTWAM_PHYSICAL_GPU"])
    helper_dir = Path("/home/zzy/skills/lock-and-reserve-gpus/scripts")
    if str(helper_dir) not in sys.path:
        sys.path.insert(0, str(helper_dir))
    from zspeed_reserve import reserve_remaining_vram

    payload = reserve_remaining_vram(
        physical_gpu=physical_gpu,
        leave_gib=float(os.environ.get("FASTWAM_ZSPEED_LEAVE_GIB", "12")),
        minimum_gib=1,
        process_name="zspeed",
        acquire_lock=False,
    )
    logging.info("In-process zspeed reservation: %s", json.dumps(payload, sort_keys=True))


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_libero.yaml")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        stream=sys.stdout,
        force=True,
    )
    start_time = time.time()
    if cfg.ckpt is None:
        raise ValueError("ckpt must not be None")
    if cfg.get("seed") is not None:
        set_global_seed(int(cfg.seed), get_worker_init_fn=False)

    output_dir = Path(os.path.expandvars(os.path.expanduser(str(cfg.EVALUATION.output_dir)))).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_dir / "resolved_config.yaml")

    model_device = _resolve_eval_device(cfg)
    model_dtype = _mixed_precision_to_model_dtype(cfg.get("mixed_precision", "bf16"))
    logging.info("Loading FastWAM model on %s with dtype=%s", model_device, model_dtype)
    model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)
    _load_model_checkpoint(model, str(cfg.ckpt))
    model = model.to(model_device).eval()
    _reserve_remaining_vram()

    dataset_stats_path = _resolve_dataset_stats_path(cfg)
    dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
    processor: FastWAMProcessor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(dataset_stats)
    logging.info("Using dataset stats: %s", dataset_stats_path)

    action_horizon_cfg = cfg.EVALUATION.get("action_horizon")
    action_horizon = (
        int(cfg.data.train.num_frames) - 1
        if action_horizon_cfg is None
        else int(action_horizon_cfg)
    )
    input_h, input_w = (int(value) for value in cfg.data.train.video_size)
    num_trials = int(cfg.EVALUATION.num_trials)
    max_tasks = int(cfg.EVALUATION.get("max_tasks", 0))
    resume = bool(cfg.EVALUATION.get("resume", True))
    seed = None if cfg.get("seed") is None else int(cfg.seed)
    suite_names = [str(name) for name in cfg.MULTIRUN.task_suite_names]
    run_signature = _run_signature(cfg, suite_names)
    logging.info("Run signature: %s", run_signature)
    task_id_filter = cfg.EVALUATION.get("task_ids")
    allowed_task_ids = None if task_id_filter is None else {int(item) for item in task_id_filter}

    benchmark_dict = benchmark.get_benchmark_dict()
    reset_query_timing()
    completed: list[dict] = []
    scheduled_tasks = 0
    for suite_name in suite_names:
        task_suite = benchmark_dict[suite_name]()
        for task_id in range(int(task_suite.n_tasks)):
            if allowed_task_ids is not None and task_id not in allowed_task_ids:
                continue
            if max_tasks > 0 and scheduled_tasks >= max_tasks:
                break
            scheduled_tasks += 1
            result_path = _task_result_path(output_dir, suite_name, task_id)
            previous = (
                _completed_result(
                    result_path,
                    num_trials=num_trials,
                    seed=seed,
                    run_signature=run_signature,
                )
                if resume
                else None
            )
            if previous is not None:
                logging.info(
                    "Resume: skip completed suite=%s task=%d success=%d/%d",
                    suite_name,
                    task_id,
                    previous["successes"],
                    previous["total_episodes"],
                )
                completed.append(previous)
                continue

            with open_dict(cfg):
                cfg.EVALUATION.task_suite_name = suite_name
                cfg.EVALUATION.task_id = task_id
            task = task_suite.get_task(task_id)
            initial_states = list(task_suite.get_task_init_states(task_id))
            while len(initial_states) < num_trials:
                initial_states.extend(initial_states[: num_trials - len(initial_states)])
            before = get_query_timing()
            task_start = time.time()
            logging.info("Task start: suite=%s task=%d trials=%d", suite_name, task_id, num_trials)
            task_result = run_single_task(
                task=task,
                initial_states=initial_states,
                model=model,
                processor=processor,
                cfg=cfg,
                video_dir=output_dir / "videos" / suite_name,
                predicted_video_dir=output_dir / "predicted_videos" / suite_name,
                action_horizon=action_horizon,
                input_w=input_w,
                input_h=input_h,
                model_device=model_device,
            )
            after = get_query_timing()
            query_count = int(after["count"]) - int(before["count"])
            query_total = float(after["total_seconds"]) - float(before["total_seconds"])
            payload = {
                "suite": suite_name,
                "task_id": task_id,
                "seed": seed,
                "run_signature": run_signature,
                "successes": int(task_result["successes"]),
                "total_episodes": num_trials,
                "success_rate": float(task_result["successes"]) / num_trials,
                "success_episodes": task_result["success_episodes"],
                "failure_episodes": task_result["failure_episodes"],
                "task_description": task_result["task_description"],
                "query_count": query_count,
                "query_time_total_seconds": query_total,
                "query_time_mean_seconds": query_total / query_count if query_count else None,
                "duration_seconds": time.time() - task_start,
            }
            _atomic_json(result_path, payload)
            completed.append(payload)
            logging.info(
                "Task final: suite=%s task=%d success=%d/%d query_mean=%.6f sec",
                suite_name,
                task_id,
                payload["successes"],
                num_trials,
                payload["query_time_mean_seconds"],
            )
        if max_tasks > 0 and scheduled_tasks >= max_tasks:
            break

    suite_summary: dict[str, dict] = {}
    for suite_name in suite_names:
        rows = [item for item in completed if item["suite"] == suite_name]
        if not rows:
            continue
        successes = sum(int(item["successes"]) for item in rows)
        episodes = sum(int(item["total_episodes"]) for item in rows)
        suite_summary[suite_name] = {
            "successes": successes,
            "episodes": episodes,
            "success_rate": successes / episodes,
        }
        logging.info(
            "FASTWAM suite final: %s %d/%d = %.4f",
            suite_name,
            successes,
            episodes,
            successes / episodes,
        )

    total_successes = sum(int(item["successes"]) for item in completed)
    total_episodes = sum(int(item["total_episodes"]) for item in completed)
    total_queries = sum(int(item["query_count"]) for item in completed)
    total_query_seconds = sum(float(item["query_time_total_seconds"]) for item in completed)
    actionaware = get_actionaware_router()
    summary = {
        "run_signature": run_signature,
        "suites": suite_summary,
        "successes": total_successes,
        "episodes": total_episodes,
        "success_rate": total_successes / total_episodes if total_episodes else None,
        "query_count": total_queries,
        "query_time_total_seconds": total_query_seconds,
        "query_time_mean_seconds": total_query_seconds / total_queries if total_queries else None,
        "duration_seconds": time.time() - start_time,
        "acceleration_config": _acceleration_config(),
        "actionaware": dict(actionaware.stats) if actionaware.enabled else None,
        "action_block_cache": get_action_block_cache_stats(),
    }
    _atomic_json(output_dir / "summary.json", summary)
    logging.info(
        "FASTWAM ALL final: %d/%d = %.4f; query_count=%d mean_query_time=%.6f sec",
        total_successes,
        total_episodes,
        summary["success_rate"] if summary["success_rate"] is not None else float("nan"),
        total_queries,
        summary["query_time_mean_seconds"] if summary["query_time_mean_seconds"] is not None else float("nan"),
    )


if __name__ == "__main__":
    main()
