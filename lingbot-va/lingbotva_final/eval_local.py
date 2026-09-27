#!/usr/bin/env python3
"""Run one or more LIBERO work items in one protected LingBot-VA process."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch.distributed as dist


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WAN_ROOT = PROJECT_ROOT / "wan_va"
LIBERO_ROOT = PROJECT_ROOT / "third_party" / "LIBERO"
for path in (str(WAN_ROOT), str(PROJECT_ROOT), str(LIBERO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from distributed.util import init_distributed  # noqa: E402
from utils import init_logger  # noqa: E402
from wan_va_server import VA_Server  # noqa: E402
from configs import VA_CONFIGS  # noqa: E402
from evaluation.libero.client import run as run_libero, run_selected  # noqa: E402
from lingbotva_final.log_utils import parse_log  # noqa: E402


SUITES = ("libero_goal", "libero_object", "libero_spatial", "libero_10")


def parse_work_plan(raw: str | None) -> list[dict[str, object]]:
    if not raw:
        return []
    payload = json.loads(raw)
    if not isinstance(payload, list) or not payload:
        raise ValueError("work-plan-json must be a non-empty JSON list")
    result = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"work item {index} must be an object")
        suite = str(item["suite"])
        task = int(item["task"])
        seed = int(item["seed"])
        test_num = int(item.get("test_num", 50))
        out_dir = Path(str(item["out_dir"])).resolve()
        if suite not in SUITES or not 0 <= task < 10 or test_num < 1:
            raise ValueError(f"invalid work item {index}: {item}")
        out_dir.relative_to(PROJECT_ROOT)
        result.append(
            {
                "suite": suite,
                "task": task,
                "seed": seed,
                "test_num": test_num,
                "out_dir": out_dir,
            }
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-num", type=int, default=50)
    parser.add_argument("--suite", choices=SUITES, default="libero_10")
    parser.add_argument("--work-plan-json")
    parser.add_argument("--task-range", type=int, nargs=2, default=(0, 10))
    parser.add_argument("--selected-task", type=int)
    parser.add_argument("--episode-indices", type=int, nargs="+")
    parser.add_argument("--seed", type=int, default=195)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--log-path", type=Path, required=True)
    args = parser.parse_args()
    work_plan = parse_work_plan(args.work_plan_json)
    if args.test_num < 1:
        raise ValueError("test-num must be positive")
    if (args.selected_task is None) != (args.episode_indices is None):
        raise ValueError("selected-task and episode-indices must be provided together")

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29623")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    init_logger()
    init_distributed(world_size=1, local_rank=0, rank=0)

    config = VA_CONFIGS["libero"]
    config.rank = 0
    config.local_rank = 0
    config.world_size = 1
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(
        "LINGBOT_RUN_CONFIG "
        + json.dumps(
            {
                "variant": "mars",
                "suite": args.suite if not work_plan else "work_plan",
                "work_plan": [
                    {**item, "out_dir": str(item["out_dir"])} for item in work_plan
                ],
                "task_range": list(args.task_range),
                "selected_task": args.selected_task,
                "episode_indices": args.episode_indices,
                "test_num": args.test_num,
                "seed": args.seed,
                "video_steps": config.num_inference_steps,
                "action_steps": config.action_num_inference_steps,
                "video_cache_mode": os.environ.get("LINGBOT_VIDEO_CACHE_MODE", "off"),
                "video_relative_l1_threshold": os.environ.get("LINGBOT_VIDEO_RELATIVE_L1_THRESHOLD"),
                "action_cache_mode": os.environ.get("LINGBOT_ACTION_CACHE_MODE", "off"),
                "action_relative_l1_threshold": os.environ.get("LINGBOT_ACTION_RELATIVE_L1_THRESHOLD"),
                "actionaware_mode": os.environ.get("LINGBOT_ACTIONAWARE_MODE", "off"),
                "actionaware_draft_steps": os.environ.get("LINGBOT_ACTIONAWARE_DRAFT_STEPS"),
                "actionaware_feature_steps": os.environ.get("LINGBOT_ACTIONAWARE_FEATURE_STEPS"),
                "actionaware_translation_threshold": os.environ.get("LINGBOT_ACTIONAWARE_TRANS_THRESHOLD"),
                "attention": "torch",
                "maxskip": None,
                "episode_seed_task_stride": 100_000,
                "query_time_definition": "generation_plus_preceding_kv_update",
            },
            sort_keys=True,
        ),
        flush=True,
    )

    try:
        model = VA_Server(config)
        if work_plan:
            item_results = []
            for item_index, item in enumerate(work_plan):
                item["out_dir"].mkdir(parents=True, exist_ok=True)
                print(
                    "LINGBOT_WORK_ITEM start "
                    + json.dumps(
                        {
                            "index": item_index,
                            **item,
                            "out_dir": str(item["out_dir"]),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                item_result = run_libero(
                    libero_benchmark=str(item["suite"]),
                    port=0,
                    out_dir=str(item["out_dir"]),
                    test_num=int(item["test_num"]),
                    task_range=[int(item["task"]), int(item["task"]) + 1],
                    save_rollout_video=False,
                    seed=int(item["seed"]),
                    model=model,
                )
                item_results.append(item_result)
                print(
                    "LINGBOT_WORK_ITEM final "
                    + json.dumps({"index": item_index, **item_result}, sort_keys=True),
                    flush=True,
                )
            total = sum(int(item["total"]) for item in item_results)
            success = sum(int(item["success"]) for item in item_results)
            result = {
                "suite": "work_plan",
                "success": success,
                "total": total,
                "success_rate": success / total if total else 0.0,
                "work_items": len(item_results),
            }
        elif args.episode_indices is not None:
            result = run_selected(
                model=model,
                libero_benchmark=args.suite,
                task_idx=args.selected_task,
                episode_indices=args.episode_indices,
                out_dir=str(args.out_dir),
                seed=args.seed,
            )
        else:
            result = run_libero(
                libero_benchmark=args.suite,
                port=0,
                out_dir=str(args.out_dir),
                test_num=args.test_num,
                task_range=list(args.task_range),
                save_rollout_video=False,
                seed=args.seed,
                model=model,
            )
        logged_query_times = []
        if args.log_path.is_file():
            query_rows, _ = parse_log(args.log_path)
            logged_query_times = [float(row["query_time_s"]) for row in query_rows]
        query_times = logged_query_times or model.query_times
        query_count = len(query_times)
        mean_query_time = sum(query_times) / query_count if query_count else 0.0
        final = {
            **result,
            "variant": "mars",
            "query_count": query_count,
            "mean_query_time_s": mean_query_time,
        }
        print("LINGBOT ALL final: " + json.dumps(final, sort_keys=True), flush=True)
        return 0
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
