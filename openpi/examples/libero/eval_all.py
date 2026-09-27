"""Evaluate π0.5 on all four 10-task LIBERO suites in one process log."""

from __future__ import annotations

import dataclasses
import logging

from main import Args as SuiteArgs
from main import EvalResult
from main import eval_libero
import tyro

DEFAULT_SUITES = "libero_spatial,libero_object,libero_goal,libero_10"


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5
    num_steps_wait: int = 10
    num_trials_per_task: int = 50
    max_tasks: int = 0
    task_start_index: int = 0
    first_task_trial_start: int = 0
    suites: str = DEFAULT_SUITES
    video_out_path: str = "data/libero/videos"
    save_videos: bool = False
    seed: int = 195
    timing_warmup_queries: int = 1


def _sum(results: list[EvalResult], field: str) -> int | float:
    return sum(getattr(result, field) for result in results)


def main(args: Args) -> None:
    suites = [suite.strip() for suite in args.suites.split(",") if suite.strip()]
    if not suites:
        raise ValueError("At least one LIBERO suite is required")

    results: list[EvalResult] = []
    for index, suite in enumerate(suites):
        logging.info("OPENPI LIBERO suite start: %s", suite)
        result = eval_libero(
            SuiteArgs(
                host=args.host,
                port=args.port,
                resize_size=args.resize_size,
                replan_steps=args.replan_steps,
                task_suite_name=suite,
                num_steps_wait=args.num_steps_wait,
                num_trials_per_task=args.num_trials_per_task,
                max_tasks=args.max_tasks,
                task_start_index=args.task_start_index,
                first_task_trial_start=args.first_task_trial_start,
                video_out_path=f"{args.video_out_path}/{suite}",
                save_videos=args.save_videos,
                seed=args.seed,
                timing_warmup_queries=args.timing_warmup_queries if index == 0 else 0,
            )
        )
        results.append(result)

    successes = int(_sum(results, "successes"))
    episodes = int(_sum(results, "episodes"))
    timed_queries = int(_sum(results, "timed_queries"))
    query_time_ms_sum = float(_sum(results, "query_time_ms_sum"))
    policy_time_ms_sum = float(_sum(results, "policy_time_ms_sum"))
    roundtrip_time_ms_sum = float(_sum(results, "roundtrip_time_ms_sum"))
    mean_query_time = query_time_ms_sum / timed_queries / 1000
    mean_policy_time = policy_time_ms_sum / timed_queries / 1000
    mean_roundtrip_time = roundtrip_time_ms_sum / timed_queries / 1000

    rates = {result.task_suite_name: result.successes / result.episodes for result in results}
    logging.info("OPENPI LIBERO suite rates: %s", rates)
    logging.info("OPENPI LIBERO ALL final: %d/%d = %.4f", successes, episodes, successes / episodes)
    logging.info(
        "OPENPI query timing final: count=%d mean_querytime=%.6f sec mean_policy=%.6f sec mean_roundtrip=%.6f sec",
        timed_queries,
        mean_query_time,
        mean_policy_time,
        mean_roundtrip_time,
    )

    actionaware_queries = int(_sum(results, "actionaware_queries"))
    if actionaware_queries:
        route_steps_sum = int(_sum(results, "actionaware_route_steps_sum"))
        full_forwards = int(_sum(results, "actionaware_full_forwards"))
        reused_forwards = int(_sum(results, "actionaware_reused_forwards"))
        score_sum = float(_sum(results, "actionaware_score_sum"))
        all_forwards = full_forwards + reused_forwards
        logging.info(
            "OPENPI ActionAware final: queries=%d mean_route_steps=%.4f "
            "full_forwards=%d reused_forwards=%d reuse_ratio=%.4f mean_score=%.6f",
            actionaware_queries,
            route_steps_sum / actionaware_queries,
            full_forwards,
            reused_forwards,
            reused_forwards / all_forwards if all_forwards else 0.0,
            score_sum / actionaware_queries,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
