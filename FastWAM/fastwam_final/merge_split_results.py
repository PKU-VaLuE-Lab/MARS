#!/usr/bin/env python3
"""Validate and merge two FastWAM LIBERO evaluation shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


EXPECTED_SUITES = ("libero_goal", "libero_object", "libero_spatial", "libero_10")
SUM_CACHE_FIELDS = (
    "queries",
    "denoise_steps",
    "full_steps",
    "reuse_steps",
    "full_blocks",
    "reused_blocks",
    "relative_l1_sum",
    "relative_l1_count",
)
MAX_CACHE_FIELDS = ("relative_l1_max", "relative_max_max")


def load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("shards", type=Path, nargs="+")
    args = parser.parse_args()

    output = args.output.resolve()
    shards = [path.resolve() for path in args.shards]
    if len(shards) < 2:
        raise ValueError("at least two shards are required")

    summaries = [load(path / "summary.json") for path in shards]
    configs = [summary["acceleration_config"] for summary in summaries]
    if any(config != configs[0] for config in configs[1:]):
        raise ValueError("shard acceleration configurations differ")

    suites: dict[str, dict] = {}
    for summary in summaries:
        for name, payload in summary["suites"].items():
            if name in suites:
                raise ValueError(f"duplicate suite across shards: {name}")
            suites[name] = payload
    if set(suites) != set(EXPECTED_SUITES):
        raise ValueError(f"expected suites {EXPECTED_SUITES}, got {sorted(suites)}")

    task_payloads: dict[tuple[str, int], tuple[Path, dict]] = {}
    for shard in shards:
        for path in sorted((shard / "task_results").glob("*.json")):
            payload = load(path)
            key = (payload["suite"], int(payload["task_id"]))
            if key in task_payloads:
                raise ValueError(f"duplicate task result: {key}")
            task_payloads[key] = (path, payload)

    expected_tasks = {(suite, task_id) for suite in EXPECTED_SUITES for task_id in range(10)}
    if set(task_payloads) != expected_tasks:
        missing = sorted(expected_tasks - set(task_payloads))
        extra = sorted(set(task_payloads) - expected_tasks)
        raise ValueError(f"task coverage mismatch; missing={missing}, extra={extra}")
    if any(payload["total_episodes"] != 50 for _, payload in task_payloads.values()):
        raise ValueError("every task must contain exactly 50 episodes")

    episodes = sum(payload["total_episodes"] for _, payload in task_payloads.values())
    successes = sum(payload["successes"] for _, payload in task_payloads.values())
    query_count = sum(payload["query_count"] for _, payload in task_payloads.values())
    query_time_total = sum(
        payload["query_time_total_seconds"] for _, payload in task_payloads.values()
    )
    if episodes != 2000:
        raise ValueError(f"expected 2000 episodes, got {episodes}")
    if episodes != sum(summary["episodes"] for summary in summaries):
        raise ValueError("task and shard episode totals differ")
    if successes != sum(summary["successes"] for summary in summaries):
        raise ValueError("task and shard success totals differ")
    if query_count != sum(summary["query_count"] for summary in summaries):
        raise ValueError("task and shard query totals differ")

    actionaware = {
        key: sum(summary["actionaware"][key] for summary in summaries)
        for key in summaries[0]["actionaware"]
    }
    cache = {
        key: sum(summary["action_block_cache"][key] for summary in summaries)
        for key in SUM_CACHE_FIELDS
    }
    cache.update(
        {
            key: max(summary["action_block_cache"][key] for summary in summaries)
            for key in MAX_CACHE_FIELDS
        }
    )
    cache["relative_l1_mean"] = cache["relative_l1_sum"] / cache["relative_l1_count"]
    cache["reuse_step_ratio"] = cache["reuse_steps"] / cache["denoise_steps"]

    source_signatures = sorted(summary["run_signature"] for summary in summaries)
    combined_signature = hashlib.sha256(
        json.dumps(
            {"source_signatures": source_signatures, "config": configs[0]},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    merged = {
        "run_signature": combined_signature,
        "source_run_signatures": source_signatures,
        "source_shards": [str(path) for path in shards],
        "suites": {suite: suites[suite] for suite in EXPECTED_SUITES},
        "successes": successes,
        "episodes": episodes,
        "success_rate": successes / episodes,
        "query_count": query_count,
        "query_time_total_seconds": query_time_total,
        "query_time_mean_seconds": query_time_total / query_count,
        "duration_seconds": max(summary["duration_seconds"] for summary in summaries),
        "acceleration_config": configs[0],
        "actionaware": actionaware,
        "action_block_cache": cache,
    }

    output.mkdir(parents=True, exist_ok=True)
    task_output = output / "task_results"
    task_output.mkdir(exist_ok=True)
    for key in sorted(task_payloads):
        source, _ = task_payloads[key]
        destination = task_output / source.name
        if destination.exists() and destination.read_bytes() != source.read_bytes():
            raise ValueError(f"refusing to replace different task result: {destination}")
        shutil.copy2(source, destination)

    temporary = output / "summary.json.tmp"
    temporary.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output / "summary.json")
    print(json.dumps(merged, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
