#!/usr/bin/env python3
"""Summarize completed LingBot logs without double-counting server/client lines."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

try:
    from lingbotva_final.log_utils import parse_log
except ModuleNotFoundError:  # Direct execution from this directory.
    from log_utils import parse_log


def parse(path: Path) -> dict[str, object]:
    query_rows, final = parse_log(path)
    if final is None:
        raise RuntimeError(f"missing completion marker in {path}")
    times = [float(row["query_time_s"]) for row in query_rows]
    if len(times) != int(final["query_count"]):
        raise RuntimeError(
            f"query count mismatch in {path}: rows={len(times)} final={final['query_count']}"
        )
    generation_times = [float(row["action_generation_time_s"]) for row in query_rows]
    kv_times = [float(row["kv_update_time_s"]) for row in query_rows]
    routes = Counter(str(row.get("action_route", "unknown")) for row in query_rows)

    def reuse_ratio(stream: str) -> float:
        full = sum(int(row.get(stream, {}).get("full_steps", 0)) for row in query_rows)
        reused = sum(int(row.get(stream, {}).get("reuse_steps", 0)) for row in query_rows)
        return reused / (full + reused) if full + reused else 0.0

    return {
        **final,
        "path": str(path.resolve()),
        "mean_query_time_s_reparsed": sum(times) / len(times) if times else 0.0,
        "mean_action_generation_time_s": (
            sum(generation_times) / len(generation_times) if generation_times else 0.0
        ),
        "mean_kv_update_time_s": sum(kv_times) / len(kv_times) if kv_times else 0.0,
        "action_routes": dict(sorted(routes.items())),
        "one_step_ratio": routes.get("1step", 0) / len(query_rows) if query_rows else 0.0,
        "video_reuse_ratio": reuse_ratio("video_cache"),
        "action_reuse_ratio": reuse_ratio("action_cache"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", type=Path, nargs="+")
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    baseline = parse(args.baseline) if args.baseline else None
    for path in args.logs:
        row = parse(path)
        if baseline is not None:
            row["speedup"] = float(baseline["mean_query_time_s"]) / float(row["mean_query_time_s"])
        print(json.dumps(row, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
