#!/usr/bin/env python3
"""Analyze a no-routing probe log and rank threshold combinations."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path


def values(raw: str) -> tuple[float, ...]:
    return tuple(float(item) for item in raw.split(",") if item.strip())


def full_count(relative_l1: list[float], threshold: float) -> int:
    full = 1
    accumulated = 0.0
    for value in relative_l1:
        accumulated += value
        if accumulated > threshold:
            full += 1
            accumulated = 0.0
    return full


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--video-thresholds", type=values, default=values(".005,.01,.02,.03,.05,.08"))
    parser.add_argument("--action-thresholds", type=values, default=values(".01,.02,.03,.05,.08,.1,.15"))
    parser.add_argument("--translation-thresholds", type=values, default=values(".05,.1,.15,.2,.3,.4"))
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()

    rows = []
    for line in args.log.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("LINGBOT_QUERY_CLIENT "):
            row = json.loads(line.split(" ", 1)[1])
            if row.get("actionaware_mode") == "probe":
                rows.append(row)
    if not rows:
        raise RuntimeError(f"no ActionAware probe queries in {args.log}")

    candidates = []
    for video_t, action_t, translation_t in itertools.product(
        args.video_thresholds, args.action_thresholds, args.translation_thresholds
    ):
        video_full = 0
        action_full = 0
        routed = 0
        routed_errors = []
        total_stack_equivalent = 0
        baseline_stack_equivalent = 0
        for row in rows:
            video_values = row["video_cache"]["relative_l1_values"]
            action_values = row["action_cache"]["relative_l1_values"]
            query_video_full = full_count(video_values, video_t)
            query_action_full = full_count(action_values, action_t)
            use_one_step = float(row["action_translation"]) > translation_t
            video_full += query_video_full
            if use_one_step:
                routed += 1
                query_action_full = 1
                probe = row.get("action_probe") or {}
                routed_errors.append(float(probe.get("pair_trans_l2_mean", 0.0)))
            action_full += query_action_full
            # One mandatory full causal-KV write follows each stream. Reused
            # residual steps are intentionally treated as near-zero only for
            # ranking; measured route latency remains the source of truth.
            kv_full = 2 if float(row.get("kv_update_time_s", 0.0)) > 0.0 else 0
            total_stack_equivalent += query_video_full + 1 + query_action_full + 1 + kv_full
            baseline_stack_equivalent += 72 + kv_full
        candidates.append(
            {
                "video_threshold": video_t,
                "action_threshold": action_t,
                "translation_threshold": translation_t,
                "one_step_ratio": routed / len(rows),
                "routed_pair_trans_l2_mean": (
                    sum(routed_errors) / len(routed_errors) if routed_errors else 0.0
                ),
                "video_full_mean": video_full / len(rows),
                "action_full_mean": action_full / len(rows),
                "stack_speed_upper_bound": baseline_stack_equivalent / total_stack_equivalent,
            }
        )
    candidates.sort(
        key=lambda row: (
            -row["stack_speed_upper_bound"],
            row["routed_pair_trans_l2_mean"],
        )
    )
    print(f"queries={len(rows)}")
    for row in candidates[: args.top]:
        print(json.dumps(row, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
