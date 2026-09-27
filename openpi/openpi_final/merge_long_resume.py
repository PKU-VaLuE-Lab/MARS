#!/usr/bin/env python3
"""Combine an interrupted LIBERO suite prefix with its exact resume suffix."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


QUERY_RE = re.compile(
    r"Query (\d+): Action query time = ([0-9.]+) sec; "
    r"policy=([0-9.]+) sec; roundtrip=([0-9.]+) sec"
)
AA_RE = re.compile(
    r"ActionAware query \d+: score=([0-9.eE+-]+) "
    r"route_steps=(\d+) full=(\d+) reused=(\d+)"
)


def episode_body(lines: list[str], *, stop_at_summary: bool) -> list[str]:
    started = False
    body: list[str] = []
    for line in lines:
        if "OPENPI LIBERO suite start:" in line:
            started = True
            continue
        if not started:
            continue
        if stop_at_summary and line.startswith("INFO:root:Total success rate:"):
            break
        body.append(line)
    return body


def completed_prefix_body(lines: list[str]) -> list[str]:
    """Drop any trailing, started-but-unfinished episode from a killed log."""
    body = episode_body(lines, stop_at_summary=False)
    final_progress = max(
        (index for index, line in enumerate(body) if line.startswith("INFO:root:# successes:")),
        default=-1,
    )
    if final_progress < 0:
        raise RuntimeError("prefix has no completed-episode progress marker")
    return body[: final_progress + 1]


def statistics(lines: list[str]) -> dict[str, float | int]:
    successes = sum(line.startswith("INFO:root:Success: True") for line in lines)
    episodes = sum(line.startswith("INFO:root:Success:") for line in lines)
    timing_count = 0
    query_sum = 0.0
    policy_sum = 0.0
    roundtrip_sum = 0.0
    aa_queries = 0
    route_steps = 0
    full_forwards = 0
    reused_forwards = 0
    score_sum = 0.0
    for line in lines:
        if match := QUERY_RE.search(line):
            # Each physical policy server has one independent cold-start query.
            if int(match.group(1)) > 1:
                timing_count += 1
                query_sum += float(match.group(2))
                policy_sum += float(match.group(3))
                roundtrip_sum += float(match.group(4))
        if match := AA_RE.search(line):
            aa_queries += 1
            score_sum += float(match.group(1))
            route_steps += int(match.group(2))
            full_forwards += int(match.group(3))
            reused_forwards += int(match.group(4))
    return {
        "successes": successes,
        "episodes": episodes,
        "timing_count": timing_count,
        "query_sum": query_sum,
        "policy_sum": policy_sum,
        "roundtrip_sum": roundtrip_sum,
        "aa_queries": aa_queries,
        "route_steps": route_steps,
        "full_forwards": full_forwards,
        "reused_forwards": reused_forwards,
        "score_sum": score_sum,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--suffix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-prefix-episodes", type=int, required=True)
    parser.add_argument("--expected-suffix-episodes", type=int, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    prefix_lines = args.prefix.read_text(encoding="utf-8", errors="replace").splitlines()
    suffix_lines = args.suffix.read_text(encoding="utf-8", errors="replace").splitlines()
    prefix_stats = statistics(prefix_lines)
    suffix_stats = statistics(suffix_lines)
    if prefix_stats["episodes"] != args.expected_prefix_episodes:
        raise RuntimeError(
            f"prefix has {prefix_stats['episodes']} episodes, expected {args.expected_prefix_episodes}"
        )
    if suffix_stats["episodes"] != args.expected_suffix_episodes:
        raise RuntimeError(
            f"suffix has {suffix_stats['episodes']} episodes, expected {args.expected_suffix_episodes}"
        )

    totals = {key: prefix_stats[key] + suffix_stats[key] for key in prefix_stats}
    episodes = int(totals["episodes"])
    successes = int(totals["successes"])
    timing_count = int(totals["timing_count"])
    aa_queries = int(totals["aa_queries"])
    all_forwards = int(totals["full_forwards"] + totals["reused_forwards"])
    if episodes != args.expected_prefix_episodes + args.expected_suffix_episodes:
        raise RuntimeError(f"combined episode count is {episodes}")
    if timing_count <= 0 or aa_queries <= 0 or all_forwards <= 0:
        raise RuntimeError("combined logs are missing timing or ActionAware diagnostics")

    output = [
        "[OpenPI final] variant=actionaware_cache_rel040 gpu=4 seed=195 "
        "suites=libero_10 trials_per_task=50 max_tasks=0 resumed=331+169",
        "[OpenPI sampler] num_steps=10 short_steps=1 action_threshold=0.7 "
        "cache_rel_l1_threshold=0.4 max_cached_steps=4",
        "INFO:root:OPENPI LIBERO suite start: libero_10",
        *completed_prefix_body(prefix_lines),
        *episode_body(suffix_lines, stop_at_summary=True),
        f"INFO:root:Total success rate: {successes / episodes}",
        f"INFO:root:Total episodes: {episodes}",
        f"INFO:root:OPENPI LIBERO suite libero_10 final: {successes}/{episodes} = {successes / episodes:.4f}",
        "INFO:root:OPENPI query timing final: "
        f"count={timing_count} mean_querytime={totals['query_sum'] / timing_count:.6f} sec "
        f"mean_policy={totals['policy_sum'] / timing_count:.6f} sec "
        f"mean_roundtrip={totals['roundtrip_sum'] / timing_count:.6f} sec",
        "INFO:root:OPENPI ActionAware final: "
        f"queries={aa_queries} mean_route_steps={totals['route_steps'] / aa_queries:.4f} "
        f"full_forwards={int(totals['full_forwards'])} "
        f"reused_forwards={int(totals['reused_forwards'])} "
        f"reuse_ratio={totals['reused_forwards'] / all_forwards:.4f} "
        f"mean_score={totals['score_sum'] / aa_queries:.6f}",
        f"INFO:root:OPENPI LIBERO suite rates: {{'libero_10': {successes / episodes}}}",
        f"INFO:root:OPENPI LIBERO ALL final: {successes}/{episodes} = {successes / episodes:.4f}",
    ]
    args.output.write_text("\n".join(output) + "\n", encoding="utf-8")
    print(f"combined {successes}/{episodes} into {args.output}")


if __name__ == "__main__":
    main()
