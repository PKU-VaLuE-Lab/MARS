#!/usr/bin/env python3
"""Continuously merge two protected LIBERO-Pro shards into one formal log."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path


COMPLETION_MARKER = "LIBERO-PRO ALL final:"
SUITE_RESULT = re.compile(
    r"LIBERO-PRO suite (libero_(?:spatial|goal|object|10)_(?:lan|object|swap|task|env)) "
    r"final: (\d+)/(\d+) = ([0-9.]+)"
)
BASES = ("goal", "object", "spatial", "10")


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def has_marker(path: Path) -> bool:
    if not path.is_file():
        return False
    with path.open("rb") as stream:
        stream.seek(0, os.SEEK_END)
        size = stream.tell()
        stream.seek(max(0, size - 4 * 1024 * 1024))
        return COMPLETION_MARKER.encode() in stream.read()


def read_suite_results(path: Path) -> dict[str, tuple[int, int]]:
    results: dict[str, tuple[int, int]] = {}
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            match = SUITE_RESULT.search(line)
            if match:
                results[match.group(1)] = (int(match.group(2)), int(match.group(3)))
    return results


def base_for_suite(suite: str) -> str:
    for base in BASES:
        if suite.startswith(f"libero_{base}_"):
            return base
    raise ValueError(f"unrecognized suite: {suite}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        help="LABEL=/absolute/path/to/shard.log (supply exactly twice)",
    )
    parser.add_argument("--expected-episodes", type=int, default=10000)
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    args = parser.parse_args()

    if len(args.source) != 2:
        parser.error("exactly two --source arguments are required")
    if args.expected_episodes < 1 or args.poll_seconds <= 0:
        parser.error("expected episodes and poll interval must be positive")

    sources: list[tuple[str, Path]] = []
    for raw in args.source:
        if "=" not in raw:
            parser.error(f"invalid --source: {raw!r}")
        label, path_text = raw.split("=", 1)
        path = Path(path_text).resolve()
        if not label.strip() or not path.is_absolute():
            parser.error(f"invalid --source: {raw!r}")
        sources.append((label.strip(), path))

    output = args.output.resolve()
    state_path = args.state.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
    else:
        state = {"offsets": {}, "header_written": False, "complete": False}
    if state.get("complete"):
        return 0

    mode = "ab" if output.exists() else "wb"
    with output.open(mode, buffering=0) as merged:
        if not state.get("header_written"):
            header = {
                "event": "liberopro_baseline_all10000_start",
                "time": timestamp(),
                "episode_accounting": "40 base tasks x 5 perturbations x 50 trials = 10000",
                "sources": {label: str(path) for label, path in sources},
            }
            merged.write((json.dumps(header, ensure_ascii=False, sort_keys=True) + "\n").encode())
            state["header_written"] = True
            atomic_json(state_path, state)

        while True:
            changed = False
            for label, source in sources:
                if not source.is_file():
                    continue
                key = str(source)
                offset = int(state["offsets"].get(key, 0))
                size = source.stat().st_size
                if size < offset:
                    event = (
                        f"\n[{timestamp()}] [{label}] source restarted/truncated; "
                        f"offset {offset} -> 0\n"
                    )
                    merged.write(event.encode())
                    offset = 0
                if size == offset:
                    state["offsets"][key] = offset
                    continue
                with source.open("rb") as stream:
                    stream.seek(offset)
                    data = stream.read(size - offset)
                boundary = (
                    f"\n[{timestamp()}] ===== {label} bytes {offset}:{size} =====\n"
                ).encode()
                merged.write(boundary)
                merged.write(data)
                if data and not data.endswith(b"\n"):
                    merged.write(b"\n")
                state["offsets"][key] = size
                changed = True

            if changed:
                atomic_json(state_path, state)

            if all(has_marker(path) for _, path in sources):
                all_results: dict[str, tuple[int, int]] = {}
                for _, path in sources:
                    for suite, result in read_suite_results(path).items():
                        if suite in all_results:
                            raise RuntimeError(f"duplicate suite result across shards: {suite}")
                        all_results[suite] = result
                expected_suites = 4 * 5
                total_episodes = sum(episodes for _, episodes in all_results.values())
                if len(all_results) != expected_suites or total_episodes != args.expected_episodes:
                    raise RuntimeError(
                        "invalid merged coverage: "
                        f"unique_suites={len(all_results)}/{expected_suites}, "
                        f"episodes={total_episodes}/{args.expected_episodes}"
                    )
                lines = [f"\n[{timestamp()}] ===== LIBERO-PRO MERGED SUMMARY ====="]
                for suite in sorted(all_results):
                    successes, episodes = all_results[suite]
                    lines.append(
                        f"LIBERO-PRO MERGED suite {suite} final: "
                        f"{successes}/{episodes} = {successes / episodes:.4f}"
                    )
                for base in BASES:
                    rows = [
                        value
                        for suite, value in all_results.items()
                        if base_for_suite(suite) == base
                    ]
                    successes = sum(value[0] for value in rows)
                    episodes = sum(value[1] for value in rows)
                    lines.append(
                        f"LIBERO-PRO MERGED base {base} final: "
                        f"{successes}/{episodes} = {successes / episodes:.4f}"
                    )
                successes = sum(value[0] for value in all_results.values())
                lines.append(
                    f"LIBERO-PRO MERGED ALL10000 final: "
                    f"{successes}/{total_episodes} = {successes / total_episodes:.4f}"
                )
                merged.write(("\n".join(lines) + "\n").encode())
                state.update(
                    complete=True,
                    completed_at=timestamp(),
                    successes=successes,
                    episodes=total_episodes,
                )
                atomic_json(state_path, state)
                return 0
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
