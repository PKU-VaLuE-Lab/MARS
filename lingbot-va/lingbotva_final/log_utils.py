"""Canonical parsing helpers for resumable LingBot-VA evaluation logs."""

from __future__ import annotations

import json
from pathlib import Path


QUERY_MARKER = "LINGBOT_QUERY_CLIENT "
EPISODE_MARKER = "LINGBOT_EPISODE "
FINAL_MARKER = "LINGBOT ALL final: "


def parse_episode_records(path: Path) -> dict[tuple[str, int, int], dict[str, object]]:
    """Return the last completed attempt for every task/episode key."""
    completed: dict[tuple[str, int, int], dict[str, object]] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith(EPISODE_MARKER):
            continue
        row = json.loads(line[len(EPISODE_MARKER):])
        fields = ("suite", "task", "episode", "query_count", "success")
        if not all(field in row for field in fields):
            raise RuntimeError(f"incomplete canonical episode row in {path}: {row}")
        key = (str(row["suite"]), int(row["task"]), int(row["episode"]))
        completed[key] = row
    return completed


def parse_log(path: Path) -> tuple[list[dict[str, object]], dict[str, object] | None]:
    """Return completed-episode query rows and the last final record.

    New logs identify every query by task, episode, and within-episode query
    index.  On a service restart, the last retry wins and queries belonging
    only to an interrupted attempt are excluded using the completed episode's
    recorded query count.  Legacy exploration logs fall back to all canonical
    client query rows.
    """
    raw_rows: list[dict[str, object]] = []
    keyed_rows: dict[tuple[str, int, int, int], dict[str, object]] = {}
    completed: dict[tuple[str, int, int], int] = {}
    final = None

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(QUERY_MARKER):
            row = json.loads(line[len(QUERY_MARKER):])
            raw_rows.append(row)
            fields = ("suite", "task", "episode", "episode_query")
            if all(field in row for field in fields):
                key = (
                    str(row["suite"]),
                    int(row["task"]),
                    int(row["episode"]),
                    int(row["episode_query"]),
                )
                keyed_rows[key] = row
        elif line.startswith(EPISODE_MARKER):
            row = json.loads(line[len(EPISODE_MARKER):])
            if "query_count" in row:
                completed[(str(row["suite"]), int(row["task"]), int(row["episode"]))] = int(
                    row["query_count"]
                )
        if FINAL_MARKER in line:
            final = json.loads(line.split(FINAL_MARKER, 1)[1])

    if not completed:
        return raw_rows, final

    selected = []
    for episode_key, query_count in sorted(completed.items()):
        for query_index in range(1, query_count + 1):
            query_key = (*episode_key, query_index)
            if query_key not in keyed_rows:
                raise RuntimeError(f"missing canonical query row {query_key} in {path}")
            selected.append(keyed_rows[query_key])
    return selected, final
