"""Regression tests for resumable canonical LingBot-VA logs."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from lingbotva_final.log_utils import parse_episode_records, parse_log


def line(prefix: str, payload: dict[str, object]) -> str:
    return prefix + json.dumps(payload, sort_keys=True)


class LogUtilsTest(unittest.TestCase):
    def test_last_completed_retry_wins_and_incomplete_attempt_is_ignored(self):
        query_key = {
            "suite": "libero_10",
            "task": 0,
            "episode": 0,
            "episode_query": 1,
        }
        old_query = {**query_key, "query_time_s": 99.0}
        new_query = {**query_key, "query_time_s": 1.25}
        old_episode = {
            "suite": "libero_10",
            "task": 0,
            "episode": 0,
            "episode_seed": 195,
            "query_count": 1,
            "success": False,
        }
        new_episode = {**old_episode, "success": True}
        incomplete_query = {
            "suite": "libero_10",
            "task": 0,
            "episode": 1,
            "episode_query": 1,
            "query_time_s": 88.0,
        }
        final = {"total": 1, "success": 1, "query_count": 1, "mean_query_time_s": 1.25}
        contents = "\n".join(
            (
                line("LINGBOT_QUERY_CLIENT ", old_query),
                line("LINGBOT_EPISODE ", old_episode),
                line("LINGBOT_QUERY_CLIENT ", new_query),
                line("LINGBOT_EPISODE ", new_episode),
                line("LINGBOT_QUERY_CLIENT ", incomplete_query),
                line("LINGBOT ALL final: ", final),
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "retry.log"
            path.write_text(contents + "\n", encoding="utf-8")
            queries, parsed_final = parse_log(path)
            episodes = parse_episode_records(path)

        self.assertEqual(queries, [new_query])
        self.assertEqual(parsed_final, final)
        self.assertEqual(set(episodes), {("libero_10", 0, 0)})
        self.assertTrue(episodes[("libero_10", 0, 0)]["success"])


if __name__ == "__main__":
    unittest.main()
