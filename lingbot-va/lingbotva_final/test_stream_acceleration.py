"""CPU-only regression tests for LingBot-VA stream residual routing."""

from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "wan_va/modules/stream_acceleration.py"
SPEC = importlib.util.spec_from_file_location("lingbot_stream_acceleration_test_module", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
StreamAccelerator = MODULE.StreamAccelerator


def make_accelerator(*, video_mode="route", action_mode="route", video_threshold=0.15, action_threshold=0.25):
    environment = {
        "LINGBOT_VIDEO_CACHE_MODE": video_mode,
        "LINGBOT_ACTION_CACHE_MODE": action_mode,
        "LINGBOT_VIDEO_RELATIVE_L1_THRESHOLD": str(video_threshold),
        "LINGBOT_ACTION_RELATIVE_L1_THRESHOLD": str(action_threshold),
        "LINGBOT_ACCEL_TRACE_VALUES": "1",
        # These deliberately small legacy-looking values must have no effect:
        # the selected policy has no consecutive-reuse/max-skip limiter.
        "LINGBOT_VIDEO_MAX_CONSECUTIVE_REUSE": "1",
        "LINGBOT_ACTION_MAX_CONSECUTIVE_REUSE": "1",
    }
    with mock.patch.dict(os.environ, environment, clear=False):
        return StreamAccelerator()


class StreamAcceleratorTest(unittest.TestCase):
    def test_route_has_no_maxskip_limit(self):
        accelerator = make_accelerator(video_threshold=10.0)
        first = torch.ones(2, 3)
        self.assertIsNone(accelerator.try_reuse("video", first, update_cache=0))
        accelerator.store_full("video", first, first + 2.0, update_cache=0)

        for index in range(1, 26):
            current = first + index * 1e-4
            reused = accelerator.try_reuse("video", current, update_cache=0)
            self.assertIsNotNone(reused)
            torch.testing.assert_close(reused, current + 2.0)

        summary = accelerator.summary("video")
        self.assertEqual(summary["full_steps"], 1)
        self.assertEqual(summary["reuse_steps"], 25)
        self.assertEqual(summary["refreshes"], 0)

    def test_accumulated_relative_l1_triggers_refresh(self):
        accelerator = make_accelerator(video_threshold=0.15)
        first = torch.ones(2, 3)
        self.assertIsNone(accelerator.try_reuse("video", first, update_cache=0))
        accelerator.store_full("video", first, first + 2.0, update_cache=0)

        second = torch.full_like(first, 1.10)
        torch.testing.assert_close(
            accelerator.try_reuse("video", second, update_cache=0), second + 2.0
        )

        third = torch.full_like(first, 1.21)
        self.assertIsNone(accelerator.try_reuse("video", third, update_cache=0))
        accelerator.store_full("video", third, third + 3.0, update_cache=0)

        fourth = torch.full_like(first, 1.27)
        torch.testing.assert_close(
            accelerator.try_reuse("video", fourth, update_cache=0), fourth + 3.0
        )
        summary = accelerator.summary("video")
        self.assertEqual(summary["full_steps"], 2)
        self.assertEqual(summary["reuse_steps"], 2)
        self.assertEqual(summary["refreshes"], 1)
        self.assertEqual(len(summary["relative_l1_values"]), 3)

    def test_persistent_kv_update_bypasses_without_mutating_stream_state(self):
        accelerator = make_accelerator(video_threshold=0.10)
        first = torch.ones(2, 3)
        self.assertIsNone(accelerator.try_reuse("video", first, update_cache=0))
        accelerator.store_full("video", first, first + 2.0, update_cache=0)

        self.assertIsNone(
            accelerator.try_reuse("video", torch.full_like(first, 10.0), update_cache=1)
        )
        accelerator.store_full(
            "video", torch.full_like(first, 10.0), torch.full_like(first, 20.0), update_cache=1
        )

        current = torch.full_like(first, 1.05)
        torch.testing.assert_close(
            accelerator.try_reuse("video", current, update_cache=0), current + 2.0
        )
        summary = accelerator.summary("video")
        self.assertEqual(summary["full_steps"], 1)
        self.assertEqual(summary["reuse_steps"], 1)
        self.assertEqual(len(summary["relative_l1_values"]), 1)

    def test_video_and_action_residuals_are_independent(self):
        accelerator = make_accelerator(video_threshold=1.0, action_threshold=1.0)
        first = torch.ones(2, 3)
        for stream, residual in (("video", 2.0), ("action", 5.0)):
            self.assertIsNone(accelerator.try_reuse(stream, first, update_cache=0))
            accelerator.store_full(stream, first, first + residual, update_cache=0)

        current = first + 0.01
        torch.testing.assert_close(
            accelerator.try_reuse("video", current, update_cache=0), current + 2.0
        )
        torch.testing.assert_close(
            accelerator.try_reuse("action", current, update_cache=0), current + 5.0
        )


if __name__ == "__main__":
    unittest.main()
