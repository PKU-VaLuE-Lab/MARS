"""Low-overhead, per-stream whole-transformer residual reuse.

The public LingBot-VA checkpoint shares one Transformer backbone between the
video and action streams.  Runtime state is nevertheless kept strictly
separate for the two streams.  The cache is local to one generated chunk and
is never mixed with LingBot-VA's persistent causal attention K/V cache.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import torch


def _mode(name: str) -> str:
    value = os.environ.get(name, "off").strip().lower()
    if value not in {"off", "probe", "route"}:
        raise ValueError(f"{name} must be one of off, probe, route; got {value!r}")
    return value


@dataclass(frozen=True)
class StreamConfig:
    mode: str
    threshold: float


@dataclass
class StreamState:
    previous_input: torch.Tensor | None = None
    residual: torch.Tensor | None = None
    accumulated_relative_l1: float = 0.0
    full_steps: int = 0
    reuse_steps: int = 0
    refreshes: int = 0
    relative_l1_values: list[float] = field(default_factory=list)
    accumulated_values: list[float] = field(default_factory=list)


class StreamAccelerator:
    """Cache one whole-stack residual independently for video and action."""

    def __init__(self) -> None:
        self.configs = {
            "video": StreamConfig(
                mode=_mode("LINGBOT_VIDEO_CACHE_MODE"),
                threshold=float(os.environ.get("LINGBOT_VIDEO_RELATIVE_L1_THRESHOLD", "0.03")),
            ),
            "action": StreamConfig(
                mode=_mode("LINGBOT_ACTION_CACHE_MODE"),
                threshold=float(os.environ.get("LINGBOT_ACTION_RELATIVE_L1_THRESHOLD", "0.10")),
            ),
        }
        for stream, config in self.configs.items():
            if config.threshold < 0:
                raise ValueError(f"{stream} relative-L1 threshold must be non-negative")
        self.trace_values = os.environ.get("LINGBOT_ACCEL_TRACE_VALUES", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.states = {"video": StreamState(), "action": StreamState()}
        active = [
            f"{stream}:{cfg.mode}@{cfg.threshold:g}"
            for stream, cfg in self.configs.items()
            if cfg.mode != "off"
        ]
        if active:
            print("[LingBot StreamCache] " + " ".join(active) + " maxskip=disabled", flush=True)

    def reset(self, stream: str | None = None) -> None:
        if stream is None:
            self.states = {"video": StreamState(), "action": StreamState()}
            return
        self._validate_stream(stream)
        self.states[stream] = StreamState()

    def _validate_stream(self, stream: str) -> None:
        if stream not in self.states:
            raise ValueError(f"unknown LingBot stream: {stream!r}")

    def try_reuse(
        self,
        stream: str,
        block_input: torch.Tensor,
        *,
        update_cache: int,
    ) -> torch.Tensor | None:
        """Return a cached whole-stack result, or ``None`` to run all blocks.

        Persistent K/V-cache writes (``update_cache != 0``) must always execute
        every Transformer block, so they bypass this cache without mutating its
        state.
        """
        self._validate_stream(stream)
        config = self.configs[stream]
        state = self.states[stream]
        if config.mode == "off" or update_cache != 0:
            return None

        if state.previous_input is None or state.residual is None:
            state.previous_input = block_input.detach()
            return None
        if (
            state.residual.shape != block_input.shape
            or state.residual.dtype != block_input.dtype
            or state.residual.device != block_input.device
        ):
            state.previous_input = block_input.detach()
            state.residual = None
            state.accumulated_relative_l1 = 0.0
            return None

        current = block_input.detach()
        previous = state.previous_input
        # Accumulate reductions in fp32, but keep the large elementwise
        # tensors in their native inference dtype.  Materializing two fp32
        # copies here adds bandwidth and allocator overhead without improving
        # the routing decision.
        relative_l1 = float(
            (
                (current - previous).abs().mean(dtype=torch.float32)
                / previous.abs().mean(dtype=torch.float32).clamp_min(1e-6)
            ).item()
        )
        state.previous_input = block_input.detach()
        state.accumulated_relative_l1 += relative_l1
        state.relative_l1_values.append(relative_l1)
        state.accumulated_values.append(state.accumulated_relative_l1)

        eligible = state.accumulated_relative_l1 <= config.threshold
        if config.mode == "route" and eligible:
            state.reuse_steps += 1
            return block_input + state.residual

        # Probe mode executes all blocks but follows the same hypothetical
        # refresh boundary, making its traces useful for threshold selection.
        if not eligible:
            state.accumulated_relative_l1 = 0.0
            state.refreshes += 1
        return None

    def store_full(
        self,
        stream: str,
        block_input: torch.Tensor,
        block_output: torch.Tensor,
        *,
        update_cache: int,
    ) -> None:
        self._validate_stream(stream)
        config = self.configs[stream]
        if config.mode == "off" or update_cache != 0:
            return
        state = self.states[stream]
        state.full_steps += 1
        if config.mode == "route" or state.residual is None or state.accumulated_relative_l1 == 0.0:
            state.residual = (block_output - block_input).detach()
        if state.previous_input is None:
            state.previous_input = block_input.detach()

    def summary(self, stream: str) -> dict[str, object]:
        self._validate_stream(stream)
        config = self.configs[stream]
        state = self.states[stream]
        compared = len(state.relative_l1_values)
        steps = state.full_steps + state.reuse_steps
        result: dict[str, object] = {
            "mode": config.mode,
            "threshold": config.threshold,
            "full_steps": state.full_steps,
            "reuse_steps": state.reuse_steps,
            "reuse_ratio": state.reuse_steps / steps if steps else 0.0,
            "refreshes": state.refreshes,
            "relative_l1_mean": (
                sum(state.relative_l1_values) / compared if compared else 0.0
            ),
            "relative_l1_max": max(state.relative_l1_values, default=0.0),
            "accumulated_max": max(state.accumulated_values, default=0.0),
        }
        if self.trace_values:
            result["relative_l1_values"] = list(state.relative_l1_values)
            result["accumulated_values"] = list(state.accumulated_values)
        return result


_ACCELERATOR: StreamAccelerator | None = None


def get_stream_accelerator() -> StreamAccelerator:
    global _ACCELERATOR
    if _ACCELERATOR is None:
        _ACCELERATOR = StreamAccelerator()
    return _ACCELERATOR


def reset_stream_acceleration(stream: str | None = None) -> None:
    get_stream_accelerator().reset(stream)


def stream_acceleration_summary(stream: str) -> dict[str, object]:
    return get_stream_accelerator().summary(stream)
