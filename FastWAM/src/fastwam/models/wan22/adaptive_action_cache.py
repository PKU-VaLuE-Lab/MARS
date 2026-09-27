"""Within-query ActionDiT residual reuse policies."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import torch


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class ActionBlockCacheConfig:
    enabled: bool
    mode: str
    relative_l1_threshold: float
    relative_max_threshold: float
    max_consecutive_reuse: int
    start_block: int


_CONFIG: ActionBlockCacheConfig | None = None
_GLOBAL_STATS = {
    "queries": 0,
    "denoise_steps": 0,
    "full_steps": 0,
    "reuse_steps": 0,
    "full_blocks": 0,
    "reused_blocks": 0,
    "relative_l1_sum": 0.0,
    "relative_l1_count": 0,
    "relative_l1_max": 0.0,
    "relative_max_max": 0.0,
}


def get_action_block_cache_config() -> ActionBlockCacheConfig:
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = ActionBlockCacheConfig(
            enabled=_env_bool("FASTWAM_ACTION_CACHE_ENABLED"),
            mode=os.environ.get("FASTWAM_ACTION_CACHE_MODE", "adaptive_tail").strip().lower(),
            relative_l1_threshold=float(
                os.environ.get("FASTWAM_ACTION_CACHE_RELATIVE_L1_THRESHOLD", "0.05")
            ),
            relative_max_threshold=float(
                os.environ.get("FASTWAM_ACTION_CACHE_RELATIVE_MAX_THRESHOLD", "inf")
            ),
            max_consecutive_reuse=int(
                os.environ.get("FASTWAM_ACTION_CACHE_MAX_CONSECUTIVE_REUSE", "2")
            ),
            start_block=int(os.environ.get("FASTWAM_ACTION_CACHE_START_BLOCK", "1")),
        )
        if _CONFIG.enabled:
            if _CONFIG.mode not in {
                "adaptive_tail",
                "first_residual",
                "interval_residual",
                "input_accum_residual",
            }:
                raise ValueError(
                    "FASTWAM_ACTION_CACHE_MODE must be one of adaptive_tail, "
                    "first_residual, interval_residual, input_accum_residual"
                )
            if _CONFIG.relative_l1_threshold < 0 or _CONFIG.relative_max_threshold < 0:
                raise ValueError("Action block-cache thresholds must be non-negative")
            if _CONFIG.max_consecutive_reuse < 1:
                raise ValueError("FASTWAM_ACTION_CACHE_MAX_CONSECUTIVE_REUSE must be positive")
            if _CONFIG.start_block < 1:
                raise ValueError("FASTWAM_ACTION_CACHE_START_BLOCK must preserve at least one anchor block")
            if _CONFIG.mode == "adaptive_tail":
                detail = (
                    f"relative_l1_threshold={_CONFIG.relative_l1_threshold} "
                    f"relative_max_threshold={_CONFIG.relative_max_threshold} "
                    f"max_consecutive_reuse={_CONFIG.max_consecutive_reuse} "
                    f"start_block={_CONFIG.start_block}"
                )
            elif _CONFIG.mode == "interval_residual":
                detail = f"probe=disabled max_consecutive_reuse={_CONFIG.max_consecutive_reuse}"
            elif _CONFIG.mode == "input_accum_residual":
                detail = (
                    "probe=disabled input_relative_l1=true "
                    f"accum_threshold={_CONFIG.relative_l1_threshold}"
                )
            else:
                detail = "probe=disabled refresh=never"
            print(f"[FastWAM ActionBlockCache] enabled: mode={_CONFIG.mode} {detail}")
    return _CONFIG


def get_action_block_cache_stats() -> dict[str, float | int] | None:
    config = get_action_block_cache_config()
    if not config.enabled:
        return None
    result = dict(_GLOBAL_STATS)
    count = int(result["relative_l1_count"])
    result["relative_l1_mean"] = (
        float(result["relative_l1_sum"]) / count if count else 0.0
    )
    steps = int(result["denoise_steps"])
    result["reuse_step_ratio"] = float(result["reuse_steps"]) / steps if steps else 0.0
    return result


class ActionBlockCacheState:
    """Per-query cache supporting block-tail and whole-stack residual reuse."""

    def __init__(self, config: ActionBlockCacheConfig, num_layers: int) -> None:
        if config.start_block >= num_layers:
            raise ValueError(
                f"FASTWAM_ACTION_CACHE_START_BLOCK={config.start_block} must be below {num_layers}"
            )
        self.config = config
        self.num_layers = int(num_layers)
        self.reference: torch.Tensor | None = None
        self.residuals: dict[int, torch.Tensor] = {}
        self.first_residual: torch.Tensor | None = None
        self.previous_input: torch.Tensor | None = None
        self.accumulated_relative_l1 = 0.0
        self.consecutive_reuse = 0
        self.query_stats = {
            "denoise_steps": 0,
            "full_steps": 0,
            "reuse_steps": 0,
            "full_blocks": 0,
            "reused_blocks": 0,
            "relative_l1_sum": 0.0,
            "relative_l1_count": 0,
            "relative_l1_max": 0.0,
            "relative_max_max": 0.0,
        }

    def _whole_cache_ready(self, block_input: torch.Tensor) -> bool:
        residual = self.first_residual
        return bool(
            residual is not None
            and residual.shape == block_input.shape
            and residual.dtype == block_input.dtype
            and residual.device == block_input.device
        )

    def _mark_whole_full(self, block_input: torch.Tensor) -> None:
        self.first_residual = None
        self.previous_input = block_input.detach()
        self.accumulated_relative_l1 = 0.0
        self.consecutive_reuse = 0
        self.query_stats["full_steps"] += 1

    def reuse_whole_residual(self, block_input: torch.Tensor) -> torch.Tensor | None:
        """Decide whole-stack reuse from a fixed window or raw input drift."""
        if self.config.mode not in {
            "first_residual",
            "interval_residual",
            "input_accum_residual",
        }:
            raise RuntimeError("reuse_whole_residual requires a whole-residual mode")
        self.query_stats["denoise_steps"] += 1
        if not self._whole_cache_ready(block_input):
            self._mark_whole_full(block_input)
            return None

        reuse = True
        if self.config.mode == "interval_residual":
            reuse = self.consecutive_reuse < self.config.max_consecutive_reuse
        elif self.config.mode == "input_accum_residual":
            if self.previous_input is None or self.previous_input.shape != block_input.shape:
                self._mark_whole_full(block_input)
                return None
            current = block_input.detach().to(torch.float32)
            previous = self.previous_input.to(torch.float32)
            relative_l1 = float(
                (
                    (current - previous).abs().mean()
                    / previous.abs().mean().clamp_min(1e-6)
                ).item()
            )
            self.previous_input = block_input.detach()
            self.accumulated_relative_l1 += relative_l1
            self.query_stats["relative_l1_sum"] += relative_l1
            self.query_stats["relative_l1_count"] += 1
            self.query_stats["relative_l1_max"] = max(
                float(self.query_stats["relative_l1_max"]), relative_l1
            )
            reuse = self.accumulated_relative_l1 <= self.config.relative_l1_threshold

        if reuse:
            self.consecutive_reuse += 1
            self.query_stats["reuse_steps"] += 1
            self.query_stats["reused_blocks"] += self.num_layers
            return block_input + self.first_residual

        self._mark_whole_full(block_input)
        return None

    def store_whole_residual(
        self,
        block_input: torch.Tensor,
        block_output: torch.Tensor,
    ) -> None:
        """Store a full ActionDiT stack residual after initialization or refresh."""
        if self.config.mode not in {
            "first_residual",
            "interval_residual",
            "input_accum_residual",
        }:
            raise RuntimeError("store_whole_residual requires a whole-residual mode")
        self.first_residual = (block_output - block_input).detach()
        self.query_stats["full_blocks"] += self.num_layers

    def decide_after_anchor(self, anchor_output: torch.Tensor) -> bool:
        self.query_stats["denoise_steps"] += 1
        expected = self.num_layers - self.config.start_block
        cache_ready = len(self.residuals) == expected
        if self.reference is None or not cache_ready:
            self.reference = anchor_output.detach().clone()
            self.consecutive_reuse = 0
            self.query_stats["full_steps"] += 1
            return False

        current = anchor_output.detach().to(torch.float32)
        reference = self.reference.to(torch.float32)
        delta = (current - reference).abs()
        denominator = reference.abs().mean().clamp_min(1e-6)
        # An infinite max threshold disables this guard.  Do not pay for a
        # full-tensor max reduction that cannot affect the cache decision.
        if math.isfinite(self.config.relative_max_threshold):
            values = torch.stack(
                (delta.mean() / denominator, delta.max() / denominator)
            ).cpu().tolist()
            relative_l1 = float(values[0])
            relative_max = float(values[1])
        else:
            relative_l1 = float((delta.mean() / denominator).item())
            relative_max = 0.0
        self.query_stats["relative_l1_sum"] += relative_l1
        self.query_stats["relative_l1_count"] += 1
        self.query_stats["relative_l1_max"] = max(
            float(self.query_stats["relative_l1_max"]), relative_l1
        )
        self.query_stats["relative_max_max"] = max(
            float(self.query_stats["relative_max_max"]), relative_max
        )
        reuse = (
            relative_l1 <= self.config.relative_l1_threshold
            and relative_max <= self.config.relative_max_threshold
            and self.consecutive_reuse < self.config.max_consecutive_reuse
        )
        if reuse:
            self.consecutive_reuse += 1
            self.query_stats["reuse_steps"] += 1
            return True

        self.reference = anchor_output.detach().clone()
        self.consecutive_reuse = 0
        self.query_stats["full_steps"] += 1
        return False

    def store_residual(
        self,
        layer_index: int,
        block_input: torch.Tensor,
        block_output: torch.Tensor,
    ) -> None:
        self.residuals[int(layer_index)] = (block_output - block_input).detach()
        self.query_stats["full_blocks"] += 1

    def reuse_residual(self, layer_index: int, block_input: torch.Tensor) -> torch.Tensor:
        self.query_stats["reused_blocks"] += 1
        return block_input + self.residuals[int(layer_index)]

    def finalize(self) -> dict[str, float | int]:
        _GLOBAL_STATS["queries"] += 1
        for key, value in self.query_stats.items():
            if key.endswith("_max"):
                _GLOBAL_STATS[key] = max(float(_GLOBAL_STATS[key]), float(value))
            else:
                _GLOBAL_STATS[key] += value
        result = dict(self.query_stats)
        count = int(result["relative_l1_count"])
        result["relative_l1_mean"] = (
            float(result["relative_l1_sum"]) / count if count else 0.0
        )
        steps = int(result["denoise_steps"])
        result["reuse_step_ratio"] = float(result["reuse_steps"]) / steps if steps else 0.0
        return result


def create_action_block_cache(num_layers: int) -> ActionBlockCacheState | None:
    config = get_action_block_cache_config()
    return ActionBlockCacheState(config, num_layers) if config.enabled else None
