"""Low-overhead action-aware diffusion-step routing for FastWAM inference."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class ActionAwareRouter:
    VALID_MODES = {"route", "probe", "force_draft", "force_full"}

    def __init__(self) -> None:
        self.enabled = _env_bool("FASTWAM_ACTIONAWARE_ENABLED")
        self.mode = os.environ.get("FASTWAM_ACTIONAWARE_MODE", "route").strip().lower()
        if self.mode not in self.VALID_MODES:
            raise ValueError(
                f"FASTWAM_ACTIONAWARE_MODE={self.mode!r}; expected one of {sorted(self.VALID_MODES)}"
            )
        self.draft_steps = int(os.environ.get("FASTWAM_ACTIONAWARE_DRAFT_STEPS", "1"))
        self.feature_steps = int(os.environ.get("FASTWAM_ACTIONAWARE_FEATURE_STEPS", "10"))
        self.trans_threshold = float(os.environ.get("FASTWAM_ACTIONAWARE_TRANS_THRESHOLD", "0.7"))
        self.report_every = int(os.environ.get("FASTWAM_ACTIONAWARE_REPORT_EVERY", "100"))
        self.trace_path = os.environ.get("FASTWAM_ACTIONAWARE_TRACE_PATH")
        self.stats_path = os.environ.get("FASTWAM_ACTIONAWARE_DATASET_STATS")
        self._bounds_cache: dict[tuple[torch.device, torch.dtype], tuple[torch.Tensor, torch.Tensor]] = {}
        self._action_min: list[float] | None = None
        self._action_max: list[float] | None = None
        self.stats = {
            "queries": 0,
            "draft": 0,
            "full": 0,
            "candidate_draft": 0,
            "candidate_full": 0,
        }

        if self.enabled:
            if not 1 <= self.draft_steps:
                raise ValueError("FASTWAM_ACTIONAWARE_DRAFT_STEPS must be positive")
            if self.feature_steps < 1:
                raise ValueError("FASTWAM_ACTIONAWARE_FEATURE_STEPS must be positive")
            if not self.stats_path:
                raise ValueError("FASTWAM_ACTIONAWARE_DATASET_STATS is required when routing is enabled")
            payload = json.loads(Path(self.stats_path).read_text(encoding="utf-8"))
            action_stats = payload["action"]["default"]
            self._action_min = action_stats["global_min"]
            self._action_max = action_stats["global_max"]
            print(
                "[FastWAM ActionAware] enabled: "
                f"mode={self.mode} draft_steps={self.draft_steps} "
                f"feature_steps={self.feature_steps} trans_threshold={self.trans_threshold}"
            )

    def validate_num_steps(self, num_steps: int) -> None:
        if self.enabled and self.draft_steps > num_steps:
            raise ValueError(
                f"draft_steps={self.draft_steps} exceeds num_inference_steps={num_steps}"
            )

    def _bounds(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        key = (value.device, value.dtype)
        cached = self._bounds_cache.get(key)
        if cached is None:
            assert self._action_min is not None and self._action_max is not None
            cached = (
                torch.as_tensor(self._action_min, device=value.device, dtype=value.dtype),
                torch.as_tensor(self._action_max, device=value.device, dtype=value.dtype),
            )
            self._bounds_cache[key] = cached
        return cached

    def denormalize(self, action: torch.Tensor) -> torch.Tensor:
        action_min, action_max = self._bounds(action)
        return 0.5 * (action + 1.0) * (action_max - action_min) + action_min

    def draft_metrics(self, draft: torch.Tensor) -> dict[str, float]:
        physical = self.denormalize(draft.to(torch.float32))
        steps = min(self.feature_steps, int(physical.shape[1]))
        trans_mag = torch.linalg.vector_norm(physical[:, :steps, :3], dim=-1)
        # The production route only gates on the mean.  Avoid the otherwise
        # unused max reduction; probe/trace runs retain it for diagnostics.
        trans_mag_mean = float(trans_mag.mean().item())
        trans_mag_max = (
            float(trans_mag.max().item())
            if self.mode == "probe" or self.trace_path
            else 0.0
        )
        return {
            "trans_mag_mean": trans_mag_mean,
            "trans_mag_max": trans_mag_max,
        }

    def candidate_uses_draft(self, metrics: dict[str, float]) -> bool:
        # Match the tested Cosmos-Policy gate semantics: low-translation
        # drafts are refined, while sufficiently decisive motion uses draft.
        return metrics["trans_mag_mean"] > self.trans_threshold

    def should_return_draft(self, candidate_draft: bool) -> bool:
        if self.mode == "force_draft":
            return True
        if self.mode in {"probe", "force_full"}:
            return False
        return candidate_draft

    def record(
        self,
        *,
        metrics: dict[str, float],
        candidate_draft: bool,
        route: str,
        draft: torch.Tensor,
        final: torch.Tensor,
    ) -> None:
        self.stats["queries"] += 1
        self.stats["candidate_draft" if candidate_draft else "candidate_full"] += 1
        self.stats["draft" if route != "full" else "full"] += 1
        if self.trace_path:
            payload: dict[str, Any] = {
                "event": "fastwam_actionaware_query",
                "query": self.stats["queries"],
                "mode": self.mode,
                "route": route,
                "draft_steps": self.draft_steps,
                "candidate_draft": candidate_draft,
                **metrics,
            }
            if self.mode == "probe":
                draft_physical = self.denormalize(draft.to(torch.float32))
                final_physical = self.denormalize(final.to(torch.float32))
                steps = min(self.feature_steps, int(draft.shape[1]), int(final.shape[1]))
                delta = draft_physical[:, :steps] - final_physical[:, :steps]
                probe_values = torch.stack(
                    (
                        delta.abs().mean(),
                        delta.abs().max(),
                        torch.linalg.vector_norm(delta[..., :3], dim=-1).mean(),
                        torch.linalg.vector_norm(delta[..., :3], dim=-1).max(),
                    )
                ).detach().cpu().tolist()
                payload.update(
                    pair_action_l1=float(probe_values[0]),
                    pair_action_linf=float(probe_values[1]),
                    pair_trans_l2_mean=float(probe_values[2]),
                    pair_trans_l2_max=float(probe_values[3]),
                )
            path = Path(self.trace_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(payload, sort_keys=True) + "\n")
        if self.report_every > 0 and self.stats["queries"] % self.report_every == 0:
            print(
                "[FastWAM ActionAware] "
                f"queries={self.stats['queries']} draft={self.stats['draft']} "
                f"full={self.stats['full']} draft_ratio={self.stats['draft'] / self.stats['queries']:.3f}"
            )


_ROUTER: ActionAwareRouter | None = None


def get_actionaware_router() -> ActionAwareRouter:
    global _ROUTER
    if _ROUTER is None:
        _ROUTER = ActionAwareRouter()
    return _ROUTER
