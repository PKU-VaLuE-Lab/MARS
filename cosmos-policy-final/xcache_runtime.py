#!/usr/bin/env python3
"""Per-block, cross-query XCache residual reuse for Cosmos-Policy.

The cache key is exactly ``(denoising_step, block_id)``.  A cache hit reuses
only that block's residual; all pre-DiT work, every front-anchor block, the
final projection, unpatchify, sampler update, and the rest of the policy stay
on their original inference paths.
"""

from __future__ import annotations

import json
import os
import types
from pathlib import Path
from typing import Any

import torch
import torch.amp as amp


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return default if raw is None else int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return default if raw is None else float(raw)


def _parse_view_groups(raw: str) -> tuple[tuple[int, ...] | None, ...]:
    groups: list[tuple[int, ...] | None] = []
    for item in raw.split(";"):
        normalized = item.strip().lower()
        if not normalized:
            continue
        if normalized == "all":
            group = None
        else:
            group = tuple(
                int(index.strip())
                for index in normalized.split(",")
                if index.strip()
            )
            if not group or len(group) != len(set(group)) or any(index < 0 for index in group):
                raise ValueError(f"Invalid COSMOS_XCACHE_VIEW_GROUPS item: {item!r}")
        if group in groups:
            raise ValueError(f"Duplicate COSMOS_XCACHE_VIEW_GROUPS item: {item!r}")
        groups.append(group)
    if not groups:
        raise ValueError("COSMOS_XCACHE_VIEW_GROUPS must contain at least one group")
    return tuple(groups)


XCACHE_NUM_STEPS = _env_int("COSMOS_XCACHE_NUM_STEPS", 5)
XCACHE_FINGERPRINT_TOKENS = _env_int("COSMOS_XCACHE_FINGERPRINT_TOKENS", 32)
XCACHE_ALPHA = _env_float("COSMOS_XCACHE_ALPHA", 0.3)
XCACHE_TAU_FLOOR = _env_float("COSMOS_XCACHE_TAU_FLOOR", 0.98)
XCACHE_MARGIN = _env_float("COSMOS_XCACHE_MARGIN", 0.02)
XCACHE_DEV_THRESHOLD = _env_float("COSMOS_XCACHE_DEV_THRESHOLD", 45.0)
XCACHE_FRONT_ANCHOR = _env_int("COSMOS_XCACHE_FRONT_ANCHOR", 1)
XCACHE_VIEW_GROUPS_RAW = os.environ.get(
    "COSMOS_XCACHE_VIEW_GROUPS",
    "all;2,6;3,7",
)
XCACHE_VIEW_GROUPS = _parse_view_groups(XCACHE_VIEW_GROUPS_RAW)
XCACHE_REL_EPS = 1e-8
XCACHE_VERBOSE = _env_bool("COSMOS_XCACHE_VERBOSE", True)
XCACHE_REPORT_EVERY = _env_int("COSMOS_XCACHE_REPORT_EVERY", 200)
XCACHE_TRACE_LIMIT = _env_int("COSMOS_XCACHE_TRACE_LIMIT", 0)
XCACHE_STATS_PATH = os.environ.get("COSMOS_XCACHE_STATS_PATH")

if XCACHE_NUM_STEPS != 5:
    raise ValueError(
        "XCache is defined against Cosmos-Policy's original five-step action "
        f"sampler, got COSMOS_XCACHE_NUM_STEPS={XCACHE_NUM_STEPS}"
    )
if XCACHE_FINGERPRINT_TOKENS < 1:
    raise ValueError("COSMOS_XCACHE_FINGERPRINT_TOKENS must be positive")
if not 0.0 < XCACHE_ALPHA <= 1.0:
    raise ValueError("COSMOS_XCACHE_ALPHA must be in (0,1]")
if not -1.0 <= XCACHE_TAU_FLOOR <= 1.0:
    raise ValueError("COSMOS_XCACHE_TAU_FLOOR must be in [-1,1]")
if XCACHE_MARGIN < 0.0:
    raise ValueError("COSMOS_XCACHE_MARGIN must be non-negative")
if XCACHE_DEV_THRESHOLD <= 0.0:
    raise ValueError("COSMOS_XCACHE_DEV_THRESHOLD must be positive")
if not 1 <= XCACHE_FRONT_ANCHOR <= 28:
    raise ValueError("COSMOS_XCACHE_FRONT_ANCHOR must be in [1,28]")
if XCACHE_TRACE_LIMIT < 0:
    raise ValueError("COSMOS_XCACHE_TRACE_LIMIT must be non-negative")


_INSTALLED_MODELS: list[torch.nn.Module] = []
_SAMPLE_INDEX_CACHE: dict[tuple[str, int | None, int, int], torch.Tensor] = {}
_FRAME_INDEX_CACHE: dict[
    tuple[str, int | None, int, tuple[int, ...] | None],
    torch.Tensor,
] = {}


def _write_jsonl(payload: dict[str, Any]) -> None:
    if not XCACHE_STATS_PATH:
        return
    path = Path(XCACHE_STATS_PATH).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")


def _new_state() -> dict[str, Any]:
    return {
        "active": False,
        "residual_cache": {},
        "fingerprint_cache": {},
        "ema": {},
        "current_query": None,
        "current_step": 0,
        "episode_query_index": 0,
        "episode_resets": 0,
        "queries": 0,
        "warmup_queries": 0,
        "full_blocks": 0,
        "skipped_blocks": 0,
        "anchor_full_blocks": 0,
        "warmup_full_blocks": 0,
        "cache_miss_full_blocks": 0,
        "gate_full_blocks": 0,
        "fallback_full_blocks": 0,
        "comparisons": 0,
        "cache_pairs_cleared": 0,
        "cosine_sum": 0.0,
        "cosine_min": None,
        "cosine_max": None,
        "deviation_sum": 0.0,
        "deviation_min": None,
        "deviation_max": None,
        "threshold_sum": 0.0,
        "per_step_full": [0] * XCACHE_NUM_STEPS,
        "per_step_skipped": [0] * XCACHE_NUM_STEPS,
        "per_block_full": [0] * 28,
        "per_block_skipped": [0] * 28,
        "fingerprint_shape": None,
        "gate_trace": [],
    }


def _get_state(net: torch.nn.Module) -> dict[str, Any]:
    state = getattr(net, "_cosmos_xcache_state", None)
    if state is None:
        state = _new_state()
        net._cosmos_xcache_state = state
    return state


def _device_key(device: torch.device) -> tuple[str, int | None]:
    return device.type, device.index


def _uniform_sample_indices(token_count: int, device: torch.device) -> torch.Tensor:
    sample_count = XCACHE_FINGERPRINT_TOKENS
    if token_count < sample_count:
        raise RuntimeError(
            "XCache fingerprint group is smaller than the requested sample: "
            f"tokens={token_count}, requested={sample_count}"
        )
    key = (*_device_key(device), token_count, sample_count)
    cached = _SAMPLE_INDEX_CACHE.get(key)
    if cached is not None:
        return cached
    if sample_count == 1:
        indices = torch.zeros(1, dtype=torch.long, device=device)
    else:
        indices = torch.div(
            torch.arange(sample_count, dtype=torch.long, device=device)
            * (token_count - 1),
            sample_count - 1,
            rounding_mode="floor",
        )
    _SAMPLE_INDEX_CACHE[key] = indices
    return indices


def _frame_indices(
    frames: int,
    group: tuple[int, ...] | None,
    device: torch.device,
) -> torch.Tensor:
    key = (*_device_key(device), frames, group)
    cached = _FRAME_INDEX_CACHE.get(key)
    if cached is not None:
        return cached
    values = tuple(range(frames)) if group is None else group
    if any(index >= frames for index in values):
        raise RuntimeError(
            "XCache view group exceeds the latent temporal dimension: "
            f"group={values}, frames={frames}"
        )
    indices = torch.tensor(values, dtype=torch.long, device=device)
    _FRAME_INDEX_CACHE[key] = indices
    return indices


def _fingerprint(block_input: torch.Tensor) -> torch.Tensor:
    """Uniform sparse tokens plus one token-mean, independently per view group."""

    if block_input.ndim != 5:
        raise RuntimeError(
            "XCache expected block input in B,T,H,W,D layout, "
            f"got shape={tuple(block_input.shape)}"
        )
    batch, frames, _height, _width, channels = block_input.shape
    fingerprints = []
    for group in XCACHE_VIEW_GROUPS:
        # ``all`` already spans the full temporal axis.  Avoid an identity
        # index_select here: it materializes another full block-input tensor
        # at every (denoising_step, block_id) gate without changing a single
        # fingerprint value.
        group_input = (
            block_input
            if group is None
            else block_input.index_select(
                1,
                _frame_indices(frames, group, block_input.device),
            )
        )
        flat = group_input.reshape(batch, -1, channels)
        sampled = flat.index_select(
            1,
            _uniform_sample_indices(flat.shape[1], block_input.device),
        )
        token_mean = flat.mean(dim=1, keepdim=True)
        fingerprints.append(torch.cat((sampled, token_mean), dim=1))
    return torch.stack(fingerprints, dim=1).detach()


def _fingerprint_metrics(
    current: torch.Tensor,
    previous: torch.Tensor,
) -> tuple[float, float]:
    """Return worst-group cosine and worst-group relative max deviation."""

    current_f32 = current.to(dtype=torch.float32)
    previous_f32 = previous.to(dtype=torch.float32)
    reduce_dims = (-1, -2)
    dot = (current_f32 * previous_f32).sum(dim=reduce_dims)
    current_norm = current_f32.square().sum(dim=reduce_dims).sqrt()
    previous_norm = previous_f32.square().sum(dim=reduce_dims).sqrt()
    cosine_by_group = dot / (current_norm * previous_norm).clamp_min(XCACHE_REL_EPS)
    deviation_by_group = (
        (current_f32 - previous_f32).abs().amax(dim=reduce_dims)
        / previous_f32.abs().mean(dim=reduce_dims).clamp_min(XCACHE_REL_EPS)
    )
    packed = torch.stack((cosine_by_group.amin(), deviation_by_group.amax()))
    cosine, deviation = packed.detach().cpu().tolist()
    return float(cosine), float(deviation)


def _record_comparison(
    state: dict[str, Any],
    cosine: float,
    deviation: float,
    threshold: float,
) -> None:
    state["comparisons"] += 1
    state["cosine_sum"] += cosine
    state["deviation_sum"] += deviation
    state["threshold_sum"] += threshold
    state["cosine_min"] = cosine if state["cosine_min"] is None else min(state["cosine_min"], cosine)
    state["cosine_max"] = cosine if state["cosine_max"] is None else max(state["cosine_max"], cosine)
    state["deviation_min"] = (
        deviation if state["deviation_min"] is None else min(state["deviation_min"], deviation)
    )
    state["deviation_max"] = (
        deviation if state["deviation_max"] is None else max(state["deviation_max"], deviation)
    )


def _record_full(
    state: dict[str, Any],
    step_index: int,
    block_index: int,
    reason: str,
) -> None:
    state["full_blocks"] += 1
    state["per_step_full"][step_index] += 1
    state["per_block_full"][block_index] += 1
    state[f"{reason}_full_blocks"] += 1


def _record_skip(state: dict[str, Any], step_index: int, block_index: int) -> None:
    state["skipped_blocks"] += 1
    state["per_step_skipped"][step_index] += 1
    state["per_block_skipped"][block_index] += 1


def reset_episode(model: torch.nn.Module) -> None:
    """Clear every residual, fingerprint, and EMA at the episode boundary."""

    net = getattr(model, "net", None)
    if net is None or not getattr(net, "_cosmos_xcache_installed", False):
        raise RuntimeError("XCache episode reset expected an installed model.net")
    state = _get_state(net)
    if state["active"]:
        raise RuntimeError("XCache cannot reset while an action query is active")
    cleared = len(state["residual_cache"])
    state["residual_cache"].clear()
    state["fingerprint_cache"].clear()
    state["ema"].clear()
    state["active"] = False
    state["current_query"] = None
    state["current_step"] = 0
    state["episode_query_index"] = 0
    state["episode_resets"] += 1
    state["cache_pairs_cleared"] += cleared
    if XCACHE_VERBOSE and state["episode_resets"] <= 3:
        print(
            "[XCache] episode reset: "
            f"episode={state['episode_resets']} cleared_cache_pairs={cleared}"
        )


def _run_block(
    block: torch.nn.Module,
    block_input: torch.Tensor,
    t_embedding: torch.Tensor,
    context_input: Any,
    rope_embedding: torch.Tensor,
    adaln_lora: torch.Tensor,
    extra_position_embedding: torch.Tensor | None,
) -> torch.Tensor:
    return block(
        block_input,
        t_embedding,
        context_input,
        rope_emb_L_1_1_D=rope_embedding,
        adaln_lora_B_T_3D=adaln_lora,
        extra_per_block_pos_emb=extra_position_embedding,
    )


def _xcache_forward(
    self,
    x_B_C_T_H_W: torch.Tensor,
    timesteps_B_T: torch.Tensor,
    crossattn_emb: torch.Tensor,
    condition_video_input_mask_B_C_T_H_W: torch.Tensor | None = None,
    fps: torch.Tensor | None = None,
    padding_mask: torch.Tensor | None = None,
    data_type: Any = None,
    intermediate_feature_ids: list[int] | None = None,
    img_context_emb: torch.Tensor | None = None,
    action: torch.Tensor | None = None,
    **kwargs,
):
    state = _get_state(self)
    if not state["active"]:
        return self._cosmos_xcache_original_forward(
            x_B_C_T_H_W=x_B_C_T_H_W,
            timesteps_B_T=timesteps_B_T,
            crossattn_emb=crossattn_emb,
            condition_video_input_mask_B_C_T_H_W=condition_video_input_mask_B_C_T_H_W,
            fps=fps,
            padding_mask=padding_mask,
            data_type=data_type,
            intermediate_feature_ids=intermediate_feature_ids,
            img_context_emb=img_context_emb,
            action=action,
            **kwargs,
        )

    step_index = int(state["current_step"])
    state["current_step"] = step_index + 1
    if step_index >= XCACHE_NUM_STEPS:
        raise RuntimeError(
            "XCache observed more than five DiT forwards in one action query: "
            f"step_index={step_index}"
        )

    if intermediate_feature_ids or action is not None:
        for block_index in range(28):
            _record_full(state, step_index, block_index, "fallback")
        return self._cosmos_xcache_original_forward(
            x_B_C_T_H_W=x_B_C_T_H_W,
            timesteps_B_T=timesteps_B_T,
            crossattn_emb=crossattn_emb,
            condition_video_input_mask_B_C_T_H_W=condition_video_input_mask_B_C_T_H_W,
            fps=fps,
            padding_mask=padding_mask,
            data_type=data_type,
            intermediate_feature_ids=intermediate_feature_ids,
            img_context_emb=img_context_emb,
            action=action,
            **kwargs,
        )

    from cosmos_policy._src.predict2.conditioner import DataType

    if data_type is None:
        data_type = DataType.VIDEO
    if not isinstance(data_type, DataType):
        raise TypeError(f"XCache expected DataType, got {type(data_type)}")

    if condition_video_input_mask_B_C_T_H_W is not None:
        if data_type == DataType.VIDEO:
            x_B_C_T_H_W = torch.cat(
                [x_B_C_T_H_W, condition_video_input_mask_B_C_T_H_W.type_as(x_B_C_T_H_W)],
                dim=1,
            )
        else:
            batch, _, frames, height, width = x_B_C_T_H_W.shape
            x_B_C_T_H_W = torch.cat(
                [
                    x_B_C_T_H_W,
                    torch.zeros(
                        (batch, 1, frames, height, width),
                        dtype=x_B_C_T_H_W.dtype,
                        device=x_B_C_T_H_W.device,
                    ),
                ],
                dim=1,
            )

    timesteps_B_T = timesteps_B_T * getattr(self, "timestep_scale", 1.0)
    (
        x_B_T_H_W_D,
        rope_emb_L_1_1_D,
        extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D,
    ) = self.prepare_embedded_sequence(
        x_B_C_T_H_W,
        fps=fps,
        padding_mask=padding_mask,
    )

    if self.use_crossattn_projection:
        crossattn_emb = self.crossattn_proj(crossattn_emb)
    if img_context_emb is not None:
        if self.extra_image_context_dim is None:
            raise RuntimeError("extra_image_context_dim must be set for img_context_emb")
        img_context_emb = self.img_context_proj(img_context_emb)
        context_input = (crossattn_emb, img_context_emb)
    else:
        context_input = crossattn_emb

    with amp.autocast("cuda", enabled=self.use_wan_fp32_strategy, dtype=torch.float32):
        if timesteps_B_T.ndim == 1:
            timesteps_B_T = timesteps_B_T.unsqueeze(1)
        t_embedding_B_T_D, adaln_lora_B_T_3D = self.t_embedder(timesteps_B_T)
        t_embedding_B_T_D = self.t_embedding_norm(t_embedding_B_T_D)

    self.affline_scale_log_info = {"t_embedding_B_T_D": t_embedding_B_T_D.detach()}
    self.affline_emb = t_embedding_B_T_D
    self.crossattn_emb = crossattn_emb
    if (
        extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D is not None
        and x_B_T_H_W_D.shape != extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D.shape
    ):
        raise RuntimeError(
            f"XCache positional embedding shape mismatch: {x_B_T_H_W_D.shape} != "
            f"{extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D.shape}"
        )

    query_context = state["current_query"]
    if query_context is None:
        raise RuntimeError("XCache active forward is missing query context")
    warmup_query = bool(query_context["warmup"])

    for block_index, block in enumerate(self.blocks):
        block_input = x_B_T_H_W_D
        if block_index < XCACHE_FRONT_ANCHOR:
            x_B_T_H_W_D = _run_block(
                block,
                block_input,
                t_embedding_B_T_D,
                context_input,
                rope_emb_L_1_1_D,
                adaln_lora_B_T_3D,
                extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D,
            )
            _record_full(state, step_index, block_index, "anchor")
            continue

        key = (step_index, block_index)
        current_fp = _fingerprint(block_input)
        if state["fingerprint_shape"] is None:
            state["fingerprint_shape"] = list(current_fp.shape)
        cached_residual = state["residual_cache"].get(key)
        previous_fp = state["fingerprint_cache"].get(key)
        compatible = (
            cached_residual is not None
            and previous_fp is not None
            and cached_residual.shape == block_input.shape
            and cached_residual.dtype == block_input.dtype
            and cached_residual.device == block_input.device
            and previous_fp.shape == current_fp.shape
            and previous_fp.dtype == current_fp.dtype
            and previous_fp.device == current_fp.device
        )

        use_cache = False
        if not warmup_query and compatible:
            cosine, deviation = _fingerprint_metrics(current_fp, previous_fp)
            previous_ema = float(state["ema"].get(key, 1.0))
            ema = XCACHE_ALPHA * cosine + (1.0 - XCACHE_ALPHA) * previous_ema
            threshold = max(XCACHE_TAU_FLOOR, ema - XCACHE_MARGIN)
            state["ema"][key] = ema
            _record_comparison(state, cosine, deviation, threshold)
            use_cache = cosine >= threshold and deviation < XCACHE_DEV_THRESHOLD
            if (
                XCACHE_TRACE_LIMIT > 0
                and block_index == XCACHE_FRONT_ANCHOR
                and len(state["gate_trace"]) < XCACHE_TRACE_LIMIT
            ):
                state["gate_trace"].append(
                    {
                        "query": int(query_context["index"]),
                        "step": step_index,
                        "block": block_index,
                        "cosine": cosine,
                        "deviation": deviation,
                        "threshold": threshold,
                        "skip": use_cache,
                    }
                )

        if use_cache:
            x_B_T_H_W_D = block_input + cached_residual
            _record_skip(state, step_index, block_index)
            continue

        x_B_T_H_W_D = _run_block(
            block,
            block_input,
            t_embedding_B_T_D,
            context_input,
            rope_emb_L_1_1_D,
            adaln_lora_B_T_3D,
            extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D,
        )
        state["residual_cache"][key] = (x_B_T_H_W_D - block_input).detach()
        state["fingerprint_cache"][key] = current_fp
        if key not in state["ema"]:
            state["ema"][key] = 1.0
        reason = "warmup" if warmup_query else ("cache_miss" if not compatible else "gate")
        _record_full(state, step_index, block_index, reason)

    x_B_T_H_W_O = self.final_layer(
        x_B_T_H_W_D,
        t_embedding_B_T_D,
        adaln_lora_B_T_3D=adaln_lora_B_T_3D,
    )
    return self.unpatchify(x_B_T_H_W_O)


def _summary_payload(state: dict[str, Any], event: str) -> dict[str, Any]:
    total_blocks = state["full_blocks"] + state["skipped_blocks"]
    comparisons = state["comparisons"]
    return {
        "event": event,
        "num_steps": XCACHE_NUM_STEPS,
        "blocks": 28,
        "fingerprint_tokens": XCACHE_FINGERPRINT_TOKENS,
        "fingerprint_shape": state["fingerprint_shape"],
        "view_groups": XCACHE_VIEW_GROUPS_RAW,
        "alpha": XCACHE_ALPHA,
        "tau_floor": XCACHE_TAU_FLOOR,
        "margin": XCACHE_MARGIN,
        "dev_threshold": XCACHE_DEV_THRESHOLD,
        "front_anchor": XCACHE_FRONT_ANCHOR,
        "episode_resets": state["episode_resets"],
        "queries": state["queries"],
        "warmup_queries": state["warmup_queries"],
        "full_blocks": state["full_blocks"],
        "skipped_blocks": state["skipped_blocks"],
        "skip_ratio": state["skipped_blocks"] / total_blocks if total_blocks else 0.0,
        "eligible_skip_ratio": state["skipped_blocks"] / comparisons if comparisons else 0.0,
        "anchor_full_blocks": state["anchor_full_blocks"],
        "warmup_full_blocks": state["warmup_full_blocks"],
        "cache_miss_full_blocks": state["cache_miss_full_blocks"],
        "gate_full_blocks": state["gate_full_blocks"],
        "fallback_full_blocks": state["fallback_full_blocks"],
        "comparisons": comparisons,
        "cache_pairs_cleared": state["cache_pairs_cleared"],
        "cosine_mean": state["cosine_sum"] / comparisons if comparisons else None,
        "cosine_min": state["cosine_min"],
        "cosine_max": state["cosine_max"],
        "deviation_mean": state["deviation_sum"] / comparisons if comparisons else None,
        "deviation_min": state["deviation_min"],
        "deviation_max": state["deviation_max"],
        "threshold_mean": state["threshold_sum"] / comparisons if comparisons else None,
        "per_step_full": list(state["per_step_full"]),
        "per_step_skipped": list(state["per_step_skipped"]),
        "per_block_full": list(state["per_block_full"]),
        "per_block_skipped": list(state["per_block_skipped"]),
        "gate_trace": list(state["gate_trace"]),
    }


def _install_on_model(model: torch.nn.Module) -> None:
    net = getattr(model, "net", None)
    if net is None or not hasattr(net, "blocks") or not hasattr(net, "final_layer"):
        raise RuntimeError("XCache expected model.net with blocks and final_layer")
    if len(net.blocks) != 28:
        raise RuntimeError(f"XCache expected exactly 28 DiT blocks, got {len(net.blocks)}")
    incompatible_markers = (
        "_cosmos_dicache_original_forward",
        "_cosmos_stepskip_original_forward",
        "_cosmos_c3cache_original_forward",
        "_cosmos_xcache_original_forward",
    )
    present = [name for name in incompatible_markers if hasattr(net, name)]
    if present:
        raise RuntimeError(f"XCache refuses an already accelerated DiT: {present}")
    sampler = getattr(model, "sampler", None)
    if sampler is not None and getattr(sampler, "_cosmos_actionaware_installed", False):
        raise RuntimeError("XCache cannot be combined with ActionAware")

    net._cosmos_xcache_original_forward = net.forward
    net._cosmos_xcache_installed = True
    net._cosmos_xcache_state = _new_state()
    net.forward = types.MethodType(_xcache_forward, net)

    original_generate = model.generate_samples_from_batch

    def generate_samples_with_xcache(self, data_batch, *args, **kwargs):
        if len(args) >= 6:
            num_steps = int(args[5])
        else:
            num_steps = int(kwargs.get("num_steps", 35))
        if num_steps != XCACHE_NUM_STEPS:
            raise ValueError(
                "XCache requires the original five-step action sampler, "
                f"got num_steps={num_steps}"
            )
        state = _get_state(net)
        if state["active"]:
            raise RuntimeError("XCache does not support nested generate calls")
        query_index = int(state["episode_query_index"])
        warmup = query_index == 0
        state["active"] = True
        state["current_step"] = 0
        state["current_query"] = {"index": query_index, "warmup": warmup}
        completed = False
        try:
            result = original_generate(data_batch, *args, **kwargs)
            observed_steps = int(state["current_step"])
            if observed_steps != XCACHE_NUM_STEPS:
                raise RuntimeError(
                    "XCache expected exactly five DiT forwards per action query, "
                    f"observed {observed_steps}"
                )
            completed = True
            return result
        finally:
            state["active"] = False
            state["current_query"] = None
            state["current_step"] = 0
            state["episode_query_index"] = query_index + 1
            if completed:
                state["queries"] += 1
                if warmup:
                    state["warmup_queries"] += 1
                if (
                    XCACHE_VERBOSE
                    and XCACHE_REPORT_EVERY > 0
                    and state["queries"] % XCACHE_REPORT_EVERY == 0
                ):
                    payload = _summary_payload(state, "xcache_summary")
                    print("[XCache] summary: " + json.dumps(payload, sort_keys=True))
                    _write_jsonl(payload)

    model.generate_samples_from_batch = types.MethodType(
        generate_samples_with_xcache,
        model,
    )
    _INSTALLED_MODELS.append(model)
    print(
        "[XCache] installed exclusive runtime patch: "
        f"blocks=28 num_steps={XCACHE_NUM_STEPS} "
        f"fingerprint_tokens={XCACHE_FINGERPRINT_TOKENS} "
        f"view_groups={XCACHE_VIEW_GROUPS_RAW!r} alpha={XCACHE_ALPHA} "
        f"tau_floor={XCACHE_TAU_FLOOR} margin={XCACHE_MARGIN} "
        f"dev_threshold={XCACHE_DEV_THRESHOLD} front_anchor={XCACHE_FRONT_ANCHOR}; "
        "cache_key=(denoising_step,block_id) other_accelerators=disabled"
    )


def install_on_cosmos_utils(cosmos_utils: Any) -> None:
    """Install XCache after the untouched Cosmos-Policy model loader."""

    current = cosmos_utils.get_model
    if getattr(current, "_cosmos_xcache_loader", False):
        return

    def get_model_with_xcache(cfg):
        model, config = current(cfg)
        _install_on_model(model)
        return model, config

    get_model_with_xcache._cosmos_xcache_loader = True
    cosmos_utils.get_model = get_model_with_xcache


def install_episode_reset(base_eval: Any) -> None:
    """Wrap base_eval.patch_runtime so every run_episode starts empty."""

    previous_patch_runtime = base_eval.patch_runtime
    if getattr(previous_patch_runtime, "_cosmos_xcache_reset_installer", False):
        return

    def patch_runtime_with_xcache_reset(module, *args, **kwargs):
        previous_patch_runtime(module, *args, **kwargs)
        if getattr(module.run_episode, "_cosmos_xcache_episode_reset", False):
            return
        original_run_episode = module.run_episode

        def run_episode_with_xcache_reset(*episode_args, **episode_kwargs):
            if len(episode_args) >= 4:
                model = episode_args[3]
            else:
                model = episode_kwargs.get("model")
            if model is None:
                raise RuntimeError("XCache could not resolve model at episode reset")
            reset_episode(model)
            return original_run_episode(*episode_args, **episode_kwargs)

        run_episode_with_xcache_reset._cosmos_xcache_episode_reset = True
        module.run_episode = run_episode_with_xcache_reset

    patch_runtime_with_xcache_reset._cosmos_xcache_reset_installer = True
    base_eval.patch_runtime = patch_runtime_with_xcache_reset


def finalize_stats() -> None:
    aggregate = _new_state()
    aggregate["models"] = len(_INSTALLED_MODELS)
    aggregate["event"] = "xcache_final"
    scalar_int_keys = (
        "episode_resets",
        "queries",
        "warmup_queries",
        "full_blocks",
        "skipped_blocks",
        "anchor_full_blocks",
        "warmup_full_blocks",
        "cache_miss_full_blocks",
        "gate_full_blocks",
        "fallback_full_blocks",
        "comparisons",
        "cache_pairs_cleared",
    )
    aggregate.update({key: 0 for key in scalar_int_keys})
    aggregate["cosine_sum"] = 0.0
    aggregate["deviation_sum"] = 0.0
    aggregate["threshold_sum"] = 0.0
    aggregate["cosine_min"] = None
    aggregate["cosine_max"] = None
    aggregate["deviation_min"] = None
    aggregate["deviation_max"] = None
    aggregate["per_step_full"] = [0] * XCACHE_NUM_STEPS
    aggregate["per_step_skipped"] = [0] * XCACHE_NUM_STEPS
    aggregate["per_block_full"] = [0] * 28
    aggregate["per_block_skipped"] = [0] * 28
    aggregate["fingerprint_shape"] = None
    aggregate["gate_trace"] = []

    for model in _INSTALLED_MODELS:
        state = _get_state(model.net)
        for key in scalar_int_keys:
            aggregate[key] += int(state[key])
        for key in ("cosine_sum", "deviation_sum", "threshold_sum"):
            aggregate[key] += float(state[key])
        for key in ("cosine_min", "deviation_min"):
            value = state[key]
            if value is not None:
                aggregate[key] = value if aggregate[key] is None else min(aggregate[key], value)
        for key in ("cosine_max", "deviation_max"):
            value = state[key]
            if value is not None:
                aggregate[key] = value if aggregate[key] is None else max(aggregate[key], value)
        for step in range(XCACHE_NUM_STEPS):
            aggregate["per_step_full"][step] += int(state["per_step_full"][step])
            aggregate["per_step_skipped"][step] += int(state["per_step_skipped"][step])
        for block in range(28):
            aggregate["per_block_full"][block] += int(state["per_block_full"][block])
            aggregate["per_block_skipped"][block] += int(state["per_block_skipped"][block])
        if aggregate["fingerprint_shape"] is None:
            aggregate["fingerprint_shape"] = state["fingerprint_shape"]
        remaining_trace = max(0, XCACHE_TRACE_LIMIT - len(aggregate["gate_trace"]))
        if remaining_trace:
            aggregate["gate_trace"].extend(state["gate_trace"][:remaining_trace])

    payload = _summary_payload(aggregate, "xcache_final")
    payload["models"] = aggregate["models"]
    print("[XCache] final: " + json.dumps(payload, sort_keys=True))
    _write_jsonl(payload)
