#!/usr/bin/env python3
"""Run LIBERO with a shallow-VAE Top-K IoU gate for future-token reuse."""

import json
import os
import sys
import types
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPT_DIR))
os.chdir(ROOT)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


# Production defaults to the zero-probe RGB gate. Shallow VAE layers remain
# available for explicit feature ablations via COSMOS_SHALLOW_GATE_LAYER.
LAYER = os.environ.get("COSMOS_SHALLOW_GATE_LAYER", "rgb").strip().lower()
TOPK_FRACTION = float(os.environ.get("COSMOS_SHALLOW_GATE_TOPK", "0.15"))
IOU_THRESHOLD = float(os.environ.get("COSMOS_SHALLOW_GATE_IOU_THRESHOLD", "0.65"))
MAX_CONSECUTIVE_SKIPS = int(os.environ.get("COSMOS_SHALLOW_GATE_MAX_CONSECUTIVE_SKIPS", "1"))
PRIMARY_MAX_CONSECUTIVE_SKIPS = int(
    os.environ.get(
        "COSMOS_SHALLOW_GATE_PRIMARY_MAX_CONSECUTIVE_SKIPS",
        str(MAX_CONSECUTIVE_SKIPS),
    )
)
WRIST_MAX_CONSECUTIVE_SKIPS = int(
    os.environ.get(
        "COSMOS_SHALLOW_GATE_WRIST_MAX_CONSECUTIVE_SKIPS",
        str(MAX_CONSECUTIVE_SKIPS),
    )
)
QUERY_LAG = int(os.environ.get("COSMOS_SHALLOW_GATE_QUERY_LAG", "1"))
CAMERA_MODE = os.environ.get("COSMOS_SHALLOW_GATE_CAMERA_MODE", "hybrid").strip().lower()
PRIMARY_GUARD_THRESHOLD = float(
    os.environ.get("COSMOS_SHALLOW_GATE_PRIMARY_GUARD_THRESHOLD", "0.45")
)
FORCE_FULL_INTERVAL = int(os.environ.get("COSMOS_SHALLOW_GATE_FORCE_FULL_INTERVAL", "0"))
FORCE_FULL_AFTER_1STEP = _env_bool("COSMOS_SHALLOW_GATE_FORCE_FULL_AFTER_1STEP", False)
SUMMARY_PATH = os.environ.get("COSMOS_SHALLOW_GATE_SUMMARY_PATH", "").strip()
TRACE_PATH = os.environ.get("COSMOS_SHALLOW_GATE_TRACE_PATH", "").strip()
DISABLE_REUSE = os.environ.get("COSMOS_SHALLOW_GATE_DISABLE_REUSE", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
    "on",
}
ANALYSIS_TOPKS = tuple(
    dict.fromkeys(
        float(item)
        for item in os.environ.get("COSMOS_SHALLOW_GATE_ANALYSIS_TOPKS", "").split(",")
        if item.strip()
    )
)
GRID_SIZE = 14
TEMPORAL_COMPRESSION_FACTOR = 4
LAYER_TO_DOWNSAMPLE_INDEX = {
    "res0": 0,
    "res1": 1,
    "down0": 2,
    "stage1_res0": 3,
    "stage1_res1": 4,
    "down1": 5,
    "stage2_res0": 6,
    "stage2_res1": 7,
    "down2": 8,
    "stage3_res0": 9,
    "stage3_res1": 10,
}
LAYER_TO_MIDDLE_INDEX = {"middle_res0": 0, "middle_attn": 1, "middle_res1": 2}
LAYER_TO_HEAD_INDEX = {"head_norm": 0, "head_act": 1, "encoder_out": 2}
SUPPORTED_LAYERS = {
    "rgb",
    "conv1",
    *LAYER_TO_DOWNSAMPLE_INDEX,
    *LAYER_TO_MIDDLE_INDEX,
    *LAYER_TO_HEAD_INDEX,
}
CAMERA_SPECS = (
    ("wrist", "current_wrist_image_latent_idx", "future_wrist_image_latent_idx"),
    ("primary", "current_image_latent_idx", "future_image_latent_idx"),
)


def _env_map(name: str, default: str = "") -> dict[str, str]:
    raw = os.environ.get(name, default)
    result: dict[str, str] = {}
    for item in raw.replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        separator = ":" if ":" in item else "=" if "=" in item else None
        if separator is None:
            raise ValueError(f"{name} entry {item!r} must use key:value or key=value")
        key, value = item.split(separator, 1)
        result[key.strip().lower()] = value.strip()
    return result


def _parse_bool_value(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value {value!r}")


SUITE_LAYERS = _env_map("COSMOS_SHALLOW_GATE_SUITE_LAYER")
SUITE_TOPK = _env_map("COSMOS_SHALLOW_GATE_SUITE_TOPK")
SUITE_IOU_THRESHOLDS = _env_map(
    "COSMOS_SHALLOW_GATE_SUITE_IOU_THRESHOLD",
    "",
)
SUITE_MAX_CONSECUTIVE_SKIPS = _env_map("COSMOS_SHALLOW_GATE_SUITE_MAX_CONSECUTIVE_SKIPS")
SUITE_CAMERA_MODES = _env_map("COSMOS_SHALLOW_GATE_SUITE_CAMERA_MODE")
SUITE_PRIMARY_GUARD_THRESHOLDS = _env_map(
    "COSMOS_SHALLOW_GATE_SUITE_PRIMARY_GUARD_THRESHOLD"
)
SUITE_FORCE_FULL_INTERVALS = _env_map("COSMOS_SHALLOW_GATE_SUITE_FORCE_FULL_INTERVAL")
SUITE_FORCE_FULL_AFTER_1STEP = _env_map("COSMOS_SHALLOW_GATE_SUITE_FORCE_FULL_AFTER_1STEP")
SUITE_ENABLED = _env_map("COSMOS_SHALLOW_GATE_SUITE_ENABLED")

SUPPORTED_CAMERA_MODES = {"hybrid", "min", "mean", "primary", "wrist", "wrist_guard"}

if LAYER not in SUPPORTED_LAYERS:
    raise ValueError(f"Unsupported shallow gate layer {LAYER!r}; choose from {sorted(SUPPORTED_LAYERS)}")
if not 0.0 < TOPK_FRACTION <= 1.0:
    raise ValueError(f"COSMOS_SHALLOW_GATE_TOPK must be in (0, 1], got {TOPK_FRACTION}")
if QUERY_LAG < 1:
    raise ValueError(f"COSMOS_SHALLOW_GATE_QUERY_LAG must be at least 1, got {QUERY_LAG}")
if PRIMARY_MAX_CONSECUTIVE_SKIPS < 0 or WRIST_MAX_CONSECUTIVE_SKIPS < 0:
    raise ValueError("Per-camera max consecutive skips must be non-negative")
if not 0.0 <= IOU_THRESHOLD <= 1.0:
    raise ValueError(f"COSMOS_SHALLOW_GATE_IOU_THRESHOLD must be in [0, 1], got {IOU_THRESHOLD}")
if CAMERA_MODE not in SUPPORTED_CAMERA_MODES:
    raise ValueError(
        f"Unsupported camera mode {CAMERA_MODE!r}; choose from {sorted(SUPPORTED_CAMERA_MODES)}"
    )
if not 0.0 <= PRIMARY_GUARD_THRESHOLD <= 1.0:
    raise ValueError("COSMOS_SHALLOW_GATE_PRIMARY_GUARD_THRESHOLD must be in [0, 1]")
if any(not 0.0 < fraction <= 1.0 for fraction in ANALYSIS_TOPKS):
    raise ValueError("COSMOS_SHALLOW_GATE_ANALYSIS_TOPKS entries must be in (0, 1]")

suite_iou_variant = "-".join(
    f"{suite}{threshold}" for suite, threshold in sorted(SUITE_IOU_THRESHOLDS.items())
)
suite_iou_part = f"_suiteiou{suite_iou_variant}" if suite_iou_variant else ""
os.environ.setdefault(
    "COSMOS_EVAL_VARIANT",
    (
        f"shallowvaeskip_{LAYER}_top{TOPK_FRACTION:g}_"
        f"iou{IOU_THRESHOLD:g}{suite_iou_part}_{CAMERA_MODE}_"
        f"maxskip{MAX_CONSECUTIVE_SKIPS}_periodic{FORCE_FULL_INTERVAL}"
    ),
)


def _suite_candidates(cfg) -> tuple[str, ...]:
    suite = str(
        getattr(cfg, "task_suite_name", "")
        or getattr(cfg, "suite", "")
        or getattr(cfg, "task_name", "")
        or ""
    ).strip().lower()
    candidates = [suite]
    task_name = str(getattr(cfg, "task_name", "") or "").strip().lower()
    if task_name:
        candidates.append(task_name)
    if suite.startswith("libero_"):
        candidates.append(suite[len("libero_") :])
    aliases = {
        "libero_10": ("libero_long", "long", "10"),
        "libero_long": ("libero_10", "long", "10"),
        "long": ("libero_10", "libero_long", "10"),
        "10": ("libero_10", "libero_long", "long"),
    }
    candidates.extend(aliases.get(suite, ()))
    return tuple(dict.fromkeys(candidate for candidate in candidates if candidate))


def _suite_value(mapping: dict[str, str], cfg, default: Any, caster) -> Any:
    for candidate in _suite_candidates(cfg):
        if candidate in mapping:
            return caster(mapping[candidate])
    return default


def _gate_config(cfg) -> tuple[str, float, float]:
    layer = str(_suite_value(SUITE_LAYERS, cfg, LAYER, str)).strip().lower()
    topk = float(_suite_value(SUITE_TOPK, cfg, TOPK_FRACTION, float))
    threshold = float(_suite_value(SUITE_IOU_THRESHOLDS, cfg, IOU_THRESHOLD, float))
    if layer not in SUPPORTED_LAYERS:
        raise ValueError(f"Unsupported shallow gate layer {layer!r}; choose from {sorted(SUPPORTED_LAYERS)}")
    if not 0.0 < topk <= 1.0:
        raise ValueError(f"Top-K fraction must be in (0, 1], got {topk}")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"IoU threshold must be in [0, 1], got {threshold}")
    return layer, topk, threshold


def _consecutive_skip_limit(cfg) -> int | None:
    value = int(_suite_value(SUITE_MAX_CONSECUTIVE_SKIPS, cfg, MAX_CONSECUTIVE_SKIPS, int))
    return value if value > 0 else None


def _camera_config(cfg) -> tuple[str, float]:
    mode = str(_suite_value(SUITE_CAMERA_MODES, cfg, CAMERA_MODE, str)).strip().lower()
    guard = float(
        _suite_value(
            SUITE_PRIMARY_GUARD_THRESHOLDS,
            cfg,
            PRIMARY_GUARD_THRESHOLD,
            float,
        )
    )
    if mode not in SUPPORTED_CAMERA_MODES:
        raise ValueError(f"Unsupported camera mode {mode!r}")
    if not 0.0 <= guard <= 1.0:
        raise ValueError(f"Primary guard threshold must be in [0, 1], got {guard}")
    return mode, guard


def _gate_enabled(cfg) -> bool:
    return bool(_suite_value(SUITE_ENABLED, cfg, True, _parse_bool_value))


def _force_full_interval(cfg) -> int | None:
    value = int(_suite_value(SUITE_FORCE_FULL_INTERVALS, cfg, FORCE_FULL_INTERVAL, int))
    return value if value > 0 else None


def _force_full_after_1step(cfg) -> bool:
    return bool(
        _suite_value(
            SUITE_FORCE_FULL_AFTER_1STEP,
            cfg,
            FORCE_FULL_AFTER_1STEP,
            _parse_bool_value,
        )
    )


_SUMMARY: dict[str, dict[str, int]] = {}
_FLOP_PROFILE_CALLABLE = None
_FLOP_PROFILE_ARGS = None


def _summary_row(suite: str) -> dict[str, int]:
    return _SUMMARY.setdefault(
        suite,
        {
            "queries": 0,
            "gate_eligible_queries": 0,
            "guard_blocked_queries": 0,
            "actionaware_guard_blocked_queries": 0,
            "wrist_limit_downgrade_queries": 0,
            "skipped_queries": 0,
            "primary_reuse_queries": 0,
            "all_reuse_queries": 0,
            "full_queries": 0,
        },
    )


def _new_state() -> dict[str, Any]:
    return {
        "gate_config": None,
        "enabled": True,
        "suite": "unknown",
        "camera_mode": CAMERA_MODE,
        "primary_guard_threshold": PRIMARY_GUARD_THRESHOLD,
        "consecutive_skip_limit": None,
        "force_full_interval": None,
        "force_full_after_1step": False,
        "query_lag": QUERY_LAG,
        "query_index": 0,
        "history": [],
        "current_latent_indices": {},
        "pending_images": None,
        "pending_features": None,
        "pending_skip": False,
        "pending_reuse_mode": None,
        "active_skip_for_call": False,
        "active_reuse_mode_for_call": None,
        "gate_pending_for_action": False,
        "consecutive_skips": 0,
        "primary_consecutive_skips": 0,
        "wrist_consecutive_skips": 0,
        "primary_consecutive_skip_limit": None,
        "wrist_consecutive_skip_limit": None,
        "task_description": "",
        "trace_queries": [],
    }


def _get_state(model: torch.nn.Module) -> dict[str, Any]:
    state = getattr(model, "_cosmos_shallowvaeskip_state", None)
    if state is None:
        state = _new_state()
        model._cosmos_shallowvaeskip_state = state
    return state


def _start_episode(model: torch.nn.Module, cfg, task_description: str) -> None:
    global _FLOP_PROFILE_ARGS
    if os.environ.get("COSMOS_SHALLOW_GATE_FLOP_PROFILE_ACTIVE") != "1":
        _FLOP_PROFILE_ARGS = None
    state = _new_state()
    state["gate_config"] = _gate_config(cfg)
    state["enabled"] = _gate_enabled(cfg)
    state["suite"] = _suite_candidates(cfg)[0]
    state["camera_mode"], state["primary_guard_threshold"] = _camera_config(cfg)
    state["consecutive_skip_limit"] = _consecutive_skip_limit(cfg)
    state["primary_consecutive_skip_limit"] = (
        PRIMARY_MAX_CONSECUTIVE_SKIPS if PRIMARY_MAX_CONSECUTIVE_SKIPS > 0 else None
    )
    state["wrist_consecutive_skip_limit"] = (
        WRIST_MAX_CONSECUTIVE_SKIPS if WRIST_MAX_CONSECUTIVE_SKIPS > 0 else None
    )
    state["force_full_interval"] = _force_full_interval(cfg)
    state["force_full_after_1step"] = _force_full_after_1step(cfg)
    state["query_lag"] = QUERY_LAG
    state["task_description"] = task_description
    model._cosmos_shallowvaeskip_state = state


def _write_trace_episode(state: dict[str, Any], success: bool) -> None:
    if not TRACE_PATH:
        return
    path = Path(TRACE_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "suite": state["suite"],
        "task": state["task_description"],
        "success": bool(success),
        "queries": state["trace_queries"],
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _tensor_index(data_batch: dict[str, Any], key: str) -> int:
    value = data_batch.get(key)
    if value is None:
        return -1
    if isinstance(value, torch.Tensor):
        return int(value.detach().flatten()[0].item())
    return int(value)


def _latent_frame(latent: torch.Tensor | None, frame_idx: int) -> torch.Tensor | None:
    if latent is None or frame_idx < 0 or latent.ndim != 5 or frame_idx >= latent.shape[2]:
        return None
    return latent[0, :, frame_idx].detach()


def _pool_feature(feature: torch.Tensor) -> torch.Tensor:
    feature = feature.detach()
    # Every gate probe contains exactly one camera frame.  Avoid launching a
    # reduction over a length-one temporal dimension on the latency-critical
    # path; retain the old reduction as a defensive fallback.
    if feature.shape[2] == 1:
        feature_2d = feature.squeeze(2).to(dtype=torch.float32)
    else:
        feature_2d = feature.to(dtype=torch.float32).mean(dim=2)
    return F.adaptive_avg_pool2d(feature_2d, (GRID_SIZE, GRID_SIZE))


def _probe_feature(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    with torch.inference_mode():
        # RGB already comes from the model's uploaded, normalized video batch.
        # Do not look up the VAE or coerce to its parameter dtype.  Pooling
        # remains FP32 so the gate decision is numerically identical.
        if layer_name == "rgb":
            return _pool_feature(image_tensor)

        encoder = model.tokenizer.model.model.encoder
        parameter = next(encoder.parameters())
        image_tensor = image_tensor.to(device=parameter.device, dtype=parameter.dtype)
        x = encoder.conv1(image_tensor)
        if layer_name == "conv1":
            return _pool_feature(x)

        target_index = LAYER_TO_DOWNSAMPLE_INDEX.get(layer_name)
        for layer_index, layer in enumerate(encoder.downsamples):
            x = layer(x)
            if target_index is not None and layer_index == target_index:
                return _pool_feature(x)

        middle_target = LAYER_TO_MIDDLE_INDEX.get(layer_name)
        for layer_index, layer in enumerate(encoder.middle):
            x = layer(x)
            if middle_target is not None and layer_index == middle_target:
                return _pool_feature(x)

        head_target = LAYER_TO_HEAD_INDEX.get(layer_name)
        for layer_index, layer in enumerate(encoder.head):
            x = layer(x)
            if head_target is not None and layer_index == head_target:
                return _pool_feature(x)

    raise RuntimeError(f"VAE encoder did not expose requested layer {layer_name}")


def _change_map_batch(current: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
    delta = torch.mean(
        torch.abs(current.to(dtype=torch.float32) - previous.to(dtype=torch.float32)),
        dim=1,
    )
    if tuple(delta.shape[-2:]) != (GRID_SIZE, GRID_SIZE):
        delta = F.adaptive_avg_pool2d(delta.unsqueeze(1), (GRID_SIZE, GRID_SIZE)).squeeze(1)
    return delta


def _change_map(current: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
    return _change_map_batch(current.unsqueeze(0), previous.unsqueeze(0))[0]


def _topk_iou_batch(
    real_maps: torch.Tensor,
    predicted_maps: torch.Tensor,
    topk_fraction: float,
) -> torch.Tensor:
    """Compute independent Top-K IoUs while batching camera-side kernels."""

    real_flat = real_maps.flatten(start_dim=1)
    predicted_flat = predicted_maps.flatten(start_dim=1)
    k = max(1, min(real_flat.shape[1], int(round(topk_fraction * real_flat.shape[1]))))
    real_indices = torch.topk(real_flat, k=k, dim=1, sorted=False).indices
    predicted_indices = torch.topk(predicted_flat, k=k, dim=1, sorted=False).indices
    predicted_mask = torch.zeros_like(predicted_flat, dtype=torch.bool)
    predicted_mask.scatter_(1, predicted_indices, True)
    intersection = predicted_mask.gather(1, real_indices).sum(dim=1)
    # torch.topk returns k distinct indices in each set, so |A union B| is
    # exactly 2k - |A intersection B|.  Avoid a second mask and two logical ops.
    union = (2 * k - intersection).clamp_min(1)
    return intersection.to(torch.float32) / union


def _topk_iou(real_map: torch.Tensor, predicted_map: torch.Tensor, topk_fraction: float) -> torch.Tensor:
    return _topk_iou_batch(real_map.unsqueeze(0), predicted_map.unsqueeze(0), topk_fraction)[0]


def _camera_gate(
    ious: dict[str, torch.Tensor],
    mode: str,
    threshold: float,
    primary_guard_threshold: float,
) -> bool:
    if mode == "min":
        values = torch.stack([ious["wrist"], ious["primary"]])
        score = values.min()
    elif mode == "mean":
        values = torch.stack([ious["wrist"], ious["primary"]])
        score = values.mean()
    elif mode in {"wrist", "wrist_guard"}:
        score = ious["wrist"]
    elif mode in {"hybrid", "primary"}:
        score = ious["primary"]
    else:
        raise RuntimeError(f"Unexpected camera mode {mode!r}")
    accepted = float(score.item()) >= threshold
    if mode == "hybrid":
        accepted = float(ious["primary"].item()) >= primary_guard_threshold
    elif mode == "wrist_guard":
        accepted = accepted and float(ious["primary"].item()) >= primary_guard_threshold
    return accepted


def _camera_specs(mode: str) -> tuple[tuple[str, str, str], ...]:
    if mode == "wrist":
        return (CAMERA_SPECS[0],)
    if mode == "primary":
        return (CAMERA_SPECS[1],)
    return CAMERA_SPECS


def _reuse_latent_frame_pairs(
    state: dict[str, Any],
    data_batch: dict[str, Any],
    reuse_mode: str,
) -> tuple[torch.Tensor, tuple[tuple[int, int], ...]] | None:
    history = state.get("history") or []
    query_lag = int(state.get("query_lag", 1))
    if len(history) < query_lag:
        return None
    source = history[-query_lag]
    previous = source.get("generated_latent")
    previous_indices = source.get("latent_indices") or {}
    current_indices = state.get("current_latent_indices") or {}
    if not isinstance(previous, torch.Tensor):
        return None

    frame_pairs = []
    camera_specs = (CAMERA_SPECS[1],) if reuse_mode == "primary" else CAMERA_SPECS
    for _, current_key, future_key in camera_specs:
        current_index = current_indices.get(current_key)
        if current_index is None:
            current_index = _tensor_index(data_batch, current_key)
        future_index = int(previous_indices.get(future_key, -1))
        if (
            current_index < 0
            or future_index < 0
            or current_index >= previous.shape[2]
            or future_index >= previous.shape[2]
        ):
            return None
        frame_pairs.append((current_index, future_index))
    return previous.detach(), tuple(frame_pairs)


def _encode_real_wrist_latent(
    model: torch.nn.Module,
    data_batch: dict[str, Any],
) -> tuple[int, torch.Tensor]:
    """Encode only the causal prefix through the wrist frame."""

    current_indices = _get_state(model).get("current_latent_indices") or {}
    current_index = current_indices.get("current_wrist_image_latent_idx")
    if current_index is None:
        current_index = _tensor_index(data_batch, "current_wrist_image_latent_idx")
    if current_index < 0:
        raise RuntimeError("Primary-camera reuse requires a current wrist image")
    video = data_batch.get("video")
    if not isinstance(video, torch.Tensor) or video.ndim != 5:
        raise RuntimeError("Primary-camera reuse requires a B,C,T,H,W video tensor")
    raw_end = _raw_frame_index(current_index) + TEMPORAL_COMPRESSION_FACTOR
    if raw_end > video.shape[2]:
        raise IndexError(f"Wrist prefix end {raw_end} exceeds video length {video.shape[2]}")
    prefix_latent = model.encode(video[:, :, :raw_end])
    if current_index >= prefix_latent.shape[2]:
        raise IndexError(
            f"Wrist latent index {current_index} exceeds prefix latent length "
            f"{prefix_latent.shape[2]}"
        )
    exact_wrist = prefix_latent[:, :, current_index].contiguous().float().detach()
    return current_index, exact_wrist


def _raw_frame_index(latent_index: int) -> int:
    if latent_index < 1:
        raise ValueError(f"Image latent index must be positive, got {latent_index}")
    return 1 + (latent_index - 1) * TEMPORAL_COMPRESSION_FACTOR


def _decide_gate(model: torch.nn.Module, data_batch: dict[str, Any]) -> dict[str, Any]:
    state = _get_state(model)
    gate_config = state.get("gate_config")
    if gate_config is None:
        raise RuntimeError("Shallow VAE skip state was not initialized for this episode")
    layer_name, topk_fraction, iou_threshold = gate_config
    state["query_index"] += 1
    summary = _summary_row(state["suite"])
    summary["queries"] += 1

    video = data_batch.get("video")
    if not isinstance(video, torch.Tensor) or video.ndim != 5:
        raise RuntimeError("Shallow VAE gate requires a B,C,T,H,W video tensor")
    if not torch.is_floating_point(video):
        raise RuntimeError("Shallow VAE gate must run after video normalization")

    camera_specs = _camera_specs(state["camera_mode"])
    current_indices = state.get("current_latent_indices") or {}
    raw_frame_indices = []
    for _, current_key, _ in camera_specs:
        latent_index = current_indices.get(current_key)
        if latent_index is None:
            # Camera latent positions are fixed for an episode.  Reading the
            # CUDA scalar once avoids two GPU-to-CPU synchronizations per query.
            latent_index = _tensor_index(data_batch, current_key)
            current_indices[current_key] = latent_index
        raw_frame_index = _raw_frame_index(latent_index)
        if raw_frame_index >= video.shape[2]:
            raise IndexError(
                f"Raw frame {raw_frame_index} for {current_key} exceeds video length {video.shape[2]}"
            )
        raw_frame_indices.append(raw_frame_index)
    state["current_latent_indices"] = current_indices

    # Reuse the model's already-uploaded and normalized video tensor. The selected
    # raw frames are the same duplicated camera frames used by the old T=1 probe.
    image_tensor = video[0, :, raw_frame_indices].permute(1, 0, 2, 3).unsqueeze(2)
    state["pending_images"] = image_tensor.detach()
    state["pending_features"] = None
    state["pending_skip"] = False
    state["pending_reuse_mode"] = None

    history = state.get("history") or []
    query_lag = int(state.get("query_lag", 1))
    source = history[-query_lag] if len(history) >= query_lag else None
    previous_images = source.get("images") if source is not None else None
    previous_generated = source.get("generated_latent") if source is not None else None
    previous_current = source.get("current_latents") or {} if source is not None else {}
    previous_indices = source.get("latent_indices") or {} if source is not None else {}

    if not isinstance(previous_images, torch.Tensor) or not isinstance(previous_generated, torch.Tensor):
        if TRACE_PATH:
            state["trace_queries"].append(
                {
                    "query_index": state["query_index"],
                    "available": False,
                    "reason": "insufficient_query_history",
                    "query_lag": query_lag,
                }
            )
        return state

    if state["camera_mode"] == "hybrid":
        consecutive_limit = state.get("primary_consecutive_skip_limit")
        consecutive_count = state["primary_consecutive_skips"]
    else:
        consecutive_limit = state.get("consecutive_skip_limit")
        consecutive_count = state["consecutive_skips"]
    blocked_by_consecutive_limit = (
        consecutive_limit is not None and consecutive_count >= consecutive_limit
    )
    force_full_interval = state.get("force_full_interval")
    blocked_by_periodic_full = (
        force_full_interval is not None
        and state["query_index"] % force_full_interval == 0
    )
    previous_actionaware_route = source.get("actionaware_route")
    blocked_by_previous_1step = (
        state.get("force_full_after_1step", False)
        and previous_actionaware_route == "1step"
    )
    blocked_by_guard = (
        blocked_by_consecutive_limit
        or blocked_by_periodic_full
        or blocked_by_previous_1step
    )
    if blocked_by_guard:
        summary["guard_blocked_queries"] += 1
    if blocked_by_previous_1step:
        summary["actionaware_guard_blocked_queries"] += 1
    if blocked_by_guard and not TRACE_PATH:
        return state

    # A forced-full query does not need a gate probe. Keep its two raw camera
    # frames and batch them with the next query. Otherwise probe only the two
    # current T=1 camera images and reuse the previous query's cached features.
    previous_features = source.get("features")
    camera_count = len(camera_specs)
    if isinstance(previous_features, dict) and all(
        camera in previous_features for camera, _, _ in camera_specs
    ):
        feature_batch = _probe_feature(model, image_tensor, layer_name)
        features = {
            camera: feature_batch[feature_index]
            for feature_index, (camera, _, _) in enumerate(camera_specs)
        }
    else:
        if previous_images.shape != image_tensor.shape:
            return state
        feature_batch = _probe_feature(
            model,
            torch.cat((previous_images, image_tensor), dim=0),
            layer_name,
        )
        previous_features = {
            camera: feature_batch[feature_index]
            for feature_index, (camera, _, _) in enumerate(camera_specs)
        }
        features = {
            camera: feature_batch[camera_count + feature_index]
            for feature_index, (camera, _, _) in enumerate(camera_specs)
        }
    state["pending_features"] = features

    if TRACE_PATH:
        change_maps: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for camera, _, future_key in camera_specs:
            current_token = previous_current.get(camera)
            future_index = int(previous_indices.get(future_key, -1))
            predicted_future = _latent_frame(previous_generated, future_index)
            if current_token is None or predicted_future is None:
                state["trace_queries"].append(
                    {"query_index": state["query_index"], "available": False, "reason": "missing_latent"}
                )
                return state
            change_maps[camera] = (
                _change_map(features[camera], previous_features[camera]),
                _change_map(predicted_future, current_token),
            )
        fractions = tuple(dict.fromkeys((topk_fraction, *ANALYSIS_TOPKS)))
        ious_by_fraction = {
            fraction: {
                camera: _topk_iou(*change_maps[camera], fraction)
                for camera, _, _ in camera_specs
            }
            for fraction in fractions
        }
        ious = ious_by_fraction[topk_fraction]
    else:
        current_features = []
        previous_feature_batch = []
        predicted_futures = []
        current_latents = []
        for camera, _, future_key in camera_specs:
            current_token = previous_current.get(camera)
            future_index = int(previous_indices.get(future_key, -1))
            predicted_future = _latent_frame(previous_generated, future_index)
            if current_token is None or predicted_future is None:
                return state
            current_features.append(features[camera])
            previous_feature_batch.append(previous_features[camera])
            predicted_futures.append(predicted_future)
            current_latents.append(current_token)
        real_maps = _change_map_batch(
            torch.stack(current_features),
            torch.stack(previous_feature_batch),
        )
        predicted_maps = _change_map_batch(
            torch.stack(predicted_futures),
            torch.stack(current_latents),
        )
        batched_ious = _topk_iou_batch(
            real_maps,
            predicted_maps,
            topk_fraction,
        )
        ious = {
            camera: batched_ious[index]
            for index, (camera, _, _) in enumerate(camera_specs)
        }

    selected_reuse_mode = "all"
    if state["camera_mode"] == "hybrid":
        # Resolve both hybrid thresholds with one device synchronization.
        primary_ok, wrist_ok = torch.stack(
            (
                ious["primary"] >= state["primary_guard_threshold"],
                ious["wrist"] >= iou_threshold,
            )
        ).tolist()
        eligible = bool(primary_ok)
        wrist_limit = state.get("wrist_consecutive_skip_limit")
        wrist_limit_reached = (
            wrist_limit is not None
            and state["wrist_consecutive_skips"] >= wrist_limit
        )
        selected_reuse_mode = "all" if wrist_ok and not wrist_limit_reached else "primary"
        if primary_ok and wrist_ok and wrist_limit_reached:
            summary["wrist_limit_downgrade_queries"] += 1
    else:
        eligible = _camera_gate(
            ious,
            state["camera_mode"],
            iou_threshold,
            state["primary_guard_threshold"],
        )
    if TRACE_PATH:
        scores = {}
        for fraction, fraction_ious in ious_by_fraction.items():
            values = {camera: float(value.detach().cpu().item()) for camera, value in fraction_ious.items()}
            camera_values = list(values.values())
            values["min"] = min(camera_values)
            values["mean"] = sum(camera_values) / len(camera_values)
            scores[f"{fraction:g}"] = values
        state["trace_queries"].append(
            {
                "query_index": state["query_index"],
                "query_lag": query_lag,
                "available": True,
                "blocked_by_guard": blocked_by_guard,
                "blocked_by_previous_1step": blocked_by_previous_1step,
                "previous_actionaware_route": previous_actionaware_route,
                "eligible": eligible,
                "selected_threshold": iou_threshold,
                "primary_threshold": state["primary_guard_threshold"],
                "selected_reuse_mode": selected_reuse_mode if eligible else None,
                "topk_scores": scores,
            }
        )
    if blocked_by_guard:
        return state
    if not eligible:
        return state
    summary["gate_eligible_queries"] += 1

    if not DISABLE_REUSE:
        state["pending_skip"] = True
        state["pending_reuse_mode"] = selected_reuse_mode
    return state


def _finalize_call(model: torch.nn.Module, result: dict[str, Any]) -> None:
    state = _get_state(model)
    summary = _summary_row(state["suite"])
    if state.pop("active_skip_for_call", False):
        state["consecutive_skips"] += 1
        state["primary_consecutive_skips"] += 1
        summary["skipped_queries"] += 1
        active_reuse_mode = state.pop("active_reuse_mode_for_call", None)
        if active_reuse_mode == "primary":
            # Primary-only reuse performs an exact wrist encode, so the wrist
            # reuse streak ends even while the primary streak continues.
            state["wrist_consecutive_skips"] = 0
            summary["primary_reuse_queries"] += 1
        elif active_reuse_mode == "all":
            state["wrist_consecutive_skips"] += 1
            summary["all_reuse_queries"] += 1
    else:
        state["active_reuse_mode_for_call"] = None
        state["consecutive_skips"] = 0
        state["primary_consecutive_skips"] = 0
        state["wrist_consecutive_skips"] = 0
        summary["full_queries"] += 1

    data_batch = result.get("data_batch") or {}
    condition_latent = result.get("orig_clean_latent_frames")
    generated_latent = result.get("generated_latent")
    current_indices = state.get("current_latent_indices") or {}
    current_latents: dict[str, torch.Tensor] = {}
    for camera, current_key, _ in CAMERA_SPECS:
        current_index = current_indices.get(current_key)
        if current_index is None:
            current_index = _tensor_index(data_batch, current_key)
            current_indices[current_key] = current_index
        latent = _latent_frame(condition_latent, current_index)
        if latent is not None:
            current_latents[camera] = latent

    state["current_latent_indices"] = current_indices
    history = state.setdefault("history", [])
    history.append(
        {
            "images": state.get("pending_images"),
            "features": state.get("pending_features"),
            "current_latents": current_latents,
            "generated_latent": generated_latent.detach() if isinstance(generated_latent, torch.Tensor) else None,
            "latent_indices": dict(result.get("latent_indices") or {}),
            "actionaware_route": getattr(
                getattr(model, "sampler", None),
                "_cosmos_actionaware_last_route",
                None,
            ),
        }
    )
    max_history = max(int(state.get("query_lag", 1)), 1)
    if len(history) > max_history:
        del history[:-max_history]

    state["pending_images"] = None
    state["pending_features"] = None
    state["pending_skip"] = False
    state["pending_reuse_mode"] = None


def _install_model_patch(model: torch.nn.Module) -> None:
    if hasattr(model, "_cosmos_shallowvaeskip_original_get_x0_fn_from_batch"):
        return
    model._cosmos_shallowvaeskip_original_get_x0_fn_from_batch = model.get_x0_fn_from_batch

    def get_x0_fn_from_batch_with_shallowvaeskip(
        self,
        data_batch: dict[str, Any],
        *args,
        **kwargs,
    ):
        state = _get_state(self)
        is_action_query = state.pop("gate_pending_for_action", False)
        if is_action_query:
            state["active_skip_for_call"] = False
            state["active_reuse_mode_for_call"] = None
            _decide_gate(self, data_batch)
        reuse = None
        if is_action_query and state.get("pending_skip"):
            active_reuse_mode = state.get("pending_reuse_mode") or "all"
            reuse = _reuse_latent_frame_pairs(state, data_batch, active_reuse_mode)
            if reuse is not None:
                previous_latent, frame_pairs = reuse
                exact_pairs = ()
                if active_reuse_mode == "primary":
                    exact_pairs = (_encode_real_wrist_latent(self, data_batch),)
                kwargs["skip_vae_encoding"] = True
                kwargs["previous_generated_latent"] = previous_latent
                state["active_skip_for_call"] = True
                state["active_reuse_mode_for_call"] = active_reuse_mode
            else:
                state["pending_skip"] = False
                state["pending_reuse_mode"] = None

        if reuse is None:
            return self._cosmos_shallowvaeskip_original_get_x0_fn_from_batch(
                data_batch,
                *args,
                **kwargs,
            )

        # The model immediately clones previous_latent on its skip path. Replace
        # only the two destination frames until that call returns, then restore them.
        # This avoids a second full-latent clone without changing model code.
        backups = {
            destination: previous_latent[:, :, destination].clone()
            for destination, _ in (*frame_pairs, *exact_pairs)
        }
        try:
            for destination, source in frame_pairs:
                previous_latent[:, :, destination].copy_(previous_latent[:, :, source])
            for destination, exact_latent in exact_pairs:
                previous_latent[:, :, destination].copy_(exact_latent)
            return self._cosmos_shallowvaeskip_original_get_x0_fn_from_batch(
                data_batch,
                *args,
                **kwargs,
            )
        finally:
            for destination, backup in backups.items():
                previous_latent[:, :, destination].copy_(backup)

    model.get_x0_fn_from_batch = types.MethodType(
        get_x0_fn_from_batch_with_shallowvaeskip,
        model,
    )


def _install_module_patches(module) -> None:
    original_get_action = module.get_action
    original_run_episode = module.run_episode

    def get_action_with_shallowvaeskip(
        cfg,
        model,
        dataset_stats,
        obs,
        task_label_or_embedding,
        seed=1,
        randomize_seed=False,
        num_denoising_steps_action=5,
        generate_future_state_and_value_in_parallel=True,
        worker_id=0,
        batch_size=1,
    ):
        global _FLOP_PROFILE_ARGS
        _FLOP_PROFILE_ARGS = (
            (cfg, model, dataset_stats, obs, task_label_or_embedding),
            {
                "seed": seed,
                "randomize_seed": randomize_seed,
                "num_denoising_steps_action": num_denoising_steps_action,
                "generate_future_state_and_value_in_parallel": generate_future_state_and_value_in_parallel,
                "worker_id": worker_id,
                "batch_size": batch_size,
            },
        )
        state = _get_state(model)
        if not state["enabled"]:
            return original_get_action(
                cfg,
                model,
                dataset_stats,
                obs,
                task_label_or_embedding,
                seed=seed,
                randomize_seed=randomize_seed,
                num_denoising_steps_action=num_denoising_steps_action,
                generate_future_state_and_value_in_parallel=generate_future_state_and_value_in_parallel,
                worker_id=worker_id,
                batch_size=batch_size,
            )
        state["gate_pending_for_action"] = True
        try:
            result = original_get_action(
                cfg,
                model,
                dataset_stats,
                obs,
                task_label_or_embedding,
                seed=seed,
                randomize_seed=randomize_seed,
                num_denoising_steps_action=num_denoising_steps_action,
                generate_future_state_and_value_in_parallel=generate_future_state_and_value_in_parallel,
                worker_id=worker_id,
                batch_size=batch_size,
            )
            _finalize_call(model, result)
            return result
        except Exception:
            state["pending_images"] = None
            state["pending_skip"] = False
            state["pending_reuse_mode"] = None
            state["active_skip_for_call"] = False
            state["active_reuse_mode_for_call"] = None
            state["gate_pending_for_action"] = False
            raise

    def run_episode_with_shallowvaeskip(*episode_args, **episode_kwargs):
        if len(episode_args) < 4:
            raise RuntimeError("Shallow VAE skip expected run_episode(cfg, env, task, model, ...)")
        cfg, _, task_description, model = episode_args[:4]
        _start_episode(model, cfg, task_description)
        result = original_run_episode(*episode_args, **episode_kwargs)
        _write_trace_episode(_get_state(model), bool(result[0]))
        return result

    module.get_action = get_action_with_shallowvaeskip
    module.run_episode = run_episode_with_shallowvaeskip


def _set_flop_profile_callable(callable_) -> None:
    """Capture the fully wrapped get_action after every accelerator is installed."""

    global _FLOP_PROFILE_CALLABLE
    _FLOP_PROFILE_CALLABLE = callable_


def _profile_flops_outside_querytime() -> str:
    """Profile one post-evaluation action query and persist its operator FLOPs."""

    output = os.environ.get("COSMOS_SHALLOW_GATE_FLOPS_PATH", "").strip()
    if not output:
        return ""
    path = Path(output).expanduser().resolve()
    payload: dict[str, Any] = {
        "status": "unavailable",
        "scope": "one representative post-evaluation action query",
        "included_in_querytime": False,
        "layer": LAYER,
    }
    try:
        if _FLOP_PROFILE_CALLABLE is None or _FLOP_PROFILE_ARGS is None:
            raise RuntimeError("no completed action query was captured")
        call_args, call_kwargs = _FLOP_PROFILE_ARGS
        from torch.profiler import ProfilerActivity, profile

        os.environ["COSMOS_SHALLOW_GATE_FLOP_PROFILE_ACTIVE"] = "1"
        try:
            torch.cuda.synchronize()
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                with_flops=True,
                record_shapes=False,
                profile_memory=False,
                with_stack=False,
            ) as prof:
                _FLOP_PROFILE_CALLABLE(*call_args, **call_kwargs)
            torch.cuda.synchronize()
        finally:
            os.environ.pop("COSMOS_SHALLOW_GATE_FLOP_PROFILE_ACTIVE", None)
        total_flops = int(sum(int(getattr(event, "flops", 0) or 0) for event in prof.key_averages()))
        payload.update(
            {
                "status": "ok",
                "flops": total_flops,
                "tflops": total_flops / 1.0e12,
                "note": "PyTorch operator FLOPs; profiler ran after all timed episodes",
            }
        )
    except Exception as exc:
        payload["error"] = f"{type(exc).__name__}: {exc}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if payload["status"] == "ok":
        return (
            f"[ShallowVAESkip FLOPs] layer={LAYER} "
            f"flops={payload['flops']} tflops={payload['tflops']:.6f} "
            "scope=post_eval_query included_in_querytime=false\n"
        )
    return (
        f"[ShallowVAESkip FLOPs] layer={LAYER} status=unavailable "
        f"error={payload.get('error', 'unknown')} included_in_querytime=false\n"
    )


from cosmos_policy.experiments.robot import cosmos_utils

_ORIGINAL_GET_MODEL = cosmos_utils.get_model


def _get_model_with_shallowvaeskip(cfg):
    model, config = _ORIGINAL_GET_MODEL(cfg)
    _install_model_patch(model)
    return model, config


cosmos_utils.get_model = _get_model_with_shallowvaeskip

import eval_wrapper

_ORIGINAL_PATCH_RUNTIME = eval_wrapper.patch_runtime


def _patch_runtime_with_shallowvaeskip(module, run_vae_decoder, rollout_videos):
    _ORIGINAL_PATCH_RUNTIME(module, run_vae_decoder, rollout_videos)
    _install_module_patches(module)


eval_wrapper.patch_runtime = _patch_runtime_with_shallowvaeskip


def _write_summary() -> None:
    if not SUMMARY_PATH:
        return
    suites = {}
    total = {
        "queries": 0,
        "gate_eligible_queries": 0,
        "guard_blocked_queries": 0,
        "actionaware_guard_blocked_queries": 0,
        "wrist_limit_downgrade_queries": 0,
        "skipped_queries": 0,
        "primary_reuse_queries": 0,
        "all_reuse_queries": 0,
        "full_queries": 0,
    }
    for suite, counters in sorted(_SUMMARY.items()):
        row = dict(counters)
        row["skip_ratio"] = row["skipped_queries"] / max(row["queries"], 1)
        row["gate_eligible_ratio"] = row["gate_eligible_queries"] / max(row["queries"], 1)
        suites[suite] = row
        for key in total:
            total[key] += counters[key]
    total["skip_ratio"] = total["skipped_queries"] / max(total["queries"], 1)
    total["gate_eligible_ratio"] = total["gate_eligible_queries"] / max(total["queries"], 1)
    payload = {
        "config": {
            "layer": LAYER,
            "topk_fraction": TOPK_FRACTION,
            "iou_threshold": IOU_THRESHOLD,
            "camera_mode": CAMERA_MODE,
            "primary_guard_threshold": PRIMARY_GUARD_THRESHOLD,
            "max_consecutive_skips": MAX_CONSECUTIVE_SKIPS,
            "primary_max_consecutive_skips": PRIMARY_MAX_CONSECUTIVE_SKIPS,
            "wrist_max_consecutive_skips": WRIST_MAX_CONSECUTIVE_SKIPS,
            "query_lag": QUERY_LAG,
            "force_full_interval": FORCE_FULL_INTERVAL,
            "force_full_after_1step": FORCE_FULL_AFTER_1STEP,
            "suite_overrides": {
                "layer": SUITE_LAYERS,
                "topk_fraction": SUITE_TOPK,
                "iou_threshold": SUITE_IOU_THRESHOLDS,
                "camera_mode": SUITE_CAMERA_MODES,
                "primary_guard_threshold": SUITE_PRIMARY_GUARD_THRESHOLDS,
                "max_consecutive_skips": SUITE_MAX_CONSECUTIVE_SKIPS,
                "force_full_interval": SUITE_FORCE_FULL_INTERVALS,
                "force_full_after_1step": SUITE_FORCE_FULL_AFTER_1STEP,
                "enabled": SUITE_ENABLED,
            },
        },
        "suites": suites,
        "all": total,
    }
    path = Path(SUMMARY_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "[ShallowVAESkip summary] "
        f"skip={total['skipped_queries']}/{total['queries']}={total['skip_ratio']:.2%}; "
        f"reuse(primary/all)={total['primary_reuse_queries']}/{total['all_reuse_queries']}; "
        f"wrist_limit_downgrades={total['wrist_limit_downgrade_queries']}; "
        f"gate_eligible={total['gate_eligible_ratio']:.2%}; path={path}"
    )


if __name__ == "__main__":
    eval_wrapper.main()
    _write_summary()
