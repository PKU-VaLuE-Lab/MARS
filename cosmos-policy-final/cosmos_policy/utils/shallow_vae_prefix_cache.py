"""Resumable Wan VAE encoder prefixes for the VAESkip feature gate.

This module is imported only by the LIBERO/LIBERO-PRO VAESkip wrappers.  It
splits the Wan encoder at the selected shallow feature layer, retains the
activation and temporal convolution state, and resumes the exact same encoder
invocation when the gate asks for a full encode.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


_DOWNSAMPLE_NAMES = {
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
_MIDDLE_NAMES = {"middle_res0": 0, "middle_attn": 1, "middle_res1": 2}
_HEAD_NAMES = {"head_norm": 0, "head_act": 1, "encoder_out": 2}


@dataclass
class EncoderPrefixCache:
    raw_vae: Any
    backend: Any
    ops: tuple[tuple[str, Any], ...]
    target_op: int
    prefix_feat_idx: int
    feat_cache: list[Any]
    activations: list[torch.Tensor]
    next_chunk: int
    input_shape: tuple[int, ...]
    input_device: torch.device
    input_dtype: torch.dtype

    def matches(self, value: torch.Tensor) -> bool:
        return (
            tuple(value.shape) == self.input_shape
            and value.device == self.input_device
            and value.dtype == self.input_dtype
        )


def _encoder_ops(encoder: torch.nn.Module) -> tuple[tuple[str, Any], ...]:
    ops: list[tuple[str, Any]] = [("cached_causal", encoder.conv1)]
    for layer in encoder.downsamples:
        kind = "cached_module" if layer.__class__.__name__ in {"ResidualBlock", "Resample"} else "plain"
        ops.append((kind, layer))
    for layer in encoder.middle:
        kind = "cached_module" if layer.__class__.__name__ == "ResidualBlock" else "plain"
        ops.append((kind, layer))
    for layer in encoder.head:
        kind = "cached_causal" if layer.__class__.__name__ == "CausalConv3d" else "plain"
        ops.append((kind, layer))
    return tuple(ops)


def _target_op(encoder: torch.nn.Module, layer_name: str) -> int:
    if layer_name == "conv1":
        return 0
    if layer_name in _DOWNSAMPLE_NAMES:
        return 1 + _DOWNSAMPLE_NAMES[layer_name]
    if layer_name in _MIDDLE_NAMES:
        return 1 + len(encoder.downsamples) + _MIDDLE_NAMES[layer_name]
    if layer_name in _HEAD_NAMES:
        return 1 + len(encoder.downsamples) + len(encoder.middle) + _HEAD_NAMES[layer_name]
    raise ValueError(f"Layer {layer_name!r} is not a resumable shallow VAE layer")


def _chunks(value: torch.Tensor, temporal_window: int) -> tuple[torch.Tensor, ...]:
    # This exactly mirrors WanVAE_.encode: one leading frame followed by
    # temporal_window-sized causal chunks and one optional remainder.
    chunks = [value[:, :, :1]]
    start = 1
    while start < value.shape[2]:
        chunks.append(value[:, :, start : start + temporal_window])
        start += temporal_window
    return tuple(chunks)


def _cached_causal(
    module: torch.nn.Module,
    value: torch.Tensor,
    feat_cache: list[Any],
    feat_idx: list[int],
) -> torch.Tensor:
    idx = feat_idx[0]
    cache_value = value[:, :, -2:].clone()
    previous = feat_cache[idx]
    if cache_value.shape[2] < 2 and previous is not None:
        cache_value = torch.cat((previous[:, :, -1:].to(cache_value.device), cache_value), dim=2)
    value = module(value, previous)
    feat_cache[idx] = cache_value
    feat_idx[0] += 1
    return value


def _run_ops(
    value: torch.Tensor,
    ops: tuple[tuple[str, Any], ...],
    start: int,
    stop: int,
    feat_cache: list[Any],
    feat_idx: list[int],
) -> torch.Tensor:
    for kind, module in ops[start:stop]:
        if kind == "cached_causal":
            value = _cached_causal(module, value, feat_cache, feat_idx)
        elif kind == "cached_module":
            value = module(value, feat_cache, feat_idx)
        else:
            value = module(value)
    return value


@torch.inference_mode()
def prepare_encoder_prefix(
    policy_model: torch.nn.Module,
    video: torch.Tensor,
    layer_name: str,
    required_frame_indices: tuple[int, ...],
) -> tuple[EncoderPrefixCache, tuple[torch.Tensor, ...]]:
    """Run through ``layer_name`` only as far as the gate-required frames.

    A Wan encoder invocation may process several latent frames in one temporal
    window (the policy checkpoints use a window larger than the VAE temporal
    compression factor).  Consequently, a policy latent index is not an
    encoder chunk index.  The caller passes raw-video frame positions here and
    this function maps them to both the containing encoder chunk and the
    selected layer's temporal coordinate.
    """

    tokenizer = policy_model.tokenizer
    backend = tokenizer.model
    if getattr(backend, "is_parallel", False):
        raise RuntimeError("Resumable shallow VAESkip does not support context-parallel VAE encoding")
    raw_vae = backend.model
    encoder = raw_vae.encoder
    parameter = next(encoder.parameters())
    encoded_input = video.to(device=parameter.device, dtype=parameter.dtype)
    chunks = _chunks(encoded_input, int(raw_vae.temporal_window))
    if not required_frame_indices:
        raise RuntimeError("Shallow VAESkip requires at least one camera frame")
    if min(required_frame_indices) < 0 or max(required_frame_indices) >= encoded_input.shape[2]:
        raise IndexError(
            f"Requested VAE frames {required_frame_indices} outside input with "
            f"{encoded_input.shape[2]} frames"
        )

    ops = _encoder_ops(encoder)
    target = _target_op(encoder, layer_name)
    # The stock WanVAE_ lazily discovers the encoder-cache size in
    # clear_cache().  Initialize that metadata once; subsequent prefix calls
    # use their own local cache and must not rebuild the unused global
    # encoder/decoder cache arrays on every query.
    if not hasattr(raw_vae, "_enc_conv_num"):
        raw_vae.clear_cache()
    feat_cache: list[Any] = [None] * int(raw_vae._enc_conv_num)
    activations: list[torch.Tensor] = []
    prefix_feat_idx: int | None = None
    temporal_window = int(raw_vae.temporal_window)
    last_frame = max(required_frame_indices)
    last_required = 0 if last_frame == 0 else 1 + (last_frame - 1) // temporal_window

    with backend.context:
        for chunk_index in range(last_required + 1):
            feat_idx = [0]
            activation = _run_ops(chunks[chunk_index], ops, 0, target + 1, feat_cache, feat_idx)
            if prefix_feat_idx is None:
                prefix_feat_idx = feat_idx[0]
            elif prefix_feat_idx != feat_idx[0]:
                raise RuntimeError("Wan encoder prefix consumed an inconsistent number of temporal cache slots")
            activations.append(activation)

    cache = EncoderPrefixCache(
        raw_vae=raw_vae,
        backend=backend,
        ops=ops,
        target_op=target,
        prefix_feat_idx=int(prefix_feat_idx or 0),
        feat_cache=feat_cache,
        activations=activations,
        next_chunk=last_required + 1,
        input_shape=tuple(encoded_input.shape),
        input_device=encoded_input.device,
        input_dtype=encoded_input.dtype,
    )
    # Determine the temporal resolution at the requested cut.  Downsample3d
    # halves only the post-leading-frame timeline; all six formal ablation cuts
    # are before the first temporal downsample, while this general mapping also
    # keeps the other explicitly supported cuts correct.
    temporal_factor = 1
    for _, module in ops[: target + 1]:
        if getattr(module, "mode", None) == "downsample3d":
            temporal_factor *= 2

    selected = []
    for raw_frame_index in required_frame_indices:
        feature_index = (
            0
            if raw_frame_index == 0
            else 1 + (raw_frame_index - 1) // temporal_factor
        )
        remaining = feature_index
        for activation in activations:
            if remaining < activation.shape[2]:
                selected.append(activation[:, :, remaining : remaining + 1])
                break
            remaining -= activation.shape[2]
        else:
            raise IndexError(
                f"Feature frame {feature_index} for raw frame {raw_frame_index} "
                f"is outside the cached {layer_name} activation"
            )
    return cache, tuple(selected)


@torch.inference_mode()
def resume_encoder_prefix(
    cache: EncoderPrefixCache,
    video: torch.Tensor,
    scale: Any,
    clear_encoder_cache: bool,
) -> torch.Tensor:
    """Finish a Wan encode from a cached gate layer without rerunning it."""

    if not cache.matches(video):
        raise RuntimeError("Cached shallow VAE prefix does not match the full-encode input")
    raw_vae = cache.raw_vae
    chunks = _chunks(video, int(raw_vae.temporal_window))

    with cache.backend.context:
        # Complete the selected prefix for chunks after the latest camera.
        for chunk_index in range(cache.next_chunk, len(chunks)):
            feat_idx = [0]
            activation = _run_ops(
                chunks[chunk_index],
                cache.ops,
                0,
                cache.target_op + 1,
                cache.feat_cache,
                feat_idx,
            )
            if feat_idx[0] != cache.prefix_feat_idx:
                raise RuntimeError("Wan encoder prefix cache position changed while resuming")
            cache.activations.append(activation)

        # Continue every temporal chunk through the suffix in its original
        # order, preserving the suffix layers' causal feature caches.
        outputs = []
        for activation in cache.activations:
            feat_idx = [cache.prefix_feat_idx]
            output = _run_ops(
                activation,
                cache.ops,
                cache.target_op + 1,
                len(cache.ops),
                cache.feat_cache,
                feat_idx,
            )
            outputs.append(output)
        encoded = torch.cat(outputs, dim=2)
        mu, _ = raw_vae.conv1(encoded).chunk(2, dim=1)
        if isinstance(scale[0], torch.Tensor):
            mu = (mu - scale[0].view(1, raw_vae.z_dim, 1, 1, 1)) * scale[1].view(
                1, raw_vae.z_dim, 1, 1, 1
            )
        else:
            mu = (mu - scale[0]) * scale[1]

    if clear_encoder_cache:
        raw_vae.clear_cache()
    return mu
