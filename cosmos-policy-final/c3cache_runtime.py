#!/usr/bin/env python3
"""Cross-query, same-denoising-step DiT residual reuse for Cosmos-Policy.

C3Cache deliberately patches only the 28-block DiT stack. Every query still
recomputes input/conditioning preparation, timestep and patch embeddings, the
final layer, unpatchify, the VAE path, and the sampler update.
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


def _parse_cached_steps(raw: str, num_steps: int) -> tuple[int, ...]:
    normalized = raw.strip().removeprefix("[").removesuffix("]").strip()
    if not normalized:
        return ()
    values = tuple(int(item.strip()) for item in normalized.split(",") if item.strip())
    if len(values) != len(set(values)):
        raise ValueError(f"COSMOS_C3CACHE_CACHED_STEPS contains duplicates: {raw!r}")
    if any(step < 0 or step >= num_steps for step in values):
        raise ValueError(
            "COSMOS_C3CACHE_CACHED_STEPS must be within "
            f"[0,{num_steps - 1}], got {values}"
        )
    return tuple(sorted(values))


C3CACHE_NUM_STEPS = _env_int("COSMOS_C3CACHE_NUM_STEPS", 5)
if C3CACHE_NUM_STEPS != 5:
    raise ValueError(
        "C3Cache is defined against Cosmos-Policy's original five-step action "
        f"sampler, got COSMOS_C3CACHE_NUM_STEPS={C3CACHE_NUM_STEPS}"
    )
C3CACHE_CACHED_STEPS = _parse_cached_steps(
    os.environ.get("COSMOS_C3CACHE_CACHED_STEPS", "0,1,2,3"),
    C3CACHE_NUM_STEPS,
)
C3CACHE_REFRESH_INTERVAL = _env_int("COSMOS_C3CACHE_REFRESH_INTERVAL", 8)
if C3CACHE_REFRESH_INTERVAL < 0:
    raise ValueError("COSMOS_C3CACHE_REFRESH_INTERVAL must be non-negative")
C3CACHE_VERBOSE = _env_bool("COSMOS_C3CACHE_VERBOSE", True)
C3CACHE_REPORT_EVERY = _env_int("COSMOS_C3CACHE_REPORT_EVERY", 200)
C3CACHE_STATS_PATH = os.environ.get("COSMOS_C3CACHE_STATS_PATH")

_INSTALLED_MODELS: list[torch.nn.Module] = []


def _write_jsonl(payload: dict[str, Any]) -> None:
    if not C3CACHE_STATS_PATH:
        return
    path = Path(C3CACHE_STATS_PATH).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")


def _new_state() -> dict[str, Any]:
    return {
        "active": False,
        "cache": {},
        "current_query": None,
        "current_step": 0,
        "episode_query_index": 0,
        "episode_resets": 0,
        "queries": 0,
        "refresh_queries": 0,
        "full_calls": 0,
        "cached_calls": 0,
        "fallback_full_calls": 0,
        "cache_entries_cleared": 0,
        "per_step_full": [0] * C3CACHE_NUM_STEPS,
        "per_step_cached": [0] * C3CACHE_NUM_STEPS,
    }


def _get_state(net: torch.nn.Module) -> dict[str, Any]:
    state = getattr(net, "_cosmos_c3cache_state", None)
    if state is None:
        state = _new_state()
        net._cosmos_c3cache_state = state
    return state


def reset_episode(model: torch.nn.Module) -> None:
    """Clear all per-step residuals at the authoritative episode boundary."""

    net = getattr(model, "net", None)
    if net is None or not getattr(net, "_cosmos_c3cache_installed", False):
        raise RuntimeError("C3Cache episode reset expected an installed model.net")
    state = _get_state(net)
    if state["active"]:
        raise RuntimeError("C3Cache cannot reset while an action query is active")
    cleared = len(state["cache"])
    state["cache"].clear()
    state["active"] = False
    state["current_query"] = None
    state["current_step"] = 0
    state["episode_query_index"] = 0
    state["episode_resets"] += 1
    state["cache_entries_cleared"] += cleared
    if C3CACHE_VERBOSE and state["episode_resets"] <= 3:
        print(
            "[C3Cache] episode reset: "
            f"episode={state['episode_resets']} cleared_step_caches={cleared}"
        )


def _is_refresh_query(query_index: int) -> bool:
    if query_index == 0:
        return True
    return C3CACHE_REFRESH_INTERVAL > 0 and query_index % C3CACHE_REFRESH_INTERVAL == 0


def _c3cache_forward(
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
        return self._cosmos_c3cache_original_forward(
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
    if step_index >= C3CACHE_NUM_STEPS:
        raise RuntimeError(
            "C3Cache observed more than five DiT forwards in one action query: "
            f"step_index={step_index}"
        )

    # These calls do not occur in the original LIBERO action-query protocol.
    # Preserve the untouched path if a caller requests them.
    if intermediate_feature_ids or action is not None:
        state["full_calls"] += 1
        state["fallback_full_calls"] += 1
        state["per_step_full"][step_index] += 1
        return self._cosmos_c3cache_original_forward(
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
        raise TypeError(f"C3Cache expected DataType, got {type(data_type)}")

    # Everything before the DiT block stack is recomputed for every step.
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
            f"C3Cache positional embedding shape mismatch: {x_B_T_H_W_D.shape} != "
            f"{extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D.shape}"
        )

    h0 = x_B_T_H_W_D
    query_context = state["current_query"]
    if query_context is None:
        raise RuntimeError("C3Cache active forward is missing query context")
    refresh_query = bool(query_context["refresh"])
    cached_residual = state["cache"].get(step_index)
    cache_compatible = (
        cached_residual is not None
        and cached_residual.shape == h0.shape
        and cached_residual.dtype == h0.dtype
        and cached_residual.device == h0.device
    )
    use_cache = (
        not refresh_query
        and step_index in C3CACHE_CACHED_STEPS
        and cache_compatible
    )

    if use_cache:
        # The only skipped computation: all 28 DiT blocks.
        x_B_T_H_W_D = h0 + cached_residual
        state["cached_calls"] += 1
        state["per_step_cached"][step_index] += 1
    else:
        for block in self.blocks:
            x_B_T_H_W_D = block(
                x_B_T_H_W_D,
                t_embedding_B_T_D,
                context_input,
                rope_emb_L_1_1_D=rope_emb_L_1_1_D,
                adaln_lora_B_T_3D=adaln_lora_B_T_3D,
                extra_per_block_pos_emb=extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D,
            )
        state["full_calls"] += 1
        state["per_step_full"][step_index] += 1
        # Only refresh queries may replace residuals; a cached step therefore
        # always uses the cache from the most recent refresh query.
        if refresh_query:
            state["cache"][step_index] = (x_B_T_H_W_D - h0).detach()

    # Final projection and unpatchify are always current-query computations.
    x_B_T_H_W_O = self.final_layer(
        x_B_T_H_W_D,
        t_embedding_B_T_D,
        adaln_lora_B_T_3D=adaln_lora_B_T_3D,
    )
    return self.unpatchify(x_B_T_H_W_O)


def _install_on_model(model: torch.nn.Module) -> None:
    net = getattr(model, "net", None)
    if net is None or not hasattr(net, "blocks") or not hasattr(net, "final_layer"):
        raise RuntimeError("C3Cache expected model.net with blocks and final_layer")
    if len(net.blocks) != 28:
        raise RuntimeError(f"C3Cache expected exactly 28 DiT blocks, got {len(net.blocks)}")
    incompatible_markers = (
        "_cosmos_dicache_original_forward",
        "_cosmos_stepskip_original_forward",
        "_cosmos_c3cache_original_forward",
    )
    present = [name for name in incompatible_markers if hasattr(net, name)]
    if present:
        raise RuntimeError(f"C3Cache refuses an already accelerated DiT: {present}")
    sampler = getattr(model, "sampler", None)
    if sampler is not None and getattr(sampler, "_cosmos_actionaware_installed", False):
        raise RuntimeError("C3Cache cannot be combined with ActionAware")

    net._cosmos_c3cache_original_forward = net.forward
    net._cosmos_c3cache_installed = True
    net._cosmos_c3cache_state = _new_state()
    net.forward = types.MethodType(_c3cache_forward, net)

    original_generate = model.generate_samples_from_batch

    def generate_samples_with_c3cache(self, data_batch, *args, **kwargs):
        if len(args) >= 6:
            num_steps = int(args[5])
        else:
            num_steps = int(kwargs.get("num_steps", 35))
        if num_steps != C3CACHE_NUM_STEPS:
            raise ValueError(
                "C3Cache requires the original five-step action sampler, "
                f"got num_steps={num_steps}"
            )
        state = _get_state(net)
        if state["active"]:
            raise RuntimeError("C3Cache does not support nested generate calls")
        query_index = int(state["episode_query_index"])
        refresh = _is_refresh_query(query_index)
        state["active"] = True
        state["current_step"] = 0
        state["current_query"] = {"index": query_index, "refresh": refresh}
        completed = False
        try:
            result = original_generate(data_batch, *args, **kwargs)
            observed_steps = int(state["current_step"])
            if observed_steps != C3CACHE_NUM_STEPS:
                raise RuntimeError(
                    "C3Cache expected exactly five DiT forwards per action query, "
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
                if refresh:
                    state["refresh_queries"] += 1
                if (
                    C3CACHE_VERBOSE
                    and C3CACHE_REPORT_EVERY > 0
                    and state["queries"] % C3CACHE_REPORT_EVERY == 0
                ):
                    total_calls = state["full_calls"] + state["cached_calls"]
                    payload = {
                        "event": "c3cache_summary",
                        "queries": state["queries"],
                        "refresh_queries": state["refresh_queries"],
                        "full_calls": state["full_calls"],
                        "cached_calls": state["cached_calls"],
                        "cache_ratio": (
                            state["cached_calls"] / total_calls if total_calls else 0.0
                        ),
                        "cached_steps": list(C3CACHE_CACHED_STEPS),
                        "refresh_interval": C3CACHE_REFRESH_INTERVAL,
                    }
                    print("[C3Cache] summary: " + json.dumps(payload, sort_keys=True))
                    _write_jsonl(payload)

    model.generate_samples_from_batch = types.MethodType(
        generate_samples_with_c3cache,
        model,
    )
    _INSTALLED_MODELS.append(model)
    print(
        "[C3Cache] installed exclusive runtime patch: "
        f"blocks=28 num_steps={C3CACHE_NUM_STEPS} "
        f"cached_steps={list(C3CACHE_CACHED_STEPS)} "
        f"refresh_interval={C3CACHE_REFRESH_INTERVAL}; "
        "other_accelerators=disabled"
    )


def install_on_cosmos_utils(cosmos_utils: Any) -> None:
    """Install C3Cache after the untouched Cosmos-Policy model loader."""

    current = cosmos_utils.get_model
    if getattr(current, "_cosmos_c3cache_loader", False):
        return

    def get_model_with_c3cache(cfg):
        model, config = current(cfg)
        _install_on_model(model)
        return model, config

    get_model_with_c3cache._cosmos_c3cache_loader = True
    cosmos_utils.get_model = get_model_with_c3cache


def install_episode_reset(base_eval: Any) -> None:
    """Wrap base_eval.patch_runtime so every run_episode starts empty."""

    previous_patch_runtime = base_eval.patch_runtime
    if getattr(previous_patch_runtime, "_cosmos_c3cache_reset_installer", False):
        return

    def patch_runtime_with_c3cache_reset(module, *args, **kwargs):
        previous_patch_runtime(module, *args, **kwargs)
        if getattr(module.run_episode, "_cosmos_c3cache_episode_reset", False):
            return
        original_run_episode = module.run_episode

        def run_episode_with_c3cache_reset(*episode_args, **episode_kwargs):
            if len(episode_args) >= 4:
                model = episode_args[3]
            else:
                model = episode_kwargs.get("model")
            if model is None:
                raise RuntimeError("C3Cache could not resolve model at episode reset")
            reset_episode(model)
            return original_run_episode(*episode_args, **episode_kwargs)

        run_episode_with_c3cache_reset._cosmos_c3cache_episode_reset = True
        module.run_episode = run_episode_with_c3cache_reset

    patch_runtime_with_c3cache_reset._cosmos_c3cache_reset_installer = True
    base_eval.patch_runtime = patch_runtime_with_c3cache_reset


def finalize_stats() -> None:
    aggregate = {
        "event": "c3cache_final",
        "cached_steps": list(C3CACHE_CACHED_STEPS),
        "refresh_interval": C3CACHE_REFRESH_INTERVAL,
        "num_steps": C3CACHE_NUM_STEPS,
        "models": len(_INSTALLED_MODELS),
        "episode_resets": 0,
        "queries": 0,
        "refresh_queries": 0,
        "full_calls": 0,
        "cached_calls": 0,
        "fallback_full_calls": 0,
        "cache_entries_cleared": 0,
        "per_step_full": [0] * C3CACHE_NUM_STEPS,
        "per_step_cached": [0] * C3CACHE_NUM_STEPS,
    }
    for model in _INSTALLED_MODELS:
        state = _get_state(model.net)
        for key in (
            "episode_resets",
            "queries",
            "refresh_queries",
            "full_calls",
            "cached_calls",
            "fallback_full_calls",
            "cache_entries_cleared",
        ):
            aggregate[key] += int(state[key])
        for step in range(C3CACHE_NUM_STEPS):
            aggregate["per_step_full"][step] += int(state["per_step_full"][step])
            aggregate["per_step_cached"][step] += int(state["per_step_cached"][step])
    total_calls = aggregate["full_calls"] + aggregate["cached_calls"]
    aggregate["cache_ratio"] = (
        aggregate["cached_calls"] / total_calls if total_calls else 0.0
    )
    print("[C3Cache] final: " + json.dumps(aggregate, sort_keys=True))
    _write_jsonl(aggregate)
