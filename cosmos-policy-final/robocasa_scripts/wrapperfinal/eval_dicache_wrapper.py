#!/usr/bin/env python3
import json
import os
import sys
import types
from pathlib import Path
from typing import Any

import torch
import torch.amp as amp


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
WRAPPER_DIR = SCRIPT_DIR / "wrapperfinal"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(WRAPPER_DIR))
os.chdir(ROOT)
os.environ.setdefault("COSMOS_EVAL_VARIANT", "dicache")


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return default if raw is None else float(raw)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return default if raw is None else int(raw)


DICACHE_PROBE_DEPTH = max(_env_int("COSMOS_DICACHE_PROBE_DEPTH", 1), 1)
DICACHE_REL_L1_THRESH = _env_float("COSMOS_DICACHE_REL_L1_THRESH", 0.08)
DICACHE_RET_RATIO = _env_float("COSMOS_DICACHE_RET_RATIO", 0.2)
DICACHE_GAMMA_MIN = _env_float("COSMOS_DICACHE_GAMMA_MIN", 1.0)
DICACHE_GAMMA_MAX = _env_float("COSMOS_DICACHE_GAMMA_MAX", 2.0)
DICACHE_REL_EPS = 1e-8
DICACHE_VERBOSE = _env_bool("COSMOS_DICACHE_VERBOSE", False)
DICACHE_REPORT_EVERY = _env_int("COSMOS_DICACHE_REPORT_EVERY", 200)
DICACHE_STATS_PATH = os.environ.get("COSMOS_DICACHE_STATS_PATH")
DICACHE_NUM_STEPS = max(_env_int("COSMOS_DICACHE_NUM_STEPS", 5), 1)


def _relative_l1(current: torch.Tensor, reference: torch.Tensor) -> float:
    current_f32 = current.detach().to(dtype=torch.float32)
    reference_f32 = reference.detach().to(dtype=torch.float32)
    numerator = torch.mean(torch.abs(current_f32 - reference_f32))
    denominator = torch.mean(torch.abs(reference_f32)).clamp_min(DICACHE_REL_EPS)
    return float((numerator / denominator).detach().cpu().item())


def _write_jsonl(path: str | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _new_state() -> dict[str, Any]:
    return {
        "cnt": 0,
        "accumulated_rel_l1_distance": 0.0,
        "previous_input": None,
        "previous_probe_states": None,
        "previous_output": None,
        "residual_cache": None,
        "probe_residual_cache": None,
        "residual_window": [],
        "probe_residual_window": [],
        "resume_flag": False,
        "global_call_idx": 0,
        "window_full_calls": 0,
        "window_skipped_calls": 0,
        "window_resumed_calls": 0,
        "total_full_calls": 0,
        "total_skipped_calls": 0,
        "total_resumed_calls": 0,
        "skipped_calls": 0,
        "last_rel_l1": None,
        "last_delta_x": None,
    }


def _clear_window_state(state: dict[str, Any]) -> None:
    total_full_calls = state["total_full_calls"]
    total_skipped_calls = state["total_skipped_calls"]
    total_resumed_calls = state["total_resumed_calls"]
    global_call_idx = state["global_call_idx"]
    skipped_calls = state["skipped_calls"]
    state.clear()
    state.update(_new_state())
    state["total_full_calls"] = total_full_calls
    state["total_skipped_calls"] = total_skipped_calls
    state["total_resumed_calls"] = total_resumed_calls
    state["global_call_idx"] = global_call_idx
    state["skipped_calls"] = skipped_calls


def _append_cache_window(window: list[torch.Tensor], value: torch.Tensor) -> None:
    if len(window) < 2:
        window.append(value)
    else:
        window[0] = window[1]
        window[1] = value


def _get_state(module: torch.nn.Module) -> dict[str, Any]:
    state = getattr(module, "_cosmos_dicache_state", None)
    if state is None:
        state = _new_state()
        module._cosmos_dicache_state = state
    return state


def _should_fallback(
    intermediate_feature_ids: list[int] | None,
    action: torch.Tensor | None,
) -> bool:
    return bool(intermediate_feature_ids) or action is not None


def _dicache_forward(
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
    if _should_fallback(intermediate_feature_ids, action):
        return self._cosmos_dicache_original_forward(
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
    assert isinstance(data_type, DataType), f"Expected DataType, got {type(data_type)}."

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
    x_B_T_H_W_D, rope_emb_L_1_1_D, extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D = self.prepare_embedded_sequence(
        x_B_C_T_H_W,
        fps=fps,
        padding_mask=padding_mask,
    )

    if self.use_crossattn_projection:
        crossattn_emb = self.crossattn_proj(crossattn_emb)

    if img_context_emb is not None:
        assert self.extra_image_context_dim is not None, "extra_image_context_dim must be set for img_context_emb."
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

    if extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D is not None:
        assert x_B_T_H_W_D.shape == extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D.shape, (
            f"{x_B_T_H_W_D.shape} != {extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D.shape}"
        )

    state = _get_state(self)
    state["global_call_idx"] += 1
    if state["cnt"] >= DICACHE_NUM_STEPS:
        _clear_window_state(state)

    num_blocks = len(self.blocks)
    probe_depth = min(DICACHE_PROBE_DEPTH, num_blocks)
    warmup_steps = int(DICACHE_RET_RATIO * DICACHE_NUM_STEPS)
    block_input = x_B_T_H_W_D
    skip_forward = False
    test_x = None
    delta_x = None
    delta_y = None
    can_probe = (
        state["cnt"] >= warmup_steps
        and state["previous_input"] is not None
        and state["previous_probe_states"] is not None
        and state["residual_cache"] is not None
        and state["previous_input"].shape == block_input.shape
        and state["previous_probe_states"].shape == block_input.shape
        and state["residual_cache"].shape == block_input.shape
    )

    if can_probe:
        test_x = block_input.clone()
        for probe_block in self.blocks[:probe_depth]:
            test_x = probe_block(
                test_x,
                t_embedding_B_T_D,
                context_input,
                rope_emb_L_1_1_D=rope_emb_L_1_1_D,
                adaln_lora_B_T_3D=adaln_lora_B_T_3D,
                extra_per_block_pos_emb=extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D,
            )
        delta_x = _relative_l1(block_input, state["previous_input"])
        delta_y = _relative_l1(test_x, state["previous_probe_states"])
        state["accumulated_rel_l1_distance"] += delta_y
        if state["accumulated_rel_l1_distance"] < DICACHE_REL_L1_THRESH:
            skip_forward = True
            state["resume_flag"] = False
        else:
            state["resume_flag"] = True
            state["accumulated_rel_l1_distance"] = 0.0
    else:
        state["resume_flag"] = False
        if state["cnt"] < warmup_steps:
            state["accumulated_rel_l1_distance"] = 0.0

    if skip_forward:
        if len(state["residual_window"]) >= 2 and len(state["probe_residual_window"]) >= 2:
            current_residual_indicator = test_x - block_input
            probe_delta = state["probe_residual_window"][1] - state["probe_residual_window"][0]
            denominator = probe_delta.detach().to(dtype=torch.float32).abs().mean().clamp_min(DICACHE_REL_EPS)
            gamma = (
                (current_residual_indicator - state["probe_residual_window"][0])
                .detach()
                .to(dtype=torch.float32)
                .abs()
                .mean()
                / denominator
            ).clamp(DICACHE_GAMMA_MIN, DICACHE_GAMMA_MAX)
            gamma = gamma.to(dtype=block_input.dtype, device=block_input.device)
            residual = state["residual_window"][0] + gamma * (state["residual_window"][1] - state["residual_window"][0])
            x_B_T_H_W_D = block_input + residual.to(dtype=block_input.dtype, device=block_input.device)
        else:
            x_B_T_H_W_D = block_input + state["residual_cache"].to(dtype=block_input.dtype, device=block_input.device)
        state["previous_input"] = block_input.detach()
        state["previous_probe_states"] = test_x.detach()
        state["window_skipped_calls"] += 1
        state["total_skipped_calls"] += 1
        state["skipped_calls"] += 1
        decision = "skip"
    else:
        probe_state = None
        if state["resume_flag"] and test_x is not None:
            x_B_T_H_W_D = test_x
            unpass_blocks = self.blocks[probe_depth:]
            probe_state = test_x
            state["window_resumed_calls"] += 1
            state["total_resumed_calls"] += 1
            decision = "resume"
        else:
            unpass_blocks = self.blocks
            decision = "full"

        for block_index, block in enumerate(unpass_blocks):
            x_B_T_H_W_D = block(
                x_B_T_H_W_D,
                t_embedding_B_T_D,
                context_input,
                rope_emb_L_1_1_D=rope_emb_L_1_1_D,
                adaln_lora_B_T_3D=adaln_lora_B_T_3D,
                extra_per_block_pos_emb=extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D,
            )
            if (not state["resume_flag"]) and block_index == probe_depth - 1:
                probe_state = x_B_T_H_W_D

        if probe_state is None:
            probe_state = x_B_T_H_W_D

        residual_cache = (x_B_T_H_W_D - block_input).detach()
        probe_residual_cache = (probe_state - block_input).detach()
        state["residual_cache"] = residual_cache
        state["probe_residual_cache"] = probe_residual_cache
        state["previous_input"] = block_input.detach()
        state["previous_probe_states"] = probe_state.detach()
        state["previous_output"] = x_B_T_H_W_D.detach()
        _append_cache_window(state["residual_window"], residual_cache)
        _append_cache_window(state["probe_residual_window"], probe_residual_cache)
        state["window_full_calls"] += 1
        state["total_full_calls"] += 1

    state["last_rel_l1"] = delta_y
    state["last_delta_x"] = delta_x
    state["cnt"] += 1

    if DICACHE_VERBOSE and DICACHE_REPORT_EVERY > 0 and state["global_call_idx"] % DICACHE_REPORT_EVERY == 0:
        total = state["total_full_calls"] + state["total_skipped_calls"]
        skip_ratio = state["total_skipped_calls"] / total if total else 0.0
        payload = {
            "event": "dicache_summary",
            "global_call_idx": state["global_call_idx"],
            "decision": decision,
            "full": state["total_full_calls"],
            "skipped": state["total_skipped_calls"],
            "resumed": state["total_resumed_calls"],
            "skip_ratio": skip_ratio,
            "last_delta_x": delta_x,
            "last_delta_y": delta_y,
            "accumulated_rel_l1_distance": state["accumulated_rel_l1_distance"],
            "threshold": DICACHE_REL_L1_THRESH,
            "probe_depth": probe_depth,
            "ret_ratio": DICACHE_RET_RATIO,
            "window_step": state["cnt"],
            "window_size": DICACHE_NUM_STEPS,
        }
        print(
            "[DiCache] "
            f"calls={state['global_call_idx']} full={state['total_full_calls']} "
            f"skipped={state['total_skipped_calls']} resumed={state['total_resumed_calls']} "
            f"skip_ratio={skip_ratio:.3f} delta_y={delta_y} thresh={DICACHE_REL_L1_THRESH:.4f} "
            f"accum={state['accumulated_rel_l1_distance']:.4f} "
            f"probe_depth={probe_depth} window_step={state['cnt']}/{DICACHE_NUM_STEPS}"
        )
        _write_jsonl(DICACHE_STATS_PATH, payload)

    if state["cnt"] >= DICACHE_NUM_STEPS:
        _clear_window_state(state)

    x_B_T_H_W_O = self.final_layer(x_B_T_H_W_D, t_embedding_B_T_D, adaln_lora_B_T_3D=adaln_lora_B_T_3D)
    return self.unpatchify(x_B_T_H_W_O)


def _install_dicache(model: torch.nn.Module) -> None:
    net = getattr(model, "net", None)
    if net is None or not hasattr(net, "blocks") or not hasattr(net, "final_layer") or not hasattr(net, "unpatchify"):
        raise RuntimeError("DiCache expected model.net with blocks/final_layer/unpatchify.")
    if hasattr(net, "_cosmos_dicache_original_forward"):
        return
    net._cosmos_dicache_original_forward = net.forward
    net.forward = types.MethodType(_dicache_forward, net)
    print(
        "[DiCache] installed runtime patch: "
        f"probe_depth={DICACHE_PROBE_DEPTH} window_size={DICACHE_NUM_STEPS} "
        f"ret_ratio={DICACHE_RET_RATIO} rel_l1_thresh={DICACHE_REL_L1_THRESH} "
        f"gamma=[{DICACHE_GAMMA_MIN},{DICACHE_GAMMA_MAX}] report_every={DICACHE_REPORT_EVERY}"
    )


from cosmos_policy.experiments.robot import cosmos_utils

_original_get_model = cosmos_utils.get_model


def _get_model_with_dicache(cfg):
    model, config = _original_get_model(cfg)
    _install_dicache(model)
    return model, config


cosmos_utils.get_model = _get_model_with_dicache

import eval_wrapper


if __name__ == "__main__":
    eval_wrapper.main()
