#!/usr/bin/env python3
import json
import os
import sys
import types
from pathlib import Path
from typing import Any

import torch
import torch.amp as amp


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPT_DIR))
os.chdir(ROOT)
os.environ.setdefault("COSMOS_EVAL_VARIANT", "stepskip")


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


STEPSKIP_REL_L1_THRESH = _env_float("COSMOS_STEPSKIP_REL_L1_THRESH", 0.25)
STEPSKIP_REL_EPS = 1e-8
STEPSKIP_VERBOSE = _env_bool("COSMOS_STEPSKIP_VERBOSE", False)
STEPSKIP_REPORT_EVERY = _env_int("COSMOS_STEPSKIP_REPORT_EVERY", 200)
STEPSKIP_STATS_PATH = os.environ.get("COSMOS_STEPSKIP_STATS_PATH")
STEPSKIP_USE_REL_L1 = _env_bool("COSMOS_STEPSKIP_USE_REL_L1", False)
STEPSKIP_NUM_STEPS = max(_env_int("COSMOS_STEPSKIP_NUM_STEPS", 5), 1)


def _relative_l1(current: torch.Tensor, reference: torch.Tensor) -> float:
    current_f32 = current.detach().to(dtype=torch.float32)
    reference_f32 = reference.detach().to(dtype=torch.float32)
    numerator = torch.mean(torch.abs(current_f32 - reference_f32))
    denominator = torch.mean(torch.abs(reference_f32)).clamp_min(STEPSKIP_REL_EPS)
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
        "previous_input": None,
        "residual_cache": None,
        "window_step": 0,
        "global_call_idx": 0,
        "full_calls": 0,
        "skipped_calls": 0,
        "last_rel_l1": None,
    }


def _get_state(module: torch.nn.Module) -> dict[str, Any]:
    state = getattr(module, "_cosmos_stepskip_state", None)
    if state is None:
        state = _new_state()
        module._cosmos_stepskip_state = state
    return state


def _should_fallback(
    intermediate_feature_ids: list[int] | None,
    action: torch.Tensor | None,
) -> bool:
    return bool(intermediate_feature_ids) or action is not None


def _stepskip_forward(
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
        return self._cosmos_stepskip_original_forward(
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

    with amp.autocast("cuda", enabled=self.use_wan_fp32_strategy, dtype=torch.float32):
        if timesteps_B_T.ndim == 1:
            timesteps_B_T = timesteps_B_T.unsqueeze(1)
        t_embedding_B_T_D, adaln_lora_B_T_3D = self.t_embedder(timesteps_B_T)
        t_embedding_B_T_D = self.t_embedding_norm(t_embedding_B_T_D)

    self.affline_scale_log_info = {"t_embedding_B_T_D": t_embedding_B_T_D.detach()}
    self.affline_emb = t_embedding_B_T_D

    if extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D is not None:
        assert x_B_T_H_W_D.shape == extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D.shape, (
            f"{x_B_T_H_W_D.shape} != {extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D.shape}"
        )

    state = _get_state(self)
    state["global_call_idx"] += 1
    block_input = x_B_T_H_W_D
    rel_l1 = None
    residual_cache = state.get("residual_cache")
    previous_input = state.get("previous_input")
    if state["window_step"] >= STEPSKIP_NUM_STEPS:
        state["window_step"] = 0
        state["residual_cache"] = None
        state["previous_input"] = None
        residual_cache = None
        previous_input = None
    has_cache = residual_cache is not None and residual_cache.shape == block_input.shape
    if residual_cache is not None and residual_cache.shape != block_input.shape:
        state["window_step"] = 0
        state["residual_cache"] = None
        state["previous_input"] = None
        residual_cache = None
        previous_input = None
        has_cache = False
    can_compare = previous_input is not None and previous_input.shape == block_input.shape
    window_step = state["window_step"]
    must_full = window_step == 0 or not has_cache
    skip_forward = False
    if (not must_full) and STEPSKIP_USE_REL_L1 and can_compare:
        rel_l1 = _relative_l1(block_input, previous_input)
        skip_forward = rel_l1 <= STEPSKIP_REL_L1_THRESH
    elif (not must_full) and (not STEPSKIP_USE_REL_L1):
        skip_forward = True

    if skip_forward:
        x_B_T_H_W_D = block_input + residual_cache
        state["skipped_calls"] += 1
        decision = "skip" if STEPSKIP_USE_REL_L1 else "window_reuse"
    else:
        if self.use_crossattn_projection:
            crossattn_emb = self.crossattn_proj(crossattn_emb)
        self.crossattn_emb = crossattn_emb

        if img_context_emb is not None:
            assert self.extra_image_context_dim is not None, "extra_image_context_dim must be set for img_context_emb."
            img_context_emb = self.img_context_proj(img_context_emb)
            context_input = (crossattn_emb, img_context_emb)
        else:
            context_input = crossattn_emb

        for block in self.blocks:
            x_B_T_H_W_D = block(
                x_B_T_H_W_D,
                t_embedding_B_T_D,
                context_input,
                rope_emb_L_1_1_D=rope_emb_L_1_1_D,
                adaln_lora_B_T_3D=adaln_lora_B_T_3D,
                extra_per_block_pos_emb=extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D,
            )
        state["residual_cache"] = (x_B_T_H_W_D - block_input).detach()
        state["full_calls"] += 1
        decision = "full"

    state["previous_input"] = block_input.detach() if STEPSKIP_USE_REL_L1 else None
    state["last_rel_l1"] = rel_l1
    state["window_step"] = window_step + 1

    if STEPSKIP_VERBOSE and STEPSKIP_REPORT_EVERY > 0 and state["global_call_idx"] % STEPSKIP_REPORT_EVERY == 0:
        total = state["full_calls"] + state["skipped_calls"]
        skip_ratio = state["skipped_calls"] / total if total else 0.0
        payload = {
            "event": "stepskip_summary",
            "global_call_idx": state["global_call_idx"],
            "decision": decision,
            "full": state["full_calls"],
            "skipped": state["skipped_calls"],
            "skip_ratio": skip_ratio,
            "last_rel_l1": rel_l1,
            "threshold": STEPSKIP_REL_L1_THRESH,
            "use_rel_l1": STEPSKIP_USE_REL_L1,
            "window_step": window_step,
            "window_size": STEPSKIP_NUM_STEPS,
        }
        print(
            "[StepSkip] "
            f"calls={state['global_call_idx']} full={state['full_calls']} "
            f"skipped={state['skipped_calls']} skip_ratio={skip_ratio:.3f} "
            f"last_rel_l1={rel_l1} thresh={STEPSKIP_REL_L1_THRESH:.4f} "
            f"use_rel_l1={STEPSKIP_USE_REL_L1} window_step={window_step + 1}/{STEPSKIP_NUM_STEPS}"
        )
        _write_jsonl(STEPSKIP_STATS_PATH, payload)

    x_B_T_H_W_O = self.final_layer(x_B_T_H_W_D, t_embedding_B_T_D, adaln_lora_B_T_3D=adaln_lora_B_T_3D)
    return self.unpatchify(x_B_T_H_W_O)


def _install_stepskip(model: torch.nn.Module) -> None:
    net = getattr(model, "net", None)
    if net is None or not hasattr(net, "blocks") or not hasattr(net, "final_layer") or not hasattr(net, "unpatchify"):
        raise RuntimeError("StepSkip expected model.net with blocks/final_layer/unpatchify.")
    if hasattr(net, "_cosmos_stepskip_original_forward"):
        return
    net._cosmos_stepskip_original_forward = net.forward
    net.forward = types.MethodType(_stepskip_forward, net)
    mode = "window_relative_l1" if STEPSKIP_USE_REL_L1 else "window_reuse_after_first"
    print(
        "[StepSkip] installed runtime patch: "
        f"mode={mode} window_size={STEPSKIP_NUM_STEPS} rel_l1_thresh={STEPSKIP_REL_L1_THRESH} "
        f"report_every={STEPSKIP_REPORT_EVERY}"
    )


def main() -> None:
    from cosmos_policy.experiments.robot import cosmos_utils

    original_get_model = cosmos_utils.get_model

    def get_model_with_stepskip(cfg):
        model, config = original_get_model(cfg)
        _install_stepskip(model)
        return model, config

    cosmos_utils.get_model = get_model_with_stepskip

    import eval_wrapper

    eval_wrapper.main()


if __name__ == "__main__":
    main()
