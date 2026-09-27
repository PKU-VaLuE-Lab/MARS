#!/usr/bin/env python3
"""Action-aware 1-step draft routing with residual-reuse continuation."""

from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPT_DIR))
os.chdir(ROOT)

# eval_stepskip_wrapper reads these at import time and installs the residual cache.
os.environ.setdefault("COSMOS_EVAL_VARIANT", "actionaware")
os.environ["COSMOS_STEPSKIP_REUSE_MODE"] = "residual"
os.environ["COSMOS_STEPSKIP_USE_REL_L1"] = "0"
os.environ["COSMOS_STEPSKIP_NUM_STEPS"] = "5"


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return default if raw is None else int(raw)


MODE = os.environ.get("COSMOS_ACTIONAWARE_MODE", "route").strip().lower()
VALID_MODES = {"route", "probe_draft", "probe_stepskip", "force_1step", "force_stepskip"}
if MODE not in VALID_MODES:
    raise ValueError(f"Invalid COSMOS_ACTIONAWARE_MODE={MODE!r}; expected one of {sorted(VALID_MODES)}")

CHUNK_SIZE = _env_int("COSMOS_ACTIONAWARE_CHUNK_SIZE", 16)
FEATURE_STEPS = _env_int("COSMOS_ACTIONAWARE_FEATURE_STEPS", 12)
STATS_PATH = os.environ.get("COSMOS_ACTIONAWARE_STATS_PATH")
DATASET_STATS_PATH = os.environ.get(
    "COSMOS_ACTIONAWARE_DATASET_STATS",
    str(ROOT / "ckpt" / "libero_dataset_statistics.json"),
)
GATE_CONFIG_PATH = os.environ.get("COSMOS_ACTIONAWARE_GATE_CONFIG")
GATE_CONFIG_JSON = os.environ.get("COSMOS_ACTIONAWARE_GATE_CONFIG_JSON")
LOG_QUERIES = _env_bool("COSMOS_ACTIONAWARE_LOG_QUERIES", MODE.startswith("probe_"))
REPORT_EVERY = _env_int("COSMOS_ACTIONAWARE_REPORT_EVERY", 200)


def _load_json(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_gate_config() -> dict[str, Any] | None:
    if GATE_CONFIG_PATH:
        return _load_json(GATE_CONFIG_PATH)
    if not GATE_CONFIG_JSON:
        return None
    config = json.loads(GATE_CONFIG_JSON)
    if not isinstance(config, dict):
        raise ValueError("COSMOS_ACTIONAWARE_GATE_CONFIG_JSON must decode to a JSON object")
    return config


GATE_CONFIG = _load_gate_config()
DATASET_STATS = _load_json(DATASET_STATS_PATH)
if DATASET_STATS is None:
    raise RuntimeError(f"Dataset statistics are required for action-aware routing: {DATASET_STATS_PATH}")

TRANS_THRESHOLD_OVERRIDE = os.environ.get("COSMOS_ACTIONAWARE_TRANS_THRESHOLD")
if TRANS_THRESHOLD_OVERRIDE is not None:
    if not GATE_CONFIG or GATE_CONFIG.get("type", "rules") != "rules":
        raise ValueError("COSMOS_ACTIONAWARE_TRANS_THRESHOLD requires a rule gate config")
    matching_rules = [rule for rule in GATE_CONFIG.get("rules", []) if rule.get("feature") == "trans_mag_mean"]
    if len(matching_rules) != 1:
        raise ValueError("COSMOS_ACTIONAWARE_TRANS_THRESHOLD requires exactly one trans_mag_mean rule")
    threshold = float(TRANS_THRESHOLD_OVERRIDE)
    matching_rules[0]["threshold"] = threshold
    GATE_CONFIG["name"] = f"actionaware_perquery_low_translation_{round(100 * threshold):03d}"

_ACTIONAWARE_SAMPLERS: list[torch.nn.Module] = []
_ACTION_STATS_CACHE: dict[tuple[torch.device, torch.dtype], tuple[torch.Tensor, torch.Tensor]] = {}


def _write_jsonl(payload: dict[str, Any]) -> None:
    if not STATS_PATH:
        return
    path = Path(STATS_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _reset_stepskip_window(net: torch.nn.Module) -> None:
    state = getattr(net, "_cosmos_stepskip_state", None)
    if state is None:
        return
    state["previous_input"] = None
    state["residual_cache"] = None
    state["output_cache"] = None
    state["window_step"] = 0
    state["last_rel_l1"] = None


def _extract_physical_actions(latent: torch.Tensor, context: dict[str, Any]) -> torch.Tensor:
    from cosmos_policy.constants import ACTION_DIM
    from cosmos_policy.experiments.robot.cosmos_utils import extract_action_chunk_from_latent_sequence

    action_indices = context["action_indices"].to(device=latent.device)
    actions = extract_action_chunk_from_latent_sequence(
        latent.to(torch.float32),
        action_shape=(CHUNK_SIZE, ACTION_DIM),
        action_indices=action_indices,
    )
    cache_key = (actions.device, actions.dtype)
    action_stats = _ACTION_STATS_CACHE.get(cache_key)
    if action_stats is None:
        action_stats = (
            torch.as_tensor(DATASET_STATS["actions_min"], dtype=actions.dtype, device=actions.device),
            torch.as_tensor(DATASET_STATS["actions_max"], dtype=actions.dtype, device=actions.device),
        )
        _ACTION_STATS_CACHE[cache_key] = action_stats
    actions_min, actions_max = action_stats
    return 0.5 * (actions + 1.0) * (actions_max - actions_min) + actions_min


def _safe_mean(value: torch.Tensor) -> torch.Tensor:
    return value.mean() if value.numel() else torch.zeros((), dtype=value.dtype, device=value.device)


def _safe_max(value: torch.Tensor) -> torch.Tensor:
    return value.max() if value.numel() else torch.zeros((), dtype=value.dtype, device=value.device)


def _safe_min(value: torch.Tensor) -> torch.Tensor:
    return value.min() if value.numel() else torch.zeros((), dtype=value.dtype, device=value.device)


def _safe_std(value: torch.Tensor) -> torch.Tensor:
    return value.std(unbiased=False) if value.numel() else torch.zeros((), dtype=value.dtype, device=value.device)


def _action_features(
    actions: torch.Tensor,
    required_names: set[str] | None = None,
) -> dict[str, torch.Tensor]:
    # The gate summarizes FEATURE_STEPS actions.  LIBERO uses 12; RoboCasa
    # overrides this with the 16 actions actually executed from its 32-action
    # prediction chunk.
    steps = min(max(FEATURE_STEPS, 3), actions.shape[1])
    action = actions[:, :steps]
    translation = action[..., :3]
    rotation = action[..., 3:6]
    gripper = action[..., 6]
    required = required_names

    def needs(*names: str) -> bool:
        return required is None or any(name in required for name in names)

    features: dict[str, torch.Tensor] = {}
    trans_mag_names = (
        "trans_mag_mean",
        "trans_mag_max",
        "trans_mag_min",
        "trans_mag_std",
        "trans_mag_first",
        "trans_mag_last",
        "trans_low_fraction",
    )
    if needs(*trans_mag_names):
        trans_mag = torch.linalg.vector_norm(translation, dim=-1)
        if needs("trans_mag_mean"):
            features["trans_mag_mean"] = _safe_mean(trans_mag)
        if needs("trans_mag_max"):
            features["trans_mag_max"] = _safe_max(trans_mag)
        if needs("trans_mag_min"):
            features["trans_mag_min"] = _safe_min(trans_mag)
        if needs("trans_mag_std"):
            features["trans_mag_std"] = _safe_std(trans_mag)
        if needs("trans_mag_first"):
            features["trans_mag_first"] = _safe_mean(trans_mag[:, 0])
        if needs("trans_mag_last"):
            features["trans_mag_last"] = _safe_mean(trans_mag[:, -1])
        if needs("trans_low_fraction"):
            features["trans_low_fraction"] = _safe_mean((trans_mag < 0.05).to(torch.float32))

    rot_mag_names = ("rot_mag_mean", "rot_mag_max", "rot_mag_min", "rot_mag_std")
    if needs(*rot_mag_names):
        rot_mag = torch.linalg.vector_norm(rotation, dim=-1)
        if needs("rot_mag_mean"):
            features["rot_mag_mean"] = _safe_mean(rot_mag)
        if needs("rot_mag_max"):
            features["rot_mag_max"] = _safe_max(rot_mag)
        if needs("rot_mag_min"):
            features["rot_mag_min"] = _safe_min(rot_mag)
        if needs("rot_mag_std"):
            features["rot_mag_std"] = _safe_std(rot_mag)

    d1_names = (
        "trans_d1_mean",
        "trans_d1_max",
        "rot_d1_mean",
        "rot_d1_max",
        "action_d1_mean",
        "action_d1_max",
        "trans_d2_mean",
        "trans_d2_max",
        "rot_d2_mean",
        "rot_d2_max",
        "action_d2_mean",
        "action_d2_max",
        "gripper_d1_max",
        "gripper_transition_fraction",
        "gripper_transitions",
    )
    d1 = action[:, 1:] - action[:, :-1] if needs(*d1_names) else None
    if needs("trans_d1_mean", "trans_d1_max"):
        trans_d1 = torch.linalg.vector_norm(d1[..., :3], dim=-1)
        if needs("trans_d1_mean"):
            features["trans_d1_mean"] = _safe_mean(trans_d1)
        if needs("trans_d1_max"):
            features["trans_d1_max"] = _safe_max(trans_d1)
    if needs("rot_d1_mean", "rot_d1_max"):
        rot_d1 = torch.linalg.vector_norm(d1[..., 3:6], dim=-1)
        if needs("rot_d1_mean"):
            features["rot_d1_mean"] = _safe_mean(rot_d1)
        if needs("rot_d1_max"):
            features["rot_d1_max"] = _safe_max(rot_d1)
    if needs("action_d1_mean", "action_d1_max"):
        all_d1 = torch.linalg.vector_norm(d1[..., :6], dim=-1)
        if needs("action_d1_mean"):
            features["action_d1_mean"] = _safe_mean(all_d1)
        if needs("action_d1_max"):
            features["action_d1_max"] = _safe_max(all_d1)

    d2_names = (
        "trans_d2_mean",
        "trans_d2_max",
        "rot_d2_mean",
        "rot_d2_max",
        "action_d2_mean",
        "action_d2_max",
    )
    d2 = d1[:, 1:] - d1[:, :-1] if needs(*d2_names) else None
    if needs("trans_d2_mean", "trans_d2_max"):
        trans_d2 = torch.linalg.vector_norm(d2[..., :3], dim=-1)
        if needs("trans_d2_mean"):
            features["trans_d2_mean"] = _safe_mean(trans_d2)
        if needs("trans_d2_max"):
            features["trans_d2_max"] = _safe_max(trans_d2)
    if needs("rot_d2_mean", "rot_d2_max"):
        rot_d2 = torch.linalg.vector_norm(d2[..., 3:6], dim=-1)
        if needs("rot_d2_mean"):
            features["rot_d2_mean"] = _safe_mean(rot_d2)
        if needs("rot_d2_max"):
            features["rot_d2_max"] = _safe_max(rot_d2)
    if needs("action_d2_mean", "action_d2_max"):
        all_d2 = torch.linalg.vector_norm(d2[..., :6], dim=-1)
        if needs("action_d2_mean"):
            features["action_d2_mean"] = _safe_mean(all_d2)
        if needs("action_d2_max"):
            features["action_d2_max"] = _safe_max(all_d2)

    if needs("endpoint_trans_delta", "endpoint_rot_delta"):
        endpoint_delta = action[:, -1] - action[:, 0]
        if needs("endpoint_trans_delta"):
            features["endpoint_trans_delta"] = _safe_mean(
                torch.linalg.vector_norm(endpoint_delta[..., :3], dim=-1)
            )
        if needs("endpoint_rot_delta"):
            features["endpoint_rot_delta"] = _safe_mean(
                torch.linalg.vector_norm(endpoint_delta[..., 3:6], dim=-1)
            )

    if needs("gripper_abs_min"):
        features["gripper_abs_min"] = _safe_min(torch.abs(gripper))
    if needs("gripper_abs_mean"):
        features["gripper_abs_mean"] = _safe_mean(torch.abs(gripper))
    if needs("gripper_d1_max"):
        features["gripper_d1_max"] = _safe_max(torch.abs(d1[..., 6]))
    if needs("gripper_range"):
        features["gripper_range"] = _safe_mean(gripper.max(dim=1).values - gripper.min(dim=1).values)
    if needs("gripper_transition_fraction", "gripper_transitions"):
        gripper_transitions = ((gripper[:, 1:] >= 0) != (gripper[:, :-1] >= 0)).to(torch.float32)
        if needs("gripper_transition_fraction"):
            features["gripper_transition_fraction"] = _safe_mean(gripper_transitions)
        if needs("gripper_transitions"):
            features["gripper_transitions"] = gripper_transitions.sum()
    return features


def _required_feature_names() -> set[str] | None:
    if MODE.startswith("probe_"):
        return None
    if not GATE_CONFIG:
        return set()
    if GATE_CONFIG.get("type", "rules") == "linear":
        return set(GATE_CONFIG["feature_names"])
    return {rule["feature"] for rule in GATE_CONFIG.get("rules", [])}


REQUIRED_FEATURE_NAMES = _required_feature_names()


def _reset_actionaware_episode() -> None:
    for sampler in _ACTIONAWARE_SAMPLERS:
        sampler._cosmos_actionaware_gate_state = {
            "query_index": 0,
            "episode_unsafe": None,
        }
        sampler._cosmos_actionaware_last_route = None


def _pair_metrics(draft: torch.Tensor, refined: torch.Tensor) -> dict[str, torch.Tensor]:
    steps = min(max(FEATURE_STEPS, 1), draft.shape[1], refined.shape[1])
    delta = draft[:, :steps] - refined[:, :steps]
    trans_l2 = torch.linalg.vector_norm(delta[..., :3], dim=-1)
    rot_l2 = torch.linalg.vector_norm(delta[..., 3:6], dim=-1)
    first4 = delta[:, : min(4, steps)]
    gripper_disagree = ((draft[:, :steps, 6] >= 0) != (refined[:, :steps, 6] >= 0)).to(torch.float32)
    return {
        "pair_action_l1": torch.abs(delta).mean(),
        "pair_action_linf": torch.abs(delta).max(),
        "pair_trans_l2_mean": _safe_mean(trans_l2),
        "pair_trans_l2_max": _safe_max(trans_l2),
        "pair_rot_l2_mean": _safe_mean(rot_l2),
        "pair_rot_l2_max": _safe_max(rot_l2),
        "pair_first4_trans_l2_mean": _safe_mean(torch.linalg.vector_norm(first4[..., :3], dim=-1)),
        "pair_first4_rot_l2_mean": _safe_mean(torch.linalg.vector_norm(first4[..., 3:6], dim=-1)),
        "pair_gripper_l1": torch.abs(delta[..., 6]).mean(),
        "pair_gripper_disagree": gripper_disagree.mean(),
    }


def _linear_score(features: dict[str, torch.Tensor], config: dict[str, Any]) -> torch.Tensor:
    names = config["feature_names"]
    vector = torch.stack([features[name] for name in names])
    mean = torch.as_tensor(config.get("mean", [0.0] * len(names)), dtype=vector.dtype, device=vector.device)
    scale = torch.as_tensor(config.get("scale", [1.0] * len(names)), dtype=vector.dtype, device=vector.device)
    weights = torch.as_tensor(config["weights"], dtype=vector.dtype, device=vector.device)
    return (((vector - mean) / scale.clamp_min(1e-8)) * weights).sum() + float(config.get("bias", 0.0))


def _gate(features: dict[str, torch.Tensor]) -> tuple[bool, float | None]:
    if not GATE_CONFIG:
        return False, None

    gate_type = GATE_CONFIG.get("type", "rules")
    if gate_type == "linear":
        score_tensor = _linear_score(features, GATE_CONFIG)
        unsafe_tensor = score_tensor >= float(GATE_CONFIG["threshold"])
        score = float(score_tensor.detach().cpu().item())
        return bool(unsafe_tensor.detach().cpu().item()), score

    if gate_type != "rules":
        raise ValueError(f"Unsupported action-aware gate type: {gate_type}")
    decisions = []
    for rule in GATE_CONFIG.get("rules", []):
        value = features[rule["feature"]]
        threshold = float(rule["threshold"])
        op = rule.get("op", ">=")
        if op in {">", ">="}:
            decisions.append(value >= threshold)
        elif op in {"<", "<="}:
            decisions.append(value <= threshold)
        else:
            raise ValueError(f"Unsupported gate rule op: {op}")
    if not decisions:
        return False, None
    stacked = torch.stack(decisions)
    unsafe_tensor = stacked.all() if GATE_CONFIG.get("logic", "any") == "all" else stacked.any()
    return bool(unsafe_tensor.detach().cpu().item()), None


def _cpu_scalars(values: dict[str, torch.Tensor]) -> dict[str, float]:
    if not values:
        return {}
    names = list(values)
    tensor = torch.stack([values[name].detach().to(torch.float32) for name in names]).cpu()
    return {name: float(value) for name, value in zip(names, tensor.tolist())}


def _actionaware_sampler_forward(
    self,
    x0_fn,
    x_sigma_max: torch.Tensor,
    num_steps: int = 5,
    sigma_min: float = 0.002,
    sigma_max: float = 80,
    rho: float = 7,
    S_churn: float = 0,
    S_min: float = 0,
    S_max: float = float("inf"),
    S_noise: float = 1,
    solver_option: str = "2ab",
):
    from cosmos_policy._src.imaginaire.modules.res_sampler import SolverConfig, differential_equation_solver, get_rev_ts

    if num_steps != 5:
        raise ValueError(f"Action-aware routing requires the original 5-step schedule, got num_steps={num_steps}")
    if solver_option != "2ab":
        raise ValueError(f"Action-aware first-result reuse currently requires solver_option='2ab', got {solver_option!r}")
    context = getattr(self, "_cosmos_actionaware_context", None)
    if context is None:
        raise RuntimeError("Action-aware sampler was called without action extraction context")

    in_dtype = x_sigma_max.dtype

    def float64_x0_fn(x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        return x0_fn(x.to(in_dtype), sigma.to(in_dtype)).to(torch.float64)

    schedule_key = (
        num_steps,
        float(sigma_min),
        float(sigma_max),
        float(rho),
        float(S_churn),
        float(S_min),
        float(S_max),
        float(S_noise),
        solver_option,
        x_sigma_max.device,
        x_sigma_max.size(0),
    )
    cached_schedule = self._cosmos_actionaware_schedule_cache.get(schedule_key)
    if cached_schedule is None:
        solver_nfe = num_steps - 1
        solver_cfg = SolverConfig(
            s_churn=S_churn,
            s_t_max=S_max,
            s_t_min=S_min,
            s_noise=S_noise,
            is_multi=True,
            rk=solver_option,
            multistep=solver_option,
        )
        sigmas = get_rev_ts(sigma_min, sigma_max, solver_nfe, rho).to(x_sigma_max.device)
        ones = torch.ones(x_sigma_max.size(0), device=x_sigma_max.device, dtype=torch.float64)
        cached_schedule = (solver_cfg, sigmas, ones)
        self._cosmos_actionaware_schedule_cache[schedule_key] = cached_schedule
    solver_cfg, sigmas, ones = cached_schedule

    # This is byte-for-byte the same x0_fn input used by the standalone 1-step sampler.
    first_result = float64_x0_fn(x_sigma_max, sigmas[0] * ones)
    gate_state = self._cosmos_actionaware_gate_state
    episode_latch = bool(MODE == "route" and GATE_CONFIG and GATE_CONFIG.get("episode_latch", False))
    if episode_latch and gate_state["query_index"] > 0:
        draft_actions = None
        features = {}
        gate_unsafe = bool(gate_state["episode_unsafe"])
        gate_score = None
    else:
        draft_actions = _extract_physical_actions(first_result, context)
        features = _action_features(draft_actions, REQUIRED_FEATURE_NAMES)
        gate_unsafe, gate_score = _gate(features)
        if episode_latch:
            gate_state["episode_unsafe"] = gate_unsafe
    gate_state["query_index"] += 1

    should_continue = MODE in {"probe_draft", "probe_stepskip", "force_stepskip"} or (MODE == "route" and gate_unsafe)
    refined_result = None
    pair = {}
    if should_continue:
        first_pending = True

        def x0_fn_with_first_result(x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
            nonlocal first_pending
            if first_pending:
                first_pending = False
                return first_result
            return float64_x0_fn(x, sigma)

        refined_result = differential_equation_solver(
            x0_fn_with_first_result,
            sigmas,
            solver_cfg,
        )(x_sigma_max)
        refined_result = x0_fn_with_first_result(refined_result, sigmas[-1] * ones)
        if MODE.startswith("probe_"):
            refined_actions = _extract_physical_actions(refined_result, context)
            pair = _pair_metrics(draft_actions, refined_actions)

    if MODE in {"probe_draft", "force_1step"}:
        selected = first_result
        route = "1step"
    elif MODE in {"probe_stepskip", "force_stepskip"}:
        selected = refined_result
        route = "stepskip"
    elif gate_unsafe:
        selected = refined_result
        route = "stepskip"
    else:
        selected = first_result
        route = "1step"

    self._cosmos_actionaware_last_route = route

    stats = self._cosmos_actionaware_stats
    stats["queries"] += 1
    stats[route] += 1
    query_id = stats["queries"]
    if LOG_QUERIES:
        payload = {
            "event": "actionaware_query",
            "query_id": query_id,
            "mode": MODE,
            "route": route,
            "gate_unsafe": gate_unsafe,
            "gate_score": gate_score,
            "features": _cpu_scalars(features),
        }
        if pair:
            payload["pair"] = _cpu_scalars(pair)
        _write_jsonl(payload)

    if REPORT_EVERY > 0 and query_id % REPORT_EVERY == 0:
        summary = {
            "event": "actionaware_summary",
            "mode": MODE,
            "queries": stats["queries"],
            "one_step": stats["1step"],
            "stepskip": stats["stepskip"],
            "stepskip_ratio": stats["stepskip"] / stats["queries"],
        }
        print(
            "[ActionAware] "
            f"queries={summary['queries']} 1step={summary['one_step']} "
            f"stepskip={summary['stepskip']} stepskip_ratio={summary['stepskip_ratio']:.3f}"
        )
        _write_jsonl(summary)

    return selected.to(in_dtype)


def _install_actionaware(model: torch.nn.Module) -> None:
    sampler = getattr(model, "sampler", None)
    net = getattr(model, "net", None)
    if sampler is None or net is None:
        raise RuntimeError("Action-aware routing expected model.sampler and model.net")
    if getattr(sampler, "_cosmos_actionaware_installed", False):
        return

    sampler._cosmos_actionaware_installed = True
    sampler._cosmos_actionaware_stats = {"queries": 0, "1step": 0, "stepskip": 0}
    sampler._cosmos_actionaware_gate_state = {"query_index": 0, "episode_unsafe": None}
    sampler._cosmos_actionaware_last_route = None
    sampler._cosmos_actionaware_context = None
    sampler._cosmos_actionaware_schedule_cache = {}
    sampler.forward = types.MethodType(_actionaware_sampler_forward, sampler)
    _ACTIONAWARE_SAMPLERS.append(sampler)

    original_generate = model.generate_samples_from_batch

    def generate_samples_actionaware(self, data_batch, *args, **kwargs):
        if len(args) >= 6:
            raise ValueError("Action-aware wrapper requires num_steps to be passed by keyword")
        kwargs["num_steps"] = 5
        _reset_stepskip_window(net)
        action_indices = data_batch.get("action_latent_idx")
        if action_indices is None:
            raise RuntimeError("Action-aware routing requires data_batch['action_latent_idx']")
        sampler._cosmos_actionaware_context = {"action_indices": action_indices.detach()}
        try:
            return original_generate(data_batch, *args, **kwargs)
        finally:
            sampler._cosmos_actionaware_context = None

    model.generate_samples_from_batch = types.MethodType(generate_samples_actionaware, model)
    gate_name = GATE_CONFIG.get("name", GATE_CONFIG.get("type", "custom")) if GATE_CONFIG else "none"
    print(
        "[ActionAware] installed: "
        f"mode={MODE} gate={gate_name} feature_steps={FEATURE_STEPS} "
        f"same_noise=true reuse_first_result=true residual_reuse=true"
    )


def _finalize_actionaware_stats() -> None:
    for sampler in _ACTIONAWARE_SAMPLERS:
        stats = sampler._cosmos_actionaware_stats
        queries = stats["queries"]
        summary = {
            "event": "actionaware_final",
            "mode": MODE,
            "gate": GATE_CONFIG.get("name", GATE_CONFIG.get("type", "custom")) if GATE_CONFIG else "none",
            "queries": queries,
            "one_step": stats["1step"],
            "stepskip": stats["stepskip"],
            "stepskip_ratio": stats["stepskip"] / queries if queries else 0.0,
            "rules": GATE_CONFIG.get("rules", []) if GATE_CONFIG else [],
        }
        print("[ActionAware] final: " + json.dumps(summary, sort_keys=True))
        _write_jsonl(summary)


# Importing this module installs the tested residual-reuse net.forward patch.
import eval_stepskip_wrapper  # noqa: E402,F401
from cosmos_policy.experiments.robot import cosmos_utils  # noqa: E402


_original_get_model = cosmos_utils.get_model


def _get_model_with_actionaware(cfg):
    model, config = _original_get_model(cfg)
    eval_stepskip_wrapper._install_stepskip(model)
    _install_actionaware(model)
    return model, config


cosmos_utils.get_model = _get_model_with_actionaware

import eval_wrapper  # noqa: E402


_original_patch_runtime = eval_wrapper.patch_runtime


def _patch_runtime_with_episode_reset(module, *args, **kwargs):
    _original_patch_runtime(module, *args, **kwargs)
    if getattr(module.run_episode, "_cosmos_actionaware_reset", False):
        return
    original_run_episode = module.run_episode

    def run_episode_with_actionaware_reset(*episode_args, **episode_kwargs):
        _reset_actionaware_episode()
        return original_run_episode(*episode_args, **episode_kwargs)

    run_episode_with_actionaware_reset._cosmos_actionaware_reset = True
    module.run_episode = run_episode_with_actionaware_reset


eval_wrapper.patch_runtime = _patch_runtime_with_episode_reset


if __name__ == "__main__":
    eval_wrapper.main()
    _finalize_actionaware_stats()
