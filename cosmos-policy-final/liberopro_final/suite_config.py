#!/usr/bin/env python3
"""Shared LIBERO-Pro final suite selection and trial-count defaults."""

from __future__ import annotations

import sys
from collections.abc import Sequence


BASE_SUITES = ("spatial", "goal", "object", "10")
PERTURBATIONS = ("lan", "object", "swap", "task", "env")
SUITE_TO_BASE = {
    f"libero_{base}_{perturbation}": f"libero_{base}"
    for base in BASE_SUITES
    for perturbation in PERTURBATIONS
}
ALL_SUITES = tuple(SUITE_TO_BASE)
DEFAULT_TRIALS_PER_PERTURBED_TASK = 10


def _aliases() -> dict[str, list[str]]:
    aliases = {"all": list(ALL_SUITES)}
    for base in BASE_SUITES:
        suites = [
            f"libero_{base}_{perturbation}"
            for perturbation in PERTURBATIONS
        ]
        aliases[base] = suites
        aliases[f"libero_{base}"] = suites
    for suite in ALL_SUITES:
        aliases[suite] = [suite]
    return aliases


SUITE_ALIASES = _aliases()


def selected_suites(selection: str) -> list[str]:
    """Expand all/base aliases to the five LIBERO-Pro perturbation suites."""
    result: list[str] = []
    for item in selection.split(","):
        key = item.strip()
        if not key:
            continue
        if key not in SUITE_ALIASES:
            raise ValueError(f"Unknown LIBERO-Pro suite selection: {key}")
        result.extend(SUITE_ALIASES[key])
    return list(dict.fromkeys(result or ALL_SUITES))


def patch_liberopro_suite_compat(module) -> None:
    """Validate every perturbed suite with its base LIBERO policy protocol."""
    original_validate_config = module.validate_config
    for pro_suite, base_suite in SUITE_TO_BASE.items():
        module.TASK_MAX_STEPS[pro_suite] = module.TASK_MAX_STEPS[base_suite]

    def validate_liberopro_config(cfg):
        original_suite = cfg.task_suite_name
        base_suite = SUITE_TO_BASE.get(original_suite)
        if base_suite is None:
            return original_validate_config(cfg)
        cfg.task_suite_name = base_suite
        try:
            return original_validate_config(cfg)
        finally:
            cfg.task_suite_name = original_suite

    module.validate_config = validate_liberopro_config


def ensure_default_trial_count(argv: list[str] | None = None) -> None:
    """Use 10 trials unless the caller explicitly selects trials."""
    args = sys.argv if argv is None else argv
    explicit_options: Sequence[str] = (
        "--num-trials-per-task",
        "--trial-indices",
    )
    for arg in args[1:]:
        if arg in explicit_options:
            return
        if any(arg.startswith(f"{option}=") for option in explicit_options):
            return
    args.extend(
        [
            "--num-trials-per-task",
            str(DEFAULT_TRIALS_PER_PERTURBED_TASK),
        ]
    )


def install_on_base_eval(base_eval) -> None:
    """Install the final-only LIBERO-Pro suite behavior on a base wrapper."""
    base_eval.selected_suites = selected_suites
    base_eval.patch_liberopro_suite_compat = patch_liberopro_suite_compat
