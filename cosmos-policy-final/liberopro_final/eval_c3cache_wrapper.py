#!/usr/bin/env python3
"""Run LIBERO-Pro with only the shared C3Cache residual accelerator."""

from __future__ import annotations

import importlib.util
import sys
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FINAL_DIR = Path(__file__).resolve().parent
CANONICAL_DIR = ROOT / "liberopro_scripts" / "wrapperfinal"
CANONICAL_PATH = CANONICAL_DIR / "eval_wrapper.py"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(FINAL_DIR))
sys.path.insert(0, str(CANONICAL_DIR))

from disable_flash_attention import install as install_no_flash

install_no_flash()

from cosmos_policy.experiments.robot import cosmos_utils
from c3cache_runtime import finalize_stats, install_episode_reset, install_on_cosmos_utils
from gpu_speed_guard import install_on_cosmos_utils as install_speed_guard

install_on_cosmos_utils(cosmos_utils)
install_speed_guard(cosmos_utils)

spec = importlib.util.spec_from_file_location(
    "_liberopro_canonical_c3cache_eval",
    CANONICAL_PATH,
)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load LIBERO-Pro eval wrapper: {CANONICAL_PATH}")
base_eval = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base_eval
spec.loader.exec_module(base_eval)

from suite_config import ensure_default_trial_count, install_on_base_eval

install_on_base_eval(base_eval)
install_episode_reset(base_eval)


def _explicit_log_path() -> Path | None:
    args = sys.argv[1:]
    for index, arg in enumerate(args):
        if arg == "--log-path" and index + 1 < len(args):
            return Path(args[index + 1]).expanduser().resolve()
        if arg.startswith("--log-path="):
            return Path(arg.split("=", 1)[1]).expanduser().resolve()
    return None


def main() -> None:
    ensure_default_trial_count()
    base_eval.main()
    final_lines = StringIO()
    with redirect_stdout(final_lines):
        finalize_stats()
    text = final_lines.getvalue()
    if not text:
        return
    print(text, end="")
    log_path = _explicit_log_path()
    if log_path is not None:
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(text)


if __name__ == "__main__":
    main()
