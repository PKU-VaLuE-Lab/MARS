#!/usr/bin/env python3
"""Run the LIBERO-Pro DiCache comparison with the canonical runtime patch."""

from __future__ import annotations

import importlib.util
import sys
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FINAL_DIR = Path(__file__).resolve().parent
CANONICAL_DIR = ROOT / "liberopro_scripts" / "wrapperfinal"
CANONICAL_PATH = CANONICAL_DIR / "eval_dicache_wrapper.py"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CANONICAL_DIR))
sys.path.insert(0, str(FINAL_DIR))

from disable_flash_attention import install as install_no_flash

install_no_flash()

spec = importlib.util.spec_from_file_location("_liberopro_canonical_dicache", CANONICAL_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load DiCache wrapper: {CANONICAL_PATH}")
dicache = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = dicache
spec.loader.exec_module(dicache)

from cosmos_policy.experiments.robot import cosmos_utils
from gpu_speed_guard import install_on_cosmos_utils

install_on_cosmos_utils(cosmos_utils)

from suite_config import ensure_default_trial_count, install_on_base_eval

install_on_base_eval(dicache.eval_wrapper)


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
    dicache.eval_wrapper.main()
    final_lines = StringIO()
    with redirect_stdout(final_lines):
        dicache._finalize_dicache_stats()
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
