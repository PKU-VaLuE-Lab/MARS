#!/usr/bin/env python3
"""Keep an idle VRAM reservation inside the evaluation CUDA process."""

from __future__ import annotations

import atexit
import ctypes
import json
import os
import time
from pathlib import Path
from typing import Any


_RESERVATION: Any | None = None
_MARKER_PATH: Path | None = None


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _set_process_name() -> None:
    name = os.environ.get("COSMOS_SPEED_GUARD_NAME", "测速度").encode("utf-8")[:15]
    try:
        libc = ctypes.CDLL(None)
        libc.prctl(15, ctypes.c_char_p(name), 0, 0, 0)  # PR_SET_NAME
    except Exception as exc:
        print(f"[测速度] process-name warning: {exc}")


def _cleanup_marker() -> None:
    global _MARKER_PATH
    if _MARKER_PATH is None:
        return
    try:
        _MARKER_PATH.unlink(missing_ok=True)
    except OSError:
        pass
    _MARKER_PATH = None


def reserve_remaining_vram() -> None:
    """Allocate one untouched tensor while retaining a configurable free margin."""
    global _MARKER_PATH, _RESERVATION

    if _RESERVATION is not None or not _env_bool("COSMOS_SPEED_GUARD_ENABLED", True):
        return

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("[测速度] CUDA is unavailable; cannot install the VRAM guard")

    leave_gib = float(os.environ.get("COSMOS_SPEED_GUARD_LEAVE_GIB", "10"))
    minimum_gib = float(os.environ.get("COSMOS_SPEED_GUARD_MIN_TENSOR_GIB", "1"))
    if leave_gib <= 0 or minimum_gib < 0:
        raise ValueError("COSMOS speed-guard memory values must be positive")

    _set_process_name()
    torch.cuda.synchronize()
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    gib = 1024**3
    requested_bytes = int(free_bytes - leave_gib * gib)
    minimum_bytes = int(minimum_gib * gib)
    if requested_bytes < minimum_bytes:
        raise RuntimeError(
            "[测速度] insufficient free VRAM for a safe reservation: "
            f"free={free_bytes / gib:.2f} GiB leave={leave_gib:.2f} GiB"
        )

    # A single uint8 tensor is deliberately left untouched. If fragmentation
    # prevents the first allocation, step down while preserving the free margin.
    step_bytes = 512 * 1024**2
    candidate_bytes = requested_bytes
    last_error: Exception | None = None
    while candidate_bytes >= minimum_bytes:
        try:
            _RESERVATION = torch.empty(
                candidate_bytes,
                dtype=torch.uint8,
                device=torch.device("cuda", torch.cuda.current_device()),
            )
            break
        except torch.OutOfMemoryError as exc:
            last_error = exc
            candidate_bytes -= step_bytes
            torch.cuda.empty_cache()

    if _RESERVATION is None:
        raise RuntimeError("[测速度] failed to reserve a safe VRAM tensor") from last_error

    torch.cuda.synchronize()
    free_after, _ = torch.cuda.mem_get_info()
    physical_gpu = os.environ.get("COSMOS_PHYSICAL_GPU", "unknown")
    marker_dir = Path(os.environ.get("COSMOS_SPEED_GUARD_MARKER_DIR", "/tmp"))
    marker_dir.mkdir(parents=True, exist_ok=True)
    _MARKER_PATH = marker_dir / f"测速度-gpu{physical_gpu}.json"
    payload = {
        "pid": os.getpid(),
        "physical_gpu": physical_gpu,
        "tensor_gib": round(candidate_bytes / gib, 3),
        "leave_target_gib": leave_gib,
        "free_after_gib": round(free_after / gib, 3),
        "total_gib": round(total_bytes / gib, 3),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    _MARKER_PATH.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "[测速度] idle tensor installed in evaluation process: "
        f"pid={os.getpid()} physical_gpu={physical_gpu} "
        f"tensor={candidate_bytes / gib:.2f} GiB free_after={free_after / gib:.2f} GiB"
    )


def install_on_cosmos_utils(cosmos_utils: Any) -> None:
    """Wrap the final get_model chain exactly once."""
    current = cosmos_utils.get_model
    if getattr(current, "_cosmos_speed_guard", False):
        return

    def get_model_with_speed_guard(cfg: Any):
        model, config = current(cfg)
        reserve_remaining_vram()
        return model, config

    get_model_with_speed_guard._cosmos_speed_guard = True
    cosmos_utils.get_model = get_model_with_speed_guard


atexit.register(_cleanup_marker)
