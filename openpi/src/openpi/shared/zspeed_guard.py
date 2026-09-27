"""Keep OpenPI inference visibly named and reserve otherwise-idle VRAM."""

from __future__ import annotations

import atexit
import ctypes
import dataclasses
import json
import os
from pathlib import Path
import time
from typing import Any


@dataclasses.dataclass
class _ReservationState:
    reservation: Any | None = None
    marker_path: Path | None = None


_STATE = _ReservationState()


def enabled() -> bool:
    return os.environ.get("OPENPI_ZSPEED_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on"}


def set_process_name() -> None:
    if not enabled():
        return
    name = os.environ.get("OPENPI_ZSPEED_PROCESS_NAME", "zspeed").encode("utf-8")[:15]
    ctypes.CDLL(None).prctl(15, ctypes.c_char_p(name), 0, 0, 0)


def reserve_remaining_vram() -> None:
    """Reserve free VRAM after the first compiled query while retaining a margin."""
    if not enabled() or _STATE.reservation is not None:
        return

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("OPENPI zspeed guard requires CUDA")

    leave_gib = float(os.environ.get("OPENPI_ZSPEED_LEAVE_GIB", "10"))
    minimum_gib = float(os.environ.get("OPENPI_ZSPEED_MINIMUM_GIB", "1"))
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    gib = 1024**3
    candidate_bytes = int(free_bytes - leave_gib * gib)
    minimum_bytes = int(minimum_gib * gib)
    step_bytes = 512 * 1024**2
    last_error: Exception | None = None
    while candidate_bytes >= minimum_bytes:
        try:
            _STATE.reservation = torch.empty(candidate_bytes, dtype=torch.uint8, device="cuda")
            break
        except torch.OutOfMemoryError as exc:
            last_error = exc
            candidate_bytes -= step_bytes
            torch.cuda.empty_cache()
    if _STATE.reservation is None:
        raise RuntimeError(
            f"OpenPI zspeed guard could not reserve VRAM while leaving {leave_gib:.1f} GiB"
        ) from last_error

    torch.cuda.synchronize()
    free_after, _ = torch.cuda.mem_get_info()
    physical_gpu = os.environ.get("OPENPI_PHYSICAL_GPU", "unknown")
    marker_dir = Path(os.environ.get("OPENPI_ZSPEED_MARKER_DIR", "/tmp"))
    marker_dir.mkdir(parents=True, exist_ok=True)
    _STATE.marker_path = marker_dir / f"openpi-zspeed-gpu{physical_gpu}.json"
    payload = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "free_after_gib": round(free_after / gib, 3),
        "leave_target_gib": leave_gib,
        "physical_gpu": physical_gpu,
        "pid": os.getpid(),
        "process_name": os.environ.get("OPENPI_ZSPEED_PROCESS_NAME", "zspeed"),
        "tensor_gib": round(candidate_bytes / gib, 3),
        "total_gib": round(total_bytes / gib, 3),
    }
    _STATE.marker_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "[OpenPI zspeed] reserved "
        f"{candidate_bytes / gib:.2f} GiB on physical GPU {physical_gpu}; "
        f"free={free_after / gib:.2f} GiB",
        flush=True,
    )


def _cleanup() -> None:
    if _STATE.marker_path is not None:
        _STATE.marker_path.unlink(missing_ok=True)


atexit.register(_cleanup)
