#!/usr/bin/env python3
"""Keep a protected GPU reserved after its evaluation queue has completed."""

from __future__ import annotations

import os
import signal
import time

from gpu_speed_guard import reserve_remaining_vram


def main() -> None:
    reserve_remaining_vram()
    physical_gpu = os.environ.get("COSMOS_PHYSICAL_GPU", "unknown")
    print(
        f"[测速度] GPU {physical_gpu} evaluation queue is complete; "
        "holding the idle tensor indefinitely.",
        flush=True,
    )

    stop = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    while not stop:
        time.sleep(30)


if __name__ == "__main__":
    main()
