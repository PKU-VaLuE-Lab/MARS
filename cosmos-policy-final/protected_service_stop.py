#!/usr/bin/env python3
"""Stop selected protected transient evaluation services without a global gap."""

from __future__ import annotations

import argparse
import os
import re
import signal
import subprocess
import time
from pathlib import Path


UNIT_PATTERN = re.compile(r"^z-[a-z0-9-]+(?:\.service)?$")
CONTROLLER_PROFILE = "cosmos-zspeed-controller"


def systemctl_show(unit: str, property_name: str) -> str:
    result = subprocess.run(
        ["systemctl", "show", unit, f"--property={property_name}", "--value"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return result.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("units", nargs="+")
    parser.add_argument("--timeout", type=float, default=45.0)
    args = parser.parse_args()

    if os.geteuid() != 0:
        raise PermissionError("protected service transitions must run as root")
    current_profile = Path("/proc/self/attr/current").read_text(encoding="utf-8").strip()
    if not current_profile.startswith(f"{CONTROLLER_PROFILE} "):
        raise RuntimeError(
            f"run through aa-exec -p {CONTROLLER_PROFILE}; current={current_profile!r}"
        )

    units = []
    pids: dict[str, int] = {}
    for raw_unit in args.units:
        if not UNIT_PATTERN.fullmatch(raw_unit):
            raise ValueError(f"refusing unexpected unit name: {raw_unit!r}")
        unit = raw_unit if raw_unit.endswith(".service") else f"{raw_unit}.service"
        units.append(unit)
        pids[unit] = int(systemctl_show(unit, "MainPID") or "0")

    subprocess.run(["systemctl", "stop", "--no-block", *units], check=True)
    for unit, pid in pids.items():
        if pid <= 1:
            continue
        try:
            os.kill(pid, signal.SIGTERM)
            print(f"controller_sigterm unit={unit} pid={pid}", flush=True)
        except ProcessLookupError:
            pass

    deadline = time.monotonic() + args.timeout
    remaining = set(units)
    while remaining and time.monotonic() < deadline:
        for unit in tuple(remaining):
            if systemctl_show(unit, "ActiveState") in {"inactive", "failed"}:
                remaining.remove(unit)
        if remaining:
            time.sleep(0.25)
    if remaining:
        raise TimeoutError(f"protected units did not stop: {sorted(remaining)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
