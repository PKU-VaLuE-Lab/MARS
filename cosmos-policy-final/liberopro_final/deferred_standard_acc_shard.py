#!/usr/bin/env python3
"""Wait for a GPU without disturbing its owner, then launch one Standard-Acc shard."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "launch_protected_eval.py"
STOPPER = ROOT / "protected_service_stop.py"
UNIT_PATTERN = re.compile(r"^z-[a-z0-9-]+$")


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def systemctl_show(unit: str, prop: str) -> str:
    return subprocess.run(
        ["systemctl", "show", unit, f"--property={prop}", "--value"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    ).stdout.strip()


def compute_pids(gpu: int) -> list[int]:
    output = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(gpu),
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout
    return [int(line.strip()) for line in output.splitlines() if line.strip()]


def wait_for_completion_json(path: Path, expected_episodes: int) -> None:
    last_report = 0.0
    while True:
        if path.is_file():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if int(payload.get("episodes", -1)) == expected_episodes:
                    print(
                        f"[{timestamp()}] prerequisite complete: {path} "
                        f"episodes={expected_episodes}",
                        flush=True,
                    )
                    return
            except (OSError, ValueError, json.JSONDecodeError):
                pass
        now = time.monotonic()
        if now - last_report >= 60:
            print(f"[{timestamp()}] waiting for completion JSON: {path}", flush=True)
            last_report = now
        time.sleep(2)


def stop_protected_unit(unit: str) -> None:
    load_state = systemctl_show(unit, "LoadState")
    if load_state in {"", "not-found"}:
        return
    subprocess.run(
        [
            "/usr/bin/aa-exec",
            "-p",
            "cosmos-zspeed-controller",
            "--",
            "/usr/bin/python3",
            str(STOPPER),
            unit,
        ],
        check=True,
    )


def wait_for_empty_gpu(gpu: int) -> None:
    last_report = 0.0
    while True:
        pids = compute_pids(gpu)
        if not pids:
            return
        now = time.monotonic()
        if now - last_report >= 60:
            print(
                f"[{timestamp()}] GPU {gpu} occupied by pids={pids}; waiting without terminating them",
                flush=True,
            )
            last_report = now
        time.sleep(2)


def ensure_exclusive(gpu: int) -> None:
    mode = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(gpu),
            "--query-gpu=compute_mode",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip().lower().replace("_", "")
    if mode != "exclusiveprocess":
        subprocess.run(["nvidia-smi", "-i", str(gpu), "-c", "EXCLUSIVE_PROCESS"], check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit", required=True, help="evaluation unit to launch")
    parser.add_argument("--gpu", type=int, choices=range(6), required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--result-log", type=Path, required=True)
    parser.add_argument("--wait-for-json", type=Path)
    parser.add_argument("--expected-episodes", type=int, default=2000)
    parser.add_argument("--stop-unit")
    args = parser.parse_args()

    if os.geteuid() != 0:
        raise PermissionError("deferred shard launcher must run as root")
    for unit in (args.unit, args.stop_unit):
        if unit and not UNIT_PATTERN.fullmatch(unit):
            raise ValueError(f"invalid unit: {unit!r}")
    if args.expected_episodes < 1:
        raise ValueError("expected episodes must be positive")

    load_state = systemctl_show(args.unit, "LoadState")
    if load_state not in {"", "not-found"}:
        if systemctl_show(args.unit, "ActiveState") == "active":
            print(f"[{timestamp()}] evaluation unit already active: {args.unit}", flush=True)
            return 0
        raise RuntimeError(f"evaluation unit already exists but is not active: {args.unit}")

    if args.wait_for_json is not None:
        wait_for_completion_json(args.wait_for_json.resolve(), args.expected_episodes)
    if args.stop_unit:
        stop_protected_unit(args.stop_unit)

    # Never evict an unrelated compute process. The queue remains here until
    # every current owner exits voluntarily, then closes the race with
    # EXCLUSIVE_PROCESS before starting the protected evaluation service.
    wait_for_empty_gpu(args.gpu)
    ensure_exclusive(args.gpu)
    wait_for_empty_gpu(args.gpu)

    command = [
        "/usr/bin/python3",
        str(LAUNCHER),
        "--unit",
        args.unit,
        "--benchmark",
        "liberopro",
        "--mechanism",
        "standard",
        "--gpu",
        str(args.gpu),
        "--trials-per-task",
        "50",
        "--seed",
        "195",
        "--selection",
        args.selection,
        "--topk",
        "0.15",
        "--wrist-iou",
        "0.98",
        "--primary-iou",
        "0.90",
        "--primary-maxskip",
        "1",
        "--wrist-maxskip",
        "1",
        "--result-log",
        str(args.result_log.resolve()),
    ]
    subprocess.run(command, check=True)
    print(
        f"[{timestamp()}] launched Standard Acc unit={args.unit} "
        f"gpu={args.gpu} selection={args.selection}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
