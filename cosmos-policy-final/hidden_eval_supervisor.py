#!/usr/bin/env python3
"""Run one evaluation with a neutral process command line and safe restart logs.

The complete specification is supplied through ``HIDDEN_EVAL_SPEC`` rather
than argv.  This keeps project paths out of ``/proc/<pid>/cmdline`` while the
evaluation itself remains the unmodified final Python entry point.
"""

from __future__ import annotations

import fcntl
import ctypes
import json
import os
import runpy
import shutil
import signal
import sys
import time
import traceback
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="microseconds")


def enter_signal_protection() -> None:
    """Enter the narrow AppArmor profile that rejects incoming SIGKILL.

    The profile otherwise grants normal runtime access and still accepts
    SIGTERM, so planned service transitions remain graceful.  Failing closed
    here is intentional: a systemd restart is safer than beginning another
    long evaluation without the requested interruption protection.
    """
    profile = os.environ.get("COSMOS_APPARMOR_PROFILE", "cosmos-zspeed-protect")
    if not profile:
        return
    library = ctypes.CDLL("libapparmor.so.1", use_errno=True)
    library.aa_change_profile.argtypes = [ctypes.c_char_p]
    library.aa_change_profile.restype = ctypes.c_int
    if library.aa_change_profile(profile.encode("utf-8")) != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number), profile)


def load_spec() -> dict[str, object]:
    raw = os.environ.get("HIDDEN_EVAL_SPEC", "")
    if not raw:
        raise RuntimeError("HIDDEN_EVAL_SPEC is required")
    spec = json.loads(raw)
    required = {
        "entry",
        "args",
        "cwd",
        "gpu",
        "env",
        "unset_env",
        "result_log",
        "launch_log",
        "queue_log",
        "archive_root",
        "completion_marker",
    }
    missing = required - spec.keys()
    if missing:
        raise ValueError(f"HIDDEN_EVAL_SPEC is missing keys: {sorted(missing)}")
    return spec


def contains_marker(path: Path, marker: str) -> bool:
    if not path.is_file() or not marker:
        return False
    with path.open("rb") as stream:
        stream.seek(0, os.SEEK_END)
        size = stream.tell()
        stream.seek(max(0, size - 4 * 1024 * 1024))
        tail = stream.read().decode("utf-8", errors="replace")
    return marker in tail


def archive_partial(paths: tuple[Path, ...], archive_root: Path) -> Path | None:
    existing = tuple(path for path in paths if path.exists())
    if not existing:
        return None
    stamp = datetime.now().astimezone().strftime("restart_%Y%m%d_%H%M%S_%f")
    destination = archive_root / stamp
    destination.mkdir(parents=True, exist_ok=False)
    for path in existing:
        shutil.move(str(path), str(destination / path.name))
    return destination


def queue_event(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", buffering=1) as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


@contextmanager
def redirected_output(path: Path, *, append: bool = False) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    saved_stdout_object = sys.stdout
    saved_stderr_object = sys.stderr
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8", buffering=1) as stream:
        os.dup2(stream.fileno(), 1)
        os.dup2(stream.fileno(), 2)
        try:
            yield
        finally:
            # Evaluation wrappers may temporarily replace and close their Tee
            # streams on an exception.  Restore the supervisor's Python stream
            # objects as well as the underlying file descriptors.
            try:
                sys.stdout.flush()
            except (OSError, ValueError):
                pass
            try:
                sys.stderr.flush()
            except (OSError, ValueError):
                pass
            sys.stdout = saved_stdout_object
            sys.stderr = saved_stderr_object
            os.dup2(saved_stdout, 1)
            os.dup2(saved_stderr, 2)
            os.close(saved_stdout)
            os.close(saved_stderr)


def hold_completed_gpu(queue_log: Path, gpu: int) -> None:
    queue_event(
        queue_log,
        {
            "event": "holder_start",
            "time": timestamp(),
            "pid": os.getpid(),
            "gpu": gpu,
        },
    )
    stopped = False

    def request_stop(_signum, _frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    while not stopped:
        time.sleep(30)


def main() -> int:
    # Set a neutral comm before importing torch or any project module.  The
    # service launcher also supplies a neutral argv[0], so broad external
    # ``pkill python`` / ``pkill -f <conda path>`` rules cannot select this job.
    ctypes.CDLL(None).prctl(15, b"zspeed", 0, 0, 0)
    enter_signal_protection()
    # ``bash exec -a`` deliberately hides the interpreter path in argv[0].
    # CPython then cannot infer ``sys.executable`` and leaves it empty, while
    # Transformer Engine legitimately uses it for a tiny package probe.  Keep
    # the neutral process argv but restore the real executable for subprocess
    # callers inside Python.
    eval_python = os.environ.get("EVAL_PY", "")
    if eval_python and Path(eval_python).is_file():
        sys.executable = eval_python
    spec = load_spec()
    entry = Path(str(spec["entry"])).resolve()
    cwd = Path(str(spec["cwd"])).resolve()
    gpu = int(spec["gpu"])
    result_log = Path(str(spec["result_log"])).resolve()
    launch_log = Path(str(spec["launch_log"])).resolve()
    queue_log = Path(str(spec["queue_log"])).resolve()
    archive_root = Path(str(spec["archive_root"])).resolve()
    completion_marker = str(spec["completion_marker"])
    resume_existing = bool(spec.get("resume_existing", False))
    lock_path = Path(f"/tmp/cosmos-policy-gpu-{gpu}.lock")

    if not entry.is_file():
        raise FileNotFoundError(entry)
    cwd.mkdir(parents=True, exist_ok=True)
    result_log.parent.mkdir(parents=True, exist_ok=True)
    archive_root.mkdir(parents=True, exist_ok=True)

    with lock_path.open("a+") as lock_stream:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
        if contains_marker(result_log, completion_marker):
            queue_event(
                queue_log,
                {
                    "event": "complete_result_reused",
                    "time": timestamp(),
                    "pid": os.getpid(),
                    "gpu": gpu,
                    "result_log": str(result_log),
                },
            )
            hold_completed_gpu(queue_log, gpu)
            return 0

        archived = (
            None
            if resume_existing
            else archive_partial((result_log, launch_log, queue_log), archive_root)
        )
        for name in spec["unset_env"]:
            os.environ.pop(str(name), None)
        for name, value in dict(spec["env"]).items():
            os.environ[str(name)] = str(value)
        # PYTHONPATH is normally consumed only while CPython initializes.
        # Hidden jobs intentionally receive their detailed environment after
        # start, so mirror it into sys.path before importing benchmark modules.
        python_path = os.environ.get("PYTHONPATH", "")
        for item in reversed(python_path.split(os.pathsep)):
            if item and item not in sys.path:
                sys.path.insert(0, item)
        # Establish this zspeed process as the GPU's CUDA-context owner before
        # the much slower model imports/checkpoint load.  In EXCLUSIVE_PROCESS
        # mode this closes the window in which an unrelated process could win
        # the device after the coordinator's final idle check.
        import torch

        torch.cuda.set_device(0)
        cuda_claim = torch.empty(1, device="cuda")
        # One in-process Inductor compiler per evaluation is ample here and
        # avoids spawning 32 short-lived Python workers on each of six cards.
        # This affects only first-use compilation, never the inference graph.
        os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
        os.environ["COSMOS_GPU_LOCK_HELD"] = "1"

        queue_event(
            queue_log,
            {
                "event": "eval_start",
                "time": timestamp(),
                "pid": os.getpid(),
                "uid": os.getuid(),
                "gpu": gpu,
                "entry": str(entry),
                "result_log": str(result_log),
                "archived_previous": str(archived) if archived is not None else None,
                "resume_existing": resume_existing,
            },
        )
        old_argv = sys.argv
        old_cwd = Path.cwd()
        rc = 0
        try:
            os.chdir(cwd)
            sys.argv = [str(entry), *(str(item) for item in spec["args"])]
            with redirected_output(launch_log, append=resume_existing):
                try:
                    runpy.run_path(str(entry), run_name="__main__")
                except SystemExit as exc:
                    rc = int(exc.code or 0) if isinstance(exc.code, (int, type(None))) else 1
                except BaseException:
                    traceback.print_exc(file=sys.__stderr__)
                    rc = 1
        finally:
            sys.argv = old_argv
            os.chdir(old_cwd)
        queue_event(
            queue_log,
            {
                "event": "eval_done",
                "time": timestamp(),
                "pid": os.getpid(),
                "uid": os.getuid(),
                "gpu": gpu,
                "rc": rc,
                "result_log": str(result_log),
            },
        )
        if rc != 0:
            return rc
        if not contains_marker(result_log, completion_marker):
            queue_event(
                queue_log,
                {
                    "event": "completion_marker_missing",
                    "time": timestamp(),
                    "gpu": gpu,
                    "marker": completion_marker,
                },
            )
            return 3
        hold_completed_gpu(queue_log, gpu)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
