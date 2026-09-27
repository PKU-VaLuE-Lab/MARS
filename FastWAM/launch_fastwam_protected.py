#!/usr/bin/env python3
"""Launch one signal-protected FastWAM evaluation on an exclusive GPU."""

from __future__ import annotations

import argparse
import json
import os
import pwd
import re
import signal
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = ROOT / ".conda/bin/python"
SUPERVISOR = Path("/data/Albus/zzy/WAM_ICLR27/cosmos-policy/hidden_eval_supervisor.py")
HOLDER = Path("/home/zzy/skills/lock-and-reserve-gpus/scripts/zspeed_reserve.py")
UNIT_PATTERN = re.compile(r"^z-fastwam-[a-z0-9-]+$")


def systemctl_show(unit: str, prop: str) -> str:
    result = subprocess.run(
        ["systemctl", "show", unit, f"--property={prop}", "--value"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    return result.stdout.strip()


def stop_verified_holder(gpu: int) -> None:
    unit = f"zspeed-reserve-gpu{gpu}.service"
    if systemctl_show(unit, "LoadState") in {"", "not-found"}:
        return
    exec_start = systemctl_show(unit, "ExecStart")
    if str(HOLDER) not in exec_start:
        raise RuntimeError(f"refusing to stop unverified holder {unit}")
    pid = int(systemctl_show(unit, "MainPID") or "0")
    subprocess.run(["systemctl", "stop", "--no-block", unit], check=True)
    if pid > 1:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if systemctl_show(unit, "ActiveState") in {"inactive", "failed"}:
            subprocess.run(["systemctl", "reset-failed", unit], check=False)
            return
        time.sleep(0.1)
    raise TimeoutError(f"holder did not stop: {unit}")


def gpu_compute_pids(gpu: int) -> list[int]:
    uuid = subprocess.run(
        ["nvidia-smi", "-i", str(gpu), "--query-gpu=uuid", "--format=csv,noheader,nounits"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    output = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout
    pids = []
    for line in output.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) == 2 and fields[0] == uuid:
            pids.append(int(fields[1]))
    return pids


def environment(args: argparse.Namespace) -> dict[str, str]:
    cache = ROOT / ".cache"
    env = {
        "HOME": str(ROOT / ".home"),
        "USER": "zzy",
        "PATH": f"{PYTHON.parent}:/home/zzy/anaconda3/bin:/usr/local/bin:/usr/bin:/bin",
        "PYTHONPATH": str(ROOT),
        "CUDA_VISIBLE_DEVICES": str(args.gpu),
        "FASTWAM_PHYSICAL_GPU": str(args.gpu),
        "FASTWAM_ZSPEED_RESERVE": "1",
        "FASTWAM_ZSPEED_LEAVE_GIB": str(args.leave_gib),
        "FASTWAM_ACTIONAWARE_ENABLED": "0" if args.variant == "baseline" else "1",
        "FASTWAM_ACTIONAWARE_MODE": "probe" if args.variant == "probe" else "route",
        "FASTWAM_ACTIONAWARE_DRAFT_STEPS": str(args.draft_steps),
        "FASTWAM_ACTIONAWARE_TRANS_THRESHOLD": str(args.trans_threshold),
        "FASTWAM_ACTIONAWARE_FEATURE_STEPS": str(args.feature_steps),
        "FASTWAM_ACTION_CACHE_ENABLED": "1" if args.action_cache else "0",
        "FASTWAM_ACTION_CACHE_MODE": args.cache_mode,
        "FASTWAM_ACTION_CACHE_RELATIVE_L1_THRESHOLD": str(args.cache_relative_l1),
        "FASTWAM_ACTION_CACHE_RELATIVE_MAX_THRESHOLD": str(args.cache_relative_max),
        "FASTWAM_ACTION_CACHE_MAX_CONSECUTIVE_REUSE": str(args.cache_max_reuse),
        "FASTWAM_ACTION_CACHE_START_BLOCK": str(args.cache_start_block),
        "FASTWAM_ACTIONAWARE_REPORT_EVERY": "100",
        "FASTWAM_ACTIONAWARE_DATASET_STATS": str(
            ROOT / "checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json"
        ),
        "FASTWAM_RESULT_LOG": str(args.result_log.resolve()),
        "DIFFSYNTH_MODEL_BASE_PATH": str(ROOT / "checkpoints"),
        "DIFFSYNTH_DOWNLOAD_SOURCE": "huggingface",
        "LIBERO_CONFIG_PATH": str(ROOT / ".libero"),
        "HF_HOME": str(cache / "huggingface"),
        "HUGGINGFACE_HUB_CACHE": str(cache / "huggingface/hub"),
        "HF_DATASETS_CACHE": str(cache / "datasets"),
        "TRANSFORMERS_CACHE": str(cache / "huggingface/transformers"),
        "MODELSCOPE_CACHE": str(cache / "modelscope"),
        "MODELSCOPE_CACHE_DIR": str(cache / "modelscope"),
        "XDG_CACHE_HOME": str(cache),
        "PIP_CACHE_DIR": str(cache / "pip"),
        "TORCH_HOME": str(cache / "torch"),
        "CUDA_CACHE_PATH": str(cache / "nv"),
        "TRITON_CACHE_DIR": str(cache / "triton"),
        "TORCHINDUCTOR_CACHE_DIR": str(cache / "torchinductor"),
        "MPLCONFIGDIR": str(cache / "matplotlib"),
        "NUMBA_CACHE_DIR": str(cache / "numba"),
        "WANDB_DIR": str(cache / "wandb"),
        "TMPDIR": str(cache / f"tmp/gpu{args.gpu}"),
        "MUJOCO_GL": "egl",
        "PYOPENGL_PLATFORM": "egl",
        "MUJOCO_EGL_DEVICE_ID": str(args.gpu),
        "TOKENIZERS_PARALLELISM": "false",
        "WANDB_MODE": "disabled",
        "PYTHONUNBUFFERED": "1",
        "PYTHONNOUSERSITE": "1",
    }
    if args.variant == "probe":
        env["FASTWAM_ACTIONAWARE_TRACE_PATH"] = str(args.trace_path.resolve())
    return env


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit", required=True)
    parser.add_argument("--gpu", type=int, choices=tuple(range(6)), required=True)
    parser.add_argument("--variant", choices=("baseline", "probe", "acc", "speed"), required=True)
    parser.add_argument("--trials", type=int, required=True)
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument("--seed", type=int, default=195)
    parser.add_argument("--draft-steps", type=int, default=1)
    parser.add_argument("--trans-threshold", type=float, default=0.7)
    parser.add_argument("--feature-steps", type=int, default=10)
    parser.add_argument("--action-cache", action="store_true")
    parser.add_argument(
        "--cache-mode",
        choices=(
            "adaptive_tail",
            "first_residual",
            "interval_residual",
            "input_accum_residual",
        ),
        default="first_residual",
    )
    parser.add_argument("--cache-relative-l1", type=float, default=0.05)
    parser.add_argument("--cache-relative-max", type=float, default=float("inf"))
    parser.add_argument("--cache-max-reuse", type=int, default=2)
    parser.add_argument("--cache-start-block", type=int, default=1)
    parser.add_argument("--leave-gib", type=float, default=12.0)
    parser.add_argument("--suites", default="libero_goal,libero_object,libero_spatial,libero_10")
    parser.add_argument("--task-ids")
    parser.add_argument("--result-log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--trace-path", type=Path)
    args = parser.parse_args()

    if os.geteuid() != 0:
        raise PermissionError("run through sudo aa-exec -p cosmos-zspeed-controller")
    if not UNIT_PATTERN.fullmatch(args.unit):
        raise ValueError(f"invalid unit: {args.unit}")
    if (
        args.trials < 1
        or args.max_tasks < 0
        or args.draft_steps < 1
        or args.cache_max_reuse < 1
        or args.cache_start_block < 1
        or args.cache_relative_l1 < 0
        or args.cache_relative_max < 0
    ):
        raise ValueError("trials/draft-steps must be positive and max-tasks non-negative")
    if systemctl_show(args.unit, "LoadState") not in {"", "not-found"}:
        raise RuntimeError(f"unit already exists: {args.unit}")
    result_log = args.result_log.resolve()
    result_log.relative_to(ROOT)
    if result_log.suffix != ".log":
        raise ValueError("result-log must end in .log")
    output_dir = (args.output_dir or result_log.with_suffix("")).resolve()
    output_dir.relative_to(ROOT)
    trace_path = args.trace_path or result_log.with_name(result_log.stem + "_trace.jsonl")
    args.trace_path = trace_path
    for directory in (
        result_log.parent,
        output_dir,
        ROOT / ".cache/huggingface",
        ROOT / ".cache/modelscope",
        ROOT / ".cache/pip",
        ROOT / ".cache/torch",
        ROOT / ".cache/datasets",
        ROOT / ".cache/matplotlib",
        ROOT / ".cache/numba",
        ROOT / ".cache/wandb",
        ROOT / ".cache/nv",
        ROOT / ".cache/triton",
        ROOT / ".cache/torchinductor",
        ROOT / ".home",
        ROOT / f".cache/tmp/gpu{args.gpu}",
    ):
        directory.mkdir(parents=True, exist_ok=True)
        user = pwd.getpwnam("zzy")
        os.chown(directory, user.pw_uid, user.pw_gid)

    mode = subprocess.run(
        ["nvidia-smi", "-i", str(args.gpu), "--query-gpu=compute_mode", "--format=csv,noheader,nounits"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip().lower().replace("_", "")
    if mode != "exclusiveprocess":
        raise RuntimeError(f"GPU {args.gpu} is not EXCLUSIVE_PROCESS")
    stop_verified_holder(args.gpu)
    pids = gpu_compute_pids(args.gpu)
    if pids:
        raise RuntimeError(f"GPU {args.gpu} still has compute pids: {pids}")

    suites = [item.strip() for item in args.suites.split(",") if item.strip()]
    hydra_args = [
        "task=libero_uncond_2cam224_1e-4",
        f"ckpt={ROOT / 'checkpoints/fastwam_release/libero_uncond_2cam224.pt'}",
        "model.redirect_common_files=false",
        f"EVALUATION.dataset_stats_path={ROOT / 'checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json'}",
        f"EVALUATION.num_trials={args.trials}",
        f"EVALUATION.output_dir={output_dir}",
        "+EVALUATION.save_rollout_video=false",
        f"+EVALUATION.max_tasks={args.max_tasks}",
        "+EVALUATION.resume=true",
        f"MULTIRUN.task_suite_names=[{','.join(suites)}]",
        f"seed={args.seed}",
    ]
    if args.task_ids:
        hydra_args.append(f"+EVALUATION.task_ids=[{args.task_ids}]")

    launch_log = result_log.with_name(result_log.stem + "_launch.log")
    queue_log = result_log.with_name(result_log.stem + "_queue.log")
    spec = {
        "entry": str(ROOT / "fastwam_final/protected_eval_wrapper.py"),
        "args": hydra_args,
        "cwd": str(ROOT),
        "gpu": args.gpu,
        "env": environment(args),
        "unset_env": [],
        "result_log": str(result_log),
        "launch_log": str(launch_log),
        "queue_log": str(queue_log),
        "archive_root": str(ROOT / "fastwam_final/explorelogs/resilient_restarts" / args.unit),
        "completion_marker": "FASTWAM ALL final:",
    }
    code = 'import os,runpy;runpy.run_path(os.environ["HIDDEN_EVAL_ENTRY"],run_name="__main__")'
    shell = f'exec -a zspeed "$EVAL_PY" -c \'{code}\''
    command = [
        "systemd-run",
        f"--unit={args.unit}",
        "--uid=zzy",
        "--gid=zzy",
        "--property=Restart=on-failure",
        "--property=RestartSec=1s",
        "--property=StartLimitIntervalSec=0",
        "--property=Nice=-10",
        "--property=OOMScoreAdjust=-500",
        f"--setenv=EVAL_PY={PYTHON}",
        f"--setenv=HIDDEN_EVAL_ENTRY={SUPERVISOR}",
        f"--setenv=HIDDEN_EVAL_SPEC={json.dumps(spec, separators=(',', ':'))}",
        f"--setenv=PYTHONPATH={ROOT}",
        f"--setenv=TMPDIR={ROOT / f'.cache/tmp/gpu{args.gpu}'}",
        "/bin/bash",
        "-c",
        shell,
    ]
    subprocess.run(command, check=True)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        pid = int(systemctl_show(args.unit, "MainPID") or "0")
        if systemctl_show(args.unit, "ActiveState") == "active" and pid > 1:
            print(json.dumps({"unit": args.unit, "pid": pid, "gpu": args.gpu, "log": str(result_log)}))
            return 0
        time.sleep(0.1)
    raise RuntimeError(f"unit failed to start: {args.unit}")


if __name__ == "__main__":
    raise SystemExit(main())
