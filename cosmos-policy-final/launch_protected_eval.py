#!/usr/bin/env python3
"""Launch one restartable, signal-protected final evaluation."""

from __future__ import annotations

import argparse
import json
import os
import pwd
import re
import subprocess
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
EVAL_PYTHON = Path("/home/zzy/anaconda3/envs/liberocosmos/bin/python")
HIDDEN_SUPERVISOR = PROJECT_ROOT / "hidden_eval_supervisor.py"
APPARMOR_POLICY = PROJECT_ROOT / "cosmos_zspeed_protect.apparmor"
CPU_ISOLATION = Path(
    "/home/zzy/skills/lock-and-reserve-gpus/scripts/zspeed_cpu_isolation.py"
)
UNIT_PATTERN = re.compile(r"^z-[a-z0-9-]+$")
SHALLOW_GATE_LAYERS = (
    "rgb",
    "conv1",
    "res0",
    "res1",
    "down0",
    "stage1_res0",
    "stage1_res1",
    "down1",
    "stage2_res0",
    "stage2_res1",
    "down2",
    "stage3_res0",
    "stage3_res1",
    "middle_res0",
    "middle_attn",
    "middle_res1",
    "head_norm",
    "head_act",
    "encoder_out",
)

MECHANISMS = {
    "baseline": "",
    "c3cache": "",
    "xcache": "",
    "dicache": "dicache",
    "standard": "actionaware,selfcross,vaeskip",
    "actionaware_selfcross": "actionaware,selfcross",
    "actionaware_vaeskip": "actionaware,vaeskip",
    "selfcross_vaeskip": "selfcross,vaeskip",
}

UNSET_ENV = [
    "COSMOS_C3CACHE_STATS_PATH",
    "COSMOS_XCACHE_NUM_STEPS",
    "COSMOS_XCACHE_FINGERPRINT_TOKENS",
    "COSMOS_XCACHE_ALPHA",
    "COSMOS_XCACHE_TAU_FLOOR",
    "COSMOS_XCACHE_MARGIN",
    "COSMOS_XCACHE_DEV_THRESHOLD",
    "COSMOS_XCACHE_FRONT_ANCHOR",
    "COSMOS_XCACHE_VIEW_GROUPS",
    "COSMOS_XCACHE_VERBOSE",
    "COSMOS_XCACHE_REPORT_EVERY",
    "COSMOS_XCACHE_TRACE_LIMIT",
    "COSMOS_XCACHE_STATS_PATH",
    "COSMOS_DICACHE_STATS_PATH",
    "COSMOS_DICACHE_WARMUP_STEPS",
    "COSMOS_SHALLOW_GATE_TRACE_PATH",
    "COSMOS_SHALLOW_GATE_ANALYSIS_TOPKS",
    "COSMOS_SHALLOW_GATE_DISABLE_REUSE",
    "COSMOS_SHALLOW_GATE_SUITE_LAYER",
    "COSMOS_SHALLOW_GATE_SUITE_TOPK",
    "COSMOS_SHALLOW_GATE_SUITE_IOU_THRESHOLD",
    "COSMOS_SHALLOW_GATE_SUITE_ENABLED",
    "COSMOS_SHALLOW_GATE_SUITE_MAX_CONSECUTIVE_SKIPS",
    "COSMOS_SHALLOW_GATE_SUITE_CAMERA_MODE",
    "COSMOS_SHALLOW_GATE_SUITE_PRIMARY_GUARD_THRESHOLD",
    "COSMOS_SHALLOW_GATE_SUITE_FORCE_FULL_INTERVAL",
    "COSMOS_SHALLOW_GATE_SUITE_FORCE_FULL_AFTER_1STEP",
]


def systemctl_show(unit: str, property_name: str) -> str:
    result = subprocess.run(
        ["systemctl", "show", unit, f"--property={property_name}", "--value"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    return result.stdout.strip()


def cpu_isolation_plan(gpu: int) -> dict[str, object]:
    result = subprocess.run(
        [str(EVAL_PYTHON), str(CPU_ISOLATION), "plan", "--gpu", str(gpu)],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return json.loads(result.stdout)


def reserve_cpu_isolation(gpu: int, unit: str) -> None:
    subprocess.run(
        [
            str(EVAL_PYTHON),
            str(CPU_ISOLATION),
            "reserve",
            "--gpu",
            str(gpu),
            "--unit",
            f"{unit}.service",
        ],
        check=True,
    )


def release_cpu_isolation(gpu: int) -> None:
    subprocess.run(
        [str(EVAL_PYTHON), str(CPU_ISOLATION), "release", "--gpu", str(gpu)],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def build_environment(args: argparse.Namespace) -> dict[str, str]:
    purelib = EVAL_PYTHON.parent.parent / "lib/python3.10/site-packages"
    nvidia_root = purelib / "nvidia"
    nvidia_components = (
        "nccl",
        "cublas",
        "cufile",
        "curand",
        "cuda_nvrtc",
        "cuda_cupti",
        "cusparse",
        "cusolver",
        "cudnn",
        "cufft",
        "nvtx",
        "cuda_runtime",
        "nvjitlink",
    )
    library_path = ":".join(
        str(nvidia_root / component / "lib") for component in nvidia_components
    )
    final_dir = PROJECT_ROOT / f"{args.benchmark}_final"
    config_dir = None
    if args.benchmark in {"libero", "liberopro"}:
        config_dir = PROJECT_ROOT / (
            ".libero" if args.benchmark == "libero" else ".liberopro"
        )
    env = {
        "HOME": "/home/zzy",
        "USER": "zzy",
        "PATH": (
            "/home/zzy/anaconda3/envs/liberocosmos/bin:/home/zzy/anaconda3/bin:"
            "/home/zzy/anaconda3/condabin:/usr/local/sbin:/usr/local/bin:"
            "/usr/sbin:/usr/bin:/sbin:/bin"
        ),
        "CUDA_HOME": str(nvidia_root),
        "CUDA_PATH": str(nvidia_root),
        "LD_LIBRARY_PATH": library_path,
        "CUDA_VISIBLE_DEVICES": str(args.gpu),
        "COSMOS_PHYSICAL_GPU": str(args.gpu),
        "COSMOS_SPEED_GUARD_ENABLED": "1",
        "COSMOS_SPEED_GUARD_LEAVE_GIB": "10",
        "COSMOS_SPEED_GUARD_MIN_TENSOR_GIB": "1",
        "COSMOS_SPEED_GUARD_NAME": "测速度",
        "HF_HOME": "/home/zzy/.cache/cosmos-policy/huggingface",
        "TRANSFORMERS_CACHE": "/home/zzy/.cache/cosmos-policy/huggingface/transformers",
        "WANDB_MODE": "disabled",
        "MUJOCO_GL": "egl",
        "PYOPENGL_PLATFORM": "egl",
        "MUJOCO_EGL_DEVICE_ID": str(args.gpu),
        "EGL_DEVICE_ID": str(args.gpu),
        "COSMOS_EVAL_VARIANT": args.mechanism,
        "COSMOS_SHALLOW_GATE_PROFILE": "uniform",
        "COSMOS_SHALLOW_GATE_LAYER": args.shallow_layer,
        "COSMOS_SHALLOW_GATE_TOPK": str(args.topk),
        "COSMOS_SHALLOW_GATE_IOU_THRESHOLD": str(args.wrist_iou),
        "COSMOS_SHALLOW_GATE_CAMERA_MODE": "hybrid",
        "COSMOS_SHALLOW_GATE_PRIMARY_GUARD_THRESHOLD": str(args.primary_iou),
        "COSMOS_SHALLOW_GATE_MAX_CONSECUTIVE_SKIPS": "1",
        "COSMOS_SHALLOW_GATE_PRIMARY_MAX_CONSECUTIVE_SKIPS": str(args.primary_maxskip),
        "COSMOS_SHALLOW_GATE_WRIST_MAX_CONSECUTIVE_SKIPS": str(args.wrist_maxskip),
        "COSMOS_SHALLOW_GATE_QUERY_LAG": "1",
        "COSMOS_SHALLOW_GATE_FORCE_FULL_INTERVAL": "0",
        "COSMOS_SHALLOW_GATE_FORCE_FULL_AFTER_1STEP": "False",
        "COSMOS_SHALLOW_GATE_SUMMARY_PATH": f"/tmp/{args.unit}-vae-summary.json",
        # A single representative post-evaluation query is profiled after all
        # timed episodes finish, so profiler overhead never enters querytime.
        "COSMOS_SHALLOW_GATE_FLOPS_PATH": str(
            args.result_log.resolve().with_suffix(".flops.json")
        ),
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
        "PYTHONUNBUFFERED": "1",
    }
    if config_dir is not None:
        env["LIBERO_CONFIG_PATH"] = str(config_dir)
    mechanisms = MECHANISMS[args.mechanism]
    mechanisms_env = {
        "libero": "LIBERO_FINAL_MECHANISMS",
        "liberopro": "LIBEROPRO_FINAL_MECHANISMS",
        "robocasa": "ROBOCASA_FINAL_MECHANISMS",
    }[args.benchmark]
    env[mechanisms_env] = mechanisms
    if args.mechanism == "c3cache":
        env.update(
            {
                "COSMOS_C3CACHE_NUM_STEPS": "5",
                "COSMOS_C3CACHE_CACHED_STEPS": args.cached_steps,
                "COSMOS_C3CACHE_REFRESH_INTERVAL": str(args.refresh_interval),
                "COSMOS_C3CACHE_VERBOSE": "True",
                "COSMOS_C3CACHE_REPORT_EVERY": "200",
            }
        )
    if args.mechanism == "xcache":
        env.update(
            {
                "COSMOS_XCACHE_NUM_STEPS": "5",
                "COSMOS_XCACHE_FINGERPRINT_TOKENS": str(args.fingerprint_tokens),
                "COSMOS_XCACHE_ALPHA": str(args.xcache_alpha),
                "COSMOS_XCACHE_TAU_FLOOR": str(args.tau_floor),
                "COSMOS_XCACHE_MARGIN": str(args.margin),
                "COSMOS_XCACHE_DEV_THRESHOLD": str(args.dev_threshold),
                "COSMOS_XCACHE_FRONT_ANCHOR": str(args.front_anchor),
                "COSMOS_XCACHE_VIEW_GROUPS": args.view_groups,
                "COSMOS_XCACHE_VERBOSE": "True",
                "COSMOS_XCACHE_REPORT_EVERY": "200",
                "COSMOS_XCACHE_TRACE_LIMIT": str(args.xcache_trace_limit),
            }
        )
    if "actionaware" in mechanisms:
        if args.benchmark == "robocasa":
            env.update(
                {
                    "COSMOS_ACTIONAWARE_MODE": "route",
                    "COSMOS_ACTIONAWARE_GATE_CONFIG_JSON": (
                        '{"episode_latch":false,"logic":"any",'
                        '"name":"robocasa_actionaware_translation_0525",'
                        '"rules":[{"feature":"trans_mag_mean","op":"<=",'
                        '"threshold":0.525}],"type":"rules"}'
                    ),
                    "COSMOS_ACTIONAWARE_TRANS_THRESHOLD": "0.525",
                    "COSMOS_ACTIONAWARE_CHUNK_SIZE": "32",
                    "COSMOS_ACTIONAWARE_FEATURE_STEPS": "16",
                    "COSMOS_ACTIONAWARE_DATASET_STATS": str(
                        PROJECT_ROOT
                        / "robocasa_scripts/ckpt/robocasa_dataset_statistics.json"
                    ),
                    "COSMOS_ACTIONAWARE_REPORT_EVERY": "200",
                }
            )
        else:
            env.update(
                {
                    "COSMOS_ACTIONAWARE_MODE": "route",
                    "COSMOS_ACTIONAWARE_GATE_CONFIG_JSON": (
                        '{"episode_latch":false,"logic":"any",'
                        '"name":"actionaware_perquery_low_translation_070",'
                        '"rules":[{"feature":"trans_mag_mean","op":"<=",'
                        '"threshold":0.7}],"type":"rules"}'
                    ),
                    "COSMOS_ACTIONAWARE_TRANS_THRESHOLD": "0.7",
                    "COSMOS_ACTIONAWARE_CHUNK_SIZE": "16",
                    "COSMOS_ACTIONAWARE_FEATURE_STEPS": "12",
                    "COSMOS_ACTIONAWARE_REPORT_EVERY": "200",
                }
            )
    if args.benchmark == "liberopro":
        env["PYTHONPATH"] = f"{PROJECT_ROOT / 'LIBERO-PRO'}:{PROJECT_ROOT}"
        temp_dir = final_dir / "cache/tmp" / f"gpu{args.gpu}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        env["TMPDIR"] = str(temp_dir)
    elif args.benchmark == "robocasa":
        if args.mechanism == "dicache":
            env.update(
                {
                    "COSMOS_DICACHE_NUM_STEPS": "5",
                    "COSMOS_DICACHE_RET_RATIO": "0.2",
                    "COSMOS_DICACHE_PROBE_DEPTH": "1",
                    "COSMOS_DICACHE_REL_L1_THRESH": "0.08",
                    "COSMOS_DICACHE_GAMMA_MIN": "1.0",
                    "COSMOS_DICACHE_GAMMA_MAX": "2.0",
                    "COSMOS_DICACHE_VERBOSE": "True",
                    "COSMOS_DICACHE_REPORT_EVERY": "200",
                }
            )
        if "vaeskip" in mechanisms:
            # Match the selected RoboCasa accuracy profile in
            # robocasa_final/eval.sh. The only experimental difference between
            # the two requested runs is warmup_queries.
            env.update(
                {
                    "COSMOS_SHALLOW_GATE_LAYER": "res1",
                    "COSMOS_SHALLOW_GATE_TOPK": str(args.topk),
                    "COSMOS_SHALLOW_GATE_IOU_THRESHOLD": "0.75",
                    "COSMOS_SHALLOW_GATE_WRIST_IOU_THRESHOLD": str(args.wrist_iou),
                    "COSMOS_SHALLOW_GATE_MAIN_IOU_THRESHOLD": str(args.primary_iou),
                    "COSMOS_SHALLOW_GATE_REUSE_MODE": "hybrid",
                    "COSMOS_SHALLOW_GATE_CAMERA_MODE": "main_min",
                    "COSMOS_SHALLOW_GATE_PRIMARY_GUARD_THRESHOLD": "0.0",
                    "COSMOS_SHALLOW_GATE_MAX_CONSECUTIVE_SKIPS": "1",
                    "COSMOS_SHALLOW_GATE_QUERY_LAG": "2",
                    "COSMOS_SHALLOW_GATE_WARMUP_QUERIES": str(
                        6 if args.warmup_queries is None else args.warmup_queries
                    ),
                    "COSMOS_SHALLOW_GATE_FORCE_FULL_INTERVAL": "0",
                    "COSMOS_SHALLOW_GATE_FORCE_FULL_AFTER_1STEP": "False",
                }
            )
        shared_root = Path("/data/Albus/zhangyongchang")
        env["PYTHONPATH"] = ":".join(
            (
                str(shared_root / "benchmark_workspace/python_targets/mujoco326_nodeps"),
                str(PROJECT_ROOT / "robocasa_scripts/vendor"),
                str(shared_root / "robocasa_workspace/robosuite"),
                str(shared_root / "github_cache/cosmos-policy/.venv/lib/python3.10/site-packages"),
                str(PROJECT_ROOT),
            )
        )
        cache_root = final_dir / "cache" / f"gpu{args.gpu}"
        xml_cache = cache_root / "xml"
        temp_dir = cache_root / "tmp"
        xml_cache.mkdir(parents=True, exist_ok=True)
        temp_dir.mkdir(parents=True, exist_ok=True)
        eval_user = pwd.getpwnam("zzy")
        for writable_dir in (cache_root, xml_cache, temp_dir):
            os.chown(writable_dir, eval_user.pw_uid, eval_user.pw_gid)
            writable_dir.chmod(0o775)
        env["COSMOS_ROBOCASA_XML_CACHE"] = str(xml_cache)
        env["TMPDIR"] = str(temp_dir)
    return env


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit", required=True)
    parser.add_argument(
        "--benchmark", choices=("libero", "liberopro", "robocasa"), required=True
    )
    parser.add_argument("--mechanism", choices=tuple(MECHANISMS), required=True)
    parser.add_argument("--gpu", type=int, choices=range(6), required=True)
    parser.add_argument("--trials-per-task", type=int, required=True)
    parser.add_argument("--seed", type=int, default=195)
    parser.add_argument("--topk", type=float, required=True)
    parser.add_argument("--wrist-iou", type=float, required=True)
    parser.add_argument("--primary-iou", type=float, required=True)
    parser.add_argument("--primary-maxskip", type=int, required=True)
    parser.add_argument("--wrist-maxskip", type=int, required=True)
    parser.add_argument(
        "--shallow-layer",
        choices=SHALLOW_GATE_LAYERS,
        default="rgb",
        help="Feature used by VAESkip (production default: rgb; res1 is retained for ablations)",
    )
    parser.add_argument("--warmup-queries", type=int)
    parser.add_argument("--cached-steps", default="0,1,2,3")
    parser.add_argument("--refresh-interval", type=int, default=8)
    parser.add_argument("--fingerprint-tokens", type=int, default=32)
    parser.add_argument("--xcache-alpha", type=float, default=0.3)
    parser.add_argument("--tau-floor", type=float, default=0.98)
    parser.add_argument("--margin", type=float, default=0.02)
    parser.add_argument("--dev-threshold", type=float, default=45.0)
    parser.add_argument("--front-anchor", type=int, default=1)
    parser.add_argument("--view-groups", default="all;2,6;3,7")
    parser.add_argument("--xcache-trace-limit", type=int, default=0)
    parser.add_argument("--selection", default="all")
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument("--result-log", type=Path, required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append to an incomplete RoboCasa result log and resume its first missing episode",
    )
    args = parser.parse_args()

    if os.geteuid() != 0:
        raise PermissionError("protected evaluations must be launched as root")
    if not UNIT_PATTERN.fullmatch(args.unit):
        raise ValueError(f"unit must match {UNIT_PATTERN.pattern!r}")
    if args.trials_per_task < 1:
        raise ValueError("trials-per-task must be positive")
    if args.primary_maxskip < 1 or args.wrist_maxskip < 1:
        raise ValueError("maxskip values must be positive")
    if args.warmup_queries is not None and args.warmup_queries < 0:
        raise ValueError("warmup-queries must be non-negative")
    if args.refresh_interval < 0:
        raise ValueError("refresh-interval must be non-negative")
    if args.fingerprint_tokens < 1:
        raise ValueError("fingerprint-tokens must be positive")
    if not 0.0 < args.xcache_alpha <= 1.0:
        raise ValueError("xcache-alpha must be in (0,1]")
    if not -1.0 <= args.tau_floor <= 1.0:
        raise ValueError("tau-floor must be in [-1,1]")
    if args.margin < 0.0:
        raise ValueError("margin must be non-negative")
    if args.dev_threshold <= 0.0:
        raise ValueError("dev-threshold must be positive")
    if not 1 <= args.front_anchor <= 28:
        raise ValueError("front-anchor must be in [1,28]")
    if not args.view_groups.strip():
        raise ValueError("view-groups must not be empty")
    if args.xcache_trace_limit < 0:
        raise ValueError("xcache-trace-limit must be non-negative")
    if args.max_tasks < 0:
        raise ValueError("max-tasks must be non-negative")
    try:
        cached_steps = tuple(
            int(item.strip())
            for item in args.cached_steps.strip().strip("[]").split(",")
            if item.strip()
        )
    except ValueError as exc:
        raise ValueError(f"invalid cached-steps: {args.cached_steps!r}") from exc
    if len(cached_steps) != len(set(cached_steps)) or any(
        step < 0 or step >= 5 for step in cached_steps
    ):
        raise ValueError(f"cached-steps must be unique values in [0,4], got {cached_steps}")
    if args.benchmark == "robocasa" and args.mechanism not in {
        "baseline",
        "c3cache",
        "dicache",
        "standard",
    }:
        raise ValueError(
            "the protected RoboCasa launcher supports baseline, C3Cache, "
            "standard, and DiCache"
        )
    if args.benchmark != "robocasa" and args.mechanism == "dicache":
        raise ValueError("dicache is currently exposed only for RoboCasa by this launcher")
    if args.resume and args.benchmark != "robocasa":
        raise ValueError("--resume is currently supported only for RoboCasa")
    if systemctl_show(args.unit, "LoadState") not in {"", "not-found"}:
        raise RuntimeError(f"systemd unit already exists: {args.unit}")
    if not CPU_ISOLATION.is_file() or not os.access(CPU_ISOLATION, os.X_OK):
        raise RuntimeError(f"zspeed CPU isolation helper is unavailable: {CPU_ISOLATION}")

    result_log = args.result_log.resolve()
    try:
        result_log.relative_to(PROJECT_ROOT)
        inside_project = True
    except ValueError:
        inside_project = False
    c3cache_log_root = Path("/home/zzy/cosmos-policy-c3cache-logs").resolve()
    try:
        result_log.relative_to(c3cache_log_root)
        inside_c3cache_log_root = True
    except ValueError:
        inside_c3cache_log_root = False
    allowed_external_c3cache_log = (
        args.mechanism == "c3cache" and inside_c3cache_log_root
    )
    if (
        (not inside_project and not allowed_external_c3cache_log)
        or result_log.suffix != ".log"
    ):
        raise ValueError(
            "result-log must be a .log path inside the project, or a C3Cache "
            f"log below {c3cache_log_root}"
        )
    launch_log = result_log.with_name(f"{result_log.stem}_launch.log")
    queue_log = result_log.with_name(f"{result_log.stem}_queue.log")
    if args.resume:
        if not result_log.is_file():
            raise FileNotFoundError(f"cannot resume missing result log: {result_log}")
    else:
        for output in (result_log, launch_log, queue_log):
            if output.exists():
                raise FileExistsError(f"refusing to overwrite {output}")
    result_log.parent.mkdir(parents=True, exist_ok=True)
    # systemd-run executes the evaluation as zzy.  A previously absent nested
    # log directory is created here by the root launcher, so hand ownership to
    # the evaluation user before the supervisor creates logs/restart archives.
    eval_user = pwd.getpwnam("zzy")
    os.chown(result_log.parent, eval_user.pw_uid, eval_user.pw_gid)
    result_log.parent.chmod(0o775)

    mode = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(args.gpu),
            "--query-gpu=compute_mode",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip().lower().replace("_", "")
    if mode != "exclusiveprocess":
        raise RuntimeError(f"GPU {args.gpu} is not EXCLUSIVE_PROCESS: {mode}")

    subprocess.run(["apparmor_parser", "-r", str(APPARMOR_POLICY)], check=True)
    final_dir = PROJECT_ROOT / f"{args.benchmark}_final"
    completion_marker = {
        "libero": "LIBERO ALL final:",
        "liberopro": "LIBERO-PRO ALL final:",
        "robocasa": "ROBOCASA ALL final:",
    }[args.benchmark]
    evaluation_env = build_environment(args)
    if args.mechanism == "dicache":
        entry_name = "eval_dicache_wrapper.py"
    elif args.mechanism == "c3cache":
        entry_name = "eval_c3cache_wrapper.py"
    elif args.mechanism == "xcache":
        entry_name = "eval_xcache_wrapper.py"
    else:
        entry_name = "eval_wrapper.py"
    archive_root = (
        result_log.parent / "resilient_restarts" / args.unit
        if args.mechanism in {"c3cache", "xcache"}
        else final_dir / "explorelogs/resilient_restarts" / args.unit
    )
    eval_args = [
        args.selection,
        "--num-trials-per-task",
        str(args.trials_per_task),
        "--max-tasks",
        str(args.max_tasks),
        "--seed",
        str(args.seed),
        "--log-path",
        str(result_log),
    ]
    if args.resume:
        eval_args.append("--resume")
    spec = {
        "entry": str(final_dir / entry_name),
        "args": eval_args,
        "cwd": str(PROJECT_ROOT),
        "gpu": args.gpu,
        "env": evaluation_env,
        "unset_env": UNSET_ENV,
        "result_log": str(result_log),
        "launch_log": str(launch_log),
        "queue_log": str(queue_log),
        "archive_root": str(archive_root),
        "completion_marker": completion_marker,
        "resume_existing": args.resume,
    }
    python_code = (
        'import os,runpy;runpy.run_path(os.environ["HIDDEN_EVAL_ENTRY"],'
        'run_name="__main__")'
    )
    shell_code = f'exec -a zspeed "$EVAL_PY" -c \'{python_code}\''
    cpu_plan = cpu_isolation_plan(args.gpu)
    early_environment = []
    for name in (
        "PYTHONPATH",
        "TMPDIR",
        "CUDA_VISIBLE_DEVICES",
        "COSMOS_PHYSICAL_GPU",
    ):
        if name in evaluation_env:
            early_environment.append(f"--setenv={name}={evaluation_env[name]}")
    command = [
        "systemd-run",
        f"--unit={args.unit}",
        "--slice=zspeed.slice",
        "--uid=zzy",
        "--gid=zzy",
        "--property=Restart=on-failure",
        "--property=RestartSec=1s",
        "--property=StartLimitIntervalSec=0",
        "--property=Nice=-10",
        "--property=OOMScoreAdjust=-500",
        f"--property=AllowedCPUs={cpu_plan['cpus']}",
        f"--property=AllowedMemoryNodes={cpu_plan['numa_node']}",
        f"--setenv=EVAL_PY={EVAL_PYTHON}",
        f"--setenv=HIDDEN_EVAL_ENTRY={HIDDEN_SUPERVISOR}",
        f"--setenv=HIDDEN_EVAL_SPEC={json.dumps(spec, ensure_ascii=False, separators=(',', ':'))}",
        *early_environment,
        "/bin/bash",
        "-c",
        shell_code,
    ]
    reserve_cpu_isolation(args.gpu, args.unit)
    try:
        subprocess.run(command, check=True)
    except BaseException:
        subprocess.run(
            ["systemctl", "stop", f"zspeed-cpu-watch-gpu{args.gpu}.service"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        release_cpu_isolation(args.gpu)
        raise

    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        state = systemctl_show(args.unit, "ActiveState")
        pid = int(systemctl_show(args.unit, "MainPID") or "0")
        if state == "active" and pid > 1:
            print(
                json.dumps(
                    {
                        "unit": args.unit,
                        "pid": pid,
                        "gpu": args.gpu,
                        "benchmark": args.benchmark,
                        "mechanism": args.mechanism,
                        "result_log": str(result_log),
                        "cpus": cpu_plan["cpus"],
                        "numa_node": cpu_plan["numa_node"],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            return 0
        time.sleep(0.1)
    subprocess.run(
        ["systemctl", "stop", f"{args.unit}.service"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    subprocess.run(
        ["systemctl", "stop", f"zspeed-cpu-watch-gpu{args.gpu}.service"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    release_cpu_isolation(args.gpu)
    raise RuntimeError(f"unit failed to become active: {args.unit}")


if __name__ == "__main__":
    raise SystemExit(main())
