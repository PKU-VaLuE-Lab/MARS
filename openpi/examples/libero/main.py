import collections
import dataclasses
import logging
import math
import pathlib
import time

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task
    max_tasks: int = 0  # Zero evaluates the complete suite.
    task_start_index: int = 0  # First task index to evaluate (for resuming an interrupted suite).
    first_task_trial_start: int = 0  # First initial-state index for only the first selected task.

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos
    save_videos: bool = True

    seed: int = 7  # Random Seed (for reproducibility)
    timing_warmup_queries: int = 1


@dataclasses.dataclass(frozen=True)
class EvalResult:
    task_suite_name: str
    successes: int
    episodes: int
    timed_queries: int
    query_time_ms_sum: float
    policy_time_ms_sum: float
    roundtrip_time_ms_sum: float
    actionaware_queries: int
    actionaware_route_steps_sum: int
    actionaware_full_forwards: int
    actionaware_reused_forwards: int
    actionaware_score_sum: float


def eval_libero(args: Args) -> EvalResult:
    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    if args.save_videos:
        pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    query_count = 0
    timed_queries = 0
    query_time_ms_sum = 0.0
    policy_time_ms_sum = 0.0
    roundtrip_time_ms_sum = 0.0
    actionaware_queries = 0
    actionaware_route_steps_sum = 0
    actionaware_full_forwards = 0
    actionaware_reused_forwards = 0
    actionaware_score_sum = 0.0
    if not 0 <= args.task_start_index < num_tasks_in_suite:
        raise ValueError(
            f"task_start_index must be in [0, {num_tasks_in_suite}), got {args.task_start_index}"
        )
    if not 0 <= args.first_task_trial_start < args.num_trials_per_task:
        raise ValueError(
            "first_task_trial_start must be in "
            f"[0, {args.num_trials_per_task}), got {args.first_task_trial_start}"
        )
    task_stop_index = (
        num_tasks_in_suite
        if args.max_tasks <= 0
        else min(args.task_start_index + args.max_tasks, num_tasks_in_suite)
    )
    for task_id in tqdm.tqdm(range(args.task_start_index, task_stop_index)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Start episodes
        task_episodes, task_successes = 0, 0
        trial_start = args.first_task_trial_start if task_id == args.task_start_index else 0
        for episode_idx in tqdm.tqdm(range(trial_start, args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            env.reset()
            action_plan = collections.deque()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []
            done = False

            logging.info(f"Starting episode {task_episodes + 1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Get preprocessed image
                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    # Save preprocessed image for replay video
                    if args.save_videos:
                        replay_images.append(img)

                    if not action_plan:
                        # Finished executing previous action chunk -- compute new chunk
                        # Prepare observations dict
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "prompt": str(task_description),
                        }

                        # Query model to get action
                        query_start = time.monotonic()
                        result = client.infer(element)
                        roundtrip_ms = (time.monotonic() - query_start) * 1000
                        query_count += 1
                        server_ms = float(result.get("server_timing", {}).get("infer_ms", roundtrip_ms))
                        policy_ms = float(result.get("policy_timing", {}).get("infer_ms", server_ms))
                        logging.info(
                            "Query %d: Action query time = %.6f sec; policy=%.6f sec; roundtrip=%.6f sec",
                            query_count,
                            server_ms / 1000,
                            policy_ms / 1000,
                            roundtrip_ms / 1000,
                        )
                        if query_count > args.timing_warmup_queries:
                            timed_queries += 1
                            query_time_ms_sum += server_ms
                            policy_time_ms_sum += policy_ms
                            roundtrip_time_ms_sum += roundtrip_ms

                        if diagnostics := result.get("actionaware"):
                            route_steps = int(diagnostics["route_steps"])
                            full_forwards = int(diagnostics["full_forwards"])
                            reused_forwards = int(diagnostics["reused_forwards"])
                            score = float(diagnostics["draft_trans_mag_mean"])
                            actionaware_queries += 1
                            actionaware_route_steps_sum += route_steps
                            actionaware_full_forwards += full_forwards
                            actionaware_reused_forwards += reused_forwards
                            actionaware_score_sum += score
                            logging.info(
                                "ActionAware query %d: score=%.6f route_steps=%d full=%d reused=%d",
                                query_count,
                                score,
                                route_steps,
                                full_forwards,
                                reused_forwards,
                            )

                        action_chunk = result["actions"]
                        assert len(action_chunk) >= args.replan_steps, (
                            f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        )
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            if args.save_videos:
                suffix = "success" if done else "failure"
                task_segment = task_description.replace(" ", "_")
                imageio.mimwrite(
                    pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}.mp4",
                    [np.asarray(x) for x in replay_images],
                    fps=10,
                )

            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")
    mean_query_time_ms = query_time_ms_sum / timed_queries if timed_queries else float("nan")
    mean_policy_time_ms = policy_time_ms_sum / timed_queries if timed_queries else float("nan")
    mean_roundtrip_time_ms = roundtrip_time_ms_sum / timed_queries if timed_queries else float("nan")
    logging.info(
        "OPENPI LIBERO suite %s final: %d/%d = %.4f",
        args.task_suite_name,
        total_successes,
        total_episodes,
        total_successes / total_episodes,
    )
    logging.info(
        "OPENPI query timing suite %s: count=%d mean_querytime=%.6f sec "
        "mean_policy=%.6f sec mean_roundtrip=%.6f sec warmup_excluded=%d",
        args.task_suite_name,
        timed_queries,
        mean_query_time_ms / 1000,
        mean_policy_time_ms / 1000,
        mean_roundtrip_time_ms / 1000,
        args.timing_warmup_queries,
    )
    return EvalResult(
        task_suite_name=args.task_suite_name,
        successes=total_successes,
        episodes=total_episodes,
        timed_queries=timed_queries,
        query_time_ms_sum=query_time_ms_sum,
        policy_time_ms_sum=policy_time_ms_sum,
        roundtrip_time_ms_sum=roundtrip_time_ms_sum,
        actionaware_queries=actionaware_queries,
        actionaware_route_steps_sum=actionaware_route_steps_sum,
        actionaware_full_forwards=actionaware_full_forwards,
        actionaware_reused_forwards=actionaware_reused_forwards,
        actionaware_score_sum=actionaware_score_sum,
    )


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
