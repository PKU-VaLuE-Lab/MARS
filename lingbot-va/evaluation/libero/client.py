import numpy as np
from wan_va.utils.Simple_Remote_Infer.deploy.websocket_client_policy import WebsocketClientPolicy
import argparse
import json
import os
from libero.libero import benchmark
import time
from libero.libero.envs import OffScreenRenderEnv
from pathlib import Path
from tqdm import tqdm
import imageio
import cv2
import torch


def _show_progress():
    return os.environ.get("LINGBOT_SHOW_PROGRESS", "0").strip().lower() in {
        "1", "true", "yes", "on",
    }


def save_video(real_obs_list, save_path, fps=15, video_names=["observation.images.agentview_rgb", "observation.images.eye_in_hand_rgb"]):
    if not real_obs_list:
        print("❌ No real observation frames")
        return

    first_obs = real_obs_list[0]
    base_h, width_base = first_obs[video_names[0]].shape[:2]
    target_size = (width_base, base_h)
    
    print(f"Saving video: {len(real_obs_list)} frames...")

    final_frames = [
        np.hstack([cv2.resize(obs[name], target_size) for name in video_names]).astype(np.uint8)
        for obs in real_obs_list
    ]

    imageio.mimsave(save_path, final_frames, fps=fps)
    print(f"✅ Video saved to: {save_path}")


def construct_single_env(env_args):
    count = 0
    env = None
    env_creation = False
    while not env_creation and count < 5:
        try:
            env = OffScreenRenderEnv(**env_args)
            env_creation = True
        except Exception as e:
            print(f"Error!!!  construct env failed: {e}")
            time.sleep(5)
            count += 1
    if count >= 5:
        return None
    return env


def _extract_obs(obs):
    """
    Extract agentview and eye_in_hand images from raw env obs dict.

    Avoids torch round-trip: the env already returns uint8 numpy arrays [H, W, C].
    We just flip the vertical axis ([::-1]) and make a contiguous copy once.
    """
    agentview = np.ascontiguousarray(obs["agentview_image"][::-1])
    eye_in_hand = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1])
    return {"observation.images.agentview_rgb": agentview, "observation.images.eye_in_hand_rgb": eye_in_hand}


def init_single_env(env_in, init_state):
    env_in.reset()
    env_in.set_init_state(init_state)
    for _ in range(5):
        obs, _, _, _ = env_in.step([0.] * 7)
    return _extract_obs(obs)


def env_one_step(env_in, action):
    obs, _, done, _ = env_in.step(action)
    return _extract_obs(obs), done


def run_one(
    model,
    libero_benchmark,
    task_idx,
    out_dir,
    episode_idx,
    seed,
    save_rollout_video=False,
):
    # Keep a task/episode pair bitwise reproducible when an exploration is
    # enlarged from (for example) 1 to 5 trials per task.  Using test_num as
    # the stride silently changed every later task's seeds between runs.
    episode_seed = int(seed) + task_idx * 100_000 + episode_idx
    np.random.seed(episode_seed)
    torch.manual_seed(episode_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(episode_seed)
    benchmark_dict = benchmark.get_benchmark_dict()
    benchmark_instance = benchmark_dict[libero_benchmark]()
    num_tasks = benchmark_instance.get_num_tasks()
    assert task_idx < num_tasks, f"Error: error id must smaller than {num_tasks}"
    prompt = benchmark_instance.get_task(task_idx).language
    env_args = {
                "bddl_file_name": benchmark_instance.get_task_bddl_file_path(task_idx),
                "camera_heights": 128,
                "camera_widths": 128,
            }
    init_states = benchmark_instance.get_task_init_states(task_idx)

    cur_env = construct_single_env(env_args)
    if cur_env is None:
        raise RuntimeError(f"failed to construct LIBERO environment for task {task_idx}")
    first_obs = init_single_env(cur_env, init_states[episode_idx % init_states.shape[0]])

    ret = model.infer(dict(reset=True, prompt=prompt))

    full_obs_list = []
    done = False
    first = True
    episode_query_count = 0
    while cur_env.env.timestep < 800:
        ret = model.infer(dict(obs=first_obs, prompt=prompt))
        action = ret['action']
        episode_query_count += 1
        if "query_metrics" in ret:
            query_metrics = {
                **ret["query_metrics"],
                "suite": libero_benchmark,
                "task": task_idx,
                "episode": episode_idx,
                "episode_seed": episode_seed,
            }
            print(
                "LINGBOT_QUERY_CLIENT " + json.dumps(query_metrics, sort_keys=True),
                flush=True,
            )

        key_frame_list = []
        assert action.shape[2] % 4 == 0
        action_per_frame = action.shape[2] // 4
        start_idx = 1 if first else 0
        for i in range(start_idx, action.shape[1]):
            for j in range(action.shape[2]):
                ee_action = action[:, i, j]
                observes, done = env_one_step(cur_env, ee_action)
                if done:
                    break
                if (j+1) % action_per_frame == 0:
                    if save_rollout_video:
                        full_obs_list.append(observes)
                    key_frame_list.append(observes)

            if done:
                break

        first = False

        if done:
            break
        else:
            model.infer(dict(obs=key_frame_list, compute_kv_cache=True, imagine=False, state=action))

    if save_rollout_video:
        out_file = Path(out_dir) / libero_benchmark / f"{task_idx}_{prompt.replace(' ', '_')}" / f"{episode_idx}_{done}.mp4"
        out_file.parent.mkdir(exist_ok=True, parents=True)
        save_video(
            real_obs_list=full_obs_list,
            save_path=out_file,
            fps=60,
            video_names=["observation.images.agentview_rgb", "observation.images.eye_in_hand_rgb"]
        )

    cur_env.close()
    return done, episode_query_count


def _write_json(payload, path):
    path.parent.mkdir(exist_ok=True, parents=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run_selected(
    model,
    libero_benchmark,
    task_idx,
    episode_indices,
    out_dir,
    seed=195,
):
    """Run exact deterministic episodes for threshold counterexample checks."""
    if not episode_indices:
        raise ValueError("episode_indices must not be empty")
    if len(set(episode_indices)) != len(episode_indices) or any(index < 0 for index in episode_indices):
        raise ValueError("episode_indices must be unique non-negative integers")

    successes = 0
    for selected_index, episode_idx in enumerate(episode_indices, start=1):
        success, episode_query_count = run_one(
            model,
            libero_benchmark,
            task_idx,
            out_dir,
            episode_idx,
            seed,
            save_rollout_video=False,
        )
        successes += int(success)
        payload = {
            "suite": libero_benchmark,
            "task": task_idx,
            "episode": episode_idx,
            "episode_seed": int(seed) + task_idx * 100_000 + episode_idx,
            "query_count": episode_query_count,
            "selected": True,
            "success": bool(success),
            "task_success": successes,
            "task_total": selected_index,
        }
        _write_json(
            payload,
            Path(out_dir) / f"selected_{libero_benchmark}_{task_idx}_{episode_idx}.json",
        )
        print("LINGBOT_EPISODE " + json.dumps(payload, sort_keys=True), flush=True)

    total = len(episode_indices)
    print(
        f"LINGBOT SUITE final: {libero_benchmark} selected task={task_idx} "
        f"{successes}/{total} = {successes / total:.4f}",
        flush=True,
    )
    return {
        "suite": libero_benchmark,
        "success": successes,
        "total": total,
        "success_rate": successes / total,
        "selected_task": task_idx,
        "episode_indices": list(episode_indices),
    }


def run(
    libero_benchmark,
    port,
    out_dir,
    test_num,
    task_range=None,
    save_rollout_video=False,
    seed=195,
    model=None,
):
    '''
        task_range: [start, end) for splitting tasks
    '''
    if task_range is None:
        benchmark_dict = benchmark.get_benchmark_dict()
        benchmark_instance = benchmark_dict[libero_benchmark]()
        num_tasks = benchmark_instance.get_num_tasks()
        progress_bar = tqdm(range(num_tasks), total=num_tasks, disable=not _show_progress())
    else:
        assert len(task_range) == 2, f'task_range: [start, end) for splitting tasks, however, task_range: {task_range}'
        num_tasks = task_range[1] - task_range[0]
        progress_bar = tqdm(
            range(task_range[0], task_range[1]),
            total=num_tasks,
            disable=not _show_progress(),
        )

    print(f"#################### Use benchmark: {libero_benchmark}, num_tasks: {num_tasks} #############")
    if model is None:
        model = WebsocketClientPolicy(port=port)

    suite_success = 0
    suite_total = 0
    for task_idx in progress_bar:
        result_file = Path(out_dir) / f"{libero_benchmark}_{task_idx}.json"
        if result_file.is_file():
            previous = json.loads(result_file.read_text(encoding="utf-8"))
            succ_num = int(previous.get("succ_num", 0))
            completed = int(previous.get("total_num", 0))
        else:
            succ_num = 0
            completed = 0
        completed = min(completed, test_num)
        episode_list = range(completed, test_num)

        for episode_idx in tqdm(
            episode_list,
            total=len(episode_list),
            disable=not _show_progress(),
        ):
            res_i, episode_query_count = run_one(
                model,
                libero_benchmark,
                task_idx,
                out_dir,
                episode_idx,
                seed,
                save_rollout_video=save_rollout_video,
            )
            succ_num += int(res_i)
            succ_rate = succ_num / (episode_idx + 1)
            print(f"Success rate: {succ_rate}, success num: {succ_num}, total num: {episode_idx + 1}")
            _write_json({
                "succ_num": succ_num,
                "total_num": episode_idx + 1,
                "succ_rate": succ_rate,
                }, result_file
            )
            print(
                "LINGBOT_EPISODE "
                + json.dumps(
                    {
                        "suite": libero_benchmark,
                        "task": task_idx,
                        "episode": episode_idx,
                        "episode_seed": int(seed) + task_idx * 100_000 + episode_idx,
                        "query_count": episode_query_count,
                        "success": bool(res_i),
                        "task_success": succ_num,
                        "task_total": episode_idx + 1,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        final = json.loads(result_file.read_text(encoding="utf-8"))
        task_success = int(final["succ_num"])
        task_total = int(final["total_num"])
        suite_success += task_success
        suite_total += task_total
        print(
            f"LINGBOT task final: suite={libero_benchmark} task={task_idx} "
            f"{task_success}/{task_total} = {task_success / task_total:.4f}",
            flush=True,
        )

    print(
        f"LINGBOT SUITE final: {libero_benchmark} {suite_success}/{suite_total} "
        f"= {suite_success / suite_total:.4f}",
        flush=True,
    )
    return {
        "suite": libero_benchmark,
        "success": suite_success,
        "total": suite_total,
        "success_rate": suite_success / suite_total,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--libero-benchmark",
        type=str,
        default="libero_10",
        choices=["libero_10", "libero_goal", "libero_spatial", "libero_object"],
        help="Benchmark name",
    )
    parser.add_argument(
        "--task-range",
        type=int,
        nargs="+",
        default=[0, 10],
        help="Task range [start, end) for splitting tasks",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=23908,
        help="WebSocket port",
    )
    parser.add_argument(
        "--test-num",
        type=int,
        default=50,
        help="Number of test episodes",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=195,
        help="Base seed; each task/episode receives a deterministic offset",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs/libero",
        help="Output directory for results",
    )
    parser.add_argument(
        "--save-rollout-video",
        action="store_true",
        help="Save rollout videos (disabled by default for timing evaluation)",
    )
    args = parser.parse_args()
    run(**vars(args))
    print("Finish all process!!!!!!!!!!!!")


if __name__ == "__main__":
    main()
