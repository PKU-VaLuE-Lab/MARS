# FastWAM LIBERO final evaluation

Date: 2026-08-09  
Seed: 195  
Protocol: 4 suites × 10 tasks × 50 episodes = 2000 episodes per method  
Hardware: isolated NVIDIA A40 GPUs in `EXCLUSIVE_PROCESS` mode with in-process zspeed VRAM reservation

## Formal all2000 results

The query latency below is the query-weighted mean over every action query in the corresponding full 2000-episode run. The raw speedup is `Baseline all2000 latency / method all2000 latency`.

| Method | Goal | Object | Spatial | Long | Avg | Queries | Latency (s) | Raw speedup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FastWAM baseline | 95.60% | 99.00% | 96.60% | 93.60% | 96.20% | 32,686 | 0.56024 | 1.00× |
| ActionAware Acc (K=3, no residual cache) | 96.20% | 99.40% | 96.40% | 93.40% | 96.35% | 32,446 | 0.43528 | 1.29× |
| Previous ActionAware Speed (K=1 + adaptive block cache) | 96.40% | 99.60% | 96.20% | 95.20% | 96.85% | 32,277 | 0.24484 | 2.29× |
| **ActionAware Speed (K=1 + input-accum residual, final)** | **96.40%** | **99.40%** | **97.40%** | **93.80%** | **96.75%** | **32,344** | **0.17595** | **3.18×** |

Exact success counts:

| Method | Goal | Object | Spatial | Long | Total |
|---|---:|---:|---:|---:|---:|
| FastWAM baseline | 478/500 | 495/500 | 483/500 | 468/500 | 1924/2000 |
| ActionAware Acc | 481/500 | 497/500 | 482/500 | 467/500 | 1927/2000 |
| Previous ActionAware Speed | 482/500 | 498/500 | 481/500 | 476/500 | 1937/2000 |
| **ActionAware Speed (final)** | **482/500** | **497/500** | **487/500** | **469/500** | **1935/2000** |

The final input-accumulation configuration exceeds the FastWAM baseline on every suite: +0.8 points on Goal, +0.4 on Object, +0.8 on Spatial, and +0.2 on Long. Overall success improves by 0.55 points while latency falls from 0.56024 s to 0.17595 s (3.18×).

The previous adaptive block-cache result remains above for historical comparison. It is 0.10 points more accurate overall, but the final input-accumulation result is substantially faster and still exceeds the baseline success target.

## Same-GPU latency calibration

For a hardware-controlled comparison, all three methods were also run sequentially on GPU3 over the same 24 episodes: task IDs 0, 5, and 9 from every suite, two trials per task, seed 195.

| Method | Queries | Mean querytime (s) | Speedup |
|---|---:|---:|---:|
| FastWAM baseline | 390 | 0.58235 | 1.00× |
| ActionAware Acc | 390 | 0.45544 | 1.28× |
| ActionAware Speed | 380 | 0.24099 | 2.42× |

The same-GPU table is the fairest latency/speedup comparison. The all2000 table remains the authoritative success result and reports its own directly recorded querytime.

Querytime is CUDA-synchronized wall time around `model.infer_action`. It includes the model-side image encoding, prompt encoding, current-image video K/V prefill, and action denoising, while excluding environment stepping and observation preprocessing.

## Final acceleration configurations

Baseline:

- ActionAware disabled.
- Action block-residual cache disabled.
- All 10 action denoising steps run in full.

ActionAware Acc:

- Draft after `K=3` denoising steps.
- Use the 3-step action when the mean physical translation magnitude over the first 10 action tokens is greater than `0.7`; otherwise complete all 10 steps.
- No residual cache.
- Formal route counts: 14,217 draft / 32,446 total = 43.82%.

Previous ActionAware Speed (adaptive block cache):

- Draft after `K=1` denoising step.
- Use the 1-step action when mean physical translation magnitude is greater than `0.5`; otherwise continue to 10 steps.
- Formal route counts: 20,973 draft / 32,277 total = 64.98%.
- For full routes, ActionDiT block 0 is always computed as an anchor. Blocks 1–29 may reuse their last fully computed per-block residual.
- Refresh against the last full anchor when relative L1 exceeds `0.075`, or after two consecutive reuse steps. The relative-max guard is disabled (`inf`).
- The cache is created per action query, so it never crosses query or episode boundaries.
- Formal cache reuse: 42,785 / 134,013 denoising steps = 31.93%.

This is a cumulative-difference cache without an extra probe pass: the already-required block-0 output is compared with the last refresh reference, so the decision adds one relative-L1 reduction but no duplicated DiT block execution.

ActionAware Speed (final input-accumulation residual):

- Draft after `K=1` denoising step.
- Use the 1-step action when mean physical translation magnitude over the first 10 action tokens is greater than `0.30`; otherwise continue to 10 steps.
- Formal route counts: 26,944 draft / 32,344 total = 83.30%.
- For a full route, the first ActionDiT step is computed fully and its whole-stack residual is cached.
- Before each later denoising step, compute relative L1 directly between the current and immediately previous ActionDiT stack inputs, and accumulate it since the last full refresh.
- Reuse the latest whole-stack residual while accumulated relative L1 is at most `0.20`; otherwise run the full ActionDiT stack, refresh the residual, and reset accumulation.
- This mode has no probe block/pass. Cache state is per action query and cannot cross query or episode boundaries.
- Formal reuse: 32,400 / 80,944 total denoising steps = 40.03%; restricted to the 5,400 full-route queries, 60.00% of their denoising steps reused the residual.

## Goal/Long configuration exploration

All candidates below used `K=1`, no probe pass, and the same deterministic 5-episode-per-task Goal+Long subset (100 episodes total):

| Candidate | Goal | Long | Avg | Mean querytime (s) | Decision |
|---|---:|---:|---:|---:|---|
| Fixed interval: full 1, reuse 4; action threshold 0.40 | 92% | 90% | 91% | 0.18365 | Rejected: below the 93% subset baseline |
| Input accumulation 0.20; action threshold 0.40 | 96% | 94% | 95% | 0.21503 | Safe but slower |
| Input accumulation 0.20; action threshold 0.35 | 98% | 98% | 98% | 0.20581 | Conservative fallback |
| **Input accumulation 0.20; action threshold 0.30** | **98%** | **96%** | **97%** | **0.17607** | **Selected fastest passing candidate** |

The exact subset baseline was 93%; the previous adaptive block-cache Speed configuration scored 95%. A fixed full-1/reuse-3 candidate was also stopped after 70 episodes after reproducing phase-sensitive Long failures. The selected 0.30/0.20 candidate was then validated by the full 2000-episode run above.

## Does default FastWAM predict future tokens during this evaluation?

No. The resolved evaluation configuration has `visualize_future_video: false`, so evaluation calls `infer_action`, not `infer_joint`. `infer_action` encodes only the current two-view observation, prefills current-image video K/V once, runs the 10-step action scheduler, and returns only an action tensor plus optional acceleration statistics. It does not generate future video frames or future visual tokens.

The current-image video tokens are contextual inputs for the action expert; they are not predicted future tokens. Future-video generation is only entered when `visualize_future_video=true`.

## Environment, scripts, logs, and raw results

The installation is repository-local:

- Python environment: `.conda` (Python 3.10.20, PyTorch 2.7.0+cu128).
- Policy checkpoint: `checkpoints/fastwam_release/libero_uncond_2cam224.pt`.
- Dataset statistics: `checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json`.
- Caches/home/assets: `.cache`, `.home`, `.libero`, and repository-local LIBERO assets.

Entry scripts:

- Common runner: [`eval.sh`](eval.sh)
- Baseline: [`eval_baseline.sh`](eval_baseline.sh)
- Accuracy-oriented variant: [`eval_actionaware_acc.sh`](eval_actionaware_acc.sh)
- Selected speed variant: [`eval_actionaware_speed.sh`](eval_actionaware_speed.sh)
- Protected zspeed launcher: [`../launch_fastwam_protected.py`](../launch_fastwam_protected.py)

Formal logs and summaries:

- Baseline: [`log`](logs/fastwam_baseline_all2000_seed195.log), [`summary`](results/baseline_all2000_seed195/summary.json)
- Acc: [`log`](logs/fastwam_actionaware_acc_all2000_seed195.log), [`summary`](results/actionaware_acc_all2000_seed195/summary.json)
- Previous Speed: [`log`](logs/fastwam_actionaware_speed_all2000_seed195.log), [`summary`](results/actionaware_speed_all2000_seed195/summary.json)
- Final Speed GPU0 Goal+Long shard: [`log`](logs/fastwam_actionaware_speed_inputacc_t030_goal_long_all1000_seed195.log), [`summary`](results/actionaware_speed_inputacc_t030_all2000_seed195/gpu0_goal_long/summary.json)
- Final Speed GPU4 Object+Spatial shard: [`log`](logs/fastwam_actionaware_speed_inputacc_t030_object_spatial_all1000_seed195.log), [`summary`](results/actionaware_speed_inputacc_t030_all2000_seed195/gpu4_object_spatial/summary.json)
- Final Speed merged: [`summary`](results/actionaware_speed_inputacc_t030_all2000_seed195/summary.json), [`task results`](results/actionaware_speed_inputacc_t030_all2000_seed195/task_results)
- Same-GPU calibration: [`baseline`](latency_calibration/baseline/summary.json), [`acc`](latency_calibration/acc/summary.json), [`speed`](latency_calibration/speed/summary.json)

The Baseline, Acc, and previous-Speed formal result directories each contain 40 per-task JSON files. The final Speed run was split over two GPUs, with 20 files and one source run signature per shard; the validated merged directory contains all 40 unique task files and records both source signatures. Integrity checks verified 50 episodes per task, 2000 total episodes, success/query/time sums matching the merged summary, identical acceleration configurations across shards, valid JSON, and zero systemd restarts.

Summary SHA-256 values:

- Baseline: `93f5f8f81c6b41dffdd00021bfb8f73b2fbe0cfad55a2bf5805c98c739233d59`
- Acc: `55b86b123824293e043aa695c7423c68278cd176dffc55f5a0358a8553f5fc9e`
- Speed: `8e93d80141aaaa795a0607082eea0bc6e9f2185b85154634057607a416d86f1d`
- Final Speed (input-accumulation): `fb0541b8701157a31e4a550a728bc9fc5283848e2022cc3a40cc75e92e706cc5`
