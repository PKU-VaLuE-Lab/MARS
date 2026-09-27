# LingBot-VA integration for MARS

This directory contains the LingBot-VA backbone used to evaluate MARS on
LIBERO-LONG. It is derived from the official
[LingBot-VA repository](https://github.com/robbyant/lingbot-va) and retains its
original license in `LICENSE.txt`.

The MARS-specific implementation consists of:

- `wan_va/modules/stream_acceleration.py`: independent whole-Transformer
  residual reuse for video and action streams;
- `wan_va/modules/model.py` and `wan_va/wan_va_server.py`: integration of
  residual reuse and action-aware diffusion-step routing;
- `evaluation/libero/client.py`: deterministic, resumable LIBERO evaluation and
  structured per-query/per-episode metrics;
- `lingbotva_final/eval_local.py`: single-GPU baseline and accelerated evaluator;
- `lingbotva_final/log_utils.py` and `summarize.py`: log validation and summary
  utilities;
- `lingbotva_final/test_*.py`: CPU regression tests for the routing and log
  parser.

Model weights, simulator assets, environments, caches, logs, and generated
results are intentionally excluded.

## Setup

Use Python 3.10+ with a CUDA build of PyTorch compatible with your system, then
install the dependencies and LIBERO:

```bash
pip install -r requirements.txt
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git third_party/LIBERO
pip install -e third_party/LIBERO
```

Download the official LingBot-VA LIBERO-LONG checkpoint and either place it at
`checkpoints/lingbot-va-posttrain-libero-long` or export its location:

```bash
export LINGBOT_MODEL_PATH=/path/to/lingbot-va-posttrain-libero-long
```

The released checkpoint used in our experiments is trained for LIBERO-LONG;
the other three LIBERO suites require their corresponding trained checkpoints.

## Evaluation

The portable launcher takes
`<variant> <gpu> <seed> <episodes-per-task> [suite] [task-start] [task-end]`.
For the complete 2,000-episode LIBERO-LONG evaluation:

```bash
# 10 tasks x 200 deterministic episodes, seed 195
bash lingbotva_final/run_libero.sh baseline 0 195 200
bash lingbotva_final/run_libero.sh accelerated 0 195 200
```

The accelerated configuration reproduces the selected thresholds: video
accumulated relative-L1 `0.15`, action accumulated relative-L1 `0.25`, and
physical-translation routing threshold `0.25`. Both variants use the official
20-step video and 50-step action schedules. Outputs and append-only logs are
written below `results/` and `logs/` by default.

You may select one task range or another suite explicitly:

```bash
bash lingbotva_final/run_libero.sh accelerated 0 196 50 libero_10 0 5
```

## Tests and summaries

```bash
python -m unittest lingbotva_final.test_stream_acceleration
python -m unittest lingbotva_final.test_log_utils
python lingbotva_final/summarize.py logs/<evaluation>.log
```

Episode seeds are derived as `seed + task_id * 100000 + episode_id`, so the
same task/episode pairs stay matched across variants and resumed runs.
