<div align="center">

<img src="./assets/mars-logo.png" width="450" alt="MARS">
<h2>Multi-Level Acceleration via Redundant Computation Skipping for WAMs</h2>

[Zengyuan Zuo](https://scholar.google.com/citations?hl=zh-CN&user=jb-4yJgAAAAJ)<sup>\*</sup>, [Yongchang Zhang](https://github.com/uuuuZYC)<sup>\*</sup>, [Guangyu Zhao](https://guangyu-zhao.github.io/), [Tianze Zhang](https://scholar.google.com/citations?user=TSzGqfAAAAAJ&hl=en), [Yiwu Zhong](https://yiwuzhong.notion.site/homepage)<sup>†</sup>

[VaLuE Lab](https://github.com/PKU-VaLuE-Lab), Peking University.

[![Paper2](https://img.shields.io/badge/Paper-arXiv-red)]()

</div>

## 📚 Overview
<p align="center"><img src="assets/overview.svg" width="90%"></p>

If MARS is helpful to your projects, please help star this repo. Thank you! :point_left:

## 🌟 Introduction

Our method is an inference-time acceleration framework for world action models. It dynamically reduces redundant computation at three complementary levels:

- **Action-Aware Diffusion Step Skip** routes coarse-grained action queries through fewer denoising steps and preserves full computation for fine-grained motion.
- **Redundancy-Aware Attention Skipping** reuses stable self- and cross-attention computation inside the action DiT.
- **Shallow-Verify Visual Token Reuse** detects visual changes independently for the primary and wrist views, then reuses unchanged image latents.

## 🛠️ Installation

Clone the repository first:

```bash
git clone https://github.com/Harbinzzy/MARS.git
cd MARS
```

### Cosmos Policy

The primary MARS implementation follows the official Cosmos Policy Docker environment:

```bash
cd cosmos-policy-final
docker build -t cosmos-policy docker
```

Setup details, checkpoints, and fine-tuning recipes:

- [Environment setup](https://github.com/NVlabs/cosmos-policy/blob/main/SETUP.md)
- [Cosmos Policy checkpoints](https://huggingface.co/collections/nvidia/cosmos-policy)
- [LIBERO fine-tuning and evaluation](https://github.com/NVlabs/cosmos-policy/blob/main/LIBERO.md)
- [RoboCasa fine-tuning and evaluation](https://github.com/NVlabs/cosmos-policy/blob/main/ROBOCASA.md)

### OpenPI / π0.5

The OpenPI adaptation is stored in `openpi/` (including the final launchers in `openpi/openpi_final/`). Install [uv](https://docs.astral.sh/uv/getting-started/installation/), restore the two upstream example dependencies, and create the environment:

```bash
cd openpi
git clone https://github.com/Physical-Intelligence/aloha.git third_party/aloha
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git third_party/libero
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

See the included [OpenPI README](./openpi/README.md) for checkpoints and π0.5 fine-tuning.

### FastWAM

The FastWAM adaptation is stored in `FastWAM/`. Its upstream environment uses Python 3.10 and CUDA 12.8 builds of PyTorch:

```bash
cd FastWAM
conda create -n fastwam python=3.10 -y
conda activate fastwam
pip install -U pip
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
pip install -e .
```

See the included [FastWAM README](./FastWAM/README.md) for released checkpoints, datasets, and model preparation.

### LingBot-VA

The LingBot-VA adaptation is stored in `lingbot-va/`. It includes the MARS
stream-residual reuse and action-aware diffusion routing implementation, the
deterministic LIBERO evaluator, result parser, and CPU regression tests. Install
the upstream dependencies and LIBERO, then place the official LIBERO-LONG
checkpoint at `lingbot-va/checkpoints/lingbot-va-posttrain-libero-long` (or set
`LINGBOT_MODEL_PATH`):

```bash
cd lingbot-va
pip install -r requirements.txt
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git third_party/LIBERO
pip install -e third_party/LIBERO
```

See the included [LingBot-VA README](./lingbot-va/README.md) for the baseline and
accelerated evaluation commands. The released LingBot-VA checkpoint used here
is specific to LIBERO-LONG.

Model weights, benchmark datasets, simulator assets, local caches, and logs are intentionally excluded from this repository. Download them through the upstream instructions and set the corresponding checkpoint/cache paths for your machine.

## 📊 Evaluation

### LIBERO

```bash
# Cosmos-Policy baseline
bash libero_final/eval.sh --gpu 0 all

# MARS: Action-Aware + Attention Skip + VAE Reuse
bash libero_final/eval_standard.sh --gpu 0 all
```

### LIBERO-PRO

```bash
# Cosmos-Policy baseline
bash liberopro_final/eval.sh --gpu 0 all

# MARS standard configuration
bash liberopro_final/eval_standard.sh --gpu 0 all
```

### RoboCasa

```bash
# Cosmos-Policy baseline
bash robocasa_final/eval.sh all --gpu 0

# MARS standard configuration
bash robocasa_final/eval_standard.sh all --gpu 0
```

### LingBot-VA / LIBERO-LONG

```bash
# Baseline, seed 195, 200 episodes per task
bash lingbot-va/lingbotva_final/run_libero.sh baseline 0 195 200

# MARS acceleration with the matched evaluation protocol
bash lingbot-va/lingbotva_final/run_libero.sh accelerated 0 195 200
```

## 📝 Citation

Paper and BibTeX will be added upon release.

```bibtex
% Coming soon.
```

## 🙏 Acknowledgments

This project is built on [NVIDIA Cosmos Policy](https://github.com/NVlabs/cosmos-policy). We also thank the authors of [LingBot-VA](https://github.com/robbyant/lingbot-va), [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO), [LIBERO-PRO](https://github.com/Zxy-MLlab/LIBERO-PRO), and [RoboCasa](https://github.com/robocasa/robocasa) for their models and benchmarks.

## 📮 Contact

Should you have any problems, please contact Zengyuan Zuo at [3565741165@qq.com](mailto:3565741165@qq.com). We will respond to your request as soon as possible!
