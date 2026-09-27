<div align="center">

<img src="./assets/mars-logo.png" width="450" alt="MARS">
<h2>Multi-Level Acceleration via Redundant Computation Skipping for WAMs</h2>

[Zengyuan Zuo](https://scholar.google.com/citations?hl=zh-CN&user=jb-4yJgAAAAJ)<sup>\*</sup>, [Yongchang Zhang](https://github.com/uuuuZYC)<sup>\*</sup>, [Guangyu Zhao](https://guangyu-zhao.github.io/), [Tianze Zhang](https://scholar.google.com/citations?user=TSzGqfAAAAAJ&hl=en), [Yiwu Zhong](https://yiwuzhong.notion.site/homepage)<sup>†</sup>

[VaLuE Lab](https://github.com/PKU-VaLuE-Lab), Peking University.

(*<b>Equal Contribution</b>)(<sup>†</sup><b>Corresponding Author</b>)

[![Paper2](https://img.shields.io/badge/Paper-arXiv-red)]()
[![Project Page](https://img.shields.io/badge/Project-Website-green)]()

</div>

<details><summary>📖 Click for the full abstract of our method</summary>

<div align="left">

> World Action Models (WAMs), which jointly predict future world states and robot actions, are emerging as a promising alternative to Vision-Language-Action (VLA) models. While efficient inference for VLAs has been widely studied, the acceleration of WAMs remains relatively underexplored. Moreover, existing WAM acceleration methods typically exploit redundancy along only a single computational axis, overlooking substantial redundancy across the full pipeline, resulting in either limited speedups or drops in success rate. To address these limitations, we propose **MARS**, a training-free multi-level inference acceleration framework that systematically exploits distinct sources of redundancy at three computational levels: (1) across Transformer blocks, we selectively skip redundant self- and cross-attention while preserving components important for action generation; (2) across diffusion denoising steps, we perform action-aware routing between direct one-step generation for coarse-grained stages and residual-reuse multi-step sampling for fine-grained stages; (3) across consecutive action chunks, we reuse previously predicted future visual tokens, guarded by a shallow VAE-based verifier that determines whether the predicted visual changes remain consistent with the current observation. These mechanisms are synergistically integrated into pretrained WAMs in a plug-and-play manner, yielding substantial end-to-end speedups while preserving task performance. Extensive experiments across multiple backbones and benchmarks demonstrate that our method achieves a superior efficiency-performance trade-off, delivering up to a 4.0× speedup in LIBERO/LIBERO-PRO and 2.1× in RoboCasa.
</details>
</div>

## 📚 Overview
<p align="center"><img src="assets/overview.svg" width="90%"></p>

If MARS is helpful to your projects, please help star this repo. Thank you! :point_left:

## 🌟 Introduction

Our method is an inference-time acceleration framework for world action models. It dynamically reduces redundant computation at three complementary levels:

- **Redundancy-Aware Attention Skipping** skips low-impact self- and cross-attention computation within each denoising step.
- **Action-Aware Diffusion Step Skip** dynamically routes each query between one-step generation and residual-reuse multi-step sampling based on motion granularity.
- **Shallow-Verify Visual Token Reuse** verifies visual consistency using shallow VAE features and reuses predicted future visual tokens when reliable.

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

```bash
cd lingbot-va
pip install -r requirements.txt
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git third_party/LIBERO
pip install -e third_party/LIBERO
```

See the included [LingBot-VA README](./lingbot-va/README.md) for the baseline and
accelerated evaluation commands. The released LingBot-VA checkpoint used here
is specific to LIBERO-LONG.

## 📊 Evaluation

### LIBERO

```bash
# Cosmos-Policy baseline
bash libero_final/eval.sh --gpu 0 all

# MARS standard configuration
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

## 📝 Citation

Paper and BibTeX will be added upon release.

```bibtex
% Coming soon.
```

## 🙏 Acknowledgments

This project is built on [NVIDIA Cosmos Policy](https://github.com/NVlabs/cosmos-policy). We also thank the authors of [Fast-WAM](https://yuantianyuan01.github.io/FastWAM/), [LingBot-VA](https://github.com/robbyant/lingbot-va), [OpenPI / π0.5](https://github.com/Physical-Intelligence/openpi), [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO), [LIBERO-PRO](https://github.com/Zxy-MLlab/LIBERO-PRO), and [RoboCasa](https://github.com/robocasa/robocasa) for their models and benchmarks.

## 📮 Contact

Should you have any problems, please contact Zengyuan Zuo at [3565741165@qq.com](mailto:3565741165@qq.com). We will respond to your request as soon as possible!
