<div align="center">

# HA-WAM

### Training-Free Hierarchical Inference Acceleration for World Action Models

[Zengyuan Zuo](https://scholar.google.com/citations?hl=zh-CN&user=jb-4yJgAAAAJ)<sup>\*</sup>,[Yongchang Zhang](https://github.com/uuuuZYC)<sup>\*</sup>, [Yiwu Zhong](https://yiwuzhong.notion.site/homepage)<sup>†</sup>

[VaLuE Lab](https://github.com/PKU-VaLuE-Lab), Peking University.

[![Paper2](https://img.shields.io/badge/Paper-arXiv-red)]()

</div>

## 📚 Overview
<p align="center"><img src="assets/overview.pdf" width="95%"></p>

If HA-WAM is helpful to your projects, please help star this repo. Thank you! :point_left:

## 🌟 Introduction

HA-WAM is an inference-time acceleration framework for world action models. It dynamically reduces redundant computation at three complementary levels:

- **Action-Aware Diffusion Step Skip** routes coarse-grained action queries through fewer denoising steps and preserves full computation for fine-grained motion.
- **Redundancy-Aware Attention Skipping** reuses stable self- and cross-attention computation inside the action DiT.
- **Shallow-Verify Visual Token Reuse** detects visual changes independently for the primary and wrist views, then reuses unchanged image latents.

## Installation

HA-WAM follows the upstream Cosmos Policy environment and checkpoint setup. See the official documentation before running an evaluation:

- [Environment setup](https://github.com/NVlabs/cosmos-policy/blob/main/SETUP.md)
- [Cosmos Policy checkpoints and training data](https://huggingface.co/collections/nvidia/cosmos-policy)
- [LIBERO fine-tuning and evaluation](https://github.com/NVlabs/cosmos-policy/blob/main/LIBERO.md)
- [RoboCasa fine-tuning and evaluation](https://github.com/NVlabs/cosmos-policy/blob/main/ROBOCASA.md)

Model weights, benchmark datasets, simulator assets, local caches, and logs are intentionally excluded from this repository. Download them through the upstream instructions and set the corresponding checkpoint/cache paths for your machine.

## Evaluation

### LIBERO

```bash
# Cosmos-Policy baseline
bash libero_final/eval.sh --gpu 0 all

# HA-WAM: Action-Aware + Attention Skip + VAE Reuse
bash libero_final/eval_standard.sh --gpu 0 all
```

### LIBERO-PRO

```bash
# Cosmos-Policy baseline
bash liberopro_final/eval.sh --gpu 0 all

# HA-WAM standard configuration
bash liberopro_final/eval_standard.sh --gpu 0 all
```

### RoboCasa

```bash
# Cosmos-Policy baseline
bash robocasa_final/eval.sh all --gpu 0

# HA-WAM standard configuration
bash robocasa_final/eval_standard.sh all --gpu 0
```

## Citation

Paper and BibTeX will be added upon release.

```bibtex
% Coming soon.
```

## Acknowledgments

This project is built on [NVIDIA Cosmos Policy](https://github.com/NVlabs/cosmos-policy). We also thank the authors of [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO), [LIBERO-PRO](https://github.com/Zxy-MLlab/LIBERO-PRO), and [RoboCasa](https://github.com/robocasa/robocasa) for their benchmarks.

## License

Please follow the licenses included in [`cosmos-policy-final`](./cosmos-policy-final/LICENSE).
