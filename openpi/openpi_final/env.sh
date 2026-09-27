#!/usr/bin/env bash

if [[ -n "${BASH_VERSION:-}" ]]; then
  openpi_env_source="${BASH_SOURCE[0]}"
elif [[ -n "${ZSH_VERSION:-}" ]]; then
  openpi_env_source="${(%):-%N}"
else
  openpi_env_source="$0"
fi
OPENPI_ROOT="$(cd -- "$(dirname -- "${openpi_env_source}")/.." && pwd)"
unset openpi_env_source
export OPENPI_ROOT
export CONDA_PKGS_DIRS="${OPENPI_ROOT}/.cache/conda/pkgs"
export CONDA_ENVS_PATH="${OPENPI_ROOT}/.cache/conda/envs"
export PIP_CACHE_DIR="${OPENPI_ROOT}/.cache/pip"
export UV_CACHE_DIR="${OPENPI_ROOT}/.cache/uv"
export XDG_CACHE_HOME="${OPENPI_ROOT}/.cache"
export HF_HOME="${OPENPI_ROOT}/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="${OPENPI_ROOT}/.cache/huggingface/hub"
export TRANSFORMERS_CACHE="${OPENPI_ROOT}/.cache/huggingface/transformers"
export OPENPI_DATA_HOME="${OPENPI_ROOT}/.cache/openpi"
export LIBERO_CONFIG_PATH="${OPENPI_ROOT}/.cache/libero"
export TMPDIR="${OPENPI_ROOT}/.cache/tmp"
export PYTHONPATH="${OPENPI_ROOT}/third_party/libero${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json

mkdir -p \
  "${CONDA_PKGS_DIRS}" \
  "${PIP_CACHE_DIR}" \
  "${UV_CACHE_DIR}" \
  "${HUGGINGFACE_HUB_CACHE}" \
  "${TRANSFORMERS_CACHE}" \
  "${OPENPI_DATA_HOME}" \
  "${LIBERO_CONFIG_PATH}" \
  "${TMPDIR}" \
  "${OPENPI_ROOT}/openpi_final/logs"

if [[ ! -f "${LIBERO_CONFIG_PATH}/config.yaml" ]]; then
  benchmark_root="${OPENPI_ROOT}/third_party/libero/libero/libero"
  printf 'assets: %s\nbddl_files: %s\nbenchmark_root: %s\ndatasets: %s\ninit_states: %s\n' \
    "${benchmark_root}/assets" \
    "${benchmark_root}/bddl_files" \
    "${benchmark_root}" \
    "${OPENPI_ROOT}/third_party/libero/libero/datasets" \
    "${benchmark_root}/init_files" \
    >"${LIBERO_CONFIG_PATH}/config.yaml"
fi
