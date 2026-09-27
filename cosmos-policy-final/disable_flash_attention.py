#!/usr/bin/env python3
"""Disable every FlashAttention route used by the final evaluation entrypoints."""

from __future__ import annotations

import os
from collections.abc import Iterable


_INSTALLED = False
_FLASH_BACKEND_NAMES = frozenset({"flash2", "flash3"})


def install() -> None:
    """Install the process-wide No-Flash policy before model modules are loaded."""
    global _INSTALLED
    if _INSTALLED:
        return

    # Keep TransformerEngine available, but forbid its optional flash-attn backend.
    os.environ["NVTE_FLASH_ATTN"] = "0"

    import torch
    from torch.nn.attention import SDPBackend

    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(True)

    # Imaginaire attention normally prefers flash2 on A40. Remove both flash
    # implementations from automatic selection and reject explicit selection.
    from cosmos_policy._src.imaginaire.attention import backends as i4_backends
    from cosmos_policy._src.imaginaire.attention import frontend as i4_frontend

    original_get_backend_list = i4_backends.get_backend_list

    def get_backend_list_without_flash(arch_tag: int) -> list[str]:
        return [
            backend
            for backend in original_get_backend_list(arch_tag)
            if backend not in _FLASH_BACKEND_NAMES
        ]

    def blocked_flash_backend(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("FlashAttention is disabled for final evaluations.")

    i4_backends.get_backend_list = get_backend_list_without_flash
    for backend in _FLASH_BACKEND_NAMES:
        i4_frontend.BACKEND_MAP[backend] = blocked_flash_backend

    # The Predict2 attention helper imports sdpa_kernel directly, so filter its
    # local backend list in addition to disabling PyTorch Flash-SDP globally.
    from cosmos_policy._src.predict2.networks import attention as predict_attention

    original_sdpa_kernel = predict_attention.sdpa_kernel

    def sdpa_kernel_without_flash(
        backends: SDPBackend | Iterable[SDPBackend],
        *args,
        **kwargs,
    ):
        if isinstance(backends, SDPBackend):
            candidates = [backends]
        else:
            candidates = list(backends)
        candidates = [
            backend
            for backend in candidates
            if backend != SDPBackend.FLASH_ATTENTION
        ]
        if not candidates:
            candidates = [SDPBackend.MATH]
        return original_sdpa_kernel(candidates, *args, **kwargs)

    predict_attention.sdpa_kernel = sdpa_kernel_without_flash
    _INSTALLED = True
    print(
        "[No-Flash default] Flash-SDP=false, I4 flash2/flash3=disabled, "
        "NVTE_FLASH_ATTN=0"
    )


install()
