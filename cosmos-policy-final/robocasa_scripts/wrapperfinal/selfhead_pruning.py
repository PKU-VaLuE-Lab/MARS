#!/usr/bin/env python3
import json
from pathlib import Path
from typing import Any

import torch


def _load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path).expanduser().resolve()
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"SelfHeadSkip config must be a JSON object: {path}")
    return cfg


def _parse_remove_heads(cfg: dict[str, Any]) -> dict[int, set[int]]:
    raw = cfg.get("remove_heads_by_block", {})
    if not isinstance(raw, dict):
        raise ValueError("SelfHeadSkip config field remove_heads_by_block must be an object.")

    parsed: dict[int, set[int]] = {}
    for block_key, heads in raw.items():
        block_idx = int(block_key)
        if not isinstance(heads, list):
            raise ValueError(f"SelfHeadSkip block {block_idx} heads must be a list.")
        parsed[block_idx] = {int(head) for head in heads}
    return parsed


def _copy_linear_selection(
    source: torch.nn.Linear,
    target: torch.nn.Linear,
    feature_indices: torch.Tensor,
    *,
    select_input: bool,
) -> None:
    with torch.no_grad():
        dimension = 1 if select_input else 0
        target.weight.copy_(source.weight.index_select(dimension, feature_indices))
        if source.bias is not None and target.bias is not None:
            if select_input:
                target.bias.copy_(source.bias)
            else:
                target.bias.copy_(source.bias.index_select(0, feature_indices))


def _prune_self_attention(attn: torch.nn.Module, remove_heads: set[int]) -> torch.nn.Module:
    original_num_heads = int(attn.n_heads)
    keep_heads = [head_idx for head_idx in range(original_num_heads) if head_idx not in remove_heads]
    if not keep_heads:
        raise ValueError("SelfHeadSkip cannot remove every self-attention head from a block.")

    head_dim = int(attn.head_dim)
    feature_indices = torch.tensor(
        [feature_idx for head_idx in keep_heads for feature_idx in range(head_idx * head_dim, (head_idx + 1) * head_dim)],
        dtype=torch.long,
        device=attn.q_proj.weight.device,
    )
    dropout = float(getattr(attn.output_dropout, "p", 0.0))
    pruned = type(attn)(
        query_dim=int(attn.query_dim),
        context_dim=None,
        n_heads=len(keep_heads),
        head_dim=head_dim,
        dropout=dropout,
        qkv_format=attn.qkv_format,
        backend=attn.backend,
        use_wan_fp32_strategy=attn.use_wan_fp32_strategy,
    ).to(device=attn.q_proj.weight.device, dtype=attn.q_proj.weight.dtype)

    _copy_linear_selection(attn.q_proj, pruned.q_proj, feature_indices, select_input=False)
    _copy_linear_selection(attn.k_proj, pruned.k_proj, feature_indices, select_input=False)
    _copy_linear_selection(attn.v_proj, pruned.v_proj, feature_indices, select_input=False)
    _copy_linear_selection(attn.output_proj, pruned.output_proj, feature_indices, select_input=True)
    pruned.q_norm.load_state_dict(attn.q_norm.state_dict())
    pruned.k_norm.load_state_dict(attn.k_norm.state_dict())
    pruned.v_norm.load_state_dict(attn.v_norm.state_dict())
    pruned.train(attn.training)
    pruned._selfheadskip_remove_heads = set(remove_heads)
    pruned._selfheadskip_original_num_heads = original_num_heads
    return pruned


def install_selfhead_pruning_from_mapping(
    model,
    remove_heads_by_block: dict[int, set[int]] | dict[int, list[int]],
    *,
    expected_num_blocks: int | None = None,
    expected_num_heads: int | None = None,
    log_fn=print,
) -> None:
    remove_heads_by_block = {
        int(block_idx): {int(head_idx) for head_idx in heads}
        for block_idx, heads in remove_heads_by_block.items()
    }
    net = getattr(model, "net", None)
    if net is None or not hasattr(net, "blocks"):
        raise RuntimeError("SelfHeadSkip expected model.net.blocks.")

    num_blocks = len(net.blocks)
    if expected_num_blocks is not None and expected_num_blocks != num_blocks:
        log_fn(
            f"[SelfHeadSkip] config num_blocks={expected_num_blocks}, "
            f"model num_blocks={num_blocks}; using model value."
        )

    patched: dict[int, list[int]] = {}
    for block_idx, remove_heads in sorted(remove_heads_by_block.items()):
        if block_idx < 0 or block_idx >= num_blocks:
            raise ValueError(f"SelfHeadSkip block index out of range 0-{num_blocks - 1}: {block_idx}")
        block = net.blocks[block_idx]
        attn = getattr(block, "self_attn", None)
        if attn is None:
            continue

        installed_heads = getattr(attn, "_selfheadskip_remove_heads", None)
        if installed_heads is not None:
            if installed_heads != remove_heads:
                raise RuntimeError(
                    f"SelfHeadSkip block {block_idx} is already installed with heads {sorted(installed_heads)}; "
                    f"cannot replace them with {sorted(remove_heads)}."
                )
            patched[block_idx] = sorted(remove_heads)
            continue

        num_heads = int(getattr(attn, "n_heads", expected_num_heads or 0))
        head_dim = int(getattr(attn, "head_dim", 0))
        invalid_heads = sorted(head for head in remove_heads if head < 0 or head >= num_heads)
        if invalid_heads:
            raise ValueError(f"SelfHeadSkip block {block_idx} head index out of range 0-{num_heads - 1}: {invalid_heads}")
        if not remove_heads:
            continue

        block.self_attn = _prune_self_attention(attn, remove_heads)
        patched[block_idx] = sorted(remove_heads)

    log_fn(
        f"[SelfHeadSkip] installed structured head pruning on {len(patched)}/{num_blocks} blocks; "
        f"removed heads={patched}"
    )


def install_selfhead_pruning(model, config_path: str | Path, log_fn=print) -> None:
    cfg = _load_config(config_path)
    install_selfhead_pruning_from_mapping(
        model,
        _parse_remove_heads(cfg),
        expected_num_blocks=int(cfg["num_blocks"]) if "num_blocks" in cfg else None,
        expected_num_heads=int(cfg["num_heads"]) if "num_heads" in cfg else None,
        log_fn=log_fn,
    )
