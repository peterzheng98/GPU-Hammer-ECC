"""Sequence-parallel attention helper for the PyTorch stress test."""
from __future__ import annotations

import torch
import torch.distributed as dist

from parallel_ctx import ParallelCtx

try:
    from flash_attn import flash_attn_func
except ImportError as exc:  # pragma: no cover - depends on optional package
    flash_attn_func = None
    _FLASH_ATTN_IMPORT_ERROR = exc
else:
    _FLASH_ATTN_IMPORT_ERROR = None


def _gather_sequence_shards(
    ctx: ParallelCtx,
    local: torch.Tensor,
) -> torch.Tensor:
    """Gather SP sequence shards in sp_rank order and concatenate them."""
    if ctx.sp_size == 1:
        return local

    shards = [torch.empty_like(local) for _ in range(ctx.sp_size)]
    dist.all_gather(shards, local.contiguous(), group=ctx.sp_group)
    return torch.cat(shards, dim=1)


def ring_attention(
    ctx: ParallelCtx,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, None]:
    """Run sequence-parallel attention over all SP K/V shards.

    Inputs use shape [heads_local, seq_local, head_dim]. FlashAttention expects
    [batch, seq, heads, head_dim], so this wrapper handles the layout conversion.
    """
    if flash_attn_func is None:
        raise ImportError(
            "ring_attention requires flash-attn. Install it with "
            "`pip install flash-attn --no-build-isolation`."
        ) from _FLASH_ATTN_IMPORT_ERROR

    k_full = _gather_sequence_shards(ctx, k)
    v_full = _gather_sequence_shards(ctx, v)

    out = flash_attn_func(
        q.transpose(0, 1).unsqueeze(0),
        k_full.transpose(0, 1).unsqueeze(0),
        v_full.transpose(0, 1).unsqueeze(0),
        dropout_p=0.0,
        softmax_scale=scale,
        causal=False,
    )
    return out.squeeze(0).transpose(0, 1).contiguous(), None
