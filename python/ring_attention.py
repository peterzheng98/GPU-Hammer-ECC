"""Sequence-parallel attention helper for the PyTorch stress test."""
from __future__ import annotations

import inspect

import torch
import torch.distributed as dist

from parallel_ctx import ParallelCtx

try:
    from flash_attn.flash_attn_interface import _flash_attn_forward
except ImportError as exc:  # pragma: no cover - depends on optional package
    _flash_attn_forward = None
    _FLASH_ATTN_IMPORT_ERROR = exc
else:
    _FLASH_ATTN_IMPORT_ERROR = None


def _flash_attention_with_lse(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run FlashAttention and return output plus per-row log-sum-exp."""
    if _flash_attn_forward is None:
        raise ImportError(
            "ring_attention requires flash-attn. Install it with "
            "`pip install flash-attn --no-build-isolation`."
        ) from _FLASH_ATTN_IMPORT_ERROR

    q_bshd = q.transpose(0, 1).unsqueeze(0).contiguous()
    k_bshd = k.transpose(0, 1).unsqueeze(0).contiguous()
    v_bshd = v.transpose(0, 1).unsqueeze(0).contiguous()

    available = inspect.signature(_flash_attn_forward).parameters
    kwargs = {
        "dropout_p": 0.0,
        "softmax_scale": scale,
        "causal": False,
        "window_size": (-1, -1),
        "alibi_slopes": None,
        "return_softmax": False,
    }
    if "softcap" in available:
        kwargs["softcap"] = 0.0

    out, softmax_lse, *_ = _flash_attn_forward(
        q_bshd,
        k_bshd,
        v_bshd,
        **{name: value for name, value in kwargs.items() if name in available},
    )
    return out.squeeze(0).transpose(0, 1).contiguous(), softmax_lse.squeeze(0)


def _merge_attention(
    out_acc: torch.Tensor,
    lse_acc: torch.Tensor,
    out_part: torch.Tensor,
    lse_part: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge two partial attention results using their log-sum-exp terms."""
    lse_max = torch.maximum(lse_acc, lse_part)
    acc_scale = torch.exp(lse_acc - lse_max)
    part_scale = torch.exp(lse_part - lse_max)
    denom = acc_scale + part_scale
    out = (
        out_acc * acc_scale.unsqueeze(-1)
        + out_part * part_scale.unsqueeze(-1)
    ) / denom.unsqueeze(-1)
    lse = lse_max + torch.log(denom)
    return out.contiguous(), lse


def _rotate_kv(
    ctx: ParallelCtx,
    k: torch.Tensor,
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate one K/V shard around the SP ring."""
    next_rank = ctx.sp_ranks[(ctx.sp_rank + 1) % ctx.sp_size]
    prev_rank = ctx.sp_ranks[(ctx.sp_rank - 1) % ctx.sp_size]
    k_next = torch.empty_like(k)
    v_next = torch.empty_like(v)

    ops = [
        dist.P2POp(dist.isend, k, next_rank, group=ctx.sp_group),
        dist.P2POp(dist.irecv, k_next, prev_rank, group=ctx.sp_group),
        dist.P2POp(dist.isend, v, next_rank, group=ctx.sp_group),
        dist.P2POp(dist.irecv, v_next, prev_rank, group=ctx.sp_group),
    ]
    for req in dist.batch_isend_irecv(ops):
        req.wait()
    return k_next, v_next


def ring_attention(
    ctx: ParallelCtx,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run sequence-parallel attention over all SP K/V shards.

    Inputs use shape [heads_local, seq_local, head_dim]. FlashAttention expects
    [batch, seq, heads, head_dim], so this wrapper handles the layout conversion
    while rotating K/V shards around the SP ring.
    """
    k_curr = k.contiguous()
    v_curr = v.contiguous()
    out_acc, lse_acc = _flash_attention_with_lse(q, k_curr, v_curr, scale)

    for _ in range(1, ctx.sp_size):
        k_curr, v_curr = _rotate_kv(ctx, k_curr, v_curr)
        out_part, lse_part = _flash_attention_with_lse(q, k_curr, v_curr, scale)
        out_acc, lse_acc = _merge_attention(out_acc, lse_acc, out_part, lse_part)

    return out_acc, lse_acc
