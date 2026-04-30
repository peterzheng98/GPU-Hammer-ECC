"""Distributed Attention Stress Test (PyTorch port).

Mirrors the C++/CUDA version:
  * PP across nodes, TP and SP intra-node
  * FlashAttention forward + Ring-Attention across SP for 1M context
  * Sustained FP16 GEMM + KV rotation + AllReduce -> ECC stress

Launch (16 ranks across 2 nodes):
    # GR: GR_WORKER_0_HOST/GR_WORKER_0_PORT are used as master.
    torchrun \
      --nnodes=2 --nproc_per_node=8 \
      --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
      main.py --pp 2 --tp 4 --sp 2 --iters 1000 --seq 1000000
"""
from __future__ import annotations
import argparse
import math
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F

from parallel_ctx import init_parallel, destroy_parallel
from ring_attention import ring_attention


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pp", type=int, default=2)
    p.add_argument("--tp", type=int, default=4)
    p.add_argument("--sp", type=int, default=2)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--seq",   type=int, default=1_000_000)
    p.add_argument("--hidden", type=int, default=8192)
    p.add_argument("--heads",  type=int, default=64)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--layers", type=int, default=80)
    p.add_argument("--microbatches", type=int, default=8)
    p.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    return p.parse_args()


def main():
    args = parse_args()
    ctx = init_parallel(args.pp, args.tp, args.sp)
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16

    S, H, Nh, D, L = args.seq, args.hidden, args.heads, args.head_dim, args.layers
    assert Nh * D == H, f"heads*head_dim must equal hidden ({Nh*D} vs {H})"
    assert S  % args.sp == 0, "seq must divide SP"
    assert Nh % args.tp == 0, "heads must divide TP"
    assert L  % args.pp == 0, "layers must divide PP"

    Sq_local = S  // args.sp
    Nh_local = Nh // args.tp
    H_local  = Nh_local * D
    L_local  = L  // args.pp

    if ctx.world_rank == 0:
        print(f"== Distributed Attention Stress (Python) ==", flush=True)
        print(f"world={ctx.world_size}  pp={args.pp} tp={args.tp} sp={args.sp}  "
              f"S={S} H={H} Nh={Nh} D={D} L={L}  iters={args.iters}  dtype={args.dtype}",
              flush=True)
        kv_gb = Nh_local * Sq_local * D * 2 / 1e9
        w_gb  = H * H_local * 2 / 1e9
        print(f"[mem] Q/K/V each: {kv_gb:.2f} GB   "
              f"W{{q,k,v,o}} each: {w_gb:.2f} GB   layers/stage: {L_local}",
              flush=True)

    dev = ctx.device

    # Per-layer weights, sharded across TP, replicated across SP/PP.
    # Wq/Wk/Wv: column-parallel  -> shape [H, H_local]
    # Wo:       row-parallel     -> shape [H_local, H]
    def randw(*shape):
        # Small init; we don't care about correctness, only stress.
        return (torch.randn(*shape, device=dev, dtype=dtype) * 0.02)

    Wq = [randw(H, H_local) for _ in range(L_local)]
    Wk = [randw(H, H_local) for _ in range(L_local)]
    Wv = [randw(H, H_local) for _ in range(L_local)]
    Wo = [randw(H_local, H) for _ in range(L_local)]

    # Activation buffer reused across layers (gradient-checkpoint style).
    X = torch.randn(Sq_local, H, device=dev, dtype=dtype) * 0.02

    # PP staging buffers
    pp_buf = torch.empty(Sq_local, H, device=dev, dtype=dtype) \
        if args.pp > 1 else None

    scale = 1.0 / math.sqrt(D)

    torch.cuda.synchronize()
    dist.barrier()
    t0 = time.time()

    for it in range(args.iters):
        for mb in range(args.microbatches):
            # PP recv
            if ctx.pp_rank > 0:
                dist.recv(pp_buf, src=ctx.pp_ranks[ctx.pp_rank - 1],
                          group=ctx.pp_group)
                X.copy_(pp_buf)

            # Forward through this stage's layers
            for l in range(L_local):
                # QKV projections (TP column-parallel)
                Q = (X @ Wq[l]).view(Sq_local, Nh_local, D).transpose(0, 1).contiguous()
                K = (X @ Wk[l]).view(Sq_local, Nh_local, D).transpose(0, 1).contiguous()
                V = (X @ Wv[l]).view(Sq_local, Nh_local, D).transpose(0, 1).contiguous()

                # SP Ring + FlashAttention
                O, _ = ring_attention(ctx, Q, K, V, scale)

                # Output projection (TP row-parallel) + intra-node AllReduce
                O_flat = O.transpose(0, 1).reshape(Sq_local, H_local)
                Y = O_flat @ Wo[l]
                dist.all_reduce(Y, op=dist.ReduceOp.SUM, group=ctx.tp_group)

                # Residual
                X = X + Y

            # PP send
            if ctx.pp_rank < ctx.pp_size - 1:
                dist.send(X.contiguous(), dst=ctx.pp_ranks[ctx.pp_rank + 1],
                          group=ctx.pp_group)

        if ctx.world_rank == 0 and it % 10 == 0:
            torch.cuda.synchronize()
            print(f"[iter {it:4d}] elapsed {time.time()-t0:.1f}s", flush=True)

    torch.cuda.synchronize()
    dist.barrier()
    if ctx.world_rank == 0:
        print("Stress run complete.", flush=True)
    destroy_parallel()


if __name__ == "__main__":
    main()
