"""Process-group setup for PP x TP x SP hybrid parallelism.

World layout (matches the C++/CUDA version):
  rank = pp_rank * (TP*SP) + tp_rank * SP + sp_rank
  - PP straddles nodes  (1 PP rank per node)
  - TP and SP stay intra-node and ride NVLink
"""
from __future__ import annotations
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


def _apply_gr_master_env() -> None:
    """Use GR worker-0 endpoint as the torch.distributed master."""
    gr_host = os.environ.get("GR_WORKER_0_HOST")
    gr_port = os.environ.get("GR_WORKER_0_PORT")
    if gr_host and not os.environ.get("MASTER_ADDR"):
        os.environ["MASTER_ADDR"] = gr_host
    if gr_port and not os.environ.get("MASTER_PORT"):
        os.environ["MASTER_PORT"] = gr_port


@dataclass
class ParallelCtx:
    world_rank: int
    world_size: int
    local_rank: int
    pp_size: int
    tp_size: int
    sp_size: int
    pp_rank: int
    tp_rank: int
    sp_rank: int
    tp_group: dist.ProcessGroup
    sp_group: dist.ProcessGroup
    pp_group: dist.ProcessGroup
    sp_ranks: list[int]   # global ranks of my SP peers, ordered by sp_rank
    pp_ranks: list[int]   # global ranks of my PP peers, ordered by pp_rank
    device: torch.device
    compute_stream: torch.cuda.Stream
    comm_stream: torch.cuda.Stream


def _make_groups(world_size: int, pp: int, tp: int, sp: int):
    """Return three lists of rank-lists for TP, SP, PP groups."""
    tp_groups, sp_groups, pp_groups = [], [], []
    for p in range(pp):
        for s in range(sp):
            tp_groups.append([p*tp*sp + t*sp + s for t in range(tp)])
        for t in range(tp):
            sp_groups.append([p*tp*sp + t*sp + s for s in range(sp)])
    for t in range(tp):
        for s in range(sp):
            pp_groups.append([p*tp*sp + t*sp + s for p in range(pp)])
    return tp_groups, sp_groups, pp_groups


def init_parallel(pp: int, tp: int, sp: int) -> ParallelCtx:
    """Init torch.distributed with NCCL backend and build sub-groups."""
    _apply_gr_master_env()
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    world_rank = dist.get_rank()
    world_size = dist.get_world_size()
    if pp * tp * sp != world_size:
        raise RuntimeError(
            f"PP*TP*SP ({pp*tp*sp}) must equal world size ({world_size})")

    local_rank = int(os.environ.get("LOCAL_RANK", world_rank % torch.cuda.device_count()))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    pp_rank = world_rank // (tp * sp)
    tp_rank = (world_rank // sp) % tp
    sp_rank = world_rank % sp

    tp_lists, sp_lists, pp_lists = _make_groups(world_size, pp, tp, sp)

    # Every rank must call new_group for every group, even ones it isn't in.
    tp_group = sp_group = pp_group = None
    my_sp_ranks: list[int] = []
    my_pp_ranks: list[int] = []
    for ranks in tp_lists:
        g = dist.new_group(ranks=ranks, backend="nccl")
        if world_rank in ranks:
            tp_group = g
    for ranks in sp_lists:
        g = dist.new_group(ranks=ranks, backend="nccl")
        if world_rank in ranks:
            sp_group = g
            my_sp_ranks = ranks
    for ranks in pp_lists:
        g = dist.new_group(ranks=ranks, backend="nccl")
        if world_rank in ranks:
            pp_group = g
            my_pp_ranks = ranks

    return ParallelCtx(
        world_rank=world_rank, world_size=world_size, local_rank=local_rank,
        pp_size=pp, tp_size=tp, sp_size=sp,
        pp_rank=pp_rank, tp_rank=tp_rank, sp_rank=sp_rank,
        tp_group=tp_group, sp_group=sp_group, pp_group=pp_group,
        sp_ranks=my_sp_ranks, pp_ranks=my_pp_ranks,
        device=device,
        compute_stream=torch.cuda.current_stream(),
        comm_stream=torch.cuda.Stream(device=device),
    )


def destroy_parallel():
    if dist.is_initialized():
        dist.destroy_process_group()
