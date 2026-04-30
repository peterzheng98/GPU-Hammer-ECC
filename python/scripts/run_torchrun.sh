#!/bin/bash
# Manual two-node launch with torchrun (no SLURM).
# GR sets GR_WORKER_0_HOST/GR_WORKER_0_PORT for the master node.
# Run on node0:  NODE_RANK=0 MASTER_ADDR=node0 bash run_torchrun.sh
# Run on node1:  NODE_RANK=1 MASTER_ADDR=node0 bash run_torchrun.sh
set -euo pipefail

NODE_RANK=${NODE_RANK:?set NODE_RANK=0 or 1}
MASTER_ADDR=${MASTER_ADDR:-${GR_WORKER_0_HOST:?set MASTER_ADDR or GR_WORKER_0_HOST}}
MASTER_PORT=${MASTER_PORT:-${GR_WORKER_0_PORT:-29500}}
NPROC=${NPROC:-8}
NNODES=${NNODES:-2}
export MASTER_ADDR MASTER_PORT

export NCCL_DEBUG=INFO
export NCCL_IB_HCA=mlx5
export NCCL_SOCKET_IFNAME=ib0
export NCCL_NET_GDR_LEVEL=5
export NCCL_P2P_LEVEL=NVL
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Background ECC monitor on this node
python ecc_monitor.py &
ECC_PID=$!

torchrun \
  --nnodes=$NNODES \
  --nproc_per_node=$NPROC \
  --node_rank=$NODE_RANK \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  main.py --pp 2 --tp 4 --sp 2 --iters 1000 --seq 1000000

kill $ECC_PID || true
