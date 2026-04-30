#!/bin/bash
#SBATCH --job-name=attn_stress_py
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1          # one launcher per node; torchrun spawns 8 workers
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --output=attn_stress_py_%j.out

set -euo pipefail
module load cuda/12.4 nccl/2.21 python/3.11 || true

# NCCL / fabric
export NCCL_DEBUG=INFO
export NCCL_IB_HCA=mlx5
export NCCL_SOCKET_IFNAME=ib0
export NCCL_NET_GDR_LEVEL=5
export NCCL_P2P_LEVEL=NVL
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Rendezvous. GR exposes worker 0 as the master endpoint.
if [[ -z "${MASTER_ADDR:-}" ]]; then
  if [[ -n "${GR_WORKER_0_HOST:-}" ]]; then
    MASTER_ADDR=$GR_WORKER_0_HOST
  else
    MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
  fi
fi
MASTER_PORT=${MASTER_PORT:-${GR_WORKER_0_PORT:-29500}}
export MASTER_ADDR MASTER_PORT

# Per-node ECC monitor
srun --ntasks-per-node=1 -N $SLURM_NNODES \
     python ecc_monitor.py &
ECC_PID=$!

# Per-node torchrun
srun --ntasks-per-node=1 --gpus-per-node=8 bash -c '
  torchrun \
    --nnodes='"$SLURM_NNODES"' \
    --nproc_per_node=8 \
    --node_rank=$SLURM_NODEID \
    --rdzv_backend=c10d \
    --rdzv_endpoint='"$MASTER_ADDR:$MASTER_PORT"' \
    main.py --pp 2 --tp 4 --sp 2 --iters 1000 --seq 1000000
'

kill $ECC_PID || true
echo "==== ECC log digest ===="
tail -n 200 /tmp/ecc_*.log 2>/dev/null || true
