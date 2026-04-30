#!/bin/bash
#SBATCH --job-name=attn_stress
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8        # 8 GPUs per node -> 16 ranks total
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=attn_stress_%j.out

set -euo pipefail

module load cuda/12.4 nccl/2.21 openmpi/4.1.6 cmake/3.27 || true

# --- NCCL / fabric tuning ---
export NCCL_DEBUG=INFO
export NCCL_IB_HCA=mlx5
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=ib0
export NCCL_NET_GDR_LEVEL=5      # GPUDirect RDMA
export NCCL_P2P_LEVEL=NVL
export OMPI_MCA_pml=ucx
export UCX_TLS=rc,cuda_copy,cuda_ipc,gdr_copy

# --- Continuous ECC monitoring on every node ---
srun --ntasks-per-node=1 -N $SLURM_NNODES \
  bash -c '
    while true; do
      ts=$(date -Iseconds)
      nvidia-smi --query-gpu=index,ecc.errors.uncorrected.volatile.total,ecc.errors.corrected.volatile.total,temperature.gpu,power.draw \
                 --format=csv,noheader \
        | sed "s/^/$ts $(hostname) /" \
        >> /tmp/ecc_$(hostname).log
      sleep 10
    done' &
ECC_MON_PID=$!

# --- Build ---
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j

# --- Run: PP across nodes (2), TP (4) and SP (2) intra-node ---
srun --mpi=pmix --gpus-per-task=1 --gpu-bind=closest \
     ./attn_stress --pp 2 --tp 4 --sp 2 --iters 1000 --seq 1000000

kill $ECC_MON_PID || true
echo "==== ECC log digest ===="
awk '{print $0}' /tmp/ecc_*.log | tail -n 200
