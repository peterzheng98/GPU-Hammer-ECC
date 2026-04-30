#!/bin/bash
# Manual launch on a 2-node cluster without SLURM.
# Provide HOSTFILE with two lines, e.g.:
#   node01 slots=8
#   node02 slots=8
set -euo pipefail

HOSTFILE=${HOSTFILE:-./hostfile}
NP=${NP:-16}     # 2 nodes * 8 GPUs

export NCCL_DEBUG=INFO
export NCCL_IB_HCA=mlx5
export NCCL_SOCKET_IFNAME=ib0
export NCCL_NET_GDR_LEVEL=5
export NCCL_P2P_LEVEL=NVL

# Background ECC monitor on each host
for h in $(awk '{print $1}' "$HOSTFILE"); do
  ssh "$h" 'nohup bash -c "
     while true; do
       ts=\$(date -Iseconds)
       nvidia-smi --query-gpu=index,ecc.errors.uncorrected.volatile.total,ecc.errors.corrected.volatile.total,temperature.gpu,power.draw \
                  --format=csv,noheader | sed \"s/^/\$ts \$(hostname) /\" \
                  >> /tmp/ecc_\$(hostname).log
       sleep 10
     done
   " >/dev/null 2>&1 &' &
done

mpirun -np "$NP" --hostfile "$HOSTFILE" \
  --map-by ppr:8:node --bind-to none \
  -x NCCL_DEBUG -x NCCL_IB_HCA -x NCCL_SOCKET_IFNAME \
  -x NCCL_NET_GDR_LEVEL -x NCCL_P2P_LEVEL \
  -x LD_LIBRARY_PATH -x PATH \
  ./build/attn_stress --pp 2 --tp 4 --sp 2 --iters 1000 --seq 1000000
