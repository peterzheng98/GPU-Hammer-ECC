#pragma once
#include <mpi.h>
#include <nccl.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(x) do { cudaError_t e = (x);                          \
  if (e != cudaSuccess) {                                                \
    fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__,               \
            cudaGetErrorString(e));                                      \
    MPI_Abort(MPI_COMM_WORLD, 1);                                        \
  } } while(0)

#define NCCL_CHECK(x) do { ncclResult_t r = (x);                         \
  if (r != ncclSuccess) {                                                \
    fprintf(stderr, "NCCL %s:%d %s\n", __FILE__, __LINE__,               \
            ncclGetErrorString(r));                                      \
    MPI_Abort(MPI_COMM_WORLD, 1);                                        \
  } } while(0)

#define MPI_CHECK(x) do { int r = (x);                                   \
  if (r != MPI_SUCCESS) {                                                \
    fprintf(stderr, "MPI %s:%d rc=%d\n", __FILE__, __LINE__, r);         \
    MPI_Abort(MPI_COMM_WORLD, 1);                                        \
  } } while(0)

// World layout: PP outermost, TP middle, SP innermost.
//   rank = pp_rank * (TP*SP) + tp_rank * SP + sp_rank
// PP straddles nodes (1 PP rank per node when world = 2*TP*SP).
// TP and SP stay intra-node, riding NVLink.
struct ParallelCtx {
  int world_rank, world_size;
  int local_rank, local_size;        // intra-node
  int pp_size, tp_size, sp_size;
  int pp_rank, tp_rank, sp_rank;

  ncclComm_t tp_comm;                // intra-node, AllReduce
  ncclComm_t sp_comm;                // intra-node, ring rotation
  ncclComm_t pp_comm;                // inter-node, P2P

  cudaStream_t compute_stream;
  cudaStream_t comm_stream;          // overlap NCCL with compute
  cudaEvent_t  comm_done;

  int device_id;
};

inline void build_subcomm(MPI_Comm world, int color, int key,
                          ncclComm_t* out) {
  MPI_Comm sub;
  MPI_CHECK(MPI_Comm_split(world, color, key, &sub));
  int sub_rank, sub_size;
  MPI_Comm_rank(sub, &sub_rank);
  MPI_Comm_size(sub, &sub_size);

  ncclUniqueId id;
  if (sub_rank == 0) ncclGetUniqueId(&id);
  MPI_CHECK(MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, sub));
  NCCL_CHECK(ncclCommInitRank(out, sub_size, id, sub_rank));
  MPI_Comm_free(&sub);
}

inline void init_parallel(ParallelCtx& ctx, int pp, int tp, int sp) {
  MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &ctx.world_rank));
  MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &ctx.world_size));
  if (pp*tp*sp != ctx.world_size) {
    if (ctx.world_rank == 0)
      fprintf(stderr, "PP*TP*SP (%d) must equal world size (%d)\n",
              pp*tp*sp, ctx.world_size);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  ctx.pp_size = pp; ctx.tp_size = tp; ctx.sp_size = sp;
  ctx.pp_rank = ctx.world_rank / (tp*sp);
  ctx.tp_rank = (ctx.world_rank / sp) % tp;
  ctx.sp_rank = ctx.world_rank % sp;

  // Determine intra-node rank via shared-memory split.
  MPI_Comm shm;
  MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                                MPI_INFO_NULL, &shm));
  MPI_Comm_rank(shm, &ctx.local_rank);
  MPI_Comm_size(shm, &ctx.local_size);
  MPI_Comm_free(&shm);

  ctx.device_id = ctx.local_rank;
  CUDA_CHECK(cudaSetDevice(ctx.device_id));

  // TP group: same (pp_rank, sp_rank), varies tp_rank
  build_subcomm(MPI_COMM_WORLD,
                ctx.pp_rank * sp + ctx.sp_rank,
                ctx.tp_rank, &ctx.tp_comm);
  // SP group: same (pp_rank, tp_rank), varies sp_rank
  build_subcomm(MPI_COMM_WORLD,
                ctx.pp_rank * tp + ctx.tp_rank,
                ctx.sp_rank, &ctx.sp_comm);
  // PP group: same (tp_rank, sp_rank), varies pp_rank  -> crosses nodes
  build_subcomm(MPI_COMM_WORLD,
                ctx.tp_rank * sp + ctx.sp_rank,
                ctx.pp_rank, &ctx.pp_comm);

  CUDA_CHECK(cudaStreamCreateWithFlags(&ctx.compute_stream,
                                       cudaStreamNonBlocking));
  CUDA_CHECK(cudaStreamCreateWithFlags(&ctx.comm_stream,
                                       cudaStreamNonBlocking));
  CUDA_CHECK(cudaEventCreateWithFlags(&ctx.comm_done,
                                      cudaEventDisableTiming));
}

inline void destroy_parallel(ParallelCtx& ctx) {
  cudaStreamDestroy(ctx.compute_stream);
  cudaStreamDestroy(ctx.comm_stream);
  cudaEventDestroy(ctx.comm_done);
  ncclCommDestroy(ctx.tp_comm);
  ncclCommDestroy(ctx.sp_comm);
  ncclCommDestroy(ctx.pp_comm);
}
