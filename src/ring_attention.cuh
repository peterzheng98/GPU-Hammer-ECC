#pragma once
#include "parallel_ctx.cuh"
#include "flash_attention.cuh"
#include <algorithm>

// Ring-Attention across the SP group:
// each rank holds Q[Sq_local] for its own slice and rotates K/V slices around
// the ring; per step it runs FlashAttention against currently-held K/V then
// merges via the LSE update.
struct RingBuffers {
  __half *K_a, *K_b, *V_a, *V_b;   // double-buffered KV slabs
  __half *O_acc, *O_part;
  float  *lse_acc, *lse_part, *lse_tmp;
  size_t kv_count;                 // elements (not bytes)
  int Sq_local, Sk_local, D, Nh;
};

inline void alloc_ring(RingBuffers& rb, int Nh, int Sq_local, int Sk_local,
                       int D) {
  rb.Nh = Nh; rb.Sq_local = Sq_local; rb.Sk_local = Sk_local; rb.D = D;
  rb.kv_count = (size_t)Nh * Sk_local * D;
  size_t kv_bytes = rb.kv_count * sizeof(__half);
  CUDA_CHECK(cudaMalloc(&rb.K_a, kv_bytes));
  CUDA_CHECK(cudaMalloc(&rb.K_b, kv_bytes));
  CUDA_CHECK(cudaMalloc(&rb.V_a, kv_bytes));
  CUDA_CHECK(cudaMalloc(&rb.V_b, kv_bytes));
  CUDA_CHECK(cudaMalloc(&rb.O_acc,  (size_t)Nh*Sq_local*D*sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&rb.O_part, (size_t)Nh*Sq_local*D*sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&rb.lse_acc,  (size_t)Nh*Sq_local*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&rb.lse_part, (size_t)Nh*Sq_local*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&rb.lse_tmp,  (size_t)Nh*Sq_local*sizeof(float)));
}

inline void free_ring(RingBuffers& rb) {
  cudaFree(rb.K_a); cudaFree(rb.K_b);
  cudaFree(rb.V_a); cudaFree(rb.V_b);
  cudaFree(rb.O_acc); cudaFree(rb.O_part);
  cudaFree(rb.lse_acc); cudaFree(rb.lse_part); cudaFree(rb.lse_tmp);
}

inline void ring_attention_forward(const ParallelCtx& ctx,
                                   const __half* Q_local,
                                   const __half* K_local,
                                   const __half* V_local,
                                   RingBuffers& rb, float scale) {
  const int sp   = ctx.sp_size;
  const int my   = ctx.sp_rank;
  const int next = (my + 1) % sp;
  const int prev = (my - 1 + sp) % sp;
  const size_t kv_bytes = rb.kv_count * sizeof(__half);

  // Stage 0: copy local KV into K_a/V_a
  CUDA_CHECK(cudaMemcpyAsync(rb.K_a, K_local, kv_bytes,
                             cudaMemcpyDeviceToDevice, ctx.compute_stream));
  CUDA_CHECK(cudaMemcpyAsync(rb.V_a, V_local, kv_bytes,
                             cudaMemcpyDeviceToDevice, ctx.compute_stream));

  constexpr int Br = 64, Bc = 64, D = 128;
  dim3 grid((rb.Sq_local + Br - 1)/Br, rb.Nh, 1);
  dim3 block(Br);
  size_t smem = (Br + 2*Bc) * D * sizeof(__half);

  // Initial pass populates O_acc / lse_acc
  flash_attn_fwd<Br, Bc, D><<<grid, block, smem, ctx.compute_stream>>>(
      Q_local, rb.K_a, rb.V_a, rb.O_acc, rb.lse_acc,
      1, rb.Nh, rb.Sq_local, rb.Sk_local, scale);

  for (int step = 1; step < sp; ++step) {
    __half* Ksrc = (step % 2) ? rb.K_a : rb.K_b;
    __half* Kdst = (step % 2) ? rb.K_b : rb.K_a;
    __half* Vsrc = (step % 2) ? rb.V_a : rb.V_b;
    __half* Vdst = (step % 2) ? rb.V_b : rb.V_a;

    // Rotate KV around the SP ring (overlapped with the previous compute)
    NCCL_CHECK(ncclGroupStart());
    NCCL_CHECK(ncclSend(Ksrc, rb.kv_count, ncclFloat16, next,
                        ctx.sp_comm, ctx.comm_stream));
    NCCL_CHECK(ncclRecv(Kdst, rb.kv_count, ncclFloat16, prev,
                        ctx.sp_comm, ctx.comm_stream));
    NCCL_CHECK(ncclSend(Vsrc, rb.kv_count, ncclFloat16, next,
                        ctx.sp_comm, ctx.comm_stream));
    NCCL_CHECK(ncclRecv(Vdst, rb.kv_count, ncclFloat16, prev,
                        ctx.sp_comm, ctx.comm_stream));
    NCCL_CHECK(ncclGroupEnd());

    // Compute waits until KV arrived
    CUDA_CHECK(cudaEventRecord(ctx.comm_done, ctx.comm_stream));
    CUDA_CHECK(cudaStreamWaitEvent(ctx.compute_stream, ctx.comm_done, 0));

    flash_attn_fwd<Br, Bc, D><<<grid, block, smem, ctx.compute_stream>>>(
        Q_local, Kdst, Vdst, rb.O_part, rb.lse_part,
        1, rb.Nh, rb.Sq_local, rb.Sk_local, scale);

    int rows = rb.Nh * rb.Sq_local;
    flash_attn_merge<<<rows, 128, 0, ctx.compute_stream>>>(
        rb.O_acc, rb.lse_acc, rb.O_part, rb.lse_part, rb.lse_tmp,
        rows, D);
    std::swap(rb.lse_acc, rb.lse_tmp);
  }
}
