// Distributed Attention Stress Test for ECC monitoring
//
// Hybrid parallelism:  PP (across nodes) x TP (intra-node) x SP (intra-node)
// Algorithms:          FlashAttention forward + Ring-Attention across SP
// Workload:            S up to 1,000,000 tokens, sustained dense FP16 GEMMs
//
// Build: see CMakeLists.txt
// Run:   scripts/run_slurm.sh  or  scripts/run_mpirun.sh

#include "parallel_ctx.cuh"
#include "flash_attention.cuh"
#include "ring_attention.cuh"
#include "softmax.cuh"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <chrono>
#include <vector>
#include <cstdio>
#include <cstring>
#include <cmath>

struct ModelCfg {
  int S  = 1'000'000;   // global sequence length
  int H  = 8192;        // hidden dim
  int Nh = 64;          // num heads
  int D  = 128;         // head dim   (Nh*D == H)
  int L  = 80;          // total transformer layers
  int micro_batches = 8;
};

static void parse_args(int argc, char** argv,
                       int& pp, int& tp, int& sp,
                       int& iters, int& seq) {
  pp=2; tp=4; sp=2; iters=100; seq=1'000'000;
  for (int i = 1; i < argc; ++i) {
    if      (!strcmp(argv[i],"--pp"))    pp    = atoi(argv[++i]);
    else if (!strcmp(argv[i],"--tp"))    tp    = atoi(argv[++i]);
    else if (!strcmp(argv[i],"--sp"))    sp    = atoi(argv[++i]);
    else if (!strcmp(argv[i],"--iters")) iters = atoi(argv[++i]);
    else if (!strcmp(argv[i],"--seq"))   seq   = atoi(argv[++i]);
  }
}

// Deterministic random fill, kept small to avoid softmax overflow.
__global__ void fill_rand_fp16(__half* p, size_t n, uint64_t seed) {
  size_t i = blockIdx.x*(size_t)blockDim.x + threadIdx.x;
  if (i >= n) return;
  uint64_t x = seed ^ (i * 0x9E3779B97F4A7C15ULL);
  x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
  x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
  x ^= x >> 33;
  float f = ((int)(x & 0xffff) - 32768) / 32768.0f;
  p[i] = __float2half(f * 0.02f);
}

__global__ void residual_add(__half* a, const __half* b, size_t n) {
  size_t i = blockIdx.x*(size_t)blockDim.x + threadIdx.x;
  if (i < n) a[i] = __hadd(a[i], b[i]);
}

// FP16 GEMM, FP32 accum, tensor cores.
// Computes Y[M,N] = X[M,K] * W[K,N] in row-major (translated to column-major
// cuBLAS).  See cuBLAS docs: trick is to swap and use OP_N/OP_N.
static void linear_fp16(cublasHandle_t h,
                        const __half* X, const __half* W, __half* Y,
                        int M, int N, int K, cudaStream_t s) {
  cublasSetStream(h, s);
  const float alpha = 1.f, beta = 0.f;
  cublasGemmEx(h,
      CUBLAS_OP_N, CUBLAS_OP_N,
      N, M, K,
      &alpha,
      W, CUDA_R_16F, N,
      X, CUDA_R_16F, K,
      &beta,
      Y, CUDA_R_16F, N,
      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

int main(int argc, char** argv) {
  MPI_CHECK(MPI_Init(&argc, &argv));
  int pp, tp, sp, iters, seq;
  parse_args(argc, argv, pp, tp, sp, iters, seq);

  ParallelCtx ctx;
  init_parallel(ctx, pp, tp, sp);

  ModelCfg cfg; cfg.S = seq;
  if (ctx.world_rank == 0) {
    printf("== Distributed Attention Stress ==\n");
    printf("world=%d  pp=%d tp=%d sp=%d  S=%d H=%d Nh=%d D=%d L=%d  iters=%d\n",
           ctx.world_size, pp, tp, sp,
           cfg.S, cfg.H, cfg.Nh, cfg.D, cfg.L, iters);
    fflush(stdout);
  }

  const int Sq_local = cfg.S / sp;
  const int Nh_local = cfg.Nh / tp;
  const int D        = cfg.D;
  const int H_local  = Nh_local * D;
  const int L_local  = cfg.L / pp;

  size_t qkv_bytes = (size_t)Nh_local * Sq_local * D * sizeof(__half);
  size_t weight_bytes = (size_t)cfg.H * H_local * sizeof(__half);

  if (ctx.world_rank == 0) {
    printf("[mem] Q/K/V each: %.2f GB  W{q,k,v,o} each: %.2f GB  layers/stage: %d\n",
           qkv_bytes/1e9, weight_bytes/1e9, L_local);
    fflush(stdout);
  }

  // Per-layer projection weights, sharded across TP, replicated across SP.
  std::vector<__half*> Wq(L_local), Wk(L_local), Wv(L_local), Wo(L_local);
  for (int l = 0; l < L_local; ++l) {
    CUDA_CHECK(cudaMalloc(&Wq[l], weight_bytes));
    CUDA_CHECK(cudaMalloc(&Wk[l], weight_bytes));
    CUDA_CHECK(cudaMalloc(&Wv[l], weight_bytes));
    CUDA_CHECK(cudaMalloc(&Wo[l], weight_bytes));
    size_t n = weight_bytes / sizeof(__half);
    int blk = 256, grd = (n+blk-1)/blk;
    fill_rand_fp16<<<grd,blk,0,ctx.compute_stream>>>(Wq[l], n, 1+l);
    fill_rand_fp16<<<grd,blk,0,ctx.compute_stream>>>(Wk[l], n, 2+l);
    fill_rand_fp16<<<grd,blk,0,ctx.compute_stream>>>(Wv[l], n, 3+l);
    fill_rand_fp16<<<grd,blk,0,ctx.compute_stream>>>(Wo[l], n, 4+l);
  }

  // Activation buffers reused across layers (gradient checkpoint style).
  __half *X, *Q, *K, *V, *Oproj;
  CUDA_CHECK(cudaMalloc(&X,     (size_t)Sq_local*cfg.H*sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&Q,     qkv_bytes));
  CUDA_CHECK(cudaMalloc(&K,     qkv_bytes));
  CUDA_CHECK(cudaMalloc(&V,     qkv_bytes));
  CUDA_CHECK(cudaMalloc(&Oproj, (size_t)Sq_local*cfg.H*sizeof(__half)));
  {
    size_t n = (size_t)Sq_local*cfg.H;
    int blk = 256, grd = (n+blk-1)/blk;
    fill_rand_fp16<<<grd,blk,0,ctx.compute_stream>>>(X, n, 7);
  }

  RingBuffers rb;
  alloc_ring(rb, Nh_local, Sq_local, Sq_local, D);

  cublasHandle_t cublas;
  cublasCreate(&cublas);
  cublasSetMathMode(cublas, CUBLAS_TENSOR_OP_MATH);

  __half *pp_recv_buf = nullptr, *pp_send_buf = nullptr;
  if (ctx.pp_size > 1) {
    CUDA_CHECK(cudaMalloc(&pp_recv_buf,
                          (size_t)Sq_local*cfg.H*sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&pp_send_buf,
                          (size_t)Sq_local*cfg.H*sizeof(__half)));
  }

  const float scale = 1.f / sqrtf((float)D);

  CUDA_CHECK(cudaDeviceSynchronize());
  MPI_Barrier(MPI_COMM_WORLD);
  auto t0 = std::chrono::high_resolution_clock::now();

  for (int it = 0; it < iters; ++it) {
    for (int mb = 0; mb < cfg.micro_batches; ++mb) {

      // PP: receive from previous stage
      if (ctx.pp_rank > 0) {
        NCCL_CHECK(ncclRecv(pp_recv_buf,
            (size_t)Sq_local*cfg.H, ncclFloat16,
            ctx.pp_rank-1, ctx.pp_comm, ctx.comm_stream));
        CUDA_CHECK(cudaEventRecord(ctx.comm_done, ctx.comm_stream));
        CUDA_CHECK(cudaStreamWaitEvent(ctx.compute_stream, ctx.comm_done, 0));
        CUDA_CHECK(cudaMemcpyAsync(X, pp_recv_buf,
            (size_t)Sq_local*cfg.H*sizeof(__half),
            cudaMemcpyDeviceToDevice, ctx.compute_stream));
      }

      // Forward through this stage's layers
      for (int l = 0; l < L_local; ++l) {
        // QKV projections (TP column-parallel: each rank owns H_local cols)
        linear_fp16(cublas, X, Wq[l], Q, Sq_local, H_local, cfg.H,
                    ctx.compute_stream);
        linear_fp16(cublas, X, Wk[l], K, Sq_local, H_local, cfg.H,
                    ctx.compute_stream);
        linear_fp16(cublas, X, Wv[l], V, Sq_local, H_local, cfg.H,
                    ctx.compute_stream);

        // Sequence-parallel FlashAttention via Ring rotation
        ring_attention_forward(ctx, Q, K, V, rb, scale);

        // Output projection (TP row-parallel) + AllReduce within TP group
        linear_fp16(cublas, rb.O_acc, Wo[l], Oproj,
                    Sq_local, cfg.H, H_local, ctx.compute_stream);
        NCCL_CHECK(ncclAllReduce(Oproj, Oproj,
            (size_t)Sq_local*cfg.H, ncclFloat16, ncclSum,
            ctx.tp_comm, ctx.compute_stream));

        // Residual: X += Oproj
        size_t n = (size_t)Sq_local*cfg.H;
        int blk = 256, grd = (n+blk-1)/blk;
        residual_add<<<grd,blk,0,ctx.compute_stream>>>(X, Oproj, n);
      }

      // PP: send to next stage
      if (ctx.pp_rank < ctx.pp_size - 1) {
        CUDA_CHECK(cudaMemcpyAsync(pp_send_buf, X,
            (size_t)Sq_local*cfg.H*sizeof(__half),
            cudaMemcpyDeviceToDevice, ctx.compute_stream));
        CUDA_CHECK(cudaEventRecord(ctx.comm_done, ctx.compute_stream));
        CUDA_CHECK(cudaStreamWaitEvent(ctx.comm_stream, ctx.comm_done, 0));
        NCCL_CHECK(ncclSend(pp_send_buf,
            (size_t)Sq_local*cfg.H, ncclFloat16,
            ctx.pp_rank+1, ctx.pp_comm, ctx.comm_stream));
      }
    } // microbatches

    if (ctx.world_rank == 0 && (it % 10 == 0)) {
      CUDA_CHECK(cudaStreamSynchronize(ctx.compute_stream));
      auto now = std::chrono::high_resolution_clock::now();
      double sec = std::chrono::duration<double>(now - t0).count();
      printf("[iter %4d] elapsed %.1fs\n", it, sec);
      fflush(stdout);
    }
  }

  CUDA_CHECK(cudaDeviceSynchronize());
  MPI_Barrier(MPI_COMM_WORLD);
  if (ctx.world_rank == 0) printf("Stress run complete.\n");

  // Cleanup
  free_ring(rb);
  for (int l = 0; l < L_local; ++l) {
    cudaFree(Wq[l]); cudaFree(Wk[l]); cudaFree(Wv[l]); cudaFree(Wo[l]);
  }
  cudaFree(X); cudaFree(Q); cudaFree(K); cudaFree(V); cudaFree(Oproj);
  if (pp_recv_buf) cudaFree(pp_recv_buf);
  if (pp_send_buf) cudaFree(pp_send_buf);
  cublasDestroy(cublas);
  destroy_parallel(ctx);
  MPI_Finalize();
  return 0;
}
