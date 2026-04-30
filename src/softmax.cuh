#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

// Standalone numerically-stable rowwise softmax (FP16 IO, FP32 accum).
// Useful for verification; the production path uses fused FlashAttention.
__global__ void softmax_rowwise_fp16(const __half* __restrict__ X,
                                     __half* __restrict__ Y,
                                     int rows, int cols) {
  extern __shared__ float smem[];
  int row = blockIdx.x;
  if (row >= rows) return;
  const __half* xp = X + row * cols;
  __half*       yp = Y + row * cols;

  // Pass 1: row max
  float m = -CUDART_INF_F;
  for (int i = threadIdx.x; i < cols; i += blockDim.x)
    m = fmaxf(m, __half2float(xp[i]));
  smem[threadIdx.x] = m;
  __syncthreads();
  for (int s = blockDim.x/2; s > 0; s >>= 1) {
    if (threadIdx.x < s)
      smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x+s]);
    __syncthreads();
  }
  float row_max = smem[0];
  __syncthreads();

  // Pass 2: sum exp
  float s = 0.f;
  for (int i = threadIdx.x; i < cols; i += blockDim.x)
    s += __expf(__half2float(xp[i]) - row_max);
  smem[threadIdx.x] = s;
  __syncthreads();
  for (int k = blockDim.x/2; k > 0; k >>= 1) {
    if (threadIdx.x < k) smem[threadIdx.x] += smem[threadIdx.x+k];
    __syncthreads();
  }
  float inv = 1.f / smem[0];

  // Pass 3: write
  for (int i = threadIdx.x; i < cols; i += blockDim.x)
    yp[i] = __float2half(__expf(__half2float(xp[i]) - row_max) * inv);
}
