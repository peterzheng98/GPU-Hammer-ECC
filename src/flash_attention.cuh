#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

// FlashAttention-1 forward (FP16 IO, FP32 accum) with online softmax.
// Computes O = softmax(QK^T / sqrt(d)) V without materializing S=QK^T.
//
// Layout: Q[B, Nh, Sq, D], K/V[B, Nh, Sk, D], O[B, Nh, Sq, D]
// Grid:   (Tr, Nh, B)   where Tr = ceil(Sq/Br)
// Block:  Br threads (each thread owns one query row in the tile)
//
// lse_out[B*Nh*Sq] receives log-sum-exp per row; required for the SP merge.
template <int Br, int Bc, int D>
__global__ void flash_attn_fwd(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    __half*       __restrict__ O,
    float*        __restrict__ lse_out,
    int B, int Nh, int Sq, int Sk,
    float scale)
{
  extern __shared__ __half smem[];
  __half* Qs = smem;                  // Br*D
  __half* Ks = Qs + Br*D;             // Bc*D
  __half* Vs = Ks + Bc*D;             // Bc*D

  int tid  = threadIdx.x;
  int bidx = blockIdx.z;
  int hidx = blockIdx.y;
  int qbid = blockIdx.x;
  int q0   = qbid * Br;

  const __half* Qp = Q + ((bidx*Nh + hidx)*Sq + q0) * D;
  const __half* Kp = K + ((bidx*Nh + hidx)*Sk    ) * D;
  const __half* Vp = V + ((bidx*Nh + hidx)*Sk    ) * D;
  __half*       Op = O + ((bidx*Nh + hidx)*Sq + q0) * D;

  // Stage Q tile to shared
  for (int i = tid; i < Br*D; i += blockDim.x) {
    int r = i / D, c = i % D;
    Qs[i] = (q0+r < Sq) ? Qp[r*D + c] : __float2half(0.f);
  }
  __syncthreads();

  // Each thread owns row `tid` (we launch blockDim.x == Br threads)
  int my_row = tid;
  bool valid = (my_row < Br) && (q0 + my_row < Sq);

  float m_i = -CUDART_INF_F;   // running max
  float l_i = 0.f;             // running sum
  float acc[D];
  #pragma unroll
  for (int t = 0; t < D; ++t) acc[t] = 0.f;

  // Iterate over K/V tiles
  for (int k0 = 0; k0 < Sk; k0 += Bc) {
    // Load K, V tiles
    for (int i = tid; i < Bc*D; i += blockDim.x) {
      int r = i / D, c = i % D;
      bool ok = (k0+r < Sk);
      Ks[i] = ok ? Kp[(k0+r)*D + c] : __float2half(0.f);
      Vs[i] = ok ? Vp[(k0+r)*D + c] : __float2half(0.f);
    }
    __syncthreads();

    if (valid) {
      // Compute logits S[j] = Q_row · K_j * scale  (j in [0,Bc))
      float S[Bc];
      #pragma unroll
      for (int j = 0; j < Bc; ++j) {
        float s = 0.f;
        #pragma unroll
        for (int t = 0; t < D; ++t)
          s += __half2float(Qs[my_row*D + t]) * __half2float(Ks[j*D + t]);
        S[j] = s * scale;
      }
      // New row max
      float m_new = m_i;
      #pragma unroll
      for (int j = 0; j < Bc; ++j) m_new = fmaxf(m_new, S[j]);

      // Rescale prior accumulator
      float alpha = __expf(m_i - m_new);
      #pragma unroll
      for (int t = 0; t < D; ++t) acc[t] *= alpha;
      float l_new = alpha * l_i;

      // Add this tile's contributions
      #pragma unroll
      for (int j = 0; j < Bc; ++j) {
        float p = __expf(S[j] - m_new);
        l_new += p;
        #pragma unroll
        for (int t = 0; t < D; ++t)
          acc[t] += p * __half2float(Vs[j*D + t]);
      }
      m_i = m_new;
      l_i = l_new;
    }
    __syncthreads();
  }

  if (valid) {
    float inv = 1.f / l_i;
    #pragma unroll
    for (int t = 0; t < D; ++t)
      Op[my_row*D + t] = __float2half(acc[t] * inv);
    if (lse_out)
      lse_out[((bidx*Nh + hidx)*Sq) + (q0 + my_row)] = m_i + __logf(l_i);
  }
}

// Merge two partial FlashAttention outputs via log-sum-exp.
// Used for Ring-Attention across the SP group.
__global__ void flash_attn_merge(
    __half* O_a, const float* lse_a,
    const __half* O_b, const float* lse_b,
    float* lse_out,
    int N, int D)
{
  int row = blockIdx.x;
  int tid = threadIdx.x;
  if (row >= N) return;
  float la = lse_a[row], lb = lse_b[row];
  float m  = fmaxf(la, lb);
  float wa = __expf(la - m);
  float wb = __expf(lb - m);
  float sum = wa + wb;
  for (int t = tid; t < D; t += blockDim.x) {
    float a = __half2float(O_a[row*D + t]);
    float b = __half2float(O_b[row*D + t]);
    O_a[row*D + t] = __float2half((a*wa + b*wb) / sum);
  }
  if (tid == 0) lse_out[row] = m + __logf(sum);
}
