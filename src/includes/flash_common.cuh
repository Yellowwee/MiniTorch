#pragma once
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>

// FP32 correctness-first FlashAttention: 4 query/key owners per block,
// 32 positions per streamed tile. Padding avoids shared-memory bank conflicts.
constexpr int FA_ROWS=4, FA_TILE=32, FA_DIM=128, FA_STRIDE=33;
__device__ inline float fa_sum(float x) {
  for(int s=16;s;s>>=1) x+=__shfl_xor_sync(0xffffffff,x,s);
  return x;
}
__device__ inline float fa_max(float x) {
  for(int s=16;s;s>>=1) x=fmaxf(x,__shfl_xor_sync(0xffffffff,x,s));
  return x;
}
inline bool fa_shape(int b,int h,int nq,int nk,int d) {
  return b>0 && h>0 && nq>0 && nk>0 && d>0 && d<=FA_DIM &&
    int64_t(b)*h*nq*d<=INT32_MAX && int64_t(b)*h*nk*d<=INT32_MAX;
}
__device__ inline bool fa_valid(int qi,int kj,int nq,int nk,float qlen,float klen,bool causal) {
  return qi<nq && kj<nk && qi<qlen && kj<klen && (!causal || kj<=qi);
}
