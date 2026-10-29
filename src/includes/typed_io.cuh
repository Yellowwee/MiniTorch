#pragma once
#include <cuda_fp16.h>
#include <cuda_bf16.h>
__device__ inline float mt_read(float x) { return x; }
__device__ inline float mt_read(__half x) { return __half2float(x); }
__device__ inline float mt_read(__nv_bfloat16 x) { return __bfloat162float(x); }
