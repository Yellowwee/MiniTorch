#pragma once
#include <cuda_runtime.h>
// Time preallocated native launches without Python dispatch/allocator overhead.
// This is a stream interval, including device-to-device copies in the fused ABI.
template <typename Launch>
int mt_benchmark(float *seconds, int repeats, Launch launch) {
  if(repeats <= 0) return cudaErrorInvalidValue;
  cudaEvent_t begin, end;
  cudaError_t status = cudaEventCreate(&begin);
  if(status != cudaSuccess) return status;
  status = cudaEventCreate(&end);
  if(status != cudaSuccess) { cudaEventDestroy(begin); return status; }
  status = cudaEventRecord(begin);
  for(int i=0; i<repeats && status==cudaSuccess; ++i)
    status = static_cast<cudaError_t>(launch());
  if(status==cudaSuccess) status=cudaEventRecord(end);
  if(status==cudaSuccess) status=cudaEventSynchronize(end);
  float ms=0;
  if(status==cudaSuccess) status=cudaEventElapsedTime(&ms,begin,end);
  cudaEventDestroy(begin); cudaEventDestroy(end);
  *seconds=ms/(1000.f*repeats);
  return status;
}
