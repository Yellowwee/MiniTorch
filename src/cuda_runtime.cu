// Stage 1: one device, default stream, owned allocations. No caching allocator yet.
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand_kernel.h>
#include <cstdint>

extern "C" {
int mt_malloc(void **p, size_t bytes) { return cudaMalloc(p, bytes); }
int mt_free(void *p) { return cudaFree(p); }
int mt_copy(void *dst, const void *src, size_t bytes, int kind) {
  return cudaMemcpy(dst, src, bytes, static_cast<cudaMemcpyKind>(kind));
}
int mt_sync() { return cudaStreamSynchronize(nullptr); }
const char *mt_error(int code) { return cudaGetErrorString(static_cast<cudaError_t>(code)); }
int mt_event_create(cudaEvent_t *e) { return cudaEventCreate(e); }
int mt_event_record(cudaEvent_t e) { return cudaEventRecord(e, nullptr); }
int mt_event_elapsed(float *ms, cudaEvent_t a, cudaEvent_t b) {
  cudaError_t e = cudaEventSynchronize(b);
  return e == cudaSuccess ? cudaEventElapsedTime(ms, a, b) : e;
}
int mt_event_destroy(cudaEvent_t e) { return cudaEventDestroy(e); }
int mt_blas_create(cublasHandle_t *h) { return cublasCreate(h); }
int mt_blas_destroy(cublasHandle_t h) { return cublasDestroy(h); }
int mt_gemm(cublasHandle_t h, float *out, const float *a, const float *b,
            int m, int n, int k) {
  const float alpha = 1, beta = 0;
  // Row-major C=A@B is column-major C^T=B^T@A^T.
  return cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                    &alpha, b, n, a, k, &beta, out, n);
}
}

__global__ void mt_fill_kernel(float *out, int size, float value) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) out[i] = value;
}
__global__ void mt_random_kernel(float *out, int size, unsigned long long seed) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    curandStatePhilox4_32_10_t state;
    curand_init(seed, i, 0, &state);
    out[i] = 1.f - curand_uniform(&state);  // [0,1)
  }
}
__global__ void mt_one_hot_kernel(float *out, const float *ids, int size, int classes) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) out[i] = (i % classes == static_cast<int>(ids[i / classes])) ? 1.f : 0.f;
}
__global__ void mt_sequence_kernel(float *out, int size, int width) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) out[i] = width ? ((i % width > i / width) ? -3.402823466e38f : 0.f) : float(i);
}
extern "C" {
int mt_fill(float *p, int n, float v) {
  if(n) mt_fill_kernel<<<(n+255)/256,256>>>(p,n,v);
  return cudaGetLastError();
}
int mt_random(float *p, int n, unsigned long long seed) {
  if(n) mt_random_kernel<<<(n+255)/256,256>>>(p,n,seed);
  return cudaGetLastError();
}
int mt_one_hot(float *out, const float *ids, int size, int classes) {
  if(size) mt_one_hot_kernel<<<(size+255)/256,256>>>(out,ids,size,classes);
  return cudaGetLastError();
}
int mt_sequence(float *out, int size, int width) {
  if(size) mt_sequence_kernel<<<(size+255)/256,256>>>(out,size,width);
  return cudaGetLastError();
}
}
