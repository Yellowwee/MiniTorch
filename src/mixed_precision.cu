// Typed device-pointer ABI. All scalar arithmetic/reductions accumulate in FP32.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cmath>

struct MTLayout { int dims; int shape[32]; int strides[32]; };

__device__ float load(const void* p, int i, int dtype) {
  if (dtype == 1) return __half2float(static_cast<const __half*>(p)[i]);
  if (dtype == 2) return __bfloat162float(static_cast<const __nv_bfloat16*>(p)[i]);
  return static_cast<const float*>(p)[i];
}
__device__ void store(void* p, int i, int dtype, float value) {
  if (dtype == 1) static_cast<__half*>(p)[i] = __float2half_rn(value);
  else if (dtype == 2) static_cast<__nv_bfloat16*>(p)[i] = __float2bfloat16_rn(value);
  else static_cast<float*>(p)[i] = value;
}
__device__ int position(int ordinal, MTLayout out, MTLayout in) {
  int pos = 0;
  for (int d = out.dims-1; d >= 0; --d) {
    int index = ordinal % out.shape[d]; ordinal /= out.shape[d];
    int j = d-(out.dims-in.dims);
    if (j >= 0 && in.shape[j] != 1) pos += index*in.strides[j];
  }
  return pos;
}
__device__ float operation(int op, float x, float y=0) {
  switch(op) {
    case 1: return x+y;
    case 2: return x*y;
    case 3: return x;
    case 4: return -x;
    case 5: return x<y;
    case 6: return x==y;
    case 7: { float z=expf(-fabsf(x)); return x>=0 ? 1/(1+z) : z/(1+z); }
    case 8: return x>0 ? x : 0;
    case 9: return x>0 ? y : 0;
    case 10: return logf(x+1e-6f);
    case 11: return y/(x+1e-6f);
    case 12: return expf(x);
    case 13: return 1/x;
    case 14: return -(1/(x*x))*y;
    case 15: return (x-y<1e-2f) && (y-x<1e-2f);
    case 16: return x>y ? x : y;
    case 17: return powf(x,y);
    case 18: return tanhf(x);
    default: return nanf("");
  }
}
__global__ void fill_kernel(void* out, int dtype, int n, float value) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n) store(out,i,dtype,value);
}
__global__ void map_kernel(void* out, int od, MTLayout o, const void* a, int ad, MTLayout x, int n, int op) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n) store(out,position(i,o,o),od,operation(op,load(a,position(i,o,x),ad)));
}
__global__ void zip_kernel(void* out, int od, MTLayout o, const void* a, int ad, MTLayout x,
                           const void* b, int bd, MTLayout y, int n, int op) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n) store(out,position(i,o,o),od,operation(op,load(a,position(i,o,x),ad),load(b,position(i,o,y),bd)));
}
__global__ void reduce_kernel(float* out, MTLayout o, const void* a, int ad, MTLayout x,
                              int dim, float start, int op) {
  __shared__ float values[256];
  int pos=position(blockIdx.x,o,x);
  float value=start;
  for(int j=threadIdx.x;j<x.shape[dim];j+=256)
    value=operation(op,value,load(a,pos+j*x.strides[dim],ad));
  values[threadIdx.x]=value;
  __syncthreads();
  for(int s=128;s;s>>=1) {
    if(threadIdx.x<s) values[threadIdx.x]=operation(op,values[threadIdx.x],values[threadIdx.x+s]);
    __syncthreads();
  }
  if(threadIdx.x==0) out[position(blockIdx.x,o,o)]=values[0];
}
extern "C" {
int mt_typed_fill(void* out, int dtype, int n, float value) {
  if(n) fill_kernel<<<(n+255)/256,256>>>(out,dtype,n,value);
  return cudaGetLastError();
}
int mt_typed_map(void* out, int od, MTLayout o, const void* a, int ad, MTLayout x, int n, int op) {
  if(n) map_kernel<<<(n+255)/256,256>>>(out,od,o,a,ad,x,n,op);
  return cudaGetLastError();
}
int mt_typed_zip(void* out, int od, MTLayout o, const void* a, int ad, MTLayout x,
                 const void* b, int bd, MTLayout y, int n, int op) {
  if(n) zip_kernel<<<(n+255)/256,256>>>(out,od,o,a,ad,x,b,bd,y,n,op);
  return cudaGetLastError();
}
int mt_typed_reduce(float* out, MTLayout o, const void* a, int ad, MTLayout x, int n, int dim, float start, int op) {
  if(n) reduce_kernel<<<n,256>>>(out,o,a,ad,x,dim,start,op);
  return cudaGetLastError();
}
int mt_low_gemm_supported(int dtype) {
  cudaDeviceProp p; int device;
  if(cudaGetDevice(&device)!=cudaSuccess || cudaGetDeviceProperties(&p,device)!=cudaSuccess) return 0;
  return dtype==1 ? (p.major>=7) : (dtype==2 && p.major>=8);
}
int mt_typed_gemm(cublasHandle_t h, void* out, const void* a, const void* b,
                  int m, int n, int k, int dtype, int out_dtype) {
  if(dtype<1 || dtype>2 || (out_dtype!=0 && out_dtype!=dtype)) return CUBLAS_STATUS_INVALID_VALUE;
  cudaDataType_t input=dtype==1 ? CUDA_R_16F : CUDA_R_16BF;
  cudaDataType_t output=out_dtype==0 ? CUDA_R_32F : input;
  const float alpha=1, beta=0;
  // COMPUTE_32F alone can still allow split-K partial sums in the output dtype.
  // Forbid that narrowing, and restore the shared handle for the FP32 path.
  cublasMath_t previous;
  cublasStatus_t status=cublasGetMathMode(h,&previous);
  if(status!=CUBLAS_STATUS_SUCCESS) return status;
  status=cublasSetMathMode(h,static_cast<cublasMath_t>(
      CUBLAS_DEFAULT_MATH | CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION));
  if(status!=CUBLAS_STATUS_SUCCESS) return status;
  // Row-major C=A B is column-major C^T=B^T A^T.
  status=cublasGemmEx(h,CUBLAS_OP_N,CUBLAS_OP_N,n,m,k,&alpha,b,input,n,a,input,k,
                     &beta,out,output,n,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT);
  cublasStatus_t restored=cublasSetMathMode(h,previous);
  return status!=CUBLAS_STATUS_SUCCESS ? status : restored;
}
}
