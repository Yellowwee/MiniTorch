#include <cuda_runtime.h>
#include <cmath>
#include "includes/typed_io.cuh"

// Every block participates in the full reduction, including irregular tails.
__device__ float reduce(float x, bool maximum=false) {
  __shared__ float values[256];
  values[threadIdx.x]=x; __syncthreads();
  for(int s=128;s;s>>=1) {
    if(threadIdx.x<s) values[threadIdx.x]=maximum ? fmaxf(values[threadIdx.x],values[threadIdx.x+s]) : values[threadIdx.x]+values[threadIdx.x+s];
    __syncthreads();
  }
  float result=values[0]; __syncthreads(); return result;
}

template<class T>
__global__ void softmax_fw(T* out,const T* x,const float* mask,int heads,int queries,int width) {
  int row=blockIdx.x, batch=row/(heads*queries);
  float m=-INFINITY,bad=0;
  for(int j=threadIdx.x;j<width;j+=256) {
    float s=mt_read(x[row*width+j])+mask[batch*width+j];
    m=fmaxf(m,s); if(isnan(s)||s==INFINITY) bad=1;
  }
  m=reduce(m,true);
  if(reduce(bad,true)>0) {
    for(int j=threadIdx.x;j<width;j+=256) out[row*width+j]=T(NAN);
    return;
  }
  float sum=0;
  for(int j=threadIdx.x;j<width;j+=256) {
    float s=mt_read(x[row*width+j])+mask[batch*width+j];
    sum+=isfinite(m)?expf(s-m):0;
  }
  sum=reduce(sum);
  for(int j=threadIdx.x;j<width;j+=256) {
    float s=mt_read(x[row*width+j])+mask[batch*width+j];
    out[row*width+j]=T(sum>0?expf(s-m)/sum:0);
  }
}
template<class T>
__global__ void softmax_bw(float* dx,const float* dy,const T* y,int width) {
  int row=blockIdx.x; float dot=0;
  for(int j=threadIdx.x;j<width;j+=256) dot+=dy[row*width+j]*mt_read(y[row*width+j]);
  dot=reduce(dot);
  for(int j=threadIdx.x;j<width;j+=256) dx[row*width+j]=mt_read(y[row*width+j])*(dy[row*width+j]-dot);
}
template<class T>
__global__ void ln_fw(T* out,float* variance,float* mean,const T* x,const float* g,const float* b,int width) {
  int row=blockIdx.x; float sum=0;
  for(int j=threadIdx.x;j<width;j+=256) sum+=mt_read(x[row*width+j]);
  float mu=reduce(sum)/width;
  // Centered two-pass variance avoids E[x^2]-E[x]^2 cancellation.
  float ss=0;
  for(int j=threadIdx.x;j<width;j+=256) { float z=mt_read(x[row*width+j])-mu; ss+=z*z; }
  float var=reduce(ss)/width+1e-5f;
  if(threadIdx.x==0) { variance[row]=var; mean[row]=mu; }
  for(int j=threadIdx.x;j<width;j+=256) out[row*width+j]=T((mt_read(x[row*width+j])-mu)*rsqrtf(var)*g[j]+b[j]);
}
template<class T>
__global__ void ln_dx(float* dx,const float* dy,const T* x,const float* g,const float* var,const float* mean,int width) {
  int row=blockIdx.x; float a=0,b=0,inv=rsqrtf(var[row]);
  for(int j=threadIdx.x;j<width;j+=256) {
    float u=dy[row*width+j]*g[j],z=(mt_read(x[row*width+j])-mean[row])*inv;
    a+=u; b+=u*z;
  }
  a=reduce(a)/width; b=reduce(b)/width;
  for(int j=threadIdx.x;j<width;j+=256) {
    float z=(mt_read(x[row*width+j])-mean[row])*inv;
    dx[row*width+j]=inv*(dy[row*width+j]*g[j]-a-z*b);
  }
}
template<class T>
__global__ void ln_affine(float* dg,float* db,const float* dy,const T* x,const float* var,const float* mean,int rows,int width) {
  int j=blockIdx.x; float a=0,b=0;
  for(int row=threadIdx.x;row<rows;row+=256) {
    float u=dy[row*width+j];
    a+=u*(mt_read(x[row*width+j])-mean[row])*rsqrtf(var[row]); b+=u;
  }
  a=reduce(a); b=reduce(b);
  if(threadIdx.x==0) { dg[j]=a; db[j]=b; }
}

// Gradients/master parameters are FP32. An integer flag is the only D2H value
// required for overflow detection, regardless of model size.
__global__ void unscale_kernel(float* g,int n,float inverse,int* bad) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n) { float x=g[i],v=x*inverse; if(!isfinite(x)||!isfinite(v)) atomicExch(bad,1); g[i]=v; }
}
extern "C" int mt_unscale(float* g,int n,float inverse,int* bad) {
  if(n) unscale_kernel<<<(n+255)/256,256>>>(g,n,inverse,bad);
  return cudaGetLastError();
}

template<class T> int launch_sm(void* out,const void* x,const float* mask,int rows,int h,int q,int w) {
  softmax_fw<<<rows,256>>>((T*)out,(const T*)x,mask,h,q,w); return cudaGetLastError();
}
extern "C" int mt_amp_softmax_fw(void* out,const void* x,const float* mask,int dtype,int rows,int h,int q,int w) {
  if(rows<=0||h<=0||q<=0||w<=0) return cudaErrorInvalidValue;
  if(dtype==1) return launch_sm<__half>(out,x,mask,rows,h,q,w);
  if(dtype==2) return launch_sm<__nv_bfloat16>(out,x,mask,rows,h,q,w);
  return cudaErrorInvalidValue;
}
extern "C" int mt_amp_softmax_bw(float* dx,const float* dy,const void* y,int dtype,int rows,int w) {
  if(rows<=0||w<=0) return cudaErrorInvalidValue;
  if(dtype==1) softmax_bw<<<rows,256>>>(dx,dy,(const __half*)y,w);
  else if(dtype==2) softmax_bw<<<rows,256>>>(dx,dy,(const __nv_bfloat16*)y,w);
  else return cudaErrorInvalidValue;
  return cudaGetLastError();
}
extern "C" int mt_amp_ln_fw(void* out,float* var,float* mean,const void* x,const float* g,const float* b,int dtype,int rows,int w) {
  if(rows<=0||w<=0) return cudaErrorInvalidValue;
  if(dtype==1) ln_fw<<<rows,256>>>((__half*)out,var,mean,(const __half*)x,g,b,w);
  else if(dtype==2) ln_fw<<<rows,256>>>((__nv_bfloat16*)out,var,mean,(const __nv_bfloat16*)x,g,b,w);
  else return cudaErrorInvalidValue;
  return cudaGetLastError();
}
template<class T> int launch_ln_bw(float* dx,float* dg,float* db,const float* dy,const void* x,const float* g,const float* var,const float* mean,int rows,int w) {
  ln_dx<<<rows,256>>>(dx,dy,(const T*)x,g,var,mean,w);
  cudaError_t e=cudaGetLastError(); if(e!=cudaSuccess) return e;
  ln_affine<<<w,256>>>(dg,db,dy,(const T*)x,var,mean,rows,w);
  return cudaGetLastError();
}
extern "C" int mt_amp_ln_bw(float* dx,float* dg,float* db,const float* dy,const void* x,const float* g,const float* var,const float* mean,int dtype,int rows,int w) {
  if(rows<=0||w<=0) return cudaErrorInvalidValue;
  if(dtype==1) return launch_ln_bw<__half>(dx,dg,db,dy,x,g,var,mean,rows,w);
  if(dtype==2) return launch_ln_bw<__nv_bfloat16>(dx,dg,db,dy,x,g,var,mean,rows,w);
  return cudaErrorInvalidValue;
}
