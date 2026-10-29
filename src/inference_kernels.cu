#include <cuda_runtime.h>
#include <cstdint>
#include <climits>
#include "includes/typed_io.cuh"

// Pretrained inference embeddings: avoid a [tokens,vocabulary] one-hot matrix.
__global__ void embedding_gather(float* dst,const float* weights,const float* ids,int tokens,int vocab,int dim) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<tokens*dim) {
    float raw=ids[i/dim];
    if(!isfinite(raw) || raw<0 || raw>=vocab || raw!=floorf(raw)) dst[i]=NAN;
    else dst[i]=weights[int(raw)*dim+i%dim];
  }
}
extern "C" int mt_embedding_gather(float* dst,const float* weights,const float* ids,int tokens,int vocab,int dim) {
  if(tokens<1||vocab<1||dim<1||int64_t(tokens)*dim>INT32_MAX||int64_t(vocab)*dim>INT32_MAX) return cudaErrorInvalidValue;
  embedding_gather<<<(int64_t(tokens)*dim+255)/256,256>>>(dst,weights,ids,tokens,vocab,dim);
  return cudaGetLastError();
}

template<class T> __global__ void append_kv(T* dst,const T* src,int count,int n,int d,int capacity,int offset) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<count) {
    int bh=i/(n*d), pos=(i/d)%n, col=i%d;
    dst[(bh*capacity+offset+pos)*d+col]=src[i];
  }
}
template<class T> __global__ void last_row(T* dst,const T* src,int batch,int n,int width) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<batch*width) dst[i]=src[((i/width)*n+n-1)*width+i%width];
}
template<class T> __global__ void row_argmax(float* dst,const T* src,int width) {
  __shared__ float values[256];
  __shared__ int indices[256];
  int t=threadIdx.x, row=blockIdx.x;
  float best=-INFINITY; int index=INT_MAX;
  // First maximum wins, including ties. NaN wins over finite, like numpy.argmax.
  for(int j=t;j<width;j+=256) {
    float v=mt_read(src[row*width+j]);
    if(index==INT_MAX || (isnan(v) && !isnan(best)) ||
       (!isnan(best) && v>best)) {best=v;index=j;}
  }
  values[t]=best; indices[t]=index; __syncthreads();
  for(int stride=128;stride;stride/=2) {
    if(t<stride) {
      float a=values[t],b=values[t+stride];int ia=indices[t],ib=indices[t+stride];
      if(ib!=INT_MAX && (ia==INT_MAX || (isnan(b)&&!isnan(a)) ||
          (!isnan(a)&&b>a) || (((isnan(a)&&isnan(b)) || a==b) && ib<ia))) {
        values[t]=b;indices[t]=ib;
      }
    }
    __syncthreads();
  }
  if(t==0) dst[row]=float(indices[0]);
}

#define DISPATCH(TYPE, CALL) if(dtype==0) {CALL(float);} else if(dtype==1) {CALL(__half);} else if(dtype==2) {CALL(__nv_bfloat16);} else return cudaErrorInvalidValue
extern "C" int mt_cache_append(void* dst,const void* src,int bh,int n,int d,int capacity,int offset,int dtype) {
  if(bh<1||n<1||d<1||offset<0||int64_t(offset)+n>capacity||int64_t(bh)*capacity*d>INT32_MAX) return cudaErrorInvalidValue;
  int count=bh*n*d;
  #define APPEND(T) append_kv<T><<<(count+255)/256,256>>>((T*)dst,(const T*)src,count,n,d,capacity,offset)
  DISPATCH(T,APPEND);
  #undef APPEND
  return cudaGetLastError();
}
extern "C" int mt_last_row(void* dst,const void* src,int b,int n,int w,int dtype) {
  if(b<1||n<1||w<1||int64_t(b)*n*w>INT32_MAX) return cudaErrorInvalidValue;
  #define LAST(T) last_row<T><<<(b*w+255)/256,256>>>((T*)dst,(const T*)src,b,n,w)
  DISPATCH(T,LAST);
  #undef LAST
  return cudaGetLastError();
}
extern "C" int mt_row_argmax(float* dst,const void* src,int rows,int width,int dtype) {
  if(rows<1||width<1||width>16777216||int64_t(rows)*width>INT32_MAX) return cudaErrorInvalidValue;
  #define ARGMAX(T) row_argmax<T><<<rows,256>>>(dst,(const T*)src,width)
  DISPATCH(T,ARGMAX);
  #undef ARGMAX
  return cudaGetLastError();
}
