#include "includes/flash_common.cuh"
#include "includes/typed_io.cuh"

template<class T>
__global__ void fa_delta(float *delta,const float *dy,const T *out,int rows,int d) {
  int lane=threadIdx.x%32,row=(blockIdx.x*blockDim.x+threadIdx.x)/32;
  float sum=0;
  if(row<rows) for(int c=lane;c<d;c+=32) sum+=dy[row*d+c]*mt_read(out[row*d+c]);
  sum=fa_sum(sum);
  if(row<rows && lane==0) delta[row]=sum;
}

// Each warp owns one dQ row; no atomic adds or quadratic scratch buffers.
template<class T>
__global__ void fa_dq(float *dq,const float *dy,const T *q,const T *k,const T *v,
 const float *lse,const float *delta,const float *qlen,const float *klen,
 int heads,int nq,int nk,int dim,bool causal) {
  __shared__ float qs[FA_ROWS][FA_DIM],dys[FA_ROWS][FA_DIM];
  __shared__ float ks[FA_DIM*FA_STRIDE],vs[FA_DIM*FA_STRIDE];
  int lane=threadIdx.x%32,warp=threadIdx.x/32,bh=blockIdx.y,batch=bh/heads;
  int qi=blockIdx.x*FA_ROWS+warp,qb=bh*nq*dim,kb=bh*nk*dim;
  for(int d=lane;d<dim;d+=32) {
    qs[warp][d]=qi<nq?mt_read(q[qb+qi*dim+d]):0;
    dys[warp][d]=qi<nq?dy[qb+qi*dim+d]:0;
  }
  float logz=qi<nq?lse[bh*nq+qi]:-INFINITY;
  float di=qi<nq?delta[bh*nq+qi]:0,acc[4]={0,0,0,0};
  for(int start=0;start<nk;start+=FA_TILE) {
    for(int t=threadIdx.x;t<FA_TILE*dim;t+=blockDim.x) {
      int j=t/dim,d=t%dim;
      ks[d*FA_STRIDE+j]=start+j<nk?mt_read(k[kb+(start+j)*dim+d]):0;
      vs[d*FA_STRIDE+j]=start+j<nk?mt_read(v[kb+(start+j)*dim+d]):0;
    }
    __syncthreads();
    float ds=0;
    if(isfinite(logz) && fa_valid(qi,start+lane,nq,nk,qlen[batch],klen[batch],causal)) {
      float score=0,dp=0;
      for(int d=0;d<dim;++d) {
        score+=qs[warp][d]*ks[d*FA_STRIDE+lane];
        dp+=dys[warp][d]*vs[d*FA_STRIDE+lane];
      }
      ds=expf(score*rsqrtf(float(dim))-logz)*(dp-di)*rsqrtf(float(dim));
    }
    for(int c=0;c<4;++c) for(int j=0;j<FA_TILE;++j) {
      float weight=__shfl_sync(0xffffffff,ds,j);
      if(lane+32*c<dim) acc[c]+=weight*ks[(lane+32*c)*FA_STRIDE+j];
    }
    __syncthreads();
  }
  if(qi<nq) for(int c=0;c<4;++c) if(lane+32*c<dim)
    dq[qb+qi*dim+lane+32*c]=isnan(logz)?NAN:acc[c];
}

// Each warp owns one K/V row and streams Q/dO tiles. Recomputing P separately
// from dQ is a deliberate deterministic first implementation, not FlashAttention-2.
template<class T>
__global__ void fa_dkv(float *dk,float *dv,const float *dy,const T *q,const T *k,const T *v,
 const float *lse,const float *delta,const float *qlen,const float *klen,
 int heads,int nq,int nk,int dim,bool causal) {
  __shared__ float ks[FA_ROWS][FA_DIM],vs[FA_ROWS][FA_DIM];
  __shared__ float qs[FA_DIM*FA_STRIDE],dys[FA_DIM*FA_STRIDE];
  int lane=threadIdx.x%32,warp=threadIdx.x/32,bh=blockIdx.y,batch=bh/heads;
  int kj=blockIdx.x*FA_ROWS+warp,qb=bh*nq*dim,kb=bh*nk*dim;
  for(int d=lane;d<dim;d+=32) {
    ks[warp][d]=kj<nk?mt_read(k[kb+kj*dim+d]):0;
    vs[warp][d]=kj<nk?mt_read(v[kb+kj*dim+d]):0;
  }
  float ak[4]={0,0,0,0},av[4]={0,0,0,0};
  bool bad=false;
  for(int start=0;start<nq;start+=FA_TILE) {
    for(int t=threadIdx.x;t<FA_TILE*dim;t+=blockDim.x) {
      int j=t/dim,d=t%dim;
      qs[d*FA_STRIDE+j]=start+j<nq?mt_read(q[qb+(start+j)*dim+d]):0;
      dys[d*FA_STRIDE+j]=start+j<nq?dy[qb+(start+j)*dim+d]:0;
    }
    __syncthreads();
    int qi=start+lane;
    float p=0,ds=0,logz=qi<nq?lse[bh*nq+qi]:-INFINITY;
    bad=__any_sync(0xffffffff,isnan(logz)) || bad;
    if(isfinite(logz) && fa_valid(qi,kj,nq,nk,qlen[batch],klen[batch],causal)) {
      float score=0,dp=0;
      for(int d=0;d<dim;++d) {
        score+=qs[d*FA_STRIDE+lane]*ks[warp][d];
        dp+=dys[d*FA_STRIDE+lane]*vs[warp][d];
      }
      p=expf(score*rsqrtf(float(dim))-logz);
      ds=p*(dp-delta[bh*nq+qi])*rsqrtf(float(dim));
    }
    for(int c=0;c<4;++c) for(int j=0;j<FA_TILE;++j) {
      float pw=__shfl_sync(0xffffffff,p,j),sw=__shfl_sync(0xffffffff,ds,j);
      int d=lane+32*c;
      if(d<dim) {
        ak[c]+=sw*qs[d*FA_STRIDE+j];
        av[c]+=pw*dys[d*FA_STRIDE+j];
      }
    }
    __syncthreads();
  }
  if(kj<nk) for(int c=0;c<4;++c) if(lane+32*c<dim) {
    dk[kb+kj*dim+lane+32*c]=bad?NAN:ak[c];
    dv[kb+kj*dim+lane+32*c]=bad?NAN:av[c];
  }
}

extern "C" int mt_flash_bw(float *dq,float *dk,float *dv,float *delta,const float *dy,
 const float *q,const float *k,const float *v,const float *out,const float *lse,
 const float *qlen,const float *klen,int b,int h,int nq,int nk,int d,bool causal) {
  if(!fa_shape(b,h,nq,nk,d) || int64_t(b)*h>65535) return cudaErrorInvalidValue;
  fa_delta<float><<<(b*h*nq+3)/4,128>>>(delta,dy,out,b*h*nq,d);
  cudaError_t err=cudaGetLastError(); if(err!=cudaSuccess) return err;
  fa_dq<float><<<dim3((nq+3)/4,b*h),128>>>(dq,dy,q,k,v,lse,delta,qlen,klen,h,nq,nk,d,causal);
  err=cudaGetLastError(); if(err!=cudaSuccess) return err;
  fa_dkv<float><<<dim3((nk+3)/4,b*h),128>>>(dk,dv,dy,q,k,v,lse,delta,qlen,klen,h,nq,nk,d,causal);
  return cudaGetLastError();
}

template<class T> int launch_typed_bw(float* dq,float* dk,float* dv,float* delta,const float* dy,
 const void* q,const void* k,const void* v,const void* out,const float* lse,
 const float* qlen,const float* klen,int b,int h,int nq,int nk,int d,bool causal) {
  fa_delta<T><<<(b*h*nq+3)/4,128>>>(delta,dy,(const T*)out,b*h*nq,d);
  cudaError_t e=cudaGetLastError(); if(e!=cudaSuccess) return e;
  fa_dq<T><<<dim3((nq+3)/4,b*h),128>>>(dq,dy,(const T*)q,(const T*)k,(const T*)v,lse,delta,qlen,klen,h,nq,nk,d,causal);
  e=cudaGetLastError(); if(e!=cudaSuccess) return e;
  fa_dkv<T><<<dim3((nk+3)/4,b*h),128>>>(dk,dv,dy,(const T*)q,(const T*)k,(const T*)v,lse,delta,qlen,klen,h,nq,nk,d,causal);
  return cudaGetLastError();
}
extern "C" int mt_flash_bw_typed(float* dq,float* dk,float* dv,float* delta,const float* dy,
 const void* q,const void* k,const void* v,const void* out,const float* lse,
 const float* qlen,const float* klen,int b,int h,int nq,int nk,int d,bool causal,int dtype) {
  if(!fa_shape(b,h,nq,nk,d) || int64_t(b)*h>65535) return cudaErrorInvalidValue;
  if(dtype==1) return launch_typed_bw<__half>(dq,dk,dv,delta,dy,q,k,v,out,lse,qlen,klen,b,h,nq,nk,d,causal);
  if(dtype==2) return launch_typed_bw<__nv_bfloat16>(dq,dk,dv,delta,dy,q,k,v,out,lse,qlen,klen,b,h,nq,nk,d,causal);
  return cudaErrorInvalidValue;
}
