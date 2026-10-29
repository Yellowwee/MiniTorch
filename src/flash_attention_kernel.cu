// Algorithm: https://arxiv.org/abs/2205.14135 (online softmax, tiled exact attention).
#include "includes/flash_common.cuh"
#include "includes/typed_io.cuh"

template<class T>
__global__ void fa_forward(T *out,float *lse,const T *q,const T *k,
 const T *v,const float *qlen,const float *klen,int heads,int nq,int nk,int dim,bool causal,
 int capacity,int query_offset) {
  __shared__ float qs[FA_ROWS][FA_DIM];
  __shared__ float ks[FA_DIM*FA_STRIDE],vs[FA_DIM*FA_STRIDE];
  int lane=threadIdx.x%32, warp=threadIdx.x/32;
  int bh=blockIdx.y, batch=bh/heads, qi=blockIdx.x*FA_ROWS+warp;
  int qb=bh*nq*dim,kb=bh*capacity*dim;
  for(int d=lane;d<dim;d+=32) qs[warp][d]=qi<nq?mt_read(q[qb+qi*dim+d]):0;
  float m=-INFINITY,l=0,acc[4]={0,0,0,0};
  for(int start=0;start<nk;start+=FA_TILE) {
    for(int t=threadIdx.x;t<FA_TILE*dim;t+=blockDim.x) {
      int j=t/dim,d=t%dim;
      ks[d*FA_STRIDE+j]=start+j<nk?mt_read(k[kb+(start+j)*dim+d]):0;
      vs[d*FA_STRIDE+j]=start+j<nk?mt_read(v[kb+(start+j)*dim+d]):0;
    }
    __syncthreads();
    int kj=start+lane;
    bool valid=qi<nq && kj<nk && (!qlen || qi<qlen[batch]) &&
      (!klen || kj<klen[batch]) && (!causal || kj<=query_offset+qi);
    float s=-INFINITY;
    if(valid) {
      s=0;
      for(int d=0;d<dim;++d) s+=qs[warp][d]*ks[d*FA_STRIDE+lane];
      s*=rsqrtf(float(dim));
    }
    float next_m=fmaxf(m,fa_max(s));
    float alpha=isfinite(m)?expf(m-next_m):0;
    float p=valid?expf(s-next_m):0;
    float next_l=alpha*l+fa_sum(p);
    #pragma unroll
    for(int c=0;c<4;++c) {
      int d=lane+32*c;
      float update=0;
      // All lanes participate, including lanes whose output column is padding.
      for(int j=0;j<FA_TILE;++j) {
        float weight=__shfl_sync(0xffffffff,p,j);
        if(d<dim) update+=weight*vs[d*FA_STRIDE+j];
      }
      acc[c]=alpha*acc[c]+update;
    }
    m=next_m; l=next_l;
    __syncthreads();
  }
  if(qi<nq) {
    for(int c=0;c<4;++c) if(lane+32*c<dim)
      out[qb+qi*dim+lane+32*c]=l>0?acc[c]/l:(isnan(l)?NAN:0);
    if(lane==0 && lse) lse[bh*nq+qi]=l>0?m+logf(l):(isnan(l)?NAN:-INFINITY);
  }
}

extern "C" int mt_flash_fw(float *out,float *lse,const float *q,const float *k,
 const float *v,const float *qlen,const float *klen,int b,int h,int nq,int nk,int d,bool causal) {
  if(!fa_shape(b,h,nq,nk,d) || int64_t(b)*h>65535) return cudaErrorInvalidValue;
  fa_forward<float><<<dim3((nq+3)/4,b*h),128>>>(out,lse,q,k,v,qlen,klen,h,nq,nk,d,causal,nk,0);
  return cudaGetLastError();
}

extern "C" int mt_flash_fw_typed(void* out,float* lse,const void* q,const void* k,
 const void* v,const float* qlen,const float* klen,int b,int h,int nq,int nk,int d,bool causal,int dtype) {
  if(!fa_shape(b,h,nq,nk,d) || int64_t(b)*h>65535) return cudaErrorInvalidValue;
  dim3 grid((nq+3)/4,b*h);
  if(dtype==1) fa_forward<__half><<<grid,128>>>((__half*)out,lse,(const __half*)q,(const __half*)k,(const __half*)v,qlen,klen,h,nq,nk,d,causal,nk,0);
  else if(dtype==2) fa_forward<__nv_bfloat16><<<grid,128>>>((__nv_bfloat16*)out,lse,(const __nv_bfloat16*)q,(const __nv_bfloat16*)k,(const __nv_bfloat16*)v,qlen,klen,h,nq,nk,d,causal,nk,0);
  else return cudaErrorInvalidValue;
  return cudaGetLastError();
}

// Inference reads the live prefix directly from capacity-strided K/V buffers.
// No LSE, backward state, full attention matrix or history compaction is needed.
extern "C" int mt_flash_inference(void* out,const void* q,const void* k,const void* v,
 int b,int h,int nq,int nk,int d,int capacity,int offset,int dtype) {
  if(!fa_shape(b,h,nq,nk,d) || capacity<nk || offset<0 || offset+nq!=nk ||
      int64_t(b)*h*capacity*d>INT32_MAX || int64_t(b)*h>65535) return cudaErrorInvalidValue;
  dim3 grid((nq+3)/4,b*h);
  if(dtype==0) fa_forward<float><<<grid,128>>>((float*)out,nullptr,(const float*)q,(const float*)k,(const float*)v,nullptr,nullptr,h,nq,nk,d,true,capacity,offset);
  else if(dtype==1) fa_forward<__half><<<grid,128>>>((__half*)out,nullptr,(const __half*)q,(const __half*)k,(const __half*)v,nullptr,nullptr,h,nq,nk,d,true,capacity,offset);
  else if(dtype==2) fa_forward<__nv_bfloat16><<<grid,128>>>((__nv_bfloat16*)out,nullptr,(const __nv_bfloat16*)q,(const __nv_bfloat16*)k,(const __nv_bfloat16*)v,nullptr,nullptr,h,nq,nk,d,true,capacity,offset);
  else return cudaErrorInvalidValue;
  return cudaGetLastError();
}
