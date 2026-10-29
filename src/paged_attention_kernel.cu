// Page-table addressing + online-softmax attention. No contiguous KV gather.
// Slab layout: [layers, 2, pages_per_slab, heads, page_size, dim].
#include "includes/flash_common.cuh"
#include "includes/typed_io.cuh"
#include <climits>

template<class T>
__device__ T* page_element(const uint64_t* slabs,const int* table,
 int batch,int head,int position,int col,int kind,int layer,int heads,int dim,
 int page_size,int pages_per_slab,int num_slabs,int table_columns) {
  int logical=position/page_size;
  if(logical<0 || logical>=table_columns) return nullptr;
  int physical=table[batch*table_columns+logical];
  if(physical<0 || int64_t(physical)>=int64_t(num_slabs)*pages_per_slab) return nullptr;
  uint64_t address=slabs[physical/pages_per_slab];
  if(!address) return nullptr;
  int slot=physical%pages_per_slab;
  int64_t i=(((int64_t(layer)*2+kind)*pages_per_slab+slot)*heads+head)*page_size*dim;
  i+=int64_t(position%page_size)*dim+col;
  return reinterpret_cast<T*>(address)+i;
}

template<class T>
__global__ void paged_append(const T* k,const T* v,const uint64_t* slabs,const int* table,
 int b,int h,int n,int d,int page_size,int pages_per_slab,int num_slabs,int columns,int layer,int offset,const int* offsets=nullptr) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=b*h*n*d) return;
  int bh=i/(n*d),pos=(i/d)%n,col=i%d;
  if(offsets) offset=offsets[bh/h];
  T* kp=page_element<T>(slabs,table,bh/h,bh%h,offset+pos,col,0,layer,h,d,page_size,pages_per_slab,num_slabs,columns);
  T* vp=page_element<T>(slabs,table,bh/h,bh%h,offset+pos,col,1,layer,h,d,page_size,pages_per_slab,num_slabs,columns);
  if(kp && vp) { *kp=k[i]; *vp=v[i]; }
}

template<class T>
__global__ void paged_forward(T* out,const T* q,const uint64_t* slabs,const int* table,
 int h,int nq,int nk,int dim,int page_size,int pages_per_slab,int num_slabs,int columns,int layer,int offset,const int* offsets=nullptr) {
  __shared__ float qs[FA_ROWS][FA_DIM];
  __shared__ float ks[FA_DIM*FA_STRIDE],vs[FA_DIM*FA_STRIDE];
  int lane=threadIdx.x%32,warp=threadIdx.x/32;
  int bh=blockIdx.y,qi=blockIdx.x*FA_ROWS+warp,qb=bh*nq*dim;
  // One sequence per batch row; every block agrees on this row's live length.
  if(offsets) { offset=offsets[bh/h]; nk=offset+1; }
  for(int d=lane;d<dim;d+=32) qs[warp][d]=qi<nq?mt_read(q[qb+qi*dim+d]):0;
  float m=-INFINITY,l=0,acc[4]={0,0,0,0};
  for(int start=0;start<nk;start+=FA_TILE) {
    for(int t=threadIdx.x;t<FA_TILE*dim;t+=blockDim.x) {
      int j=t/dim,d=t%dim,pos=start+j;
      float kval=0,vval=0;
      if(pos<nk) {
        const T* kp=page_element<T>(slabs,table,bh/h,bh%h,pos,d,0,layer,h,dim,page_size,pages_per_slab,num_slabs,columns);
        const T* vp=page_element<T>(slabs,table,bh/h,bh%h,pos,d,1,layer,h,dim,page_size,pages_per_slab,num_slabs,columns);
        kval=kp?mt_read(*kp):NAN; vval=vp?mt_read(*vp):NAN;
      }
      ks[d*FA_STRIDE+j]=kval;vs[d*FA_STRIDE+j]=vval;
    }
    __syncthreads();
    int kj=start+lane;
    bool valid=qi<nq && kj<nk && kj<=offset+qi;
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
      for(int j=0;j<FA_TILE;++j) {
        float weight=__shfl_sync(0xffffffff,p,j);
        if(d<dim) update+=weight*vs[d*FA_STRIDE+j];
      }
      acc[c]=alpha*acc[c]+update;
    }
    m=next_m;l=next_l;
    __syncthreads();
  }
  if(qi<nq) for(int c=0;c<4;++c) if(lane+32*c<dim)
    out[qb+qi*dim+lane+32*c]=l>0?acc[c]/l:(isnan(l)?NAN:0);
}

inline bool paged_shape(int b,int h,int nq,int nk,int d,int page_size,int pages_per_slab,
 int num_slabs,int columns,int layer,int layers,int offset) {
  return fa_shape(b,h,nq,nk,d) && int64_t(b)*h<=65535 &&
    (page_size==8 || page_size==16 || page_size==32) && pages_per_slab>0 && num_slabs>0 &&
    int64_t(pages_per_slab)*num_slabs<=INT32_MAX && columns>0 && int64_t(b)*columns<=INT32_MAX &&
    nk<=int64_t(columns)*page_size && layers>0 && layer>=0 && layer<layers &&
    offset>=0 && int64_t(offset)+nq==nk;
}

extern "C" int mt_paged_append(const void* k,const void* v,const uint64_t* slabs,const int* table,
 int b,int h,int n,int d,int page_size,int pages_per_slab,int num_slabs,int columns,
 int layer,int layers,int offset,int dtype) {
  if(int64_t(offset)+n>INT32_MAX || !paged_shape(b,h,n,offset+n,d,page_size,pages_per_slab,num_slabs,columns,layer,layers,offset))
    return cudaErrorInvalidValue;
  int blocks=(int64_t(b)*h*n*d+255)/256;
  #define APPEND(T) paged_append<T><<<blocks,256>>>((const T*)k,(const T*)v,slabs,table,b,h,n,d,page_size,pages_per_slab,num_slabs,columns,layer,offset)
  if(dtype==0) {APPEND(float);} else if(dtype==1) {APPEND(__half);} else if(dtype==2) {APPEND(__nv_bfloat16);} else return cudaErrorInvalidValue;
  #undef APPEND
  return cudaGetLastError();
}

extern "C" int mt_paged_attention(void* out,const void* q,const uint64_t* slabs,const int* table,
 int b,int h,int nq,int nk,int d,int page_size,int pages_per_slab,int num_slabs,int columns,
 int layer,int layers,int offset,int dtype) {
  if(!paged_shape(b,h,nq,nk,d,page_size,pages_per_slab,num_slabs,columns,layer,layers,offset)) return cudaErrorInvalidValue;
  dim3 grid((nq+3)/4,b*h);
  #define FORWARD(T) paged_forward<T><<<grid,128>>>((T*)out,(const T*)q,slabs,table,h,nq,nk,d,page_size,pages_per_slab,num_slabs,columns,layer,offset)
  if(dtype==0) {FORWARD(float);} else if(dtype==1) {FORWARD(__half);} else if(dtype==2) {FORWARD(__nv_bfloat16);} else return cudaErrorInvalidValue;
  #undef FORWARD
  return cudaGetLastError();
}

// Ragged decode: Q/K/V [batch,heads,1,dim], offsets[batch] are committed lengths.
// Python validates offsets and owns all referenced pages throughout the launch.
extern "C" int mt_paged_decode_batch(void* out,const void* q,const void* k,const void* v,
 const uint64_t* slabs,const int* table,const int* offsets,
 int b,int h,int d,int page_size,int pages_per_slab,int num_slabs,int columns,
 int layer,int layers,int dtype) {
  if(!out || !q || !k || !v || !slabs || !table || !offsets ||
      !paged_shape(b,h,1,1,d,page_size,pages_per_slab,num_slabs,columns,layer,layers,0) ||
      int64_t(columns)*page_size>INT32_MAX) return cudaErrorInvalidValue;
  int blocks=(int64_t(b)*h*d+255)/256;
  dim3 grid(1,b*h);
  #define DECODE(T) \
    paged_append<T><<<blocks,256>>>((const T*)k,(const T*)v,slabs,table,b,h,1,d,page_size,pages_per_slab,num_slabs,columns,layer,0,offsets); \
    {cudaError_t e=cudaGetLastError();if(e!=cudaSuccess) return e;} \
    paged_forward<T><<<grid,128>>>((T*)out,(const T*)q,slabs,table,h,1,1,d,page_size,pages_per_slab,num_slabs,columns,layer,0,offsets)
  if(dtype==0) {DECODE(float);} else if(dtype==1) {DECODE(__half);} else if(dtype==2) {DECODE(__nv_bfloat16);} else return cudaErrorInvalidValue;
  #undef DECODE
  return cudaGetLastError();
}
