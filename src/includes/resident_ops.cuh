// Device-pointer ABI alongside the original host-pointer launchers in combine.cu.
// Metadata is passed by value as kernel arguments; no per-op metadata allocation.
struct MTLayout { int dims; int shape[32]; int strides[32]; };
__device__ int mt_position(int ordinal, MTLayout out, MTLayout in) {
  int pos = 0;
  for (int d = out.dims - 1; d >= 0; --d) {
    int index = ordinal % out.shape[d];
    ordinal /= out.shape[d];
    int j = d - (out.dims - in.dims);
    if (j >= 0 && in.shape[j] != 1) pos += index * in.strides[j];
  }
  return pos;
}
__global__ void mt_map_kernel(float *out, MTLayout o, const float *a, MTLayout x, int n, int op) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < n) out[mt_position(i,o,o)] = fn(op,a[mt_position(i,o,x)]);
}
__global__ void mt_zip_kernel(float *out, MTLayout o, const float *a, MTLayout x,
                             const float *b, MTLayout y, int n, int op) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < n) out[mt_position(i,o,o)] = fn(op,a[mt_position(i,o,x)],b[mt_position(i,o,y)]);
}
__global__ void mt_reduce_kernel(float *out, MTLayout o, const float *a, MTLayout x,
                                int dim, float start, int op) {
  // One block per output; supported reductions are associative add/mul/max.
  __shared__ float values[256];
  int pos = mt_position(blockIdx.x,o,x);
  float value = start;
  for(int j=threadIdx.x; j<x.shape[dim]; j+=blockDim.x)
    value = fn(op,value,a[pos+j*x.strides[dim]]);
  values[threadIdx.x] = value;
  __syncthreads();
  for(int step=128;step;step>>=1) {
    if(threadIdx.x<step) values[threadIdx.x] = fn(op,values[threadIdx.x],values[threadIdx.x+step]);
    __syncthreads();
  }
  if(threadIdx.x==0) out[mt_position(blockIdx.x,o,o)] = values[0];
}
extern "C" {
int mt_map(float *out, MTLayout o, const float *a, MTLayout x, int n, int op) {
  if(n) mt_map_kernel<<<(n+255)/256,256>>>(out,o,a,x,n,op);
  return cudaGetLastError();
}
int mt_zip(float *out, MTLayout o, const float *a, MTLayout x, const float *b, MTLayout y, int n, int op) {
  if(n) mt_zip_kernel<<<(n+255)/256,256>>>(out,o,a,x,b,y,n,op);
  return cudaGetLastError();
}
int mt_reduce(float *out, MTLayout o, const float *a, MTLayout x, int n, int dim, float start, int op) {
  if(n) mt_reduce_kernel<<<n,256>>>(out,o,a,x,dim,start,op);
  return cudaGetLastError();
}
}
