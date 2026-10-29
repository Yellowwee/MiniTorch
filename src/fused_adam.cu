// FP32 packed Adam, legacy MiniTorch epsilon/bias-correction convention.
#include <cuda_runtime.h>

struct AdamTile { int offset, count, parameter; };

__global__ void packed_adam(float *p, const float *g, float *m, float *v,
                            const AdamTile *tiles, const float *step_sizes,
                            float beta1, float beta2, float one_minus_beta1,
                            float one_minus_beta2, float eps) {
  const AdamTile tile = tiles[blockIdx.x];
  const float step = step_sizes[tile.parameter];
  if (step < 0 || threadIdx.x >= tile.count) return; // grad=None: skip state too
  const int i = tile.offset + threadIdx.x;
  const float grad = g[i];
  const float mi = beta1*m[i] + one_minus_beta1*grad;
  const float vi = beta2*v[i] + one_minus_beta2*(grad*grad);
  m[i] = mi;
  v[i] = vi;
  p[i] -= step*mi/(sqrtf(vi)+eps);
}

extern "C" int mt_packed_adam(float *p, const float *g, float *m, float *v,
                              const AdamTile *tiles, const float *step_sizes,
                              int tile_count, float beta1, float beta2,
                              float one_minus_beta1, float one_minus_beta2, float eps) {
  if (tile_count > 0)
    packed_adam<<<tile_count,256>>>(p,g,m,v,tiles,step_sizes,beta1,beta2,
                                  one_minus_beta1,one_minus_beta2,eps);
  return cudaGetLastError();
}
