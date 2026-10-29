#include "includes/block_reduce.h"
#include "includes/kernels.h"
#include "includes/cuda_util.h"

#include <cooperative_groups.h>
#include <cstddef>

namespace cg = cooperative_groups;
namespace lightseq {
namespace cuda {

const float LN_EPSILON = 1e-5f;
#define TILE_DIM 32


/**
@brief: ker_layer_norm
Standard layer normalization.
It will not only output the layer norm result,
  but also outputs variance.
  may also output means, depends on whether
  the means argument is nullptr

@thread
gridDim.x = batch_size * seq_len
blockDim.x = hidden_size

@param
ln_res: [batch_size * seq_len, hidden_size], ln result.
vars: [batch_size * seq_len], variance per token
means: [batch_size * seq_len], means per token, can be nullptr
inp: [batch_size * seq_len, hidden_size], ln input.
scale: [hidden_size], ln scale
bias: [hidden_size], ln bias
*/
template <typename T>
__global__ void ker_layer_norm(T *ln_res, T *vars, T *means, const T *inp,
                               const T *scale, const T *bias, int hidden_size) {
  
  /// BEGIN ASSIGN4_2_1
  /// TODO
  // Hints:
  // 1. Compute x and x^2 with reinterpret_cast by casting to float4 for speedup
  // 2. Compute reduce sum with blockReduce and add epsilon with LN_EPSILON
  // 3. Compute layernorm result with reinterpret_cast by casting to float4 for speedup
  
  // Step 1
  float l_sum = 0;
  const float4 *inp_f4 = reinterpret_cast<const float4 *>(inp) + blockIdx.x * hidden_size;  
  // Step 2
  float l_sum_sq = 0;
  const float4 *scale_f4 = reinterpret_cast<const float4 *>(scale);
  const float4 *bias_f4 = reinterpret_cast<const float4 *>(bias);

  float4 *out_f4 = reinterpret_cast<float4 *>(ln_res) + blockIdx.x * hidden_size;
  for(uint idx = threadIdx.x; idx < hidden_size; idx += blockDim.x)
  {
    float4 val = inp_f4[idx];
    l_sum_sq += val.x * val.x + val.y * val.y + val.z * val.z + val.w * val.w;
    l_sum += val.x + val.y + val.z + val.w;
  }

  float stats[2] = {l_sum, l_sum_sq};
  blockReduce<ReduceType::kSum, 2>(stats);

  __shared__ float shared_mean;
  __shared__ float shared_inv_std;

  if(threadIdx.x == 0)
  {
    const float mean = stats[0] / (4.0f *hidden_size);
    float var = stats[1] / (4.0f * hidden_size) - mean * mean;

    var = fmaxf(var, 0.0f);
    var += LN_EPSILON;
    shared_mean = mean;
    shared_inv_std = rsqrtf(var);

    vars[blockIdx.x] = var;
    if(means != nullptr)  means[blockIdx.x] = mean;
  }

  __syncthreads();
  // Step 3
  for(uint idx=threadIdx.x;idx < hidden_size;idx += blockDim.x)
  {
    const float4 value = inp_f4[idx];
    const float4 gamma = scale_f4[idx];
    const float4 beta = bias_f4[idx];

    float4 result;
    result.x = gamma.x * ((value.x - shared_mean) * shared_inv_std) + beta.x;
    result.y = gamma.y * ((value.y - shared_mean) * shared_inv_std) + beta.y;
    result.z = gamma.z * ((value.z - shared_mean) * shared_inv_std) + beta.z;
    result.w = gamma.w * ((value.w - shared_mean) * shared_inv_std) + beta.w;
    out_f4[idx] = result;
  }
  /// END ASSIGN4_2_1
}

extern "C" {
void launch_layernorm(float *ln_res, float *vars, float *means,
                              const float *inp, const float *scale,
                              const float *bias, int batch_size, int hidden_dim,
                              cudaStream_t stream) {
  if (hidden_dim % 4 != 0) {
    throw std::runtime_error("violate hidden_dim % 4 = 0");
  }
  int float_size = sizeof(float);
  int input_size = batch_size * hidden_dim * float_size;
  int scale_size = hidden_dim * float_size;
  int bias_size = hidden_dim * float_size;
  int output_size = batch_size * hidden_dim * float_size;
  int mean_size = batch_size * float_size;
  int var_size = batch_size * float_size;


  float *d_ln_res, *d_vars, *d_means, *d_inp, *d_scale, *d_bias;
  cudaMalloc((void **)&d_ln_res, output_size);
  cudaMalloc((void **)&d_vars, var_size);
  cudaMalloc((void **)&d_means, mean_size);
  cudaMalloc((void **)&d_inp, input_size);
  cudaMalloc((void **)&d_scale, scale_size);
  cudaMalloc((void **)&d_bias, bias_size);

  cudaMemcpy(d_inp, inp, input_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_scale, scale, scale_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_bias, bias, bias_size, cudaMemcpyHostToDevice);

  // For using float4
  hidden_dim >>= 2;
  int nthread = min(((hidden_dim + 31) / 32) * 32, MAX_THREADS);
  dim3 grid_dim(batch_size);
  dim3 block_dim(nthread);

  ker_layer_norm<float><<<grid_dim, block_dim, 0, stream>>>(
      d_ln_res, d_vars, d_means, d_inp, d_scale, d_bias, hidden_dim);

  // Copy back to the host
  cudaMemcpy(ln_res, d_ln_res, output_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(vars, d_vars, var_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(means, d_means, mean_size, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  // Check CUDA execution
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "launch_layernorm Error: %s\n", cudaGetErrorString(err));
    // Handle the error (e.g., by exiting the program)
    exit(EXIT_FAILURE);
  }

  // Free memory on device
  cudaFree(d_ln_res);
  cudaFree(d_vars);
  cudaFree(d_means);
  cudaFree(d_inp);
  cudaFree(d_scale);
  cudaFree(d_bias);

}
}

/**
@brief: ker_ln_bw_dgamma_dbetta
Layer norm backward kernel, compute the gradient of gamma and betta.
dbetta = sum(dout, dim=0)
dgamma = sum(xhat * dout, dim=0)
xhat = (input - mean) * rsqrt(var) or
  (output - betta) / gamma

@thread
gridDim.x = hidden_size / 32
blockDim.x = 32
blockDim.y = 32

@param
gamma_grad: [hidden_size], gradient of gamma
betta_grad: [hidden_size], gradient of betta
out_grad: [batch_size * seq_len, hidden_size], gradient of betta ln output
inp_or_out: [batch_size * seq_len, hidden_size], ln output if means is nullptr
  ln input if means is not nullptr
gamma: [hidden_size], gamma of ln,
  used to compute xhat, maybe nullptr
betta: [hidden_size], betta of ln,
  used to compute xhat, maybe nullptr
vars: [batch_size * seq_len], variance of ln forward,
  used to compute xhat, maybe nullptr
means: [batch_size * seq_len], mean of ln forward,
  used to compute xhat, maybe nullptr
(gamma && betta) ^ (vars && means) should be true
*/
template <typename T>
__global__ void ker_ln_bw_dgamma_dbetta(T *gamma_grad, T *betta_grad,
                                        const T *out_grad,
                                        const T *inp, const T *gamma,
                                        const T *betta, const T *vars,
                                        const T *means, int rows, int width) {

  /// BEGIN ASSIGN4_2_2
  /// TODO
  // Hints:
  // 1. Compute the partial gradients by looping across inp rows
  // 2. Store the partial gradients in the shared memory arrays
  // 3. Compute the reduce sum of the shared memory arrays with g.shfl_down
  //      -> More hints about `g.shfl_down`:
  //      -> https://developer.nvidia.com/blog/cooperative-groups/#:~:text=Using%20thread_block_tile%3A%3Ashfl_down()%20to%20simplify%20our%20warp%2Dlevel%20reduction%20does%20benefit%20our%20code%3A%20it%20simplifies%20it%20and%20eliminates%20the%20need%20for%20shared%20memory
  //      -> The highlighted line gives you a conceptual understanding of what the g.shfl_down is doing. Usually, the threads inside a block need to load everything to shared memory and work together to reduce the result (like what you have implemented in the hw1 for reduce function). 
  //      -> Now g.shfl_down helps you do so without consuming any shared memory. g.shfl_down makes it more efficient.
  // 4. Assign the final result to the correct position in the global output

  __shared__ float betta_buffer[TILE_DIM][TILE_DIM];
  __shared__ float gamma_buffer[TILE_DIM][TILE_DIM];

  cg::thread_block b = cg::this_thread_block();
  cg::thread_block_tile<TILE_DIM> g = cg::tiled_partition<TILE_DIM>(b);

  // Step 1
  int col = blockIdx.x * TILE_DIM + threadIdx.y;
  float dbeta_sum = 0.f, dgamma_sum = 0.f;

  if(col < width)
  {
    for(int row=threadIdx.x;row<rows;row+=TILE_DIM)
    {
      int offset = row * width + col;
      float dout = (float)out_grad[offset];

      float x_hat;
      if(means!=nullptr)
      {
        x_hat = (static_cast<float>(inp[offset]) - static_cast<float>(means[row])) * rsqrt(static_cast<float>(vars[row]));
      }
      else
      {
        x_hat = (static_cast<float>(inp[offset]) - static_cast<float>(betta[col])) / static_cast<float>(gamma[col]);
      }
      dbeta_sum += dout;
      dgamma_sum += x_hat * dout;
    }
  }

  for(int offset=TILE_DIM / 2;offset>0;offset /= 2)
  {
    dbeta_sum += g.shfl_down(dbeta_sum, offset);
    dgamma_sum += g.shfl_down(dgamma_sum, offset);
  }

  if(threadIdx.x == 0 && col < width)
  {
    betta_grad[col] = static_cast<T>(dbeta_sum);
    gamma_grad[col] = static_cast<T>(dgamma_sum);
  }
  /// END ASSIGN4_2_2
}

/**
@brief: ker_ln_bw_dinp
Layer norm backward kernel, compute the gradient of input.
dinp = (dxhat - (sum(dxhat) + xhat * sum(dxhat * xhat)) / hidden_dim)
  * rsqrt(var)
xhat = (input - mean) * rsqrt(var) if mean is not nullptr
       (output - betta) / gamma if mean is nullptr
dxhat = dout * gamma


@thread
gridDim.x = batch_size * seq_len
blockDim.x = hidden_size

@param
inp_grad: [batch_size * seq_len, hidden_size], gradient of betta ln output
out_grad: [batch_size * seq_len, hidden_size], gradient of betta ln output
residual_grad: [batch_size * seq_len, hidden_size], gradient of residual input,
  usually appear in pre-layer-norm for transformer layer, maybe nullptr
inp_or_out: [batch_size * seq_len, hidden_size], ln output if means is nullptr
  ln input if means is not nullptr
gamma: [hidden_size], gamma of ln,
  used to compute xhat and dxhat
betta: [hidden_size], betta of ln,
  used to compute xhat, maybe nullptr
vars: [batch_size * seq_len], variance of ln forward,
  used to compute xhat and dinp
means: [batch_size * seq_len], mean of ln forward,
  used to compute xhat, maybe nullptr
*/
template <typename T>
__global__ void ker_ln_bw_dinp(T *inp_grad, const T *out_grad, const T *inp,
                               const T *gamma, const T *betta, const T *vars,
                               const T *means, int hidden_dim) {
  
  /// BEGIN ASSIGN4_2_2
  /// TODO
  // Hints:
  // 1. Compute dxhat=dy*w with reinterpret_cast by casting to float4 for speedup
  // 2. Compute xhat with reinterpret_cast by casting to float4 for speedup
  // 3. Compute reduce sum for dxhat and dxhat*xhat with blockReduce
  // 4. Compute final gradient
  
  // Step 1
  int row = blockIdx.x;
  int col = threadIdx.x;

  const float4* out_grad_f4 = reinterpret_cast<const float4*>(out_grad);
  const float4* inp_f4 = reinterpret_cast<const float4*>(inp);
  const float4* gamma_f4 = reinterpret_cast<const float4*>(gamma);
  const float4* betta_f4 = reinterpret_cast<const float4*>(betta);
  float4* inp_grad_f4 = reinterpret_cast<float4*>(inp_grad);

  float4 dx_hat = {0.f, 0.f, 0.f, 0.f};
  float4 x_hat = {0.f, 0.f, 0.f, 0.f};

  float inv_std = rsqrtf(vars[row]);
  // Step 2
  if(col < hidden_dim)
  {
    float4 dy = out_grad_f4[row * hidden_dim + col];
    float4 w_g = gamma_f4[col];

    dx_hat.x = dy.x * w_g.x;
    dx_hat.y = dy.y * w_g.y;
    dx_hat.z = dy.z * w_g.z;
    dx_hat.w = dy.w * w_g.w;

    float4 value = inp_f4[row * hidden_dim + col];
    if(means != nullptr)
    {
      float mean = means[row];
      x_hat.x = (value.x - mean) * inv_std;
      x_hat.y = (value.y - mean) * inv_std;
      x_hat.z = (value.z - mean) * inv_std;
      x_hat.w = (value.w - mean) * inv_std;
    }
    else
    {
      float4 beta = betta_f4[col];
      x_hat.x = (value.x - beta.x) / w_g.x;
      x_hat.y = (value.y - beta.y) / w_g.y;
      x_hat.z = (value.z - beta.z) / w_g.z;
      x_hat.w = (value.w - beta.w) / w_g.w;
    }
  }
  // Step 3
  float stats[2];
  stats[0] = dx_hat.x + dx_hat.y + dx_hat.z + dx_hat.w;
  stats[1] = dx_hat.x * x_hat.x + dx_hat.y * x_hat.y + dx_hat.z * x_hat.z + dx_hat.w * x_hat.w;
  blockReduce<ReduceType::kSum, 2>(stats);

  __shared__ float shared_sum_dx_hat;
  __shared__ float shared_sum_dx_hat_x_hat;

  if(threadIdx.x == 0)
  {
    shared_sum_dx_hat = stats[0];
    shared_sum_dx_hat_x_hat = stats[1];
  }
  __syncthreads();
  // Step 4
  if(col < hidden_dim)
  {
    float width = static_cast<float>(hidden_dim * 4);
    float4 dinp;

    dinp.x = (dx_hat.x - (shared_sum_dx_hat + x_hat.x * shared_sum_dx_hat_x_hat) / width) * inv_std;
    dinp.y = (dx_hat.y - (shared_sum_dx_hat + x_hat.y * shared_sum_dx_hat_x_hat) / width) * inv_std;
    dinp.z = (dx_hat.z - (shared_sum_dx_hat + x_hat.z * shared_sum_dx_hat_x_hat) / width) * inv_std;
    dinp.w = (dx_hat.w - (shared_sum_dx_hat + x_hat.w * shared_sum_dx_hat_x_hat) / width) * inv_std;

    inp_grad_f4[row * hidden_dim + col] = dinp;
  }
  /// END ASSIGN4_2_2
}
extern "C" {
void launch_layernorm_bw(float *gamma_grad, float *betta_grad, float *inp_grad,
                         const float *out_grad, const float *inp, const float *gamma,
                         const float *betta, const float *vars,
                         const float *means, int batch_size, int hidden_dim,
                         cudaStream_t stream_1, cudaStream_t stream_2) {
  
  // Allocate device memory
  float *d_gamma_grad, *d_betta_grad, *d_inp_grad, *d_out_grad, *d_inp, *d_gamma, *d_betta, *d_vars, *d_means;
  int grad_output_size = batch_size * hidden_dim * sizeof(float);
  int gamma_betta_size = hidden_dim * sizeof(float);
  int vars_means_size = batch_size * sizeof(float);

  cudaMalloc((void **)&d_gamma_grad, gamma_betta_size);
  cudaMalloc((void **)&d_betta_grad, gamma_betta_size);
  cudaMalloc((void **)&d_inp_grad, grad_output_size);
  cudaMalloc((void **)&d_out_grad, grad_output_size);
  cudaMalloc((void **)&d_inp, grad_output_size);
  cudaMalloc((void **)&d_gamma, gamma_betta_size);
  cudaMalloc((void **)&d_betta, gamma_betta_size);
  cudaMalloc((void **)&d_vars, vars_means_size);
  cudaMalloc((void **)&d_means, vars_means_size);

  // Copy memory to device
  cudaMemcpy((void *)d_out_grad, out_grad, grad_output_size, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_inp, inp, grad_output_size, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_gamma, gamma, gamma_betta_size, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_betta, betta, gamma_betta_size, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_vars, vars, vars_means_size, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_means, means, vars_means_size, cudaMemcpyHostToDevice);

  // Launch kernels
  // Compute grad of gamma and betta
  // This calculates the number of blocks needed to cover the data along the specified dimension, rounds it up.
  dim3 grid_dim((hidden_dim + TILE_DIM - 1) / TILE_DIM);
  dim3 block_dim(TILE_DIM, TILE_DIM);
  ker_ln_bw_dgamma_dbetta<float><<<grid_dim, block_dim, 0, stream_1>>>(
      d_gamma_grad, d_betta_grad, d_out_grad, d_inp, d_gamma, d_betta, d_vars,
      d_means, batch_size, hidden_dim);

  // Compute grad of input
  if (hidden_dim % 4 != 0 || hidden_dim > 4096) {
    throw std::runtime_error("hidden_dim % 4 != 0 || hidden_dim > 4096");
  }
  hidden_dim >>= 2;
  int nthread = min(((hidden_dim + 31) / 32) * 32, MAX_THREADS);
  ker_ln_bw_dinp<<<batch_size, nthread, 0, stream_2>>>(
      d_inp_grad, d_out_grad, d_inp, d_gamma, d_betta, d_vars, d_means, hidden_dim);

  // Synchronize and check for errors
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "launch_layernorm_bw Error: %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Copy back to host
  cudaMemcpy(gamma_grad, d_gamma_grad, gamma_betta_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(betta_grad, d_betta_grad, gamma_betta_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(inp_grad, d_inp_grad, grad_output_size, cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(d_gamma_grad);
  cudaFree(d_betta_grad);
  cudaFree(d_inp_grad);
  cudaFree((void *)d_out_grad);
  cudaFree((void *)d_inp);
  cudaFree((void *)d_gamma);
  cudaFree((void *)d_betta);
  cudaFree((void *)d_vars);
  cudaFree((void *)d_means);
}}
}}

namespace lightseq { namespace cuda {
extern "C" int mt_layernorm_fw(float *out, float *vars, float *means,
 const float *inp, const float *gamma, const float *beta, int rows, int width) {
  if(rows<=0 || width<=0 || width%4 || width>4096) return cudaErrorInvalidValue;
  int vectors=width/4, threads=((vectors+31)/32)*32;
  ker_layer_norm<float><<<rows,threads>>>(out,vars,means,inp,gamma,beta,vectors);
  return cudaGetLastError();
}
extern "C" int mt_layernorm_bw(float *dg, float *db, float *dx,
 const float *dy, const float *x, const float *g, const float *b,
 const float *vars, const float *means, int rows, int width) {
  if(rows<=0 || width<=0 || width%4 || width>4096) return cudaErrorInvalidValue;
  ker_ln_bw_dgamma_dbetta<float><<<(width+31)/32,dim3(32,32)>>>(dg,db,dy,x,g,b,vars,means,rows,width);
  cudaError_t status = cudaGetLastError();
  if(status!=cudaSuccess) return status;
  int vectors=width/4, threads=((vectors+31)/32)*32;
  ker_ln_bw_dinp<float><<<rows,threads>>>(dx,dy,x,g,b,vars,means,vectors);
  return cudaGetLastError();
}
}}

#include "includes/device_benchmark.cuh"
namespace lightseq { namespace cuda {
extern "C" int mt_bench_layernorm_fw(float *seconds, int repeats, float *out, float *var, float *mean, const float *x, const float *g, const float *b, int rows, int width) {
 return mt_benchmark(seconds,repeats,[&]() { return mt_layernorm_fw(out,var,mean,x,g,b,rows,width); });
}
extern "C" int mt_bench_layernorm_bw(float *seconds, int repeats, float *dg, float *db, float *dx, const float *dy, const float *x, const float *g, const float *b, const float *var, const float *mean, int rows, int width) {
 return mt_benchmark(seconds,repeats,[&]() { return mt_layernorm_bw(dg,db,dx,dy,x,g,b,var,mean,rows,width); });
}
}}
