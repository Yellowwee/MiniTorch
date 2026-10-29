#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p minitorch/cuda_kernels
NVCC="${NVCC:-nvcc}"
for source in combine softmax_kernel layernorm_kernel cuda_runtime flash_attention_kernel flash_attention_backward fused_adam mixed_precision amp_kernels inference_kernels paged_attention_kernel; do
    extra=()
    if [[ "$source" == cuda_runtime || "$source" == mixed_precision ]]; then extra=(-lcublas); fi
    if [[ "$source" == fused_adam ]]; then extra=(--fmad=false); fi
    "$NVCC" -std=c++17 -O3 --shared "src/$source.cu" -Xcompiler -fPIC \
        "${extra[@]}" -o "minitorch/cuda_kernels/$source.so"
done
