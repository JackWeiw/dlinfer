#include "hip/hip_runtime.h"
// 2024 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#include <torch/all.h>
#include <ATen/cuda/HIPContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "cuda_compat.h"
#include "dispatch_utils.h"

#ifdef USE_ROCM
  #include "quantization/fp8/amd/quant_utils.cuh"
#else
  #include "quantization/fp8/nvidia/quant_utils.cuh"
#endif

#include <algorithm>
#include <cassert>
#include <map>
#include <vector>

#ifdef USE_ROCM
  #include <hip/hip_bf16.h>
  #include "hip/hip_runtime.h"
  #include <hip/hip_runtime.h>
typedef __hip_bfloat16 __nv_bfloat16;
#endif

namespace vllm {

template <typename scalar_t, typename cache_t, Fp8KVCacheDataType kv_dt>
__global__ void reshape_and_cache_kernel(
    const scalar_t* __restrict__ key,    // [num_tokens, num_heads, head_size]
    const scalar_t* __restrict__ value,  // [num_tokens, num_heads, head_size]
    cache_t* __restrict__ key_cache,     // [num_blocks, num_heads, head_size/x,
                                         // block_size, x]
    cache_t* __restrict__ value_cache,   // [num_blocks, num_heads, head_size,
                                         // block_size]
    const int64_t* __restrict__ slot_mapping,  // [num_tokens]
    const int key_stride, const int value_stride, const int num_heads,
    const int head_size, const int block_size, const int x, const float k_scale,
    const float v_scale) {
  const int64_t token_idx = blockIdx.x;
  const int64_t slot_idx = slot_mapping[token_idx];
  if (slot_idx < 0) {
    // Padding token that should be ignored.
    return;
  }

  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;

  const int n = num_heads * head_size;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const int64_t src_key_idx = token_idx * key_stride + i;
    const int64_t src_value_idx = token_idx * value_stride + i;

    const int head_idx = i / head_size;
    const int head_offset = i % head_size;
    const int x_idx = head_offset / x;
    const int x_offset = head_offset % x;

    const int64_t tgt_key_idx =
        block_idx * num_heads * (head_size / x) * block_size * x +
        head_idx * (head_size / x) * block_size * x + x_idx * block_size * x +
        block_offset * x + x_offset;
    const int64_t tgt_value_idx =
        block_idx * num_heads * head_size * block_size +
        head_idx * head_size * block_size + head_offset * block_size +
        block_offset;
    scalar_t tgt_key = key[src_key_idx];
    scalar_t tgt_value = value[src_value_idx];
    if constexpr (kv_dt == Fp8KVCacheDataType::kAuto) {
      key_cache[tgt_key_idx] = tgt_key;
      value_cache[tgt_value_idx] = tgt_value;
    } else {
      key_cache[tgt_key_idx] =
          fp8::scaled_convert<cache_t, scalar_t, kv_dt>(tgt_key, k_scale);
      value_cache[tgt_value_idx] =
          fp8::scaled_convert<cache_t, scalar_t, kv_dt>(tgt_value, v_scale);
    }
  }
}

template <typename scalar_t, typename cache_t, Fp8KVCacheDataType kv_dt>
__global__ void reshape_and_cache_kernel_layout(
    const scalar_t* __restrict__ key,    // [num_tokens, num_heads, head_size]
    const scalar_t* __restrict__ value,  // [num_tokens, num_heads, head_size]
    cache_t* __restrict__ key_cache,     // [num_blocks, num_heads, head_size/x,
                                         // block_size, x]
    cache_t* __restrict__ value_cache,   // [num_blocks, num_heads, head_size,
                                         // block_size]
    const int64_t* __restrict__ slot_mapping,  // [num_tokens]
    const int key_stride, const int value_stride, const int num_heads,
    const int head_size, const int block_size, const int x,
    const float kv_scale, bool is_deepseek) {
  const int64_t token_idx = blockIdx.x;
  const int64_t slot_idx = slot_mapping[token_idx];
  if (slot_idx < 0) {
    // Padding token that should be ignored.
    return;
  }

  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;

  const int n = num_heads * head_size;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const int64_t src_key_idx = token_idx * key_stride + i;
    const int64_t src_value_idx = token_idx * value_stride + i;

    const int head_idx = i / head_size;
    const int head_offset = i % head_size;

    int64_t tgt_key_idx = 0;
    if (is_deepseek) {
      const int x_idx = head_offset / x;
      const int x_offset = head_offset % x;
      tgt_key_idx =
        block_idx * num_heads * head_size * block_size +
        head_idx * head_size * block_size + x_idx * block_size * x +
        block_offset * x + x_offset;
    } else {
      tgt_key_idx = block_idx * num_heads * head_size * block_size +
      head_idx * head_size * block_size + block_offset * head_size +
      head_offset;
    }

    const int64_t tgt_value_idx =
        block_idx * num_heads * head_size * block_size +
        head_idx * head_size * block_size + block_offset * head_size +
        head_offset;
    scalar_t tgt_key = key[src_key_idx];
    scalar_t tgt_value = value[src_value_idx];
    if constexpr (kv_dt == Fp8KVCacheDataType::kAuto) {
      key_cache[tgt_key_idx] = tgt_key;
      value_cache[tgt_value_idx] = tgt_value;
    } else {
      key_cache[tgt_key_idx] =
          fp8::scaled_convert<cache_t, scalar_t, kv_dt>(tgt_key, kv_scale);
      value_cache[tgt_value_idx] =
          fp8::scaled_convert<cache_t, scalar_t, kv_dt>(tgt_value, kv_scale);
    }
  }
}
template <typename scalar_t, typename cache_t, Fp8KVCacheDataType kv_dt>
__global__ void reshape_and_cache_kernel_layout_opt(
    const scalar_t* __restrict__ key,    // [num_tokens, num_heads, head_size]
    const scalar_t* __restrict__ value,  // [num_tokens, num_heads, head_size]
    cache_t* __restrict__ key_cache,     // [num_blocks, num_heads, head_size/x,
                                         // block_size, x]
    cache_t* __restrict__ value_cache,   // [num_blocks, num_heads, head_size,
                                         // block_size]
    const int64_t* __restrict__ slot_mapping,  // [num_tokens]
    const int key_stride, const int value_stride, const int num_heads,
    const int head_size, const int block_size, const int x,
    const float kv_scale, bool is_deepseek) {
  const int64_t token_idx = blockIdx.x;
  const int64_t slot_idx = slot_mapping[token_idx];
  if (slot_idx < 0) {
    // Padding token that should be ignored.
    return;
  }

  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;
  const int n = num_heads * head_size / 8;
  for (int t = threadIdx.x; t < n; t += blockDim.x) {
    int i = t << 3;
    const int64_t src_key_idx = token_idx * key_stride + i;
    const int64_t src_value_idx = token_idx * value_stride + i;

    const int head_idx = i / head_size;
    const int head_offset = i % head_size;

    int64_t tgt_key_idx = 0;
    if (is_deepseek) {
      const int x_idx = head_offset / x;
      const int x_offset = head_offset % x;
      tgt_key_idx =
        block_idx * num_heads * head_size * block_size +
        head_idx * head_size * block_size + x_idx * block_size * x +
        block_offset * x + x_offset;
    } else {
      tgt_key_idx = block_idx * num_heads * head_size * block_size +
      head_idx * head_size * block_size + block_offset * head_size +
      head_offset;
    }

    const int64_t tgt_value_idx =
        block_idx * num_heads * head_size * block_size +
        head_idx * head_size * block_size + block_offset * head_size +
        head_offset;

    *(float4*)(key_cache + tgt_key_idx) = *(float4*)(key + src_key_idx);
    *(float4*)(value_cache + tgt_value_idx) = *(float4*)(value + src_value_idx);
    /*
    scalar_t tgt_key = key[src_key_idx];
    scalar_t tgt_value = value[src_value_idx];
    if constexpr (kv_dt == Fp8KVCacheDataType::kAuto) {
      key_cache[tgt_key_idx] = tgt_key;
      value_cache[tgt_value_idx] = tgt_value;
    } else {
      key_cache[tgt_key_idx] =
          fp8::scaled_convert<cache_t, scalar_t, kv_dt>(tgt_key, kv_scale);
      value_cache[tgt_value_idx] =
          fp8::scaled_convert<cache_t, scalar_t, kv_dt>(tgt_value, kv_scale);
    }
    */
  }
}


template <typename scalar_t, typename cache_t, Fp8KVCacheDataType kv_dt>
__global__ void reshape_and_cache_flash_kernel(
    const scalar_t* __restrict__ key,    // [num_tokens, num_heads, head_size]
    const scalar_t* __restrict__ value,  // [num_tokens, num_heads, head_size]
    cache_t* __restrict__ key_cache,     // [num_blocks, block_size, num_heads,
                                         // head_size]
    cache_t* __restrict__ value_cache,   // [num_blocks, block_size, num_heads,
                                         // head_size]
    const int64_t* __restrict__ slot_mapping,  // [num_tokens]
    const int block_stride, const int key_stride, const int value_stride,
    const int num_heads, const int head_size, const int block_size,
    const float k_scale, const float v_scale) {
  const int64_t token_idx = blockIdx.x;
  const int64_t slot_idx = slot_mapping[token_idx];
  // NOTE: slot_idx can be -1 if the token is padded
  if (slot_idx < 0) {
    return;
  }
  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;
  const int n = num_heads * head_size;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const int64_t src_key_idx = token_idx * key_stride + i;
    const int64_t src_value_idx = token_idx * value_stride + i;
    const int head_idx = i / head_size;
    const int head_offset = i % head_size;
    const int64_t tgt_key_value_idx = block_idx * block_stride +
                                      block_offset * num_heads * head_size +
                                      head_idx * head_size + head_offset;
    scalar_t tgt_key = key[src_key_idx];
    scalar_t tgt_value = value[src_value_idx];
    if constexpr (kv_dt == Fp8KVCacheDataType::kAuto) {
      key_cache[tgt_key_value_idx] = tgt_key;
      value_cache[tgt_key_value_idx] = tgt_value;
    } else {
      key_cache[tgt_key_value_idx] =
          fp8::scaled_convert<cache_t, scalar_t, kv_dt>(tgt_key, k_scale);
      value_cache[tgt_key_value_idx] =
          fp8::scaled_convert<cache_t, scalar_t, kv_dt>(tgt_value, v_scale);
    }
  }
}
}  // namespace vllm

// KV_T is the stored data type of kv-cache.
// CACHE_T is the data type of key and value tensors.
// KV_DTYPE is the real data type of kv-cache.
#define CALL_RESHAPE_AND_CACHE(KV_T, CACHE_T, KV_DTYPE)               \
  vllm::reshape_and_cache_kernel<KV_T, CACHE_T, KV_DTYPE>             \
      <<<grid, block, 0, stream>>>(                                   \
          reinterpret_cast<KV_T*>(key.data_ptr()),                    \
          reinterpret_cast<KV_T*>(value.data_ptr()),                  \
          reinterpret_cast<CACHE_T*>(key_cache.data_ptr()),           \
          reinterpret_cast<CACHE_T*>(value_cache.data_ptr()),         \
          slot_mapping.data_ptr<int64_t>(), key_stride, value_stride, \
          num_heads, head_size, block_size, x, k_scale, v_scale);

void reshape_and_cache(
    torch::Tensor& key,    // [num_tokens, num_heads, head_size]
    torch::Tensor& value,  // [num_tokens, num_heads, head_size]
    torch::Tensor&
        key_cache,  // [num_blocks, num_heads, head_size/x, block_size, x]
    torch::Tensor&
        value_cache,  // [num_blocks, num_heads, head_size, block_size]
    torch::Tensor& slot_mapping,  // [num_tokens]
    const std::string& kv_cache_dtype, const double k_scale,
    const double v_scale) {
  int num_tokens = key.size(0);
  int num_heads = key.size(1);
  int head_size = key.size(2);
  int block_size = key_cache.size(3);
  int x = key_cache.size(4);

  int key_stride = key.stride(0);
  int value_stride = value.stride(0);

  dim3 grid(num_tokens);
  dim3 block(std::min(num_heads * head_size, 512));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(key));
  const hipStream_t stream = at::cuda::getCurrentCUDAStream();

  DISPATCH_BY_KV_CACHE_DTYPE(key.dtype(), kv_cache_dtype,
                             CALL_RESHAPE_AND_CACHE)
}


#define CALL_RESHAPE_AND_CACHE_LAYOUT(KV_T, CACHE_T, KV_DTYPE)               \
  vllm::reshape_and_cache_kernel_layout<KV_T, CACHE_T, KV_DTYPE>             \
      <<<grid, block, 0, stream>>>(                                   \
          reinterpret_cast<KV_T*>(key.data_ptr()),                    \
          reinterpret_cast<KV_T*>(value.data_ptr()),                  \
          reinterpret_cast<CACHE_T*>(key_cache.data_ptr()),           \
          reinterpret_cast<CACHE_T*>(value_cache.data_ptr()),         \
          slot_mapping.data_ptr<int64_t>(), key_stride, value_stride, \
          num_heads, head_size, block_size, x, kv_scale, is_deepseek);

#define CALL_RESHAPE_AND_CACHE_LAYOUT_OPT(KV_T, CACHE_T, KV_DTYPE)               \
  vllm::reshape_and_cache_kernel_layout_opt<KV_T, CACHE_T, KV_DTYPE>             \
      <<<grid, block, 0, stream>>>(                                   \
          reinterpret_cast<KV_T*>(key.data_ptr()),                    \
          reinterpret_cast<KV_T*>(value.data_ptr()),                  \
          reinterpret_cast<CACHE_T*>(key_cache.data_ptr()),           \
          reinterpret_cast<CACHE_T*>(value_cache.data_ptr()),         \
          slot_mapping.data_ptr<int64_t>(), key_stride, value_stride, \
          num_heads, head_size, block_size, x, kv_scale, is_deepseek);
          
void reshape_and_cache_new(
    torch::Tensor& key,    // [num_tokens, num_heads, head_size]
    torch::Tensor& value,  // [num_tokens, num_heads, head_size]
    torch::Tensor&
        key_cache,  // [num_blocks, num_heads, head_size/x, block_size, x]
    torch::Tensor&
        value_cache,  // [num_blocks, num_heads, head_size, block_size]
    torch::Tensor& slot_mapping,  // [num_tokens]
    const std::string& kv_cache_dtype, const double kv_scale, const double v_scale) {
  int num_tokens = key.size(0);
  int num_heads = key.size(1);
  int head_size = key.size(2);

  int x;
  int block_size;
  bool is_deepseek = value_cache.size(-1) == 576;
  if (is_deepseek) {
    // [num_blocks, num_heads, head_size / 16, block_size, 16]
    x = key_cache.size(4);
    block_size = key_cache.size(3);
  } else {
    // [num_blocks, num_heads, block_size, head_size]
    x = 16;
    block_size = key_cache.size(2);
  }

  int key_stride = key.stride(0);
  int value_stride = value.stride(0);

  dim3 grid(num_tokens);
  dim3 block(std::min(num_heads * head_size, 512));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(key));
  const hipStream_t stream = at::cuda::getCurrentCUDAStream();
  
  if (kv_cache_dtype == "auto") {
    if (key.dtype() == at::ScalarType::Float) {
      CALL_RESHAPE_AND_CACHE_LAYOUT(float, float, vllm::Fp8KVCacheDataType::kAuto);
    } else if (key.dtype() == at::ScalarType::Half) {
      if((x & 7) == 0 && (head_size & 7) == 0) {
        CALL_RESHAPE_AND_CACHE_LAYOUT_OPT(half, half, vllm::Fp8KVCacheDataType::kAuto);
      } else {
        CALL_RESHAPE_AND_CACHE_LAYOUT(half, half, vllm::Fp8KVCacheDataType::kAuto);
      }
    }
    
    else if (key.dtype() == at::ScalarType::BFloat16) {
      if((x & 7) == 0 && (head_size & 7) == 0) {
        CALL_RESHAPE_AND_CACHE_LAYOUT_OPT(__nv_bfloat16, __nv_bfloat16, vllm::Fp8KVCacheDataType::kAuto);
      } else {
        CALL_RESHAPE_AND_CACHE_LAYOUT(__nv_bfloat16, __nv_bfloat16, vllm::Fp8KVCacheDataType::kAuto);
      }
    }
    
  }
 /* 
  else if (kv_cache_dtype == "fp8_e5m2") {
    if (key.dtype() == at::ScalarType::Float) {
      CALL_RESHAPE_AND_CACHE_LAYOUT(float, uint8_t, vllm::Fp8KVCacheDataType::kAuto);
    } else if (key.dtype() == at::ScalarType::Half) {
      CALL_RESHAPE_AND_CACHE_LAYOUT(uint16_t, uint8_t, vllm::Fp8KVCacheDataType::kAuto);
    } else if (key.dtype() == at::ScalarType::BFloat16) {
      CALL_RESHAPE_AND_CACHE_LAYOUT(__nv_bfloat16, uint8_t, vllm::Fp8KVCacheDataType::kAuto);
    }
  }
  */
  else {
    TORCH_CHECK(false, "Unsupported data type of kv cache: ", kv_cache_dtype);
  }

}
