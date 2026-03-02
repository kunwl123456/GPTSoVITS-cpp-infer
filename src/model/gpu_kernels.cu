
#include "GPTSoVITS/model/gpu_kernels.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace GPTSoVITS::Model::GPU {

// ============ Type Conversion Kernels ============

// FP32 to FP16 conversion kernel
__global__ void ConvertFP32ToFP16Kernel(const float* __restrict__ input,
                                        half* __restrict__ output,
                                        int64_t numel) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numel) {
    output[idx] = __float2half(input[idx]);
  }
}

// FP16 to FP32 conversion kernel
__global__ void ConvertFP16ToFP32Kernel(const half* __restrict__ input,
                                        float* __restrict__ output,
                                        int64_t numel) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numel) {
    output[idx] = __half2float(input[idx]);
  }
}

// Int32 to Int64 conversion kernel
__global__ void ConvertInt32ToInt64Kernel(const int32_t* __restrict__ input,
                                          int64_t* __restrict__ output,
                                          int64_t numel) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numel) {
    output[idx] = static_cast<int64_t>(input[idx]);
  }
}

// Int64 to Int32 conversion kernel
__global__ void ConvertInt64ToInt32Kernel(const int64_t* __restrict__ input,
                                          int32_t* __restrict__ output,
                                          int64_t numel) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numel) {
    output[idx] = static_cast<int32_t>(input[idx]);
  }
}

// Generic type conversion (FP32 to other types)
__global__ void ConvertFP32ToInt32Kernel(const float* __restrict__ input,
                                        int32_t* __restrict__ output,
                                        int64_t numel) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numel) {
    output[idx] = static_cast<int32_t>(input[idx]);
  }
}

__global__ void ConvertFP32ToInt64Kernel(const float* __restrict__ input,
                                        int64_t* __restrict__ output,
                                        int64_t numel) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numel) {
    output[idx] = static_cast<int64_t>(input[idx]);
  }
}

// Generic type conversion (Int32 to other types)
__global__ void ConvertInt32ToFP32Kernel(const int32_t* __restrict__ input,
                                        float* __restrict__ output,
                                        int64_t numel) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numel) {
    output[idx] = static_cast<float>(input[idx]);
  }
}

// Generic type conversion (Int64 to other types)
__global__ void ConvertInt64ToFP32Kernel(const int64_t* __restrict__ input,
                                        float* __restrict__ output,
                                        int64_t numel) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numel) {
    output[idx] = static_cast<float>(input[idx]);
  }
}

// ============ Sampling Kernels ============

// Softmax kernel (for top-k sampling)
__global__ void SoftmaxKernel(float* __restrict__ logits,
                              int64_t numel,
                              float temperature) {
  extern __shared__ float sdata[];

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int tid = threadIdx.x;

  // Find max value for numerical stability
  float max_val = -INFINITY;
  for (int i = tid; i < numel; i += blockDim.x) {
    if (logits[i] > max_val) {
      max_val = logits[i];
    }
  }

  // Reduce max value across threads
  sdata[tid] = max_val;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
    }
    __syncthreads();
  }

  max_val = sdata[0];
  __syncthreads();

  // Apply temperature and exp
  float sum = 0.0f;
  for (int i = tid; i < numel; i += blockDim.x) {
    if (temperature != 1.0f) {
      logits[i] = (logits[i] - max_val) / temperature;
    } else {
      logits[i] = logits[i] - max_val;
    }
    // Clamp 防止 exp 溢出/下溢
    if (logits[i] > 50.0f) logits[i] = 50.0f;
    else if (logits[i] < -50.0f) logits[i] = -50.0f;
    logits[i] = expf(logits[i]);
    // 检查 NaN/Inf
    if (!isfinite(logits[i])) logits[i] = 0.0f;
    sum += logits[i];
  }

  // Reduce sum across threads
  sdata[tid] = sum;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] = sdata[tid] + sdata[tid + s];
    }
    __syncthreads();
  }

  sum = sdata[0];
  __syncthreads();

  // Normalize
  if (sum > 0.0f) {
    for (int i = tid; i < numel; i += blockDim.x) {
      logits[i] /= sum;
    }
  }
}

// ============ GPU Top-K Sampling Kernel ============

// Philox random number generator (stateless, deterministic)
__device__ __forceinline__ uint4 philox4x32(uint4 counter, uint2 key) {
  const uint32_t PHILOX_M4x32_0 = 0xD2511F53;
  const uint32_t PHILOX_M4x32_1 = 0xCD9E8D57;
  const uint32_t PHILOX_W32_0 = 0x9E3779B9;
  const uint32_t PHILOX_W32_1 = 0xBB67AE85;

  #pragma unroll
  for (int i = 0; i < 10; ++i) {
    counter.x *= PHILOX_M4x32_0;
    counter.y *= PHILOX_M4x32_1;
    counter.x ^= key.x;
    counter.y ^= key.y;
    key.x += PHILOX_W32_0;
    key.y += PHILOX_W32_1;

    uint32_t tmp = counter.x;
    counter.x = counter.y;
    counter.y = counter.z;
    counter.z = counter.w;
    counter.w = tmp;
  }
  return counter;
}

// Generate a random float in [0, 1)
__device__ __forceinline__ float random_uniform(uint64_t* rng_state) {
  // Use Philox to generate random numbers
  uint4 counter = make_uint4(
      static_cast<uint32_t>((*rng_state) & 0xFFFFFFFF),
      static_cast<uint32_t>((*rng_state) >> 32),
      static_cast<uint32_t>((*rng_state) >> 16),
      static_cast<uint32_t>((*rng_state) >> 8)
  );
  uint2 key = make_uint2(0x12345678, 0x87654321);
  
  uint4 result = philox4x32(counter, key);
  
  // Update state
  *rng_state = static_cast<uint64_t>(result.x) | (static_cast<uint64_t>(result.y) << 32);
  
  // Convert to float in [0, 1)
  return static_cast<float>(result.z) / static_cast<float>(0xFFFFFFFF);
}

// FP32 Top-K sampling kernel
__global__ void SampleTopKFP32Kernel(
    const float* __restrict__ topk_values,
    const int64_t* __restrict__ topk_indices,
    int64_t k,
    float temperature,
    int64_t* __restrict__ out_token,
    uint64_t* __restrict__ rng_state) {
  
  extern __shared__ float probs[];
  
  int tid = threadIdx.x;
  
  // Load and process topk_values
  float max_val = -INFINITY;
  
  // Find max (single block, small k)
  for (int i = tid; i < k; i += blockDim.x) {
    float v = topk_values[i];
    if (temperature != 1.0f) v = v / temperature;
    probs[i] = v;
    if (v > max_val) max_val = v;
  }
  __syncthreads();
  
  // Compute exp(v - max) and sum
  float sum = 0.0f;
  for (int i = tid; i < k; i += blockDim.x) {
    float v = probs[i] - max_val;
    // Clamp
    if (v > 50.0f) v = 50.0f;
    else if (v < -50.0f) v = -50.0f;
    probs[i] = expf(v);
    if (!isfinite(probs[i])) probs[i] = 0.0f;
    sum += probs[i];
  }
  __syncthreads();
  
  // Sum reduction
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      probs[tid] += probs[tid + s];
    }
    __syncthreads();
  }
  sum = probs[0];
  __syncthreads();
  
  // Normalize to get probabilities
  if (sum > 1e-10f) {
    for (int i = tid; i < k; i += blockDim.x) {
      probs[i] /= sum;
    }
  }
  __syncthreads();
  
  // Sample using CDF (only thread 0)
  if (tid == 0) {
    float r = random_uniform(rng_state);
    float cdf = 0.0f;
    int64_t selected_idx = k - 1;  // fallback
    
    for (int64_t i = 0; i < k; ++i) {
      cdf += probs[i];
      if (r <= cdf) {
        selected_idx = i;
        break;
      }
    }
    
    *out_token = topk_indices[selected_idx];
  }
}

// FP16 Top-K sampling kernel (optimized for FP16 model output)
__global__ void SampleTopKFP16Kernel(
    const half* __restrict__ topk_values,
    const int64_t* __restrict__ topk_indices,
    int64_t k,
    float temperature,
    int64_t* __restrict__ out_token,
    uint64_t* __restrict__ rng_state) {
  
  extern __shared__ float probs[];
  
  int tid = threadIdx.x;
  
  // Load and process topk_values (convert FP16 to FP32)
  float max_val = -INFINITY;
  
  for (int i = tid; i < k; i += blockDim.x) {
    float v = __half2float(topk_values[i]);
    if (temperature != 1.0f) v = v / temperature;
    probs[i] = v;
    if (v > max_val) max_val = v;
  }
  __syncthreads();
  
  // Compute exp(v - max) and sum
  float sum = 0.0f;
  for (int i = tid; i < k; i += blockDim.x) {
    float v = probs[i] - max_val;
    if (v > 50.0f) v = 50.0f;
    else if (v < -50.0f) v = -50.0f;
    probs[i] = expf(v);
    if (!isfinite(probs[i])) probs[i] = 0.0f;
    sum += probs[i];
  }
  __syncthreads();
  
  // Sum reduction
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      probs[tid] += probs[tid + s];
    }
    __syncthreads();
  }
  sum = probs[0];
  __syncthreads();
  
  // Normalize
  if (sum > 1e-10f) {
    for (int i = tid; i < k; i += blockDim.x) {
      probs[i] /= sum;
    }
  }
  __syncthreads();
  
  // Sample using CDF (only thread 0)
  if (tid == 0) {
    float r = random_uniform(rng_state);
    float cdf = 0.0f;
    int64_t selected_idx = k - 1;
    
    for (int64_t i = 0; i < k; ++i) {
      cdf += probs[i];
      if (r <= cdf) {
        selected_idx = i;
        break;
      }
    }
    
    *out_token = topk_indices[selected_idx];
  }
}

// ============ Host Functions ============

cudaError_t LaunchTypeConversionKernel(const void* input,
                                       void* output,
                                       int64_t numel,
                                       DataType input_type,
                                       DataType output_type,
                                       cudaStream_t stream) {
  if (numel <= 0) {
    return cudaSuccess;
  }

  int block_size = 256;
  int grid_size = (numel + block_size - 1) / block_size;

  switch (input_type) {
    case DataType::kFloat32:
      switch (output_type) {
        case DataType::kFloat16:
          ConvertFP32ToFP16Kernel<<<grid_size, block_size, 0, stream>>>(
              static_cast<const float*>(input),
              static_cast<half*>(output),
              numel);
          break;
        case DataType::kInt32:
          ConvertFP32ToInt32Kernel<<<grid_size, block_size, 0, stream>>>(
              static_cast<const float*>(input),
              static_cast<int32_t*>(output),
              numel);
          break;
        case DataType::kInt64:
          ConvertFP32ToInt64Kernel<<<grid_size, block_size, 0, stream>>>(
              static_cast<const float*>(input),
              static_cast<int64_t*>(output),
              numel);
          break;
        default:
          return cudaErrorNotSupported;
      }
      break;

    case DataType::kFloat16:
      switch (output_type) {
        case DataType::kFloat32:
          ConvertFP16ToFP32Kernel<<<grid_size, block_size, 0, stream>>>(
              static_cast<const half*>(input),
              static_cast<float*>(output),
              numel);
          break;
        default:
          return cudaErrorNotSupported;
      }
      break;

    case DataType::kInt32:
      switch (output_type) {
        case DataType::kFloat32:
          ConvertInt32ToFP32Kernel<<<grid_size, block_size, 0, stream>>>(
              static_cast<const int32_t*>(input),
              static_cast<float*>(output),
              numel);
          break;
        case DataType::kInt64:
          ConvertInt32ToInt64Kernel<<<grid_size, block_size, 0, stream>>>(
              static_cast<const int32_t*>(input),
              static_cast<int64_t*>(output),
              numel);
          break;
        default:
          return cudaErrorNotSupported;
      }
      break;

    case DataType::kInt64:
      switch (output_type) {
        case DataType::kFloat32:
          ConvertInt64ToFP32Kernel<<<grid_size, block_size, 0, stream>>>(
              static_cast<const int64_t*>(input),
              static_cast<float*>(output),
              numel);
          break;
        case DataType::kInt32:
          ConvertInt64ToInt32Kernel<<<grid_size, block_size, 0, stream>>>(
              static_cast<const int64_t*>(input),
              static_cast<int32_t*>(output),
              numel);
          break;
        default:
          return cudaErrorNotSupported;
      }
      break;

    default:
      return cudaErrorNotSupported;
  }

  return cudaGetLastError();
}

cudaError_t LaunchSoftmaxKernel(float* logits,
                                int64_t numel,
                                float temperature,
                                cudaStream_t stream) {
  if (numel <= 0) {
    return cudaSuccess;
  }

  int block_size = 256;
  int grid_size = 1; // Use single block for softmax (assumes small k)
  int shared_mem_size = block_size * sizeof(float);

  SoftmaxKernel<<<grid_size, block_size, shared_mem_size, stream>>>(
      logits, numel, temperature);

  return cudaGetLastError();
}

cudaError_t LaunchSampleTopKKernel(const void* topk_values,
                                   const int64_t* topk_indices,
                                   int64_t k,
                                   float temperature,
                                   int64_t* out_token,
                                   uint64_t* rng_state,
                                   cudaStream_t stream) {
  if (k <= 0) {
    return cudaErrorInvalidValue;
  }

  // Use 64 threads (enough for k <= 50)
  int block_size = 64;
  if (k > 64) block_size = 128;
  if (k > 128) block_size = 256;
  
  int shared_mem_size = k * sizeof(float);

  SampleTopKFP32Kernel<<<1, block_size, shared_mem_size, stream>>>(
      static_cast<const float*>(topk_values),
      topk_indices,
      k,
      temperature,
      out_token,
      rng_state);

  return cudaGetLastError();
}

cudaError_t LaunchSampleTopKFP16Kernel(const void* topk_values,
                                       const int64_t* topk_indices,
                                       int64_t k,
                                       float temperature,
                                       int64_t* out_token,
                                       uint64_t* rng_state,
                                       cudaStream_t stream) {
  if (k <= 0) {
    return cudaErrorInvalidValue;
  }

  int block_size = 64;
  if (k > 64) block_size = 128;
  if (k > 128) block_size = 256;
  
  int shared_mem_size = k * sizeof(float);

  SampleTopKFP16Kernel<<<1, block_size, shared_mem_size, stream>>>(
      static_cast<const half*>(topk_values),
      topk_indices,
      k,
      temperature,
      out_token,
      rng_state);

  return cudaGetLastError();
}

} // namespace GPTSoVITS::Model::GPU