
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
    logits[i] = expf(logits[i]);
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

} // namespace GPTSoVITS::Model::GPU