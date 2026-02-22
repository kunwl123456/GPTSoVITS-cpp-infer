//
// Created by 19254 on 2026/2/8.
//
// CUDA kernels for GPU-accelerated tensor operations
//

#ifndef GPT_SOVITS_CPP_GPU_KERNELS_H
#define GPT_SOVITS_CPP_GPU_KERNELS_H

#include "GPTSoVITS/model/tensor.h"

#ifdef WITH_CUDA
#include <cuda_runtime.h>
#endif

namespace GPTSoVITS::Model::GPU {

#ifdef WITH_CUDA

/**
 * @brief Launch type conversion kernel on GPU
 * @param input Input data pointer
 * @param output Output data pointer
 * @param numel Number of elements
 * @param input_type Input data type
 * @param output_type Output data type
 * @param stream CUDA stream (default: 0 for default stream)
 * @return cudaError_t CUDA error code
 */
cudaError_t LaunchTypeConversionKernel(const void* input,
                                       void* output,
                                       int64_t numel,
                                       DataType input_type,
                                       DataType output_type,
                                       cudaStream_t stream = 0);

/**
 * @brief Launch softmax kernel on GPU
 * @param logits Pointer to logits data (will be modified in-place)
 * @param numel Number of elements
 * @param temperature Temperature for softmax
 * @param stream CUDA stream (default: 0 for default stream)
 * @return cudaError_t CUDA error code
 */
cudaError_t LaunchSoftmaxKernel(float* logits,
                                int64_t numel,
                                float temperature,
                                cudaStream_t stream = 0);

#else

// Stubs for non-CUDA builds
inline cudaError_t LaunchTypeConversionKernel(const void*,
                                              void*,
                                              int64_t,
                                              DataType,
                                              DataType,
                                              void* = nullptr) {
  return cudaErrorNotSupported;
}

inline cudaError_t LaunchSoftmaxKernel(float*,
                                       int64_t,
                                       float,
                                       void* = nullptr) {
  return cudaErrorNotSupported;
}

#endif // WITH_CUDA

} // namespace GPTSoVITS::Model::GPU

#endif // GPT_SOVITS_CPP_GPU_KERNELS_H