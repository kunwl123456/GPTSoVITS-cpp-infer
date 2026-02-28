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

/**
 * @brief GPU Top-K sampling kernel
 * 
 * Performs softmax + multinomial sampling entirely on GPU.
 * This eliminates D2H/H2D round trips during autoregressive generation.
 * 
 * @param topk_values Top-K values [1, K] (FP16 or FP32)
 * @param topk_indices Top-K indices [1, K] (int64)
 * @param k Number of top-k candidates
 * @param temperature Temperature for softmax (default: 1.0)
 * @param out_token Output sampled token [1] (int64)
 * @param rng_state Random number generator state (will be updated)
 * @param stream CUDA stream
 * @return cudaError_t
 */
cudaError_t LaunchSampleTopKKernel(const void* topk_values,
                                   const int64_t* topk_indices,
                                   int64_t k,
                                   float temperature,
                                   int64_t* out_token,
                                   uint64_t* rng_state,
                                   cudaStream_t stream = 0);

/**
 * @brief GPU Top-K sampling kernel with FP16 input (optimized)
 */
cudaError_t LaunchSampleTopKFP16Kernel(const void* topk_values,
                                       const int64_t* topk_indices,
                                       int64_t k,
                                       float temperature,
                                       int64_t* out_token,
                                       uint64_t* rng_state,
                                       cudaStream_t stream = 0);

#else

// Stubs for non-CUDA builds
inline int LaunchTypeConversionKernel(const void*,
                                      void*,
                                      int64_t,
                                      DataType,
                                      DataType,
                                      void* = nullptr) {
  return -1; // Operation not supported
}

inline int LaunchSoftmaxKernel(float*,
                                int64_t,
                                float,
                                void* = nullptr) {
  return -1; // Operation not supported
}

#endif // WITH_CUDA

} // namespace GPTSoVITS::Model::GPU

#endif // GPT_SOVITS_CPP_GPU_KERNELS_H