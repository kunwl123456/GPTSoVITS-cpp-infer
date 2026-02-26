//
// Created by iFlow CLI on 2026/2/20.
//

#include "GPTSoVITS/model/gpu_utils.h"

#include <algorithm>
#include <cmath>
#include <random>

#include "GPTSoVITS/Utils/exception.h"
#include "GPTSoVITS/plog.h"

#ifdef WITH_CUDA
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <curand.h>
#include "GPTSoVITS/model/gpu_kernels.h"
#endif

namespace GPTSoVITS::Model::GPU {

namespace {
// CPU 版本的 softmax + top-k 采样
std::unique_ptr<Tensor> SampleTopKCPU(
    const Tensor* topk_values,
    const Tensor* topk_indices,
    float temperature) {
  // 确保输入在 CPU 上
  auto values_cpu = topk_values->IsCPU() ?
      topk_values->Clone() : topk_values->ToCPU();
  auto indices_cpu = topk_indices->IsCPU() ?
      topk_indices->Clone() : topk_indices->ToCPU();

  int batch = topk_values->Shape()[0];
  int k = topk_values->Shape()[1];

  // 创建输出张量
  auto output = Tensor::Empty({batch, 1}, DataType::kInt64, Device(DeviceType::kCPU));
  int64_t* output_ptr = output->Data<int64_t>();

  float* values_ptr = values_cpu->Data<float>();
  int64_t* indices_ptr = indices_cpu->Data<int64_t>();

  // 对每个 batch 进行采样
  for (int b = 0; b < batch; ++b) {
    float* batch_values = values_ptr + b * k;
    int64_t* batch_indices = indices_ptr + b * k;

    // 减去最大值 (数值稳定性，必须做)
    float max_val = *std::max_element(batch_values, batch_values + k);
    for (int i = 0; i < k; ++i) {
      batch_values[i] = batch_values[i] - max_val;
    }

    // 应用温度 (只有 temperature != 1.0 时才除)
    if (temperature != 1.0f && temperature > 1e-6f) {
      for (int i = 0; i < k; ++i) {
        batch_values[i] /= temperature;
      }
    }

    // Clamp 防止 exp 溢出/下溢
    for (int i = 0; i < k; ++i) {
      if (batch_values[i] > 50.0f) batch_values[i] = 50.0f;
      else if (batch_values[i] < -50.0f) batch_values[i] = -50.0f;
    }

    // Softmax
    float sum = 0.0f;
    for (int i = 0; i < k; ++i) {
      batch_values[i] = std::exp(batch_values[i]);
      if (!std::isfinite(batch_values[i])) batch_values[i] = 0.0f;
      sum += batch_values[i];
    }

    // 归一化
    for (int i = 0; i < k; ++i) {
      batch_values[i] /= sum;
    }

    // 多项式采样
    float r = static_cast<float>(rand()) / RAND_MAX;
    float cumulative = 0.0f;
    for (int i = 0; i < k; ++i) {
      cumulative += batch_values[i];
      if (r <= cumulative) {
        output_ptr[b] = batch_indices[i];
        break;
      }
    }

    // 如果未命中（数值精度问题），选择最后一个
    if (cumulative < r) {
      output_ptr[b] = batch_indices[k - 1];
    }
  }

  // 如果原始输入在 GPU 上，将结果搬回 GPU
  if (topk_values->IsCUDA()) {
    return output->ToDevice(topk_values->GetDevice());
  }

  return output;
}

// CPU 版本的类型转换
std::unique_ptr<Tensor> ConvertTypeCPU(
    const Tensor* src,
    DataType target_dtype) {
  // 如果已经在 CPU 上，直接使用现有实现
  if (src->IsCPU()) {
    return src->ToType(target_dtype);
  }

  // 如果在 GPU 上，先搬到 CPU，转换后再搬回去
  PrintWarn("[GPUTypeConverter] CUDA not available, falling back to CPU conversion");
  auto cpu_src = src->ToCPU();
  auto cpu_dst = cpu_src->ToType(target_dtype);
  return cpu_dst->ToDevice(src->GetDevice());
}

}  // namespace

// ============ GPUSampler ============

struct GPUSampler::Impl {
  std::mt19937 rng;
#ifdef WITH_CUDA
  curandGenerator_t curand_gen;
#endif

  Impl() : rng(std::random_device{}()) {
#ifdef WITH_CUDA
    curandStatus_t err = curandCreateGenerator(&curand_gen, CURAND_RNG_PSEUDO_DEFAULT);
    if (err != CURAND_STATUS_SUCCESS) {
      const char* err_str = "unknown";
      switch (err) {
        case CURAND_STATUS_SUCCESS: err_str = "success"; break;
        case CURAND_STATUS_VERSION_MISMATCH: err_str = "version mismatch"; break;
        case CURAND_STATUS_NOT_INITIALIZED: err_str = "not initialized"; break;
        case CURAND_STATUS_ALLOCATION_FAILED: err_str = "allocation failed"; break;
        case CURAND_STATUS_TYPE_ERROR: err_str = "type error"; break;
        case CURAND_STATUS_OUT_OF_RANGE: err_str = "out of range"; break;
        case CURAND_STATUS_LENGTH_NOT_MULTIPLE: err_str = "length not multiple"; break;
        case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED: err_str = "double precision required"; break;
        case CURAND_STATUS_LAUNCH_FAILURE: err_str = "launch failure"; break;
        case CURAND_STATUS_PREEXISTING_FAILURE: err_str = "preexisting failure"; break;
        case CURAND_STATUS_INITIALIZATION_FAILED: err_str = "initialization failed"; break;
        case CURAND_STATUS_ARCH_MISMATCH: err_str = "arch mismatch"; break;
        case CURAND_STATUS_INTERNAL_ERROR: err_str = "internal error"; break;
      }
      PrintError("[GPUSampler] Failed to create CURAND generator: {}", err_str);
    }
#endif
  }

  ~Impl() {
#ifdef WITH_CUDA
    if (curand_gen) {
      curandDestroyGenerator(curand_gen);
    }
#endif
  }
};

GPUSampler::GPUSampler() : impl_(std::make_unique<Impl>()) {}

GPUSampler::~GPUSampler() = default;

std::unique_ptr<Tensor> GPUSampler::SampleTopK(
    const Tensor* topk_values,
    const Tensor* topk_indices,
    float temperature) {
#ifdef WITH_CUDA
  if (topk_values->IsCUDA() && topk_indices->IsCUDA()) {
    // 使用 GPU 加速 softmax 计算，但 multinomial 采样仍在 CPU 上进行
    try {
      // 在 GPU 上拷贝并应用 softmax
      auto softmax_values = topk_values->Clone();
      cudaError_t err = LaunchSoftmaxKernel(
          softmax_values->Data<float>(),
          softmax_values->ElementCount(),
          temperature);

      if (err != cudaSuccess) {
        PrintWarn("[GPUSampler] CUDA softmax failed: {}, falling back to CPU", cudaGetErrorString(err));
        return SampleTopKCPU(topk_values, topk_indices, temperature);
      }

      // 同步等待kernel完成
      cudaStreamSynchronize(nullptr);

      // 将 softmax 后的值和索引拷贝到 CPU 进行采样
      auto values_cpu = softmax_values->ToCPU();
      auto indices_cpu = topk_indices->ToCPU();

      // 使用 GPU 加速后的值进行 CPU 采样
      return SampleTopKCPU(values_cpu.get(), indices_cpu.get(), temperature);
    } catch (const std::exception& e) {
      PrintWarn("[GPUSampler] CUDA sampling exception: {}, falling back to CPU", e.what());
      return SampleTopKCPU(topk_values, topk_indices, temperature);
    }
  }
#endif

  // 回退到 CPU 版本
  return SampleTopKCPU(topk_values, topk_indices, temperature);
}

std::unique_ptr<Tensor> GPUSampler::SampleMultinomial(
    const Tensor* logits,
    float temperature) {
  // TODO: 实现完整的多项式采样
  THROW_ERROR("[GPUSampler] SampleMultinomial not yet implemented");
}

// ============ GPUTypeConverter ============

std::unique_ptr<Tensor> GPUTypeConverter::ConvertType(
    const Tensor* src,
    DataType target_dtype) {
  // 如果类型相同，直接返回拷贝
  if (src->Type() == target_dtype) {
    return src->Clone();
  }

#ifdef WITH_CUDA
  if (src->IsCUDA()) {
    // 使用 CUDA kernel 进行类型转换
    try {
      // 在目标设备上分配内存
      auto dst = Tensor::Empty(src->Shape(), target_dtype, src->GetDevice());

      // 启动类型转换 kernel
      cudaError_t err = LaunchTypeConversionKernel(
          src->Data(),
          dst->Data(),
          src->ElementCount(),
          src->Type(),
          target_dtype);

      if (err != cudaSuccess) {
        PrintWarn("[GPUTypeConverter] CUDA kernel failed: {}, falling back to CPU", cudaGetErrorString(err));
        return ConvertTypeCPU(src, target_dtype);
      }

      // 同步等待kernel完成
      cudaStreamSynchronize(nullptr);

      return dst;
    } catch (const std::exception& e) {
      PrintWarn("[GPUTypeConverter] CUDA conversion exception: {}, falling back to CPU", e.what());
      return ConvertTypeCPU(src, target_dtype);
    }
  }
#endif

  // 回退到 CPU 版本
  return ConvertTypeCPU(src, target_dtype);
}

bool GPUTypeConverter::IsConversionSupported(DataType src_dtype, DataType target_dtype) {
  // FP8 转换仅在 GPU 上支持
  if (src_dtype == DataType::kFloat8 || target_dtype == DataType::kFloat8) {
#ifdef WITH_CUDA
    // 检查 CUDA 架构是否支持 FP8
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    return prop.major >= 9;  // Hopper (9.0+) 支持 FP8
#else
    return false;
#endif
  }

  // 其他转换都支持
  return true;
}

// ============ GPUCacheManager ============

std::unique_ptr<Tensor> GPUCacheManager::PreallocateKVCache(
    int64_t max_seq_len,
    int num_layers,
    int num_heads,
    int head_dim,
    DataType dtype,
    const Device& device) {
  // [num_layers, batch, num_heads, max_seq_len, head_dim]
  std::vector<int64_t> cache_shape = {
      num_layers,
      1,           // batch（只支持 batch=1）
      num_heads,
      max_seq_len,
      head_dim
  };

  return Tensor::Empty(cache_shape, dtype, device);
}

void GPUCacheManager::UpdateCacheInPlace(
    Tensor* cache,
    const Tensor* new_data,
    int64_t offset) {
  // 在 CPU 或 GPU 上直接拷贝数据到指定位置
  size_t element_size = Tensor::ElementSize(cache->Type());
  size_t copy_size = new_data->ByteSize();

  uint8_t* cache_ptr = static_cast<uint8_t*>(cache->Data());
  uint8_t* src_ptr = static_cast<uint8_t*>(new_data->Data());

  // 计算目标偏移位置
  uint8_t* dst_ptr = cache_ptr + offset * element_size;

  // 拷贝数据
  if (cache->IsCPU()) {
    std::memcpy(dst_ptr, src_ptr, copy_size);
  } else {
#ifdef WITH_CUDA
    cudaMemcpy(dst_ptr, src_ptr, copy_size, cudaMemcpyDeviceToDevice);
#endif
  }
}

}  // namespace GPTSoVITS::Model::GPU