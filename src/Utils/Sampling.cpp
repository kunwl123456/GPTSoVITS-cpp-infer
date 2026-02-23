//
// Created by 19254 on 2026/2/22.
//
// 采样工具实现
//

#include "GPTSoVITS/Utils/Sampling.h"

#ifdef WITH_CUDA
#include <cuda_runtime_api.h>
#include <driver_types.h>
#endif


#include <algorithm>
#include <cmath>
#include <numeric>

#include "GPTSoVITS/plog.h"

namespace GPTSoVITS {
namespace Utils {

namespace {
// 线程局部 RNG
std::mt19937& GetGlobalRNG() {
  static thread_local std::mt19937 rng([]() {
    std::random_device rd;
    std::seed_seq seq{rd(), rd(), rd(), rd(), rd(), rd(), rd(), rd()};
    return std::mt19937(seq);
  }());
  return rng;
}

std::vector<float>& GetThreadLocalProbsBuffer() {
  static thread_local std::vector<float> probs;
  return probs;
}
}  // namespace

// ============================================================================
// Sampler 类实现
// ============================================================================

Sampler::Sampler() : rng_(std::random_device{}()) {}

Sampler::~Sampler() = default;

void Sampler::SetSeed(unsigned int seed) {
  rng_.seed(seed);
  // 同时重置全局 RNG
  GetGlobalRNG().seed(seed);
}

int64_t Sampler::Sample(const Model::Tensor* logits, const SamplingConfig& config) {
  if (!logits || logits->ElementCount() == 0) {
    PrintError("[Sampler] Invalid input: null or empty tensor");
    return 0;
  }
  
  // 确保数据在CPU上并转换为float32
  std::unique_ptr<Model::Tensor> logits_cpu;
  if (logits->Type() == Model::DataType::kFloat32 && logits->IsCPU()) {
    logits_cpu = logits->Clone();
  } else if (logits->Type() == Model::DataType::kFloat32) {
    logits_cpu = logits->ToCPU();
  } else {
    // 非float32类型需要显式转换
    logits_cpu = logits->IsCPU() 
        ? logits->ToType(Model::DataType::kFloat32)
        : logits->ToCPU()->ToType(Model::DataType::kFloat32);
  }
  
  const float* data = logits_cpu->Data<float>();
  size_t size = static_cast<size_t>(logits_cpu->ElementCount());
  
  switch (config.strategy) {
    case SamplingStrategy::kGreedy: {
      // 贪婪采样：返回最大值索引
      return std::distance(data, std::max_element(data, data + size));
    }
    case SamplingStrategy::kTopK:
    case SamplingStrategy::kTopP:
    case SamplingStrategy::kTopKTopP: {
      // 分配工作缓冲区
      if (work_probs_.size() < size) {
        work_probs_.resize(size);
        work_indices_.resize(size);
      }
      
      // 计算softmax
      StableSoftmax(data, work_probs_.data(), size, config.temperature);
      
      // 创建索引序列
      std::iota(work_indices_.begin(), work_indices_.begin() + size, 0);
      
      // 根据策略采样
      if (config.strategy == SamplingStrategy::kTopK) {
        // Top-K: 先排序取前K个，再采样
        std::partial_sort(
            work_indices_.begin(),
            work_indices_.begin() + std::min(static_cast<size_t>(config.top_k), size),
            work_indices_.begin() + size,
            [&](int64_t a, int64_t b) { return work_probs_[a] > work_probs_[b]; });
        
        size_t k = std::min(static_cast<size_t>(config.top_k), size);
        float sum = 0.0f;
        for (size_t i = 0; i < k; ++i) {
          sum += work_probs_[work_indices_[i]];
        }
        
        std::discrete_distribution<int> dist(
            k, 0, k,
            [&](double i) { return work_probs_[work_indices_[static_cast<size_t>(i)]] / sum; });
        
        return work_indices_[dist(rng_)];
      } else {
        // Top-P 或 Top-K+Top-P
        // 先按概率降序排序
        std::sort(work_indices_.begin(), work_indices_.begin() + size,
            [&](int64_t a, int64_t b) { return work_probs_[a] > work_probs_[b]; });
        
        // 计算累积概率，找到top-p边界
        float cumsum = 0.0f;
        size_t cutoff = 0;
        for (size_t i = 0; i < size; ++i) {
          cumsum += work_probs_[work_indices_[i]];
          cutoff = i + 1;
          if (cumsum >= config.top_p) break;
        }
        
        // 如果配置了top-k，限制最大候选数
        if (config.strategy == SamplingStrategy::kTopKTopP) {
          cutoff = std::min(cutoff, static_cast<size_t>(config.top_k));
        }
        
        // 从截断后的分布采样
        std::discrete_distribution<int> dist(
            cutoff, 0, cutoff,
            [&](double i) { return work_probs_[work_indices_[static_cast<size_t>(i)]]; });
        
        return work_indices_[dist(rng_)];
      }
    }
    default:
      return SampleFromProbs(work_probs_.data(), size, rng_);
  }
}

int64_t Sampler::SampleFromTopK(
    const Model::Tensor* topk_values,
    const Model::Tensor* topk_indices,
    float temperature) {
  // 直接使用全局函数
  return SampleTopK(topk_values, topk_indices, temperature);
}

int64_t Sampler::SampleZeroCopy(
    const Model::Tensor* logits_gpu,
    float* cpu_buffer,
    const SamplingConfig& config) {
  if (!logits_gpu || logits_gpu->ElementCount() == 0) {
    return 0;
  }
  
  size_t size = static_cast<size_t>(logits_gpu->ElementCount());
  
  // 将GPU数据拷贝到预分配的CPU缓冲区
  if (logits_gpu->IsCPU()) {
    std::memcpy(cpu_buffer, logits_gpu->Data<float>(), size * sizeof(float));
  } else {
#ifdef WITH_CUDA
    cudaError_t err = cudaMemcpy(
        cpu_buffer, logits_gpu->Data<float>(), 
        size * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
      PrintError("[Sampler] D2H copy failed: {}", cudaGetErrorString(err));
      return 0;
    }
    cudaDeviceSynchronize();
#else
    // 无CUDA支持，使用ToCPU
    auto cpu_tensor = logits_gpu->ToCPU();
    std::memcpy(cpu_buffer, cpu_tensor->Data<float>(), size * sizeof(float));
#endif
  }
  
  // 计算softmax并采样
  if (work_probs_.size() < size) {
    work_probs_.resize(size);
  }
  
  StableSoftmax(cpu_buffer, work_probs_.data(), size, config.temperature);
  
  // 贪婪采样
  return SampleFromProbs(work_probs_.data(), size, rng_);
}

// ============================================================================
// 全局函数实现
// ============================================================================

void StableSoftmax(
    const float* logits,
    float* probs,
    size_t size,
    float temperature) {
  if (size == 0) return;
  
  // 找最大值 (数值稳定性关键)
  float max_val = logits[0];
  for (size_t i = 1; i < size; ++i) {
    if (logits[i] > max_val) max_val = logits[i];
  }
  
  // 计算 exp((x - max) / temperature)
  //  topk_values = topk_values - np.max(topk_values)
  //  probs = np.exp(topk_values)
  float sum = 0.0f;
  for (size_t i = 0; i < size; ++i) {
    float logit = logits[i] - max_val;  // 先减去最大值
    if (temperature != 1.0f && temperature > 1e-6f) {
      logit /= temperature;  // 只有当 temperature != 1.0 时才除以温度
    }
    // Clamp 防止 exp 溢出/下溢
    if (logit > 50.0f) {
      logit = 50.0f;
    } else if (logit < -50.0f) {
      logit = -50.0f;
    }
    probs[i] = std::exp(logit);
    // 检查 NaN/Inf
    if (!std::isfinite(probs[i])) {
      probs[i] = 0.0f;
    }
    sum += probs[i];
  }
  
  // 归一化
  // Python: probs /= np.sum(probs)
  if (sum > 1e-10f) {
    float inv_sum = 1.0f / sum;
    for (size_t i = 0; i < size; ++i) {
      probs[i] *= inv_sum;
    }
  } else {
    // 所有概率都接近0，使用均匀分布
    float uniform = 1.0f / static_cast<float>(size);
    for (size_t i = 0; i < size; ++i) {
      probs[i] = uniform;
    }
  }
}

int SampleFromProbs(const float* probs, size_t size, std::mt19937& rng) {
  if (size == 0) return 0;
  
  try {
    std::discrete_distribution<int> dist(probs, probs + size);
    return dist(rng);
  } catch (const std::exception& e) {
    PrintError("[Sampler] discrete_distribution failed: {}", e.what());
    // Fallback: 返回最大概率索引
    return std::distance(probs, std::max_element(probs, probs + size));
  }
}

int64_t SampleTopK(
    const Model::Tensor* topk_values,
    const Model::Tensor* topk_indices,
    float temperature) {
  if (!topk_values || !topk_indices || topk_values->ElementCount() == 0) {
    PrintError("[SampleTopK] Invalid input: null or empty tensor");
    return 0;
  }

  // 一次性完成类型转换和设备迁移，减少中间拷贝
  std::unique_ptr<Model::Tensor> values_cpu_owner;
  std::unique_ptr<Model::Tensor> indices_cpu_owner;
  
  const float* values_ptr = nullptr;
  const int64_t* indices_ptr = nullptr;
  int64_t k = topk_values->ElementCount();
  
  // 处理 values 必须是 float32 且在 CPU
  if (topk_values->IsCPU() && topk_values->Type() == Model::DataType::kFloat32) {
    values_ptr = topk_values->Data<float>();
  } else if (topk_values->IsCPU()) {
    // 仅需类型转换
    values_cpu_owner = topk_values->ToType(Model::DataType::kFloat32);
    values_ptr = values_cpu_owner->Data<float>();
  } else {
    // 需要设备迁移 + 类型转换
    values_cpu_owner = topk_values->ToCPU();
    if (values_cpu_owner->Type() != Model::DataType::kFloat32) {
      values_cpu_owner = values_cpu_owner->ToType(Model::DataType::kFloat32);
    }
    values_ptr = values_cpu_owner->Data<float>();
  }
  
  // 处理 indices 必须是 int64 且在 CPU
  if (topk_indices->IsCPU() && topk_indices->Type() == Model::DataType::kInt64) {
    indices_ptr = topk_indices->Data<int64_t>();
  } else if (topk_indices->IsCPU()) {
    indices_cpu_owner = topk_indices->ToType(Model::DataType::kInt64);
    indices_ptr = indices_cpu_owner->Data<int64_t>();
  } else {
    indices_cpu_owner = topk_indices->ToCPU();
    if (indices_cpu_owner->Type() != Model::DataType::kInt64) {
      indices_cpu_owner = indices_cpu_owner->ToType(Model::DataType::kInt64);
    }
    indices_ptr = indices_cpu_owner->Data<int64_t>();
  }
  
  // 检查温度参数有效性
  if (temperature <= 1e-6f) {
    temperature = 1.0f;
  }
  
  // 使用线程局部缓冲区，避免每次分配
  auto& probs = GetThreadLocalProbsBuffer();
  if (probs.size() < static_cast<size_t>(k)) {
    probs.resize(static_cast<size_t>(k));
  }
  
  // ================================================================
  //   topk_values = topk_values - np.max(topk_values)
  //   probs = np.exp(topk_values)
  //   probs /= np.sum(probs)
  // ================================================================
  
  // 找最大值 (数值稳定性)
  float max_val = values_ptr[0];
  for (int64_t i = 1; i < k; ++i) {
    if (values_ptr[i] > max_val) max_val = values_ptr[i];
  }
  
  // 计算 exp((x - max) / temperature) 并求和
  // Python: topk_values = topk_values - np.max(topk_values)
  //         probs = np.exp(topk_values)
  float sum = 0.0f;
  for (int64_t i = 0; i < k; ++i) {
    float logit = values_ptr[i] - max_val;  // 先减去最大值
    if (temperature != 1.0f && temperature > 1e-6f) {
      logit /= temperature;  // 只有当 temperature != 1.0 时才除以温度
    }
    // Clamp 防止 exp 溢出/下溢
    if (logit > 50.0f) {
      logit = 50.0f;
    } else if (logit < -50.0f) {
      logit = -50.0f;
    }
    probs[i] = std::exp(logit);
    // 检查 NaN/Inf
    if (!std::isfinite(probs[i])) {
      probs[i] = 0.0f;
    }
    sum += probs[i];
  }
  
  // 归一化
  if (sum > 1e-10f) {
    float inv_sum = 1.0f / sum;
    for (int64_t i = 0; i < k; ++i) {
      probs[i] *= inv_sum;
    }
  } else {
    // 均匀分布 fallback（如果频繁触发就是有问题）
    float uniform = 1.0f / static_cast<float>(k);
    for (int64_t i = 0; i < k; ++i) {
      probs[i] = uniform;
    }
  }
  
  // 全局 RNG 多项式采样
  try {
    auto& rng = GetGlobalRNG();
    std::discrete_distribution<int> dist(probs.begin(), probs.begin() + k);
    int choice = dist(rng);
    return indices_ptr[choice];
  } catch (const std::exception& e) {
    PrintError("[SampleTopK] discrete_distribution failed: {}", e.what());
    // Fallback: argmax
    int max_idx = 0;
    float max_prob = probs[0];
    for (int64_t i = 1; i < k; ++i) {
      if (probs[i] > max_prob) {
        max_prob = probs[i];
        max_idx = static_cast<int>(i);
      }
    }
    return indices_ptr[max_idx];
  }
}

}  // namespace Utils
}  // namespace GPTSoVITS
