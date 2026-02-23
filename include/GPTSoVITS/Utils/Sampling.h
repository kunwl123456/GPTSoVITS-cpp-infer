//
// Created by 19254 on 2026/2/22.
//
// 采样工具
// 提供数值稳定的softmax实现，避免精度问题
//

#ifndef GPT_SOVITS_CPP_UTILS_SAMPLING_H
#define GPT_SOVITS_CPP_UTILS_SAMPLING_H

#include <cstdint>
#include <memory>
#include <vector>
#include <random>

#include "GPTSoVITS/model/tensor.h"

namespace GPTSoVITS {
namespace Utils {

/**
 * @brief 采样策略枚举
 */
enum class SamplingStrategy {
  kTopK,        ///< 仅Top-K采样
  kTopP,        ///< 仅Top-P采样 (Nucleus Sampling)
  kTopKTopP,    ///< Top-K + Top-P 组合采样
  kGreedy,      ///< 贪婪采样 (argmax)
};

/**
 * @brief 采样配置
 */
struct SamplingConfig {
  SamplingStrategy strategy = SamplingStrategy::kTopK;
  
  // Top-K 参数
  int top_k = 15;
  
  // Top-P 参数 (Nucleus Sampling)
  float top_p = 0.6f;
  
  // 温度参数
  float temperature = 1.0f;
  
  // 验证配置有效性
  bool Validate() const {
    if (top_k <= 0) return false;
    if (top_p <= 0.0f || top_p > 1.0f) return false;
    if (temperature <= 0.0f) return false;
    return true;
  }
};

/**
 * @brief 采样器类
 */
class Sampler {
public:
  Sampler();
  ~Sampler();
  
  /**
   * @brief 从logits张量采样
   * 
   * 自动处理类型转换（FP16/FP32），保证数值稳定性
   * 
   * @param logits 输入logits张量 (可以是FP16或FP32)
   * @param config 采样配置
   * @return 采样得到的token索引
   */
  int64_t Sample(const Model::Tensor* logits, const SamplingConfig& config);
  
  /**
   * @brief 从top-k结果采样
   * 
   * 直接从模型的top-k输出采样，避免额外的排序开销
   * 
   * @param topk_values Top-K的值 (logits)
   * @param topk_indices Top-K的索引
   * @param temperature 温度
   * @return 采样得到的token索引
   */
  int64_t SampleFromTopK(
      const Model::Tensor* topk_values,
      const Model::Tensor* topk_indices,
      float temperature = 1.0f);
  
  /**
   * @brief 零拷贝采样 - 使用预分配的CPU缓冲区
   * 
   * 避免每次采样都分配新的CPU内存
   * 
   * @param logits_gpu GPU上的logits张量
   * @param cpu_buffer 预分配的CPU缓冲区 (足够容纳logits数据)
   * @param config 采样配置
   * @return 采样得到的token索引
   */
  int64_t SampleZeroCopy(
      const Model::Tensor* logits_gpu,
      float* cpu_buffer,
      const SamplingConfig& config);
  
  /**
   * @brief 设置随机种子 (用于可复现性)
   */
  void SetSeed(unsigned int seed);
  
private:
  std::mt19937 rng_;
  
  // 内部工作缓冲区 (用于零拷贝模式)
  std::vector<float> work_probs_;
  std::vector<int64_t> work_indices_;
};

// ============================================================================
// 全局采样
// ============================================================================

/**
 * @brief Top-K采样
 * 
 * 参考 Python 的 sample_topk 实现：
 * - 数值稳定的softmax
 * - 自动处理FP16/FP32类型
 * 
 * @param topk_values Top-K的值
 * @param topk_indices Top-K的索引  
 * @param temperature 温度参数
 * @return 采样得到的token索引
 */
int64_t SampleTopK(
    const Model::Tensor* topk_values,
    const Model::Tensor* topk_indices,
    float temperature = 1.0f);

/**
 * @brief 数值稳定的Softmax计算
 * 
 * 参考 numpy 实现：
 * - 先减最大值
 * - 再计算exp
 * - 最后归一化
 * 
 * @param logits 输入logits
 * @param probs 输出概率 (需预分配)
 * @param size 元素数量
 * @param temperature 温度参数
 */
void StableSoftmax(
    const float* logits,
    float* probs,
    size_t size,
    float temperature = 1.0f);

/**
 * @brief 从概率分布采样
 * 
 * @param probs 概率分布
 * @param size 元素数量
 * @param rng 随机数生成器
 * @return 采样索引
 */
int SampleFromProbs(const float* probs, size_t size, std::mt19937& rng);

}  // namespace Utils
}  // namespace GPTSoVITS

#endif  // GPT_SOVITS_CPP_UTILS_SAMPLING_H
