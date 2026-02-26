//
// Created by 19254 on 2026/2/8.
//

#ifndef GPT_SOVITS_CPP_SAMPLE_CONFIG_H
#define GPT_SOVITS_CPP_SAMPLE_CONFIG_H

#include <cstdint>

namespace GPTSoVITS::Model {

/**
 * @brief 采样配置结构
 *
 * 用于控制生成过程中的采样策略，包括 Top-K、Top-P (nucleus) 和温度参数。
 *
 * 默认值：
 * - top_k = 20
 * - top_p = 0.6
 * - temperature = 0.6
 */
struct SampleConfig {
  /**
   * @brief Top-K 采样参数
   *
   * 只从概率最高的 K 个 token 中采样。
   * 设置为 0 表示禁用 Top-K 筛选。
   */
  int32_t top_k = 20;

  /**
   * @brief Top-P (Nucleus) 采样参数
   *
   * 从累积概率达到 P 的最小 token 集合中采样。
   * 范围: [0.0, 1.0]
   * 设置为 1.0 或更大表示禁用 Top-P 筛选。
   */
  float top_p = 0.6f;

  /**
   * @brief 温度参数
   *
   * 控制输出的随机性：
   * - 温度 < 1.0: 输出更集中，更确定性
   * - 温度 = 1.0: 标准采样
   * - 温度 > 1.0: 输出更随机，更多样性
   * 必须大于 0
   */
  float temperature = 0.6f;

  /**
   * @brief 默认配置（python项目默认）
   */
  static SampleConfig Default() {
    return SampleConfig{};
  }

  /**
   * @brief 确定性配置（更少的随机性）
   */
  static SampleConfig Deterministic() {
    return SampleConfig{10, 0.3f, 0.3f};
  }

  /**
   * @brief 创造性配置（更多的随机性）
   */
  static SampleConfig Creative() {
    return SampleConfig{50, 0.9f, 1.0f};
  }

  /**
   * @brief 验证配置有效性
   * @return 是否有效
   */
  bool Validate() const {
    if (top_k < 0) {
      return false;
    }
    if (top_p < 0.0f || top_p > 1.0f) {
      return false;
    }
    if (temperature <= 0.0f) {
      return false;
    }
    return true;
  }

  /**
   * @brief 检查是否使用 Top-K 采样
   */
  [[nodiscard]] bool UseTopK() const {
    return top_k > 0;
  }

  /**
   * @brief 检查是否使用 Top-P 采样
   */
  [[nodiscard]] bool UseTopP() const {
    return top_p < 1.0f;
  }
};

/**
 * @brief 推理性能统计
 */
struct InferStats {
  int    gpt_tokens     = 0;    // GPT 生成的 token 数
  double gpt_time_s     = 0.0;  // GPT 自回归生成耗时（秒）
  double sovits_time_s  = 0.0;  // SoVITS 解码耗时（秒）

  double TokensPerSec() const {
    return gpt_time_s > 0.0 ? gpt_tokens / gpt_time_s : 0.0;
  }
};

}  // namespace GPTSoVITS::Model

#endif  // GPT_SOVITS_CPP_SAMPLE_CONFIG_H