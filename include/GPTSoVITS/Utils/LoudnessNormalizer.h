//
// Created by Huiyicc on 25-2-23.
// 响度归一化
//

#ifndef GPT_SOVITS_CPP_LOUDNESSNORMALIZER_H
#define GPT_SOVITS_CPP_LOUDNESSNORMALIZER_H

#include <vector>
#include <cmath>

namespace GPTSoVITS {

/**
 * @brief 响度归一化配置
 */
struct LoudnessConfig {
  // 目标 RMS 值 (0.0 - 1.0)，对应 dBFS:
  // 0.1 ≈ -20dBFS (较低)
  // 0.18 ≈ -15dBFS (中等)
  // 0.25 ≈ -12dBFS (较高)
  float target_rms = 0.18f;
  
  // 最大增益限制 (防止过度放大噪声)
  float max_gain = 10.0f;
  
  // 最小增益限制 (防止过度衰减)
  float min_gain = 0.1f;
  
  // RMS 计算窗口大小 (采样点数，用于流式场景)
  int rms_window_size = 2048;
  
  // 流式增益平滑系数 (0-1，越大越平滑)
  float smoothing_factor = 0.9f;
  
  // 是否启用峰值限制 (防止削波)
  bool enable_peak_limiting = true;
  
  // 峰值限制阈值
  float peak_threshold = 0.95f;
};

/**
 * @brief 响度归一化器
 *
 * 批量模式：对整段音频进行 RMS 归一化
 * 流式模式：逐块处理，平滑增益调整
 */
class LoudnessNormalizer {
public:
  explicit LoudnessNormalizer(const LoudnessConfig& config = LoudnessConfig());
  
  // ============ 批量处理 ============
  
  /**
   * @brief 计算 RMS 值
   * @param samples 音频样本
   * @return RMS 值
   */
  float CalculateRMS(const std::vector<float>& samples) const;
  
  /**
   * @brief 计算峰值
   * @param samples 音频样本
   * @return 峰值绝对值
   */
  float CalculatePeak(const std::vector<float>& samples) const;
  
  /**
   * @brief RMS 归一化 (批量)
   * @param samples 音频样本 (会被修改)
   * @return 应用的增益值
   */
  float NormalizeRMS(std::vector<float>& samples) const;
  
  /**
   * @brief 峰值归一化 (批量)
   * @param samples 音频样本 (会被修改)
   * @param target_peak 目标峰值 (默认 0.9)
   * @return 应用的增益值
   */
  float NormalizePeak(std::vector<float>& samples, float target_peak = 0.9f) const;
  
  /**
   * @brief 组合归一化 (RMS + 峰值限制)
   * 
   * 先进行 RMS 归一化，再进行峰值限制防止削波
   * 
   * @param samples 音频样本 (会被修改)
   * @return 应用的总增益值
   */
  float NormalizeCombined(std::vector<float>& samples) const;
  
  // ============ 流式处理 ============
  
  /**
   * @brief 重置流式状态
   * 
   * 开始新的流式处理前调用
   */
  void Reset();
  
  /**
   * @brief 流式 RMS 归一化
   * 
   * 对每个音频块进行归一化，使用平滑增益避免突变
   * 
   * @param samples 音频样本 (会被修改)
   * @return 当前使用的增益值
   */
  float NormalizeStreaming(std::vector<float>& samples);
  
  /**
   * @brief 获取当前平滑增益
   */
  float GetCurrentGain() const { return smoothed_gain_; }
  
  /**
   * @brief 设置目标 RMS
   */
  void SetTargetRMS(float target_rms);
  
  /**
   * @brief 获取配置
   */
  const LoudnessConfig& GetConfig() const { return config_; }
  
private:
  LoudnessConfig config_;
  float smoothed_gain_ = 1.0f;  // 平滑后的增益值
  float running_rms_ = 0.0f;    // 运行 RMS (用于流式)
  int samples_processed_ = 0;   // 已处理的采样点数
  
  /**
   * @brief 应用增益并进行峰值限制
   */
  void ApplyGainWithPeakLimit(std::vector<float>& samples, float gain) const;
  
  /**
   * @brief 计算平滑增益
   */
  float CalculateSmoothedGain(float target_gain);
};

}  // namespace GPTSoVITS

#endif  // GPT_SOVITS_CPP_LOUDNESSNORMALIZER_H
