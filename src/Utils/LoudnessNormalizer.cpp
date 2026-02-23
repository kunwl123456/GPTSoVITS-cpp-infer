//
// Created by Huiyicc on 25-2-23.
// 响度归一化
//

#include "GPTSoVITS/Utils/LoudnessNormalizer.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace GPTSoVITS {

LoudnessNormalizer::LoudnessNormalizer(const LoudnessConfig& config)
    : config_(config) {
  // 验证配置
  config_.target_rms = std::clamp(config_.target_rms, 0.01f, 1.0f);
  config_.max_gain = std::max(config_.max_gain, 1.0f);
  config_.min_gain = std::min(config_.min_gain, 1.0f);
  config_.smoothing_factor = std::clamp(config_.smoothing_factor, 0.0f, 0.99f);
}

float LoudnessNormalizer::CalculateRMS(const std::vector<float>& samples) const {
  if (samples.empty()) {
    return 0.0f;
  }
  
  double sum_sq = 0.0;
  for (const auto& s : samples) {
    sum_sq += static_cast<double>(s) * s;
  }
  
  return static_cast<float>(std::sqrt(sum_sq / samples.size()));
}

float LoudnessNormalizer::CalculatePeak(const std::vector<float>& samples) const {
  if (samples.empty()) {
    return 0.0f;
  }
  
  float max_abs = 0.0f;
  for (const auto& s : samples) {
    float abs_val = std::abs(s);
    if (abs_val > max_abs) {
      max_abs = abs_val;
    }
  }
  
  return max_abs;
}

float LoudnessNormalizer::NormalizeRMS(std::vector<float>& samples) const {
  if (samples.empty()) {
    return 1.0f;
  }
  
  float rms = CalculateRMS(samples);
  
  // 防止除以零
  if (rms < 1e-8f) {
    return 1.0f;
  }
  
  // 计算目标增益
  float gain = config_.target_rms / rms;
  
  // 限制增益范围
  gain = std::clamp(gain, config_.min_gain, config_.max_gain);
  
  // 应用增益
  for (auto& s : samples) {
    s *= gain;
  }
  
  return gain;
}

float LoudnessNormalizer::NormalizePeak(std::vector<float>& samples, float target_peak) const {
  if (samples.empty()) {
    return 1.0f;
  }
  
  float peak = CalculatePeak(samples);
  
  if (peak < 1e-8f) {
    return 1.0f;
  }
  
  float gain = target_peak / peak;
  gain = std::clamp(gain, config_.min_gain, config_.max_gain);
  
  for (auto& s : samples) {
    s *= gain;
  }
  
  return gain;
}

float LoudnessNormalizer::NormalizeCombined(std::vector<float>& samples) const {
  if (samples.empty()) {
    return 1.0f;
  }
  
  // 第一步：RMS 归一化
  float rms = CalculateRMS(samples);
  if (rms < 1e-8f) {
    return 1.0f;
  }
  
  float rms_gain = config_.target_rms / rms;
  rms_gain = std::clamp(rms_gain, config_.min_gain, config_.max_gain);
  
  // 应用 RMS 增益
  for (auto& s : samples) {
    s *= rms_gain;
  }
  
  // 第二步：峰值限制 (如果启用)
  float peak_gain = 1.0f;
  if (config_.enable_peak_limiting) {
    float peak = CalculatePeak(samples);
    if (peak > config_.peak_threshold) {
      peak_gain = config_.peak_threshold / peak;
      for (auto& s : samples) {
        s *= peak_gain;
      }
    }
  }
  
  return rms_gain * peak_gain;
}

void LoudnessNormalizer::Reset() {
  smoothed_gain_ = 1.0f;
  running_rms_ = 0.0f;
  samples_processed_ = 0;
}

float LoudnessNormalizer::NormalizeStreaming(std::vector<float>& samples) {
  if (samples.empty()) {
    return smoothed_gain_;
  }
  
  // 计算当前块的 RMS
  float block_rms = CalculateRMS(samples);
  
  // 更新运行 RMS (指数加权移动平均)
  if (samples_processed_ == 0) {
    running_rms_ = block_rms;
  } else {
    // 使用平滑因子更新
    float alpha = 1.0f - config_.smoothing_factor;
    running_rms_ = config_.smoothing_factor * running_rms_ + alpha * block_rms;
  }
  
  // 计算目标增益
  float target_gain = 1.0f;
  if (running_rms_ > 1e-8f) {
    target_gain = config_.target_rms / running_rms_;
    target_gain = std::clamp(target_gain, config_.min_gain, config_.max_gain);
  }
  
  // 平滑增益变化
  smoothed_gain_ = CalculateSmoothedGain(target_gain);
  
  // 应用增益并进行峰值限制
  ApplyGainWithPeakLimit(samples, smoothed_gain_);
  
  samples_processed_ += static_cast<int>(samples.size());
  
  return smoothed_gain_;
}

void LoudnessNormalizer::SetTargetRMS(float target_rms) {
  config_.target_rms = std::clamp(target_rms, 0.01f, 1.0f);
}

float LoudnessNormalizer::CalculateSmoothedGain(float target_gain) {
  // 使用指数平滑避免增益突变
  float alpha = 1.0f - config_.smoothing_factor;
  return config_.smoothing_factor * smoothed_gain_ + alpha * target_gain;
}

void LoudnessNormalizer::ApplyGainWithPeakLimit(std::vector<float>& samples, float gain) const {
  // 首先应用增益
  for (auto& s : samples) {
    s *= gain;
  }
  
  // 如果启用峰值限制，检查并限制
  if (config_.enable_peak_limiting) {
    float peak = CalculatePeak(samples);
    if (peak > config_.peak_threshold) {
      float limit_gain = config_.peak_threshold / peak;
      for (auto& s : samples) {
        s *= limit_gain;
      }
    }
  }
}

}  // namespace GPTSoVITS
