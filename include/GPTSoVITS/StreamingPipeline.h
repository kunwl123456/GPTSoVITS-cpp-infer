#ifndef GPT_SOVITS_CPP_STREAMING_PIPELINE_H
#define GPT_SOVITS_CPP_STREAMING_PIPELINE_H

#include <functional>
#include <memory>
#include <string>
#include <vector>
#include <deque>

#include "GPTSoVITS/AudioTools.h"
#include "GPTSoVITS/EdgePipeline.h"
#include "GPTSoVITS/model/tensor.h"

namespace GPTSoVITS {

/**
 * @brief 流推理配置
 */
struct StreamingConfig {
  int chunk_length = 24;          // 分块长度（token 数）
  float pause_length = 0.3f;      // 段落间停顿（秒）
  int fade_length = 1280;         // 淡入淡出长度（采样点数）
  int h_len = 512;                // 历史token长度（用于平滑过渡）
  int l_len = 16;                 // 前瞻token长度（用于平滑过渡）
  bool enable_fade = true;        // 是否启用淡入淡出
  bool enable_mute_matrix = false; // 是否使用静音矩阵分割
  float mute_threshold = 0.3f;    // 静音矩阵分割阈值
  std::string mute_matrix_path;   // 静音矩阵文件路径（可选）
};

/**
 * @brief 音频分块
 */
struct AudioChunk {
  std::vector<float> audio_data;  // 音频数据
  bool is_first;                  // 是否是第一个分块
  bool is_last;                   // 是否是最后一个分块
  int segment_index;              // 段落索引
  int chunk_index;                // 分块索引
  float duration;                 // 音频时长（秒）
};

/**
 * @brief 流推理 Pipeline
 *
 */
class StreamingPipeline {
public:
  /**
   * @brief 音频分块回调函数
   * @param chunk 音频分块
   */
  using AudioChunkCallback = std::function<void(const AudioChunk&)>;

  /**
   * @brief 构造函数
   * @param edge_pipeline 边缘推理 Pipeline
   * @param config 流推理配置
   */
  explicit StreamingPipeline(
      std::shared_ptr<EdgePipeline> edge_pipeline,
      const StreamingConfig& config = StreamingConfig());

  virtual ~StreamingPipeline() = default;

  /**
   * @brief 流式推理说话人
   * @param speaker_name 说话人名称
   * @param text 目标文本
   * @param text_lang 文本语言
   * @param callback 音频分块回调
   * @param sample_config 采样配置
   * @param noise_scale 噪声缩放
   * @param speed 速度
   * @return 是否成功
   */
  bool InferSpeakerStreaming(
      const std::string& speaker_name,
      const std::string& text,
      const std::string& text_lang,
      AudioChunkCallback callback,
      const Model::SampleConfig& sample_config,
      float noise_scale = 0.5f,
      float speed = 1.0f);

  /**
   * @brief 设置流推理配置
   * @param config 新的配置
   */
  void SetConfig(const StreamingConfig& config) { m_config = config; }

  /**
   * @brief 获取当前配置
   * @return 当前配置
   */
  const StreamingConfig& GetConfig() const { return m_config; }

  /**
   * @brief 加载静音矩阵（用于智能分割）
   * @param path 静音矩阵文件路径
   * @return 是否成功
   */
  bool LoadMuteMatrix(const std::string& path);

  /**
   * @brief 检查是否已加载静音矩阵
   */
  bool HasMuteMatrix() const { return m_mute_matrix != nullptr; }

private:
  std::shared_ptr<EdgePipeline> m_edge_pipeline;
  StreamingConfig m_config;
  std::unique_ptr<Model::Tensor> m_mute_matrix;  // 静音矩阵 (1025,)

  /**
   * @brief 流式处理单个文本段落
   * 
   * 核心流推理逻辑：边生成边解码
   */
  std::vector<float> ProcessSegmentStreaming(
      const SpeakerInfo& speaker_info,
      const std::string& segment,
      int segment_index,
      AudioChunkCallback callback,
      float temperature,
      float noise_scale,
      float speed,
      const std::vector<float>& prev_fade_out);

  /**
   * @brief 使用静音矩阵查找最佳分割点
   * @param tokens 生成的tokens
   * @param min_length 最小分割长度
   * @return 分割点索引，如果没有找到返回 -1
   */
  int FindBestSplitPoint(const std::vector<int64_t>& tokens, int min_length) const;

  /**
   * @brief 解码音频分块
   * 
   * 使用历史和前瞻token实现平滑过渡
   * 
   * @param chunk_tokens 当前分块的tokens
   * @param history_tokens 历史tokens（用于平滑过渡）
   * @param lookahead_tokens 前瞻tokens（用于平滑过渡）
   * @param speaker_info 说话人信息
   * @param target_phones 目标音素序列
   * @param noise_scale 噪声缩放
   * @param speed 速度
   * @return 解码后的音频数据
   */
  std::vector<float> DecodeChunk(
      const std::vector<int64_t>& chunk_tokens,
      const std::vector<int64_t>& history_tokens,
      const std::vector<int64_t>& lookahead_tokens,
      const SpeakerInfo& speaker_info,
      const std::vector<int64_t>& target_phones,
      float noise_scale,
      float speed);

  /**
   * @brief 应用淡入淡出
   */
  std::vector<float> ApplyFade(
      const std::vector<float>& audio,
      const std::vector<float>& fade_in,
      const std::vector<float>& fade_out);

  /**
   * @brief 生成停顿音频
   */
  std::vector<float> GeneratePause(float duration, int sampling_rate);

  /**
   * @brief 采样top-k token
   */
  static int64_t SampleTopK(
      const Model::Tensor* topk_values,
      const Model::Tensor* topk_indices,
      float temperature);
};

}  // namespace GPTSoVITS

#endif  // GPT_SOVITS_CPP_STREAMING_PIPELINE_H
