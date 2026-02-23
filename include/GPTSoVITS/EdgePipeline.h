//
// Created by iFlow CLI on 2026/2/20.
//

#ifndef GPT_SOVITS_CPP_EDGE_PIPELINE_H
#define GPT_SOVITS_CPP_EDGE_PIPELINE_H

#include <filesystem>
#include <memory>
#include <string>

#include "GPTSoVITS/G2P/Pipline.h"
#include "GPTSoVITS/PipelineBase.h"
#include "GPTSoVITS/Text/Sentence.h"
#include "GPTSoVITS/model/gpt_encoder.h"
#include "GPTSoVITS/model/gpt_step.h"
#include "GPTSoVITS/model/sovits.h"

namespace GPTSoVITS::Model {
struct SampleConfig;
}

namespace GPTSoVITS {

/**
 * @brief 边缘推理 Pipeline
 *
 * 这是一个轻量级的 Pipeline，只加载推理所需的模型：
 * - BERT (用于目标文本处理)
 * - GPT Encoder (用于编码)
 * - GPT Step (用于生成)
 * - SoVITS (用于音频生成)
 *
 * 不需要的模型（仅 CreateSpeaker 需要）：
 * - SSL, VQ, Spectrogram, SV Embedding
 *
 * 说话人数据通过 ImportSpeaker 从打包文件加载。
 */
class EdgePipeline : public PipelineBase {
public:
  /**
   * @brief 构造函数 - 只加载推理所需的模型
   * @param config 配置 JSON 字符串
   * @param model_path 模型文件目录路径
   * @param g2p_pipline G2P 管道（需要 BERT 处理器）
   * @param gpt_encoder_model GPT 编码器模型
   * @param gpt_step_model GPT 步进模型
   * @param sovits_model SoVITS 模型
   */
  EdgePipeline(const std::string& config, const std::string& model_path,
               const std::shared_ptr<G2P::G2PPipline>& g2p_pipline,
               const std::shared_ptr<Model::GPTEncoderModel>& gpt_encoder_model,
               const std::shared_ptr<Model::GPTStepModel>& gpt_step_model,
               const std::shared_ptr<Model::SoVITSModel>& sovits_model);

  ~EdgePipeline() override = default;

  /**
   * @brief 从文件导入说话人数据
   * @param input_path 说话人数据包路径
   * @param speaker_name 可选的新名称
   * @return 是否成功
   */
  bool ImportSpeaker(const std::string& input_path,
                     const std::string& speaker_name = "");

  /**
   * @brief 推理说话人（与 GPTSoVITSPipline 相同的接口）
   * @param speaker_name 说话人名称
   * @param text 目标文本
   * @param text_lang 文本语言
   * @param temperature 温度参数（已弃用，使用 sample_config）
   * @param noise_scale 噪声缩放
   * @param speed 速度
   * @return 生成的音频
   *
   * @deprecated 请使用带 SampleConfig 参数的版本
   */
  std::unique_ptr<AudioTools> InferSpeaker(const std::string& speaker_name,
                                           const std::string& text,
                                           const std::string& text_lang = "zh",
                                           float temperature = 1.0f,
                                           float noise_scale = 0.5f,
                                           float speed = 1.0f);

  /**
   * @brief 推理说话人（推荐版本，支持完整采样配置）
   * @param speaker_name 说话人名称
   * @param text 目标文本
   * @param text_lang 文本语言
   * @param sample_config 采样配置
   * @param noise_scale 噪声缩放
   * @param speed 速度
   * @return 生成的音频
   */
  std::unique_ptr<AudioTools> InferSpeaker(
      const std::string& speaker_name, const std::string& text,
      const std::string& text_lang, const Model::SampleConfig& sample_config,
      float noise_scale = 0.5f, float speed = 1.0f);

  // ============ 流推理支持方法 ============

  /**
   * @brief 获取 GPT Encoder 模型
   */
  [[nodiscard]] std::shared_ptr<Model::GPTEncoderModel> GetGPTEncoderModel() const {
    return m_gpt_encoder_model;
  }

  /**
   * @brief 获取 GPT Step 模型
   */
  [[nodiscard]] std::shared_ptr<Model::GPTStepModel> GetGPTStepModel() const {
    return m_gpt_step_model;
  }

  /**
   * @brief 获取 SoVITS 模型
   */
  [[nodiscard]] std::shared_ptr<Model::SoVITSModel> GetSoVITSModel() const {
    return m_sovits_model;
  }

  /**
   * @brief 获取 G2P Pipeline
   */
  [[nodiscard]] std::shared_ptr<G2P::G2PPipline> GetG2PPipeline() const {
    return m_g2p_pipline;
  }

  /**
   * @brief 获取模型信息
   * @return 模型信息字符串
   */
  [[nodiscard]] std::string GetModelInfo() const override;

private:
  std::shared_ptr<G2P::G2PPipline> m_g2p_pipline;
  std::shared_ptr<Model::GPTEncoderModel> m_gpt_encoder_model;
  std::shared_ptr<Model::GPTStepModel> m_gpt_step_model;
  std::shared_ptr<Model::SoVITSModel> m_sovits_model;
};

}  // namespace GPTSoVITS

#endif  // GPT_SOVITS_CPP_EDGE_PIPELINE_H
