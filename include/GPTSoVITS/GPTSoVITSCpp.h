//
// Created by Huiyicc on 2026/1/17.
//

#ifndef GPT_SOVITS_CPP_GPTSOVITSCPP_H
#define GPT_SOVITS_CPP_GPTSOVITSCPP_H

#include <filesystem>
#include <memory>
#include <string>

#include "GPTSoVITS/G2P/Pipline.h"
#include "GPTSoVITS/PipelineBase.h"
#include "GPTSoVITS/Text/Sentence.h"
#include "GPTSoVITS/model/gpt_encoder.h"
#include "GPTSoVITS/model/gpt_step.h"
#include "GPTSoVITS/model/sovits.h"
#include "GPTSoVITS/model/spectrogram.h"
#include "GPTSoVITS/model/sv_embedding.h"
#include "GPTSoVITS/model/ssl.h"
#include "GPTSoVITS/model/vq.h"


namespace GPTSoVITS {

void SetGlobalResourcesPath(const std::string& path);
std::filesystem::path GetGlobalResourcesPath();

/**
 * @brief 完整版 Pipeline
 *
 * 支持说话人创建和推理功能，包含所有模型
 */
class GPTSoVITSPipline : public PipelineBase {
public:
  explicit GPTSoVITSPipline(
      const std::string& config,
      const std::shared_ptr<G2P::G2PPipline>& g2p_pipline,
      const std::shared_ptr<Model::SSLModel>& ssl_model,
      const std::shared_ptr<Model::VQModel>& vq_model,
      const std::shared_ptr<Model::SpectrogramModel>& spectrogram_model,
      const std::shared_ptr<Model::SVEmbeddingModel>& sv_embedding_model,
      const std::shared_ptr<Model::GPTEncoderModel>& gpt_encoder_model,
      const std::shared_ptr<Model::GPTStepModel>& gpt_step_model,
      const std::shared_ptr<Model::SoVITSModel>& sovits_model);

  ~GPTSoVITSPipline() override = default;

  /**
   * @brief 创建说话人
   */
  const SpeakerInfo& CreateSpeaker(const std::string& speaker_name,
                                   const std::string& ref_audio_lang,
                                   const std::filesystem::path& ref_audio_path,
                                   const std::string& ref_audio_text);

  /**
   * @brief 推理说话人
   */
  std::unique_ptr<AudioTools> InferSpeaker(const std::string& speaker_name,
                                           const std::string& text,
                                           const std::string& text_lang = "zh",
                                           float temperature = 1.0f,
                                           float noise_scale = 0.5f,
                                           float speed = 1.0f);

  /**
   * @brief 导出说话人数据到文件
   */
  bool ExportSpeaker(const std::string& speaker_name,
                     const std::string& output_path,
                     bool include_audio = false);

  /**
   * @brief 从文件导入说话人数据
   */
  bool ImportSpeaker(const std::string& input_path,
                     const std::string& speaker_name = "");

private:
  std::shared_ptr<G2P::G2PPipline> m_g2p_pipline;
  std::shared_ptr<Model::SSLModel> m_ssl_model;
  std::shared_ptr<Model::VQModel> m_vq_model;
  std::shared_ptr<Model::SpectrogramModel> m_spectrogram_model;
  std::shared_ptr<Model::SVEmbeddingModel> m_sv_embedding_model;
  std::shared_ptr<Model::GPTEncoderModel> m_gpt_encoder_model;
  std::shared_ptr<Model::GPTStepModel> m_gpt_step_model;
  std::shared_ptr<Model::SoVITSModel> m_sovits_model;

  // Helper methods
  static int64_t SampleTopK(const Model::Tensor* topk_values,
                            const Model::Tensor* topk_indices,
                            float temperature);
  static std::unique_ptr<Model::Tensor> ConcatTensor(const Model::Tensor* a,
                                                     const Model::Tensor* b,
                                                     int axis);

  // 检测模型精度
  void DetectModelPrecision();
};

}  // namespace GPTSoVITS

#endif  // GPT_SOVITS_CPP_GPTSOVITSCPP_H
