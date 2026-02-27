//
// Created by Huiyicc on 2026/2/17.
//

#ifndef GSV_CPP_SOVITS_H
#define GSV_CPP_SOVITS_H

#include <memory>
#include <vector>

#include "GPTSoVITS/model/base.h"
#include "GPTSoVITS/model/tensor.h"
#include "GPTSoVITS/Utils/exception.h"
#include "GPTSoVITS/model/backend/backend_config.h"

namespace GPTSoVITS::Model {

/**
 * @brief SoVITS Model for final audio generation
 *
 * SoVITS takes the generated semantic tokens and converts them to audio
 * using the reference speaker's timbre information (spectrogram and SV embedding).
 */
class SoVITSModel {
protected:
  std::unique_ptr<BaseModel> m_model;
  int m_sample_rate = 32000;
  int m_hop_length = 320;
  int m_win_length = 1280;
  int m_filter_length = 1024;
  int m_mel_bins = 128;
  int m_sv_dim = 512;  // Default for v2, Pro uses 20480

  // 缓存标量输入 tensor
  std::unique_ptr<Tensor> m_noise_scale_tensor;
  std::unique_ptr<Tensor> m_speed_tensor;
  float m_cached_noise_scale = -1.0f;
  float m_cached_speed = -1.0f;

public:
  SoVITSModel() = default;
  virtual ~SoVITSModel() = default;

  /**
   * @brief Initialize model with backend
   * @tparam MODEL_BACKEND Backend type (e.g., ONNXBackend)
   * @param model_path Path to sovits.onnx
   * @param device Device to run on
   * @param work_thread_num Number of threads
   */
  template <typename MODEL_BACKEND>
  void Init(const std::string& model_path,
            const Device& device = DeviceType::kCPU,
            int work_thread_num = 1) {
    m_model = std::make_unique<MODEL_BACKEND>();
    if (!m_model->Load(model_path, device, work_thread_num)) {
      THROW_ERRORN("Failed to load SoVITSModel from: {}", model_path);
    }
  }

  template <typename MODEL_BACKEND>
  void Init(const std::string& model_path, const BackendConfig& config) {
    m_model = std::make_unique<MODEL_BACKEND>();
    if (!m_model->Load(model_path, config)) {
      THROW_ERRORN("Failed to load SoVITSModel from: {}", model_path);
    }
  }

  /**
   * @brief Run SoVITS inference to generate audio
   *
   * @param pred_semantic Generated semantic tokens [1, 1, seq_len]
   * @param text_seq Target phoneme sequence [1, seq_len]
   * @param refer_spec Reference spectrogram [1, mel_bins, spec_len]
   * @param sv_emb Speaker verification embedding [1, sv_dim]
   * @param noise_scale Noise scale for generation (default 0.5)
   * @param speed Speed factor for generation (default 1.0)
   * @return Generated audio samples as vector of floats
   */
  std::vector<float> Generate(Tensor* pred_semantic,
                              Tensor* text_seq,
                              Tensor* refer_spec,
                              Tensor* sv_emb,
                              float noise_scale = 0.5f,
                              float speed = 1.0f);

  /**
   * @brief Run SoVITS inference returning Tensor
   *
   * @param pred_semantic Generated semantic tokens [1, 1, seq_len]
   * @param text_seq Target phoneme sequence [1, seq_len]
   * @param refer_spec Reference spectrogram [1, mel_bins, spec_len]
   * @param sv_emb Speaker verification embedding [1, sv_dim]
   * @param noise_scale Noise scale for generation (default 0.5)
   * @param speed Speed factor for generation (default 1.0)
   * @return Generated audio as Tensor [1, num_samples]
   */
  std::unique_ptr<Tensor> GenerateTensor(Tensor* pred_semantic,
                                         Tensor* text_seq,
                                         Tensor* refer_spec,
                                         Tensor* sv_emb,
                                         float noise_scale = 0.5f,
                                         float speed = 1.0f);

  /**
   * @brief Get sample rate
   */
  [[nodiscard]] int GetSampleRate() const { return m_sample_rate; }

  /**
   * @brief Get SV embedding dimension
   */
  [[nodiscard]] int GetSVDim() const { return m_sv_dim; }

  /**
   * @brief Set SV embedding dimension (for Pro models)
   */
  void SetSVDim(int dim) { m_sv_dim = dim; }

  /**
   * @brief Get the underlying model
   */
  [[nodiscard]] const BaseModel* GetModel() const { return m_model.get(); }
};

}  // namespace GPTSoVITS::Model

#endif  // GSV_CPP_SOVITS_H