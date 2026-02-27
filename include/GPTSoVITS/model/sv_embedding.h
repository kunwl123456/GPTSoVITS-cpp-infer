//
// Created by 回忆 on 2026/2/19.
//

#ifndef GSV_CPP_SV_EMBEDDING_H
#define GSV_CPP_SV_EMBEDDING_H

#include <memory>
#include <vector>

#include "GPTSoVITS/model/base.h"
#include "GPTSoVITS/model/tensor.h"
#include "GPTSoVITS/Utils/exception.h"
#include "GPTSoVITS/model/backend/backend_config.h"

namespace GPTSoVITS {
class AudioTools;
}
namespace GPTSoVITS::Model {
class SVEmbeddingModel {
protected:
  std::unique_ptr<BaseModel> m_model;

public:
  virtual ~SVEmbeddingModel() = default;

  template <typename MODEL_BACKEND>
  void Init(const std::string& model_path,
            const Device& device = DeviceType::kCPU, int work_thread_num = 1) {
    m_model = std::make_unique<MODEL_BACKEND>();
    if (!m_model->Load(model_path, device, work_thread_num)) {
      THROW_ERRORN("Failed to load SVEmbeddingModel from: {}", model_path);
    }
  }

  template <typename MODEL_BACKEND>
  void Init(const std::string& model_path, const BackendConfig& config) {
    m_model = std::make_unique<MODEL_BACKEND>();
    if (!m_model->Load(model_path, config)) {
      THROW_ERRORN("Failed to load SVEmbeddingModel from: {}", model_path);
    }
  }

  std::unique_ptr<Tensor>  ComputeEmbedding(const std::vector<float>& audio_16k) const;
};
}  // namespace GPTSoVITS::Model

#endif  // GSV_CPP_SV_EMBEDDING_H
