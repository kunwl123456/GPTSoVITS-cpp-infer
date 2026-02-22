//
// Created by 19254 on 2026/2/8.
//

#include "GPTSoVITS/Core/ModelPool.h"

#include <filesystem>
#include <mutex>

#include "GPTSoVITS/GPTSoVITSCpp.h"
#include "GPTSoVITS/model/CNBertModel.h"
#include "GPTSoVITS/model/backend/onnx_backend.h"
#include "GPTSoVITS/model/gpt_encoder.h"
#include "GPTSoVITS/model/gpt_step.h"
#include "GPTSoVITS/model/sovits.h"
#include "GPTSoVITS/model/spectrogram.h"
#include "GPTSoVITS/model/ssl.h"
#include "GPTSoVITS/model/sv_embedding.h"
#include "GPTSoVITS/model/vq.h"
#include "GPTSoVITS/plog.h"

namespace GPTSoVITS::Model {

// ============ 辅助函数 ============

std::vector<ModelType> GetModelTypesInGroup(ModelGroup group) {
  switch (group) {
    case ModelGroup::kAll:
      return {
          ModelType::kBert,
          ModelType::kSSL,
          ModelType::kVQ,
          ModelType::kSpectrogram,
          ModelType::kSVEmbedding,
          ModelType::kGPTEncoder,
          ModelType::kGPTStep,
          ModelType::kSoVITS
      };
    case ModelGroup::kInference:
      return {
          ModelType::kBert,
          ModelType::kGPTEncoder,
          ModelType::kGPTStep,
          ModelType::kSoVITS
      };
    case ModelGroup::kCreation:
      return {
          ModelType::kSSL,
          ModelType::kVQ,
          ModelType::kSpectrogram,
          ModelType::kSVEmbedding
      };
    default:
      return {};
  }
}

std::string ModelTypeToString(ModelType type) {
  switch (type) {
    case ModelType::kBert: return "bert";
    case ModelType::kSSL: return "ssl";
    case ModelType::kVQ: return "vq";
    case ModelType::kSpectrogram: return "spectrogram";
    case ModelType::kSVEmbedding: return "sv_embedding";
    case ModelType::kGPTEncoder: return "gpt_encoder";
    case ModelType::kGPTStep: return "gpt_step";
    case ModelType::kSoVITS: return "sovits";
    default: return "unknown";
  }
}

ModelType StringToModelType(const std::string& str) {
  if (str == "bert") return ModelType::kBert;
  if (str == "ssl") return ModelType::kSSL;
  if (str == "vq") return ModelType::kVQ;
  if (str == "spectrogram") return ModelType::kSpectrogram;
  if (str == "sv_embedding") return ModelType::kSVEmbedding;
  if (str == "gpt_encoder") return ModelType::kGPTEncoder;
  if (str == "gpt_step") return ModelType::kGPTStep;
  if (str == "sovits") return ModelType::kSoVITS;
  return ModelType::kBert;  // 默认
}

std::string ModelPool::GetModelFileName(ModelType type) {
  switch (type) {
    case ModelType::kBert: return "bert.onnx";
    case ModelType::kSSL: return "ssl.onnx";
    case ModelType::kVQ: return "vq_encoder.onnx";
    case ModelType::kSpectrogram: return "spectrogram.onnx";
    case ModelType::kSVEmbedding: return "sv_embedding.onnx";
    case ModelType::kGPTEncoder: return "gpt_encoder.onnx";
    case ModelType::kGPTStep: return "gpt_step.onnx";
    case ModelType::kSoVITS: return "sovits.onnx";
    default: return "";
  }
}

// ============ ModelPool 实现 ============

ModelPool& ModelPool::Instance() {
  static ModelPool instance;
  return instance;
}

ModelPool::ModelPool() = default;

ModelPool::~ModelPool() = default;

void ModelPool::SetDeviceContext(Core::DeviceContext* ctx) {
  std::lock_guard<std::mutex> lock(mutex_);
  device_ctx_ = ctx;
}

Core::DeviceContext* ModelPool::GetDeviceContext() const {
  return device_ctx_;
}

void ModelPool::RegisterModel(ModelType type, const ModelConfig& config) {
  std::lock_guard<std::mutex> lock(mutex_);
  configs_[type] = config;
  PrintDebug("[ModelPool] Registered model: {} -> {}", ModelTypeToString(type), config.path);
}

void ModelPool::RegisterModels(const std::unordered_map<ModelType, ModelConfig>& configs) {
  for (const auto& [type, config] : configs) {
    RegisterModel(type, config);
  }
}

void ModelPool::RegisterModelGroup(
    ModelGroup group,
    const std::string& base_path,
    const Device& device,
    DataType precision) {
  auto types = GetModelTypesInGroup(group);

  for (auto type : types) {
    ModelConfig config;
    config.path = base_path + "/" + GetModelFileName(type);
    config.device = device;
    config.precision = precision;
    config.lazy_load = true;

    // // BERT 需要 tokenizer
    // if (type == ModelType::kBert) {
    //
    // }

    RegisterModel(type, config);
  }

  PrintInfo("[ModelPool] Registered model group: {} models at {}",
            types.size(), base_path);
}

std::shared_ptr<BaseModel> ModelPool::GetBaseModel(ModelType type) {
  return GetModel<BaseModel>(type);
}

std::shared_ptr<void> ModelPool::CreateModel(ModelType type) {
  auto config_it = configs_.find(type);
  if (config_it == configs_.end()) {
    PrintError("[ModelPool] Model not registered: {}", ModelTypeToString(type));
    return nullptr;
  }

  const auto& config = config_it->second;

  PrintInfo("[ModelPool] Creating model: {} on device {}",
            ModelTypeToString(type),
            config.device.type == DeviceType::kCUDA ? "CUDA" : "CPU");

  try {
    switch (type) {
      case ModelType::kBert: {
        auto model = std::make_shared<CNBertModel>();
        auto tokenizer_path = (GetGlobalResourcesPath() / "bert_tokenizer.json").string();
        model->Init<ONNXBackend>(config.path, tokenizer_path, config.device, config.thread_num);
        return model;
      }
      case ModelType::kSSL: {
        auto model = std::make_shared<SSLModel>();
        model->Init<ONNXBackend>(config.path, config.device, config.thread_num);
        return model;
      }
      case ModelType::kVQ: {
        auto model = std::make_shared<VQModel>();
        model->Init<ONNXBackend>(config.path, config.device, config.thread_num);
        return model;
      }
      case ModelType::kSpectrogram: {
        auto model = std::make_shared<SpectrogramModel>();
        model->Init<ONNXBackend>(config.path, config.device, config.thread_num);
        return model;
      }
      case ModelType::kSVEmbedding: {
        auto model = std::make_shared<SVEmbeddingModel>();
        model->Init<ONNXBackend>(config.path, config.device, config.thread_num);
        return model;
      }
      case ModelType::kGPTEncoder: {
        auto model = std::make_shared<GPTEncoderModel>();
        model->Init<ONNXBackend>(config.path, config.device, config.thread_num);
        return model;
      }
      case ModelType::kGPTStep: {
        auto model = std::make_shared<GPTStepModel>();
        model->Init<ONNXBackend>(config.path, config.device, config.thread_num);
        return model;
      }
      case ModelType::kSoVITS: {
        auto model = std::make_shared<SoVITSModel>();
        model->Init<ONNXBackend>(config.path, config.device, config.thread_num);
        return model;
      }
      default:
        PrintError("[ModelPool] Unknown model type: {}", static_cast<int>(type));
        return nullptr;
    }
  } catch (const std::exception& e) {
    PrintError("[ModelPool] Failed to create model {}: {}",
               ModelTypeToString(type), e.what());
    return nullptr;
  }
}

void ModelPool::PreloadModelGroup(ModelGroup group) {
  auto types = GetModelTypesInGroup(group);
  for (auto type : types) {
    GetModel<BaseModel>(type);
  }
  PrintInfo("[ModelPool] Preloaded {} models", types.size());
}

void ModelPool::UnloadModel(ModelType type) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = models_.find(type);
  if (it != models_.end()) {
    models_.erase(it);
    PrintInfo("[ModelPool] Unloaded model: {}", ModelTypeToString(type));
  }
}

void ModelPool::UnloadModelGroup(ModelGroup group) {
  auto types = GetModelTypesInGroup(group);
  for (auto type : types) {
    UnloadModel(type);
  }
}

void ModelPool::UnloadAll() {
  std::lock_guard<std::mutex> lock(mutex_);
  models_.clear();
  PrintInfo("[ModelPool] Unloaded all models");
}

bool ModelPool::HasModel(ModelType type) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return models_.find(type) != models_.end();
}

bool ModelPool::IsRegistered(ModelType type) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return configs_.find(type) != configs_.end();
}

size_t ModelPool::LoadedModelCount() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return models_.size();
}

size_t ModelPool::GetMemoryUsage() const {
  // 估计值，实际内存使用需要从后端获取
  std::lock_guard<std::mutex> lock(mutex_);
  size_t total = 0;
  // TODO: 实现精确的内存估算
  return total;
}

const ModelConfig* ModelPool::GetConfig(ModelType type) const {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = configs_.find(type);
  return it != configs_.end() ? &it->second : nullptr;
}

}  // namespace GPTSoVITS::Model
