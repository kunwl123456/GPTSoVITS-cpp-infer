//
// Created by iFlow CLI on 2026/2/22.
// TensorRT 后端预留实现
//

#include "GPTSoVITS/model/backend/tensorrt_backend.h"

#include "GPTSoVITS/plog.h"

namespace GPTSoVITS::Model {

// ============================================================================
// BackendFactory 实现
// ============================================================================

std::unique_ptr<BaseModel> BackendFactory::CreateBackend(BackendType type) {
  switch (type) {
    case BackendType::kONNX:
      // ONNX 后端已实现，由各模型类直接使用
      PrintWarn("[BackendFactory] ONNX backend is implemented in model classes directly");
      return nullptr;
    case BackendType::kTensorRT:
      PrintWarn("[BackendFactory] TensorRT backend is not yet implemented");
      // TODO: 未来返回 TensorRT 后端实例
      // return std::make_unique<TensorRTBackend>();
      return nullptr;
    case BackendType::kOpenVINO:
      PrintWarn("[BackendFactory] OpenVINO backend is not yet implemented");
      return nullptr;
    case BackendType::kCustom:
      PrintWarn("[BackendFactory] Custom backend requires manual registration");
      return nullptr;
    default:
      PrintError("[BackendFactory] Unknown backend type");
      return nullptr;
  }
}

bool BackendFactory::IsBackendAvailable(BackendType type) {
  switch (type) {
    case BackendType::kONNX:
      return true;  // ONNX Runtime 始终可用
    case BackendType::kTensorRT:
#ifdef WITH_TENSORRT
      return true;
#else
      return false;
#endif
    case BackendType::kOpenVINO:
#ifdef WITH_OPENVINO
      return true;
#else
      return false;
#endif
    case BackendType::kCustom:
      return true;  // 自定义后端始终可用
    default:
      return false;
  }
}

std::vector<BackendType> BackendFactory::GetAvailableBackends() {
  std::vector<BackendType> available;
  if (IsBackendAvailable(BackendType::kONNX)) {
    available.push_back(BackendType::kONNX);
  }
  if (IsBackendAvailable(BackendType::kTensorRT)) {
    available.push_back(BackendType::kTensorRT);
  }
  if (IsBackendAvailable(BackendType::kOpenVINO)) {
    available.push_back(BackendType::kOpenVINO);
  }
  available.push_back(BackendType::kCustom);
  return available;
}

}  // namespace GPTSoVITS::Model
