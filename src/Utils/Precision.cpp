//
// Created by 19254 on 2026/2/22.
//
// 精度上下文管理实现
//

#include "GPTSoVITS/Utils/Precision.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>

#include "GPTSoVITS/model/base.h"
#include "GPTSoVITS/plog.h"

namespace GPTSoVITS {
namespace Utils {

// ============================================================================
// PrecisionContext 实现
// ============================================================================

PrecisionContext::PrecisionContext() 
    : mode_(PrecisionMode::kAuto),
      compute_dtype_(Model::DataType::kFloat32),
      cache_dtype_(Model::DataType::kFloat32),
      output_dtype_(Model::DataType::kFloat32),
      version_("v2"),
      auto_detected_(false) {
}

void PrecisionContext::SetMode(PrecisionMode mode) {
  mode_ = mode;
  auto_detected_ = false;
  
  switch (mode) {
    case PrecisionMode::kFP32:
      compute_dtype_ = Model::DataType::kFloat32;
      cache_dtype_ = Model::DataType::kFloat32;
      output_dtype_ = Model::DataType::kFloat32;
      break;
    case PrecisionMode::kFP16:
      compute_dtype_ = Model::DataType::kFloat16;
      cache_dtype_ = Model::DataType::kFloat16;
      output_dtype_ = Model::DataType::kFloat16;
      break;
    case PrecisionMode::kMixed:
      // 混合精度：计算用FP16，输出用FP32
      compute_dtype_ = Model::DataType::kFloat16;
      cache_dtype_ = Model::DataType::kFloat16;
      output_dtype_ = Model::DataType::kFloat32;
      break;
    case PrecisionMode::kAuto:
      // Auto模式等待检测
      break;
    // case PrecisionMode::kINT8Dyn:
    // case PrecisionMode::kINT8Static:
    //   PrintWarn("[PrecisionContext] INT8 mode not yet implemented, using FP32");
    //   compute_dtype_ = Model::DataType::kFloat32;
    //   cache_dtype_ = Model::DataType::kFloat32;
    //   output_dtype_ = Model::DataType::kFloat32;
    //   break;
  }
}

void PrecisionContext::DetectFromModel(Model::BaseModel* model) {
  if (!model) {
    PrintError("[PrecisionContext] Null model for detection");
    return;
  }
  
  // 获取输入名称列表
  auto input_names = model->GetInputNames();
  std::unordered_map<std::string, std::string> input_types;
  
  for (const auto& name : input_names) {
    // 直接使用 DataType，转换为字符串
    auto dtype = model->GetInputDataType(name);
    input_types[name] = DataTypeToString(dtype);
  }
  
  DetectFromInputTypes(input_types);
}

void PrecisionContext::DetectFromInputTypes(
    const std::unordered_map<std::string, std::string>& input_types) {
  
  // 检测 bert_feature 的类型来确定模型精度
  //   for node in self.sess_gpt_enc.get_inputs():
  //     if node.name == "bert_feature" and node.type == 'tensor(float16)':
  //       self.precision = np.float16
  
  auto it = input_types.find("bert_feature");
  if (it != input_types.end()) {
    std::string type_str = it->second;
    // 转换为小写便于匹配
    std::transform(type_str.begin(), type_str.end(), type_str.begin(), ::tolower);
    
    if (type_str.find("float16") != std::string::npos ||
        type_str.find("fp16") != std::string::npos) {
      compute_dtype_ = Model::DataType::kFloat16;
      PrintInfo("[PrecisionContext] Detected FP16 model inputs. Enabling FP16 mode.");
    } else {
      compute_dtype_ = Model::DataType::kFloat32;
      PrintInfo("[PrecisionContext] Detected FP32 model inputs.");
    }
  }
  
  // 检测 k_cache 的类型来确定 Cache 精度
  auto cache_it = input_types.find("k_cache");
  if (cache_it != input_types.end()) {
    DetectCacheDType(cache_it->second);
  }
  
  auto_detected_ = true;
}

void PrecisionContext::DetectCacheDType(const std::string& cache_type) {
  std::string type_str = cache_type;
  std::transform(type_str.begin(), type_str.end(), type_str.begin(), ::tolower);
  
  if (type_str.find("float16") != std::string::npos ||
      type_str.find("fp16") != std::string::npos) {
    cache_dtype_ = Model::DataType::kFloat16;
    PrintInfo("[PrecisionContext] Detected FP16 KV Cache.");
  } else {
    cache_dtype_ = Model::DataType::kFloat32;
    PrintInfo("[PrecisionContext] Detected FP32 KV Cache.");
  }
}

bool PrecisionContext::IsCompatible(Model::DataType source, Model::DataType target) const {
  // 相同类型始终兼容
  if (source == target) return true;
  
  // FP16 <-> FP32 兼容
  if ((source == Model::DataType::kFloat16 && target == Model::DataType::kFloat32) ||
      (source == Model::DataType::kFloat32 && target == Model::DataType::kFloat16)) {
    return true;
  }
  
  // 整型之间兼容（需要范围检查）
  if (source == Model::DataType::kInt32 && target == Model::DataType::kInt64) {
    return true;  // int32 -> int64 安全
  }
  if (source == Model::DataType::kInt64 && target == Model::DataType::kInt32) {
    return false;  // int64 -> int32 可能溢出，需要运行时检查
  }
  
  // 浮点到整型不兼容（需要显式转换）
  if ((source == Model::DataType::kFloat32 || source == Model::DataType::kFloat16) &&
      (target == Model::DataType::kInt32 || target == Model::DataType::kInt64)) {
    return false;
  }
  
  return false;
}

std::string PrecisionContext::ToString() const {
  std::ostringstream oss;
  oss << "PrecisionContext{";
  oss << "mode=" << static_cast<int>(mode_);
  oss << ", compute=" << DataTypeToString(compute_dtype_);
  oss << ", cache=" << DataTypeToString(cache_dtype_);
  oss << ", output=" << DataTypeToString(output_dtype_);
  oss << ", version=" << version_;
  oss << ", auto_detected=" << auto_detected_;
  oss << "}";
  return oss.str();
}

Model::DataType PrecisionContext::ParseOnnxType(const std::string& onnx_type) {
  std::string type_str = onnx_type;
  std::transform(type_str.begin(), type_str.end(), type_str.begin(), ::tolower);
  
  if (type_str.find("float16") != std::string::npos || 
      type_str.find("fp16") != std::string::npos) {
    return Model::DataType::kFloat16;
  } else if (type_str.find("float") != std::string::npos ||
             type_str.find("fp32") != std::string::npos) {
    return Model::DataType::kFloat32;
  } else if (type_str.find("int64") != std::string::npos) {
    return Model::DataType::kInt64;
  } else if (type_str.find("int32") != std::string::npos) {
    return Model::DataType::kInt32;
  } else if (type_str.find("int8") != std::string::npos) {
    return Model::DataType::kInt8;
  } else if (type_str.find("uint8") != std::string::npos) {
    return Model::DataType::kUInt8;
  }
  
  PrintWarn("[PrecisionContext] Unknown ONNX type: {}, defaulting to FP32", onnx_type);
  return Model::DataType::kFloat32;
}

std::string PrecisionContext::DataTypeToString(Model::DataType dtype) {
  switch (dtype) {
    case Model::DataType::kFloat32: return "FP32";
    case Model::DataType::kFloat16: return "FP16";
    case Model::DataType::kInt32: return "INT32";
    case Model::DataType::kInt64: return "INT64";
    case Model::DataType::kInt8: return "INT8";
    case Model::DataType::kUInt8: return "UINT8";
    default: return "UNKNOWN";
  }
}

// ============================================================================
// SafeNumericCast 实现
// ============================================================================

template<>
float SafeNumericCast<float, int64_t>(int64_t value) {
  // int64 -> float 可能有精度损失，但不会溢出
  return static_cast<float>(value);
}

template<>
int64_t SafeNumericCast<int64_t, float>(float value) {
  // float -> int64 需要检查范围
  if (!std::isfinite(value)) {
    return 0;
  }
  
  constexpr float min_val = static_cast<float>(std::numeric_limits<int64_t>::min());
  constexpr float max_val = static_cast<float>(std::numeric_limits<int64_t>::max());
  
  if (value < min_val) return std::numeric_limits<int64_t>::min();
  if (value > max_val) return std::numeric_limits<int64_t>::max();
  
  return static_cast<int64_t>(value);
}

template<>
float SafeNumericCast<float, double>(double value) {
  // double -> float 可能有精度损失
  if (!std::isfinite(value)) {
    if (std::isinf(value) && value > 0) return std::numeric_limits<float>::max();
    if (std::isinf(value) && value < 0) return std::numeric_limits<float>::lowest();
    return 0.0f;
  }
  
  return static_cast<float>(value);
}

template<>
int SafeNumericCast<int, float>(float value) {
  if (!std::isfinite(value)) return 0;
  
  if (value < static_cast<float>(std::numeric_limits<int>::min())) {
    return std::numeric_limits<int>::min();
  }
  if (value > static_cast<float>(std::numeric_limits<int>::max())) {
    return std::numeric_limits<int>::max();
  }
  
  return static_cast<int>(value);
}

}  // namespace Utils
}  // namespace GPTSoVITS
