//
// Created by iFlow CLI on 2026/2/22.
// TensorRT 后端预留接口
//

#ifndef GPT_SOVITS_CPP_TENSORRT_BACKEND_H
#define GPT_SOVITS_CPP_TENSORRT_BACKEND_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "GPTSoVITS/model/backend/backend_config.h"
#include "GPTSoVITS/model/base.h"
#include "GPTSoVITS/model/device.h"

namespace GPTSoVITS::Model {

/**
 * @brief 后端类型枚举
 */
enum class BackendType {
  kONNX,       // ONNX Runtime
  kTensorRT,   // TensorRT (预留)
  kOpenVINO,   // OpenVINO (预留)
  kCustom      // 自定义后端
};

/**
 * @brief INT8 量化配置
 */
struct INT8QuantConfig {
  // 是否启用 INT8 量化
  bool enabled = false;

  // 校准数据路径
  std::string calibration_data_path;

  // 校准批次大小
  int calibration_batch_size = 1;

  // 校准方法
  enum class CalibrationMethod {
    kMinMax,     // Min-Max 校准
    kEntropy,    // 熵校准 (推荐)
    kPercentile  // 百分位校准
  };
  CalibrationMethod calibration_method = CalibrationMethod::kEntropy;

  // 每层量化配置（可选）
  struct LayerQuantConfig {
    std::string layer_name;
    bool quantize_input = true;
    bool quantize_output = true;
    // 指定该层的精度，空则使用全局配置
    std::string weight_precision;
    std::string activation_precision;
  };
  std::vector<LayerQuantConfig> layer_configs;
};

/**
 * @brief TensorRT 专用配置
 */
struct TensorRTConfig {
  // 基础后端配置
  BackendConfig base_config;

  // INT8 量化配置
  INT8QuantConfig int8_config;

  // TensorRT 优化级别 (0-5)
  int optimization_level = 3;

  // 是否启用 FP16 精度
  bool enable_fp16 = true;

  // 是否启用 INT8 精度
  bool enable_int8 = false;

  // 是否启用 TF32 精度 (Ampere+ GPU)
  bool enable_tf32 = true;

  // 是否启用 CUDA Graph
  bool enable_cuda_graph = false;

  // 是否允许 GPU 内存增长
  bool allow_gpu_memory_growth = true;

  // DLA (Deep Learning Accelerator) 配置
  struct DLAConfig {
    bool enabled = false;
    int device_id = 0;
    bool allow_gpu_fallback = true;
  };
  DLAConfig dla_config;

  // 最大工作空间大小 (字节)
  int64_t max_workspace_size = 1LL << 30;  // 1GB

  // 最小/最优/最大优化配置 (动态形状)
  struct OptimizationProfile {
    std::vector<int64_t> min_shape;
    std::vector<int64_t> opt_shape;
    std::vector<int64_t> max_shape;
  };
  std::vector<OptimizationProfile> optimization_profiles;

  // TensorRT 引擎缓存路径
  std::string engine_cache_path;

  // 是否强制重新构建引擎
  bool force_rebuild = false;

  /**
   * @brief 验证配置
   */
  void Validate() {
    base_config.Validate();
    if (optimization_level < 0) optimization_level = 0;
    if (optimization_level > 5) optimization_level = 5;
    if (max_workspace_size < 0) max_workspace_size = 0;
  }

  /**
   * @brief 获取默认 TensorRT 配置
   */
  static TensorRTConfig GetDefault() {
    TensorRTConfig config;
    config.base_config.device = Device(DeviceType::kCUDA, 0);
    config.base_config.precision = PrecisionMode::kAuto;
    config.Validate();
    return config;
  }

  /**
   * @brief 获取高性能 TensorRT 配置
   */
  static TensorRTConfig GetHighPerformance() {
    TensorRTConfig config;
    config.base_config.device = Device(DeviceType::kCUDA, 0);
    config.base_config.precision = PrecisionMode::kFP16;
    config.enable_fp16 = true;
    config.enable_cuda_graph = true;
    config.optimization_level = 5;
    config.max_workspace_size = 4LL << 30;  // 4GB
    config.Validate();
    return config;
  }

  /**
   * @brief 获取 INT8 量化配置
   */
  static TensorRTConfig GetINT8Config(const std::string& calibration_path) {
    TensorRTConfig config;
    config.base_config.device = Device(DeviceType::kCUDA, 0);
    config.base_config.precision = PrecisionMode::kINT8;
    config.enable_fp16 = true;
    config.enable_int8 = true;
    config.int8_config.enabled = true;
    config.int8_config.calibration_data_path = calibration_path;
    config.int8_config.calibration_method = INT8QuantConfig::CalibrationMethod::kEntropy;
    config.Validate();
    return config;
  }
};

/**
 * @brief TensorRT 后端接口 (预留)
 *
 * 未来实现时继承此接口，提供 TensorRT 专用功能
 */
class ITensorRTBackend {
public:
  virtual ~ITensorRTBackend() = default;

  /**
   * @brief 从 ONNX 模型构建 TensorRT 引擎
   * @param onnx_path ONNX 模型路径
   * @param config TensorRT 配置
   * @return 是否成功
   */
  virtual bool BuildFromONNX(const std::string& onnx_path,
                             const TensorRTConfig& config) = 0;

  /**
   * @brief 从序列化数据加载引擎
   * @param engine_path 引擎文件路径
   * @return 是否成功
   */
  virtual bool LoadEngine(const std::string& engine_path) = 0;

  /**
   * @brief 序列化当前引擎到文件
   * @param engine_path 输出文件路径
   * @return 是否成功
   */
  virtual bool SerializeEngine(const std::string& engine_path) = 0;

  /**
   * @brief 执行 INT8 校准
   * @param calibration_data 校准数据目录
   * @return 是否成功
   */
  virtual bool CalibrateINT8(const std::string& calibration_data) = 0;

  /**
   * @brief 获取引擎信息
   */
  virtual std::string GetEngineInfo() const = 0;

  /**
   * @brief 获取支持的精度列表
   */
  virtual std::vector<PrecisionMode> GetSupportedPrecisions() const = 0;
};

/**
 * @brief 后端工厂 (预留)
 *
 * 用于根据后端类型创建相应的模型实例
 */
class BackendFactory {
public:
  /**
   * @brief 创建后端实例
   * @param type 后端类型
   * @return 后端实例指针
   */
  static std::unique_ptr<BaseModel> CreateBackend(BackendType type);

  /**
   * @brief 检查后端是否可用
   * @param type 后端类型
   * @return 是否可用
   */
  static bool IsBackendAvailable(BackendType type);

  /**
   * @brief 获取可用的后端列表
   */
  static std::vector<BackendType> GetAvailableBackends();
};

}  // namespace GPTSoVITS::Model

#endif  // GPT_SOVITS_CPP_TENSORRT_BACKEND_H
