//
// TensorRT 推理后端
//

#ifndef GPT_SOVITS_CPP_TENSORRT_BACKEND_H
#define GPT_SOVITS_CPP_TENSORRT_BACKEND_H

#ifdef WITH_CUDA
#include <driver_types.h>



#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "GPTSoVITS/model/backend/backend_config.h"
#include "GPTSoVITS/model/base.h"

namespace GPTSoVITS::Model {

/**
 * @brief TensorRT 专用配置
 */
struct TensorRTConfig {
  BackendConfig base_config;

  // 优化级别 0-5，越高构建越慢但推理越快
  int optimization_level = 3;

  // 精度开关
  bool enable_fp16 = true;
  bool enable_fp8  = false;  // 仅 Hopper/Ada 架构支持，Load 时自动检测
  bool enable_tf32 = false;   // 一般用不上

  // 最大工作空间
  int64_t max_workspace_size = 2LL << 30;  // 2GB

  // 引擎缓存目录；空表示不缓存
  // {model_stem}_{device_id}_{precision}.engine
  std::string engine_cache_dir;

  // 强制重新构建（忽略缓存）
  bool force_rebuild = false;

  static TensorRTConfig GetDefault() {
    TensorRTConfig c;
    c.base_config.device    = Device(DeviceType::kCUDA, 0);
    c.base_config.precision = PrecisionMode::kFP16;
    return c;
  }
};

/**
 * @brief TensorRT 推理后端
 *
 * 规则：
 *   - 传入 .engine 文件 → 直接加载序列化引擎
 *   - 传入 .onnx 文件   → 用 nvonnxparser 构建引擎；
 *                         若配置了 engine_cache_dir 则保存缓存
 *
 * 动态 shape 的 optimization profile 根据模型文件名自动配置
 */
class TensorRTBackend : public BaseModel {
public:
  TensorRTBackend();
  ~TensorRTBackend() override;

  // BaseModel 接口
  bool Load(const std::string& model_path, const Device& device,
            int work_thread_num) override;
  bool Load(const std::string& model_path, const BackendConfig& config) override;

  // TensorRT 专用加载接口
  bool Load(const std::string& model_path, const TensorRTConfig& trt_config);

  void Forward(const std::unordered_map<std::string, Tensor*>& inputs,
               std::unordered_map<std::string, std::unique_ptr<Tensor>>& outputs) override;
  void Forward(const std::unordered_map<std::string, Tensor*>& inputs,
               std::vector<std::unique_ptr<Tensor>>& outputs) override;
  bool ForwardWithPreallocatedOutput(
      const std::unordered_map<std::string, Tensor*>& inputs,
      std::unordered_map<std::string, Tensor*>& outputs) override;

  const std::vector<std::string>& GetInputNames() const override;
  const std::vector<std::string>& GetOutputNames() const override;
  DataType GetInputDataType(const std::string& name) const override;
  DataType GetOutputDataType(const std::string& name) const override;

  void Synchronize();

  cudaStream_t GetStream() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;

  bool BuildFromONNX(const std::string& onnx_path, const TensorRTConfig& cfg);
  bool LoadEngine(const std::string& engine_path);
  bool SaveEngine(const std::string& engine_path);
  void CollectIOMetadata();
  DataType DetermineInputType(DataType model_type) const;

  bool InferCore(const std::unordered_map<std::string, Tensor*>& inputs,
                 const std::unordered_map<std::string, Tensor*>& outputs);
};

/**
 * @brief 后端工厂
 */
class BackendFactory {
public:
  static std::unique_ptr<BaseModel> CreateBackend(BackendType type);
  static bool IsBackendAvailable(BackendType type);
  static std::vector<BackendType> GetAvailableBackends();
};

}  // namespace GPTSoVITS::Model
#endif
#endif  // GPT_SOVITS_CPP_TENSORRT_BACKEND_H
