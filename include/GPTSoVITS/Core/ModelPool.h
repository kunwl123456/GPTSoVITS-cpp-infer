//
// Created by 19254 on 2026/2/8.
//
// 模型注册池 - 统一管理模型实例，支持共享权重和多设备
//

#ifndef GPT_SOVITS_CPP_MODEL_POOL_H
#define GPT_SOVITS_CPP_MODEL_POOL_H

#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "GPTSoVITS/Core/DeviceContext.h"
#include "GPTSoVITS/model/device.h"
#include "GPTSoVITS/model/tensor.h"

namespace GPTSoVITS::Model {

// 前向声明
class BaseModel;
class CNBertModel;
class SSLModel;
class VQModel;
class SpectrogramModel;
class SVEmbeddingModel;
class GPTEncoderModel;
class GPTStepModel;
class SoVITSModel;

/**
 * @brief 模型类型枚举
 */
enum class ModelType {
  kBert,           // BERT 模型
  kSSL,            // SSL 模型
  kVQ,             // VQ 编码器
  kSpectrogram,    // 频谱图
  kSVEmbedding,    // 说话人验证嵌入
  kGPTEncoder,     // GPT 编码器
  kGPTStep,        // GPT 步进模型
  kSoVITS,         // SoVITS 声码器
};

/**
 * @brief 模型配置
 */
struct ModelConfig {
  std::string path;                         // 模型路径
  Device device = Device(DeviceType::kCPU);  // 运行设备
  int thread_num = 1;                        // 线程数
  DataType precision = DataType::kFloat32;   // 精度
  bool enable_cache = true;                  // 是否缓存模型实例
  bool lazy_load = false;                    // 是否延迟加载

  // tokenizer 路径（BERT 需要）
  std::string tokenizer_path;
};

/**
 * @brief 模型组定义
 */
enum class ModelGroup {
  kAll,        // 全部模型
  kInference,  // 仅推理模型 (BERT, GPTEncoder, GPTStep, SoVITS)
  kCreation,   // 仅说话人创建模型 (SSL, VQ, Spectrogram, SVEmbedding)
};

/**
 * @brief 获取模型组包含的模型类型
 */
std::vector<ModelType> GetModelTypesInGroup(ModelGroup group);

/**
 * @brief 模型类型转字符串
 */
std::string ModelTypeToString(ModelType type);

/**
 * @brief 字符串转模型类型
 */
ModelType StringToModelType(const std::string& str);

/**
 * @brief 模型注册池 - 统一管理模型实例
 *
 * 单例模式，全局唯一
 * 模型共享，避免重复加载
 * 设备感知，自动管理模型设备
 * 线程安全
 *
 * @code
 * auto& pool = ModelPool::Instance();
 * pool.SetDeviceContext(&device_ctx);
 * pool.RegisterModel(ModelType::kSoVITS, config);
 * auto sovits = pool.GetModel<SoVITSModel>(ModelType::kSoVITS);
 * @endcode
 */
class ModelPool {
public:
  /**
   * @brief 获取单例实例
   */
  static ModelPool& Instance();

  // 禁止拷贝和移动
  ModelPool(const ModelPool&) = delete;
  ModelPool& operator=(const ModelPool&) = delete;

  // ============ 配置 ============

  /**
   * @brief 设置设备上下文
   */
  void SetDeviceContext(Core::DeviceContext* ctx);

  /**
   * @brief 获取设备上下文
   */
  [[nodiscard]] Core::DeviceContext* GetDeviceContext() const;

  // ============ 模型注册 ============

  /**
   * @brief 注册模型配置
   * @param type 模型类型
   * @param config 模型配置
   */
  void RegisterModel(ModelType type, const ModelConfig& config);

  /**
   * @brief 批量注册模型配置
   */
  void RegisterModels(const std::unordered_map<ModelType, ModelConfig>& configs);

  /**
   * @brief 注册模型组
   * @param group 模型组
   * @param base_path 模型基础路径
   * @param device 设备
   * @param precision 精度
   */
  void RegisterModelGroup(
      ModelGroup group,
      const std::string& base_path,
      const Device& device,
      DataType precision = DataType::kFloat32);

  // ============ 模型获取 ============

  /**
   * @brief 获取模型实例
   * @tparam ModelT 模型类型
   * @param type 模型类型标识
   * @return 模型共享指针（如果不存在则创建）
   */
  template <typename ModelT>
  std::shared_ptr<ModelT> GetModel(ModelType type);

  /**
   * @brief 获取底层 BaseModel 指针
   */
  std::shared_ptr<BaseModel> GetBaseModel(ModelType type);

  // ============ 模型管理 ============

  /**
   * @brief 预加载模型组
   */
  void PreloadModelGroup(ModelGroup group);

  /**
   * @brief 卸载模型（释放内存）
   */
  void UnloadModel(ModelType type);

  /**
   * @brief 卸载模型组
   */
  void UnloadModelGroup(ModelGroup group);

  /**
   * @brief 卸载所有模型
   */
  void UnloadAll();

  // ============ 状态查询 ============

  /**
   * @brief 检查模型是否已加载
   */
  [[nodiscard]] bool HasModel(ModelType type) const;

  /**
   * @brief 检查模型是否已注册
   */
  [[nodiscard]] bool IsRegistered(ModelType type) const;

  /**
   * @brief 获取已加载模型数量
   */
  [[nodiscard]] size_t LoadedModelCount() const;

  /**
   * @brief 获取内存使用量（估计值）
   */
  [[nodiscard]] size_t GetMemoryUsage() const;

  /**
   * @brief 获取模型配置
   */
  [[nodiscard]] const ModelConfig* GetConfig(ModelType type) const;

private:
  ModelPool();
  ~ModelPool();

  // 内部创建模型
  std::shared_ptr<void> CreateModel(ModelType type);

  // 获取模型路径
  static std::string GetModelFileName(ModelType type);

  mutable std::mutex mutex_;
  Core::DeviceContext* device_ctx_ = nullptr;

  std::unordered_map<ModelType, ModelConfig> configs_;
  std::unordered_map<ModelType, std::shared_ptr<void>> models_;
};

// ============ 模板实现 ============

template <typename ModelT>
std::shared_ptr<ModelT> ModelPool::GetModel(ModelType type) {
  std::lock_guard<std::mutex> lock(mutex_);

  // 检查是否已加载
  auto it = models_.find(type);
  if (it != models_.end()) {
    return std::static_pointer_cast<ModelT>(it->second);
  }

  // 检查是否已注册
  auto config_it = configs_.find(type);
  if (config_it == configs_.end()) {
    return nullptr;
  }

  // 创建模型
  auto model = CreateModel(type);
  if (!model) {
    return nullptr;
  }

  models_[type] = model;
  return std::static_pointer_cast<ModelT>(model);
}

}  // namespace GPTSoVITS::Model

#endif  // GPT_SOVITS_CPP_MODEL_POOL_H
