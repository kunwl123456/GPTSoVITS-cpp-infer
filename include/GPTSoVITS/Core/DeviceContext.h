//
// Created by 19254 on 2026/2/8.
//
// 设备上下文 - 统一管理设备选择、内存池、流同步
//

#ifndef GPT_SOVITS_CPP_DEVICE_CONTEXT_H
#define GPT_SOVITS_CPP_DEVICE_CONTEXT_H

#include <cstddef>
#include <memory>
#include <string>

#include "GPTSoVITS/model/device.h"
#include "GPTSoVITS/model/tensor.h"

namespace GPTSoVITS::Core {

/**
 * @brief 设备上下文配置
 */
struct DeviceConfig {
  Model::DeviceType preferred_device = Model::DeviceType::kCPU;
  int device_id = 0;

  // 计算精度
  Model::DataType compute_precision = Model::DataType::kFloat32;

  // 线程数
  int thread_num = 1;

  // 内存池配置
  bool enable_memory_pool = false;
  size_t memory_pool_size_mb = 512;

  // 流配置
  bool enable_stream = false;

  // 打印设备信息
  bool verbose = true;
};

/**
 * @brief 设备能力信息
 */
struct DeviceCapabilities {
  bool supports_fp16 = false;
  bool supports_int8 = false;
  bool supports_tensorrt = false;
  size_t total_memory = 0;
  size_t available_memory = 0;
  std::string device_name;
  int compute_capability_major = 0;
  int compute_capability_minor = 0;
};

/**
 * @brief 设备上下文 - 统一管理设备相关操作
 */
class DeviceContext {
public:
  /**
   * @brief 构造函数
   * @param config 设备配置
   */
  explicit DeviceContext(const DeviceConfig& config = {});

  /**
   * @brief 析构函数
   */
  ~DeviceContext();

  // 禁止拷贝
  DeviceContext(const DeviceContext&) = delete;
  DeviceContext& operator=(const DeviceContext&) = delete;

  // 允许移动
  DeviceContext(DeviceContext&&) noexcept;
  DeviceContext& operator=(DeviceContext&&) noexcept;

  // ============ 设备信息 ============

  /**
   * @brief 获取当前设备
   */
  [[nodiscard]] Model::Device GetDevice() const;

  /**
   * @brief 获取设备类型
   */
  [[nodiscard]] Model::DeviceType GetDeviceType() const;

  /**
   * @brief 获取设备ID
   */
  [[nodiscard]] int GetDeviceId() const;

  /**
   * @brief 获取计算精度
   */
  [[nodiscard]] Model::DataType GetComputePrecision() const;

  /**
   * @brief 设置计算精度
   */
  void SetComputePrecision(Model::DataType precision);

  /**
   * @brief 获取设备能力
   */
  [[nodiscard]] const DeviceCapabilities& GetCapabilities() const;

  /**
   * @brief 获取配置
   */
  [[nodiscard]] const DeviceConfig& GetConfig() const;

  // ============ 同步操作 ============

  /**
   * @brief 同步设备（等待所有操作完成）
   */
  void Synchronize();

  /**
   * @brief 创建 CUDA 流（如果启用）
   * @return 流指针
   */
  void* GetOrCreateStream();

  // ============ 内存管理 ============

  /**
   * @brief 分配内存
   * @param size 字节数
   * @return 内存指针
   */
  void* Allocate(size_t size);

  /**
   * @brief 释放内存
   * @param ptr 内存指针
   */
  void Deallocate(void* ptr);

  /**
   * @brief 获取已分配内存大小
   */
  [[nodiscard]] size_t GetAllocatedSize() const;

  // ============ Tensor 工厂方法 ============

  /**
   * @brief 在当前设备上创建空 Tensor
   */
  std::unique_ptr<Model::Tensor> CreateTensor(
      const std::vector<int64_t>& shape,
      Model::DataType dtype);

  /**
   * @brief 将 Tensor 移动到当前设备
   */
  std::unique_ptr<Model::Tensor> ToDevice(const Model::Tensor* tensor);

  /**
   * @brief 将 Tensor 转换为当前计算精度
   */
  std::unique_ptr<Model::Tensor> ToPrecision(const Model::Tensor* tensor);

  // ============ 静态方法 ============

  /**
   * @brief 检测设备能力
   */
  static DeviceCapabilities DetectCapabilities(Model::DeviceType type, int device_id);

  /**
   * @brief 获取可用设备数量
   */
  static int GetDeviceCount(Model::DeviceType type);

  /**
   * @brief 设置默认设备上下文
   */
  static void SetDefault(DeviceContext* ctx);

  /**
   * @brief 获取默认设备上下文
   */
  static DeviceContext* GetDefault();

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace GPTSoVITS::Core

#endif  // GPT_SOVITS_CPP_DEVICE_CONTEXT_H
