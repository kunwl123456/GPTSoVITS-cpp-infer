//
// Created by 19254 on 2026/2/8.
//
// 说话人特征 - 设备感知的内存管理
//

#ifndef GPT_SOVITS_CPP_SPEAKER_FEATURES_H
#define GPT_SOVITS_CPP_SPEAKER_FEATURES_H

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "GPTSoVITS/model/device.h"
#include "GPTSoVITS/model/tensor.h"

namespace GPTSoVITS {

namespace Bert {
struct BertRes;
}

namespace Core {
class DeviceContext;
}

/**
 * @brief 说话人特征类型
 */
enum class FeatureType {
  kPhoneSeq,      // 音素序列
  kBertSeq,       // BERT 特征序列
  kVQCodes,       // VQ 编码
  kReferSpec,     // 参考频谱图
  kSVEmbedding,   // 说话人验证嵌入
};

/**
 * @brief 说话人元数据
 */
struct SpeakerMetadata {
  std::string name;
  std::string lang;
  std::string created_at;
  std::string model_version;
  int sv_dim = 20480;
  int max_seq_len = 1000;
};

/**
 * @brief 说话人特征
 *
 * @code
 * SpeakerFeatures speaker("firefly", "zh");
 * speaker.SetVQCodes(std::move(vq_codes));
 * 
 * // 在 GPU 上使用
 * Tensor* vq_on_gpu = speaker.GetVQCodes(Device(DeviceType::kCUDA, 0));
 * 
 * // 释放 GPU 缓存，保留 CPU 数据
 * speaker.ReleaseDeviceCache(Device(DeviceType::kCUDA, 0));
 * @endcode
 */
class SpeakerFeatures {
public:
  /**
   * @brief 构造函数
   */
  explicit SpeakerFeatures(const std::string& name, const std::string& lang = "zh");

  /**
   * @brief 从元数据构造
   */
  explicit SpeakerFeatures(const SpeakerMetadata& metadata);

  /**
   * @brief 析构函数
   */
  ~SpeakerFeatures();

  // 禁止拷贝
  SpeakerFeatures(const SpeakerFeatures&) = delete;
  SpeakerFeatures& operator=(const SpeakerFeatures&) = delete;

  // 允许移动
  SpeakerFeatures(SpeakerFeatures&&) noexcept;
  SpeakerFeatures& operator=(SpeakerFeatures&&) noexcept;

  // ============ 基本信息 ============

  /**
   * @brief 获取说话人名称
   */
  [[nodiscard]] const std::string& Name() const;

  /**
   * @brief 获取说话人语言
   */
  [[nodiscard]] const std::string& Lang() const;

  /**
   * @brief 获取元数据
   */
  [[nodiscard]] const SpeakerMetadata& GetMetadata() const;

  /**
   * @brief 设置元数据
   */
  void SetMetadata(const SpeakerMetadata& metadata);

  // ============ 特征访问（设备感知） ============

  /**
   * @brief 获取音素序列
   * @param target_device 目标设备（如果数据不在该设备上，会自动迁移）
   * @return 音素序列 Tensor 指针
   */
  Model::Tensor* GetPhoneSeq(Model::Device target_device);

  /**
   * @brief 获取 BERT 特征序列
   */
  Model::Tensor* GetBertSeq(Model::Device target_device);

  /**
   * @brief 获取 VQ 编码
   */
  Model::Tensor* GetVQCodes(Model::Device target_device);

  /**
   * @brief 获取参考频谱图
   */
  Model::Tensor* GetReferSpec(Model::Device target_device);

  /**
   * @brief 获取说话人验证嵌入
   */
  Model::Tensor* GetSVEmbedding(Model::Device target_device);

  // ============ 特征设置 ============

  /**
   * @brief 设置音素序列
   */
  void SetPhoneSeq(std::unique_ptr<Model::Tensor> tensor);

  /**
   * @brief 设置 BERT 特征序列
   */
  void SetBertSeq(std::unique_ptr<Model::Tensor> tensor);

  /**
   * @brief 设置 VQ 编码
   */
  void SetVQCodes(std::unique_ptr<Model::Tensor> tensor);

  /**
   * @brief 设置参考频谱图
   */
  void SetReferSpec(std::unique_ptr<Model::Tensor> tensor);

  /**
   * @brief 设置说话人验证嵌入
   */
  void SetSVEmbedding(std::unique_ptr<Model::Tensor> tensor);

  /**
   * @brief 从 BertRes 设置
   */
  void SetFromBertRes(std::shared_ptr<Bert::BertRes> bert_res);

  // ============ 设备管理 ============

  /**
   * @brief 迁移到指定设备（保留 CPU 原始数据，创建设备缓存）
   */
  void EnsureOnDevice(Model::Device device);

  /**
   * @brief 释放设备缓存（保留 CPU 原始数据）
   */
  void ReleaseDeviceCache(Model::Device device);

  /**
   * @brief 释放所有设备缓存
   */
  void ReleaseAllDeviceCaches();

  /**
   * @brief 检查特征是否在指定设备上可用
   */
  [[nodiscard]] bool IsAvailableOnDevice(Model::Device device) const;

  // ============ 内存管理 ============

  /**
   * @brief 获取内存使用量
   */
  [[nodiscard]] size_t GetMemoryUsage() const;

  /**
   * @brief 获取设备缓存内存使用量
   */
  [[nodiscard]] size_t GetDeviceCacheMemoryUsage(Model::Device device) const;

  // ============ 序列化支持 ============

  /**
   * @brief 序列化到字节缓冲区
   */
  bool SerializeToBuffer(std::vector<uint8_t>& buffer) const;

  /**
   * @brief 从字节缓冲区反序列化
   */
  static std::unique_ptr<SpeakerFeatures> DeserializeFromBuffer(
      const std::vector<uint8_t>& buffer);

  /**
   * @brief 获取 BertRes
   */
  [[nodiscard]] std::shared_ptr<Bert::BertRes> GetBertRes() const;

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace GPTSoVITS

#endif  // GPT_SOVITS_CPP_SPEAKER_FEATURES_H
