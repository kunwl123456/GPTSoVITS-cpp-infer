//
// Created by 19254 on 2026/2/8.
//
// 说话人管理器 - 统一管理说话人特征
//

#ifndef GPT_SOVITS_CPP_SPEAKER_MANAGER_H
#define GPT_SOVITS_CPP_SPEAKER_MANAGER_H

#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "GPTSoVITS/SpeakerFeatures.h"

namespace GPTSoVITS {

namespace Core {
class DeviceContext;
}

namespace G2P {
class G2PPipline;
}

namespace Model {
class ModelPool;
}

/**
 * @brief 说话人导入选项
 */
struct SpeakerImportOptions {
  std::string rename;          // 重命名
  bool validate = true;        // 是否验证数据包
  bool lazy_load = false;      // 是否延迟加载特征
  bool keep_cpu_copy = true;   // 是否保留 CPU 副本
};

/**
 * @brief 说话人导出选项
 */
struct SpeakerExportOptions {
  bool include_audio = false;  // 是否包含音频数据
  bool compress = false;       // 是否压缩
};

/**
 * @brief 说话人管理器 - 统一管理说话人特征
 *
 * 单例模式
 * 创建、导入、导出说话人
 * 设备感知
 * 线程安全
 *
 * @code
 * auto& manager = SpeakerManager::Instance();
 * manager.SetDeviceContext(&device_ctx);
 * 
 * // 导入说话人
 * auto* speaker = manager.ImportFromPackage("firefly.gsppkg");
 * 
 * // 获取说话人特征
 * Tensor* vq = speaker->GetVQCodes(device);
 * @endcode
 */
class SpeakerManager {
public:
  /**
   * @brief 获取单例实例
   */
  static SpeakerManager& Instance();

  // 禁止拷贝和移动
  SpeakerManager(const SpeakerManager&) = delete;
  SpeakerManager& operator=(const SpeakerManager&) = delete;

  // ============ 配置 ============

  /**
   * @brief 设置设备上下文
   */
  void SetDeviceContext(Core::DeviceContext* ctx);

  /**
   * @brief 设置模型池
   */
  void SetModelPool(Model::ModelPool* pool);

  /**
   * @brief 设置 G2P Pipeline（用于创建说话人时的 BERT 特征提取）
   */
  void SetG2PPipeline(std::shared_ptr<G2P::G2PPipline> pipeline);

  /**
   * @brief 获取设备上下文
   */
  [[nodiscard]] Core::DeviceContext* GetDeviceContext() const;

  // ============ 创建说话人（需要全量模型） ============

  /**
   * @brief 创建说话人
   * @param name 说话人名称
   * @param lang 语言
   * @param ref_audio_path 参考音频路径
   * @param ref_text 参考文本
   * @return 说话人特征指针（失败返回 nullptr）
   */
  SpeakerFeatures* CreateSpeaker(
      const std::string& name,
      const std::string& lang,
      const std::filesystem::path& ref_audio_path,
      const std::string& ref_text);

  // ============ 导入/导出 ============

  /**
   * @brief 从数据包导入说话人
   * @param path 数据包路径
   * @param options 导入选项
   * @return 说话人特征指针（失败返回 nullptr）
   */
  SpeakerFeatures* ImportFromPackage(
      const std::string& path,
      const SpeakerImportOptions& options = {});

  /**
   * @brief 从数据包导入说话人（简化版本）
   * @param path 数据包路径
   * @param rename 重命名（可选）
   * @return 说话人特征指针
   */
  SpeakerFeatures* ImportFromPackage(
      const std::string& path,
      const std::string& rename);

  /**
   * @brief 导出说话人到数据包
   * @param name 说话人名称
   * @param path 输出路径
   * @param options 导出选项
   * @return 是否成功
   */
  bool ExportToPackage(
      const std::string& name,
      const std::string& path,
      const SpeakerExportOptions& options = {});

  // ============ 管理操作 ============

  /**
   * @brief 获取说话人
   */
  SpeakerFeatures* GetSpeaker(const std::string& name);

  /**
   * @brief 检查说话人是否存在
   */
  [[nodiscard]] bool HasSpeaker(const std::string& name) const;

  /**
   * @brief 移除说话人
   */
  bool RemoveSpeaker(const std::string& name);

  /**
   * @brief 列出所有说话人
   */
  [[nodiscard]] std::vector<std::string> ListSpeakers() const;

  /**
   * @brief 获取说话人数量
   */
  [[nodiscard]] size_t SpeakerCount() const;

  // ============ 批量操作 ============

  /**
   * @brief 预加载说话人特征到设备
   */
  void PreloadToDevice(const std::string& name, Model::Device device);

  /**
   * @brief 预加载所有说话人特征到设备
   */
  void PreloadAllToDevice(Model::Device device);

  /**
   * @brief 释放所有说话人的设备缓存
   */
  void ReleaseAllDeviceCaches();

  // ============ 遍历 ============

  /**
   * @brief 遍历所有说话人
   */
  void ForEach(const std::function<void(const std::string&, SpeakerFeatures&)>& callback);

  // ============ 验证 ============

  /**
   * @brief 验证数据包
   */
  [[nodiscard]] static bool ValidatePackage(const std::string& path);

  /**
   * @brief 获取数据包信息
   */
  [[nodiscard]] static std::unique_ptr<SpeakerMetadata> GetPackageInfo(const std::string& path);

private:
  SpeakerManager();
  ~SpeakerManager();

  mutable std::mutex mutex_;
  Core::DeviceContext* device_ctx_ = nullptr;
  Model::ModelPool* model_pool_ = nullptr;
  std::shared_ptr<G2P::G2PPipline> g2p_pipeline_;

  std::unordered_map<std::string, std::unique_ptr<SpeakerFeatures>> speakers_;
};

}  // namespace GPTSoVITS

#endif  // GPT_SOVITS_CPP_SPEAKER_MANAGER_H
