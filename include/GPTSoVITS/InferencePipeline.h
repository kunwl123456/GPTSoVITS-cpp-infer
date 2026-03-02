//
// Created by 19254 on 2026/2/8.
//
// 统一推理管道 - 支持全量端/推理端场景
//

#ifndef GPT_SOVITS_CPP_INFERENCE_PIPELINE_H
#define GPT_SOVITS_CPP_INFERENCE_PIPELINE_H

#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "AudioTools.h"
#include "GPTSoVITS/Core/DeviceContext.h"
#include "GPTSoVITS/Core/ModelPool.h"
#include "GPTSoVITS/model/backend/backend_config.h"
#include "GPTSoVITS/SpeakerManager.h"
#include "GPTSoVITS/model/sample_config.h"

namespace GPTSoVITS {

class AudioTools;

/**
 * @brief Pipeline 运行模式
 */
enum class PipelineMode {
  kFull,    // 完整模式：加载全部模型，支持创建说话人和推理
  kEdge,    // 边缘模式：仅加载推理模型，从数据包加载说话人
  kStream,  // 流式模式：支持实时音频生成
};

/**
 * @brief Pipeline 配置
 */
struct PipelineConfig {
  // 运行模式
  PipelineMode mode = PipelineMode::kFull;

  // 模型路径
  std::string model_path;

  // 配置文件路径（可选，默认为 model_path/config.json）
  std::string config_path;

  // 设备配置
  Model::DeviceType device_type = Model::DeviceType::kCPU;
  int device_id = 0;
  Model::DataType compute_precision = Model::DataType::kFloat32;

  // 性能配置
  int thread_num = 1;
  bool enable_memory_pool = false;

  // G2P 配置
  std::string resources_path;
  std::string default_lang = "zh";

  // 推理后端配置
  Model::BackendType backend = Model::BackendType::kAuto;
  std::string engine_cache_dir;  // TRT 引擎缓存目录，空表示不缓存

  // 日志配置
  bool verbose = true;

  /**
   * @brief 创建默认配置（完整模式，CPU）
   */
  static PipelineConfig Default() { return {}; }

  /**
   * @brief 创建边缘模式配置
   */
  static PipelineConfig Edge(const std::string& model_path,
                              Model::DeviceType device = Model::DeviceType::kCPU,
                              int device_id = 0) {
    PipelineConfig config;
    config.mode = PipelineMode::kEdge;
    config.model_path = model_path;
    config.device_type = device;
    config.device_id = device_id;
    return config;
  }

  /**
   * @brief 创建完整模式配置
   */
  static PipelineConfig Full(const std::string& model_path,
                              Model::DeviceType device = Model::DeviceType::kCPU,
                              int device_id = 0) {
    PipelineConfig config;
    config.mode = PipelineMode::kFull;
    config.model_path = model_path;
    config.device_type = device;
    config.device_id = device_id;
    return config;
  }

  /**
   * @brief 创建 CUDA 完整模式配置
   */
  static PipelineConfig FullCUDA(const std::string& model_path, int device_id = 0) {
    return Full(model_path, Model::DeviceType::kCUDA, device_id);
  }
};

/**
 * @brief 流式回调类型
 */
using AudioChunkCallback = std::function<void(const AudioChunk&)>;

/**
 * @brief 统一推理管道
 *
 * @code
 * // 完整模式
 * PipelineConfig config = PipelineConfig::FullCUDA("/path/to/models");
 * InferencePipeline pipeline(config);
 * 
 * pipeline.CreateSpeaker("speaker1", "zh", "/path/to/ref.wav", "参考文本");
 * pipeline.ExportSpeaker("speaker1", "speaker1.gsppkg");
 * 
 * auto audio = pipeline.Infer("speaker1", "要生成的文本");
 * audio->SaveToFile("output.wav");
 * 
 * // 边缘模式
 * PipelineConfig config = PipelineConfig::Edge("/path/to/models");
 * InferencePipeline pipeline(config);
 * 
 * pipeline.ImportSpeaker("speaker1.gsppkg");
 * auto audio = pipeline.Infer("speaker1", "要生成的文本");
 * @endcode
 */
class InferencePipeline {
public:
  /**
   * @brief 构造函数
   */
  explicit InferencePipeline(const PipelineConfig& config);

  /**
   * @brief 析构函数
   */
  ~InferencePipeline();

  // 禁止拷贝
  InferencePipeline(const InferencePipeline&) = delete;
  InferencePipeline& operator=(const InferencePipeline&) = delete;

  // 允许移动
  InferencePipeline(InferencePipeline&&) noexcept;
  InferencePipeline& operator=(InferencePipeline&&) noexcept;

  // ============ 配置 ============

  /**
   * @brief 获取运行模式
   */
  [[nodiscard]] PipelineMode GetMode() const;

  /**
   * @brief 获取配置
   */
  [[nodiscard]] const PipelineConfig& GetConfig() const;

  /**
   * @brief 获取模型信息字符串
   */
  [[nodiscard]] std::string GetModelInfo() const;

  /**
   * @brief 检查是否支持说话人创建
   */
  [[nodiscard]] bool CanCreateSpeaker() const;

  // ============ 说话人管理 ============

  /**
   * @brief 创建说话人（仅 Full 模式）
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

  /**
   * @brief 导入说话人数据包
   * @param package_path 数据包路径
   * @param rename 重命名（可选）
   * @return 是否成功
   */
  bool ImportSpeaker(const std::string& package_path, const std::string& rename = "");

  /**
   * @brief 导出说话人数据包
   * @param name 说话人名称
   * @param output_path 输出路径
   * @param include_audio 是否包含音频数据
   * @return 是否成功
   */
  bool ExportSpeaker(const std::string& name, const std::string& output_path, bool include_audio = false);

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

  // ============ 推理接口 ============

  /**
   * @brief 同步推理
   * @param speaker_name 说话人名称
   * @param text 目标文本
   * @param lang 语言（默认使用配置中的默认语言）
   * @param sample_config 采样配置
   * @param noise_scale 噪声缩放
   * @param speed 速度
   * @return 生成的音频
   */
  std::unique_ptr<AudioTools> Infer(
      const std::string& speaker_name,
      const std::string& text,
      const std::string& lang = "",
      const Model::SampleConfig& sample_config = {},
      float noise_scale = 0.5f,
      float speed = 1.0f,
      Model::InferStats* stats = nullptr,
      std::function<void()> on_first_segment = nullptr);

  /**
   * @brief 简化推理接口（使用默认参数）
   */
  std::unique_ptr<AudioTools> Infer(
      const std::string& speaker_name,
      const std::string& text,
      float temperature);

  // ============ 流式推理 ============

  /**
   * @brief 流式推理
   * @param speaker_name 说话人名称
   * @param text 目标文本
   * @param lang 语言
   * @param callback 音频分块回调
   * @param sample_config 采样配置
   * @param noise_scale 噪声缩放
   * @param speed 速度
   * @return 是否成功
   */
  bool InferStreaming(
      const std::string& speaker_name,
      const std::string& text,
      const std::string& lang,
      AudioChunkCallback callback,
      const Model::SampleConfig& sample_config = {},
      float noise_scale = 0.5f,
      float speed = 1.0f);

  // ============ 底层组件访问 ============

  /**
   * @brief 获取设备上下文
   */
  [[nodiscard]] Core::DeviceContext& GetDeviceContext();

  /**
   * @brief 获取模型池
   */
  [[nodiscard]] Model::ModelPool& GetModelPool();

  /**
   * @brief 获取说话人管理器
   */
  [[nodiscard]] SpeakerManager& GetSpeakerManager();

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace GPTSoVITS

#endif  // GPT_SOVITS_CPP_INFERENCE_PIPELINE_H
