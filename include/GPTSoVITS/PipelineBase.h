//
// Created by iFlow CLI on 2026/2/22.
// Pipeline 基类 - 提供共同的说话人管理和配置功能
//

#ifndef GPT_SOVITS_CPP_PIPELINE_BASE_H
#define GPT_SOVITS_CPP_PIPELINE_BASE_H

#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "GPTSoVITS/AudioTools.h"
#include "GPTSoVITS/model/bert.h"
#include "GPTSoVITS/model/tensor.h"

namespace GPTSoVITS {

class _JsonImpl;

/**
 * @brief 说话人信息
 *
 * 存储说话人的参考音频特征，用于推理时进行音色迁移
 */
class SpeakerInfo {
public:
  std::string m_speaker_name;
  std::string m_speaker_lang;
  std::shared_ptr<Bert::BertRes> m_bert_res;
  std::unique_ptr<Model::Tensor> m_ssl_content;
  std::unique_ptr<Model::Tensor> m_vq_codes;
  std::unique_ptr<AudioTools> m_speaker_16k;
  std::unique_ptr<AudioTools> m_speaker_32k;
  std::unique_ptr<Model::Tensor> m_refer_spec;
  std::unique_ptr<Model::Tensor> m_sv_emb;

  [[nodiscard]] std::string SpeakerName() const { return m_speaker_name; }
  [[nodiscard]] std::string SpeakerLang() const { return m_speaker_lang; }
  [[nodiscard]] std::shared_ptr<Bert::BertRes> BertRes() const { return m_bert_res; }
};

/**
 * @brief Pipeline 配置参数
 *
 * 从 config.json 读取的公共配置
 */
struct PipelineConfigParams {
  int sampling_rate = 32000;
  int max_len = 1000;        // GPT KV cache 预分配最大长度
  int hop_length = 640;
  int filter_length = 2048;
  int mel_bins = 128;
  int sv_dim = 20480;        // SV embedding 维度
  std::string model_version = "v2";
};

/**
 * @brief Pipeline 基类
 *
 * 提供共同的说话人管理和配置功能
 */
class PipelineBase {
protected:
  std::map<std::string, SpeakerInfo> m_speaker_map;
  std::shared_ptr<_JsonImpl> m_config;
  PipelineConfigParams m_config_params;
  Model::DataType m_compute_precision = Model::DataType::kFloat32;

public:
  virtual ~PipelineBase() = default;

  // ============ 说话人管理 ============

  /**
   * @brief 列出所有已创建的说话人
   */
  std::vector<std::string> ListSpeakers() const;

  /**
   * @brief 移除说话人
   * @param speaker_name 说话人名称
   * @return 是否成功
   */
  bool RemoveSpeaker(const std::string& speaker_name);

  /**
   * @brief 检查说话人是否存在
   */
  bool HasSpeaker(const std::string& speaker_name) const;

  /**
   * @brief 获取说话人信息
   * @return 说话人信息指针（不存在返回 nullptr）
   */
  const SpeakerInfo* GetSpeakerInfo(const std::string& speaker_name) const;

  // ============ 配置访问 ============

  /**
   * @brief 获取采样率
   */
  [[nodiscard]] int GetSamplingRate() const { return m_config_params.sampling_rate; }

  /**
   * @brief 获取配置参数
   */
  [[nodiscard]] const PipelineConfigParams& GetConfigParams() const { return m_config_params; }

  /**
   * @brief 获取计算精度
   */
  [[nodiscard]] Model::DataType GetComputePrecision() const { return m_compute_precision; }

  /**
   * @brief 获取模型版本
   */
  [[nodiscard]] const std::string& GetModelVersion() const { return m_config_params.model_version; }

  /**
   * @brief 获取模型信息
   */
  [[nodiscard]] virtual std::string GetModelInfo() const;

protected:
  /**
   * @brief 从 JSON 配置初始化参数
   */
  void InitializeConfig();

  /**
   * @brief 获取说话人信息（可修改版本）
   */
  SpeakerInfo* GetSpeakerInfoMutable(const std::string& speaker_name);
};

}  // namespace GPTSoVITS

#endif  // GPT_SOVITS_CPP_PIPELINE_BASE_H
