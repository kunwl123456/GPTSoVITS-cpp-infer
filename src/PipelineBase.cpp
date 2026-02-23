//
// Created by iFlow CLI on 2026/2/22.
//

#include "GPTSoVITS/PipelineBase.h"

#include "GPTSoVITS/plog.h"
#include "GPTSoVITS/G2P/SymbolManager.h"
#include <sstream>
#include "nlohmann/json.hpp"

namespace GPTSoVITS {

// JSON 配置实现（共享）
class _JsonImpl {
public:
  nlohmann::json data;
  void Parse(const std::string& content) {
    data = nlohmann::json::parse(content);
  }
};

std::vector<std::string> PipelineBase::ListSpeakers() const {
  std::vector<std::string> names;
  names.reserve(m_speaker_map.size());
  for (const auto& [name, _] : m_speaker_map) {
    names.push_back(name);
  }
  return names;
}

bool PipelineBase::RemoveSpeaker(const std::string& speaker_name) {
  auto iter = m_speaker_map.find(speaker_name);
  if (iter == m_speaker_map.end()) {
    PrintWarn("[PipelineBase] Speaker '{}' not found", speaker_name);
    return false;
  }
  m_speaker_map.erase(iter);
  PrintInfo("[PipelineBase] Removed speaker: {}", speaker_name);
  return true;
}

bool PipelineBase::HasSpeaker(const std::string& speaker_name) const {
  return m_speaker_map.find(speaker_name) != m_speaker_map.end();
}

const SpeakerInfo* PipelineBase::GetSpeakerInfo(const std::string& speaker_name) const {
  auto iter = m_speaker_map.find(speaker_name);
  if (iter == m_speaker_map.end()) {
    return nullptr;
  }
  return &iter->second;
}

SpeakerInfo* PipelineBase::GetSpeakerInfoMutable(const std::string& speaker_name) {
  auto iter = m_speaker_map.find(speaker_name);
  if (iter == m_speaker_map.end()) {
    return nullptr;
  }
  return &iter->second;
}

void PipelineBase::InitializeConfig() {
  if (!m_config || !m_config->data.is_object()) {
    PrintWarn("[PipelineBase] No valid config, using defaults");
    return;
  }

  // 从 config.json 读取基本参数
  if (m_config->data.contains("data")) {
    auto& data = m_config->data["data"];
    m_config_params.sampling_rate = data.value<int>("sampling_rate", 32000);
    m_config_params.max_len = data.value<int>("max_len", 1000);
    m_config_params.hop_length = data.value<int>("hop_length", 640);
    m_config_params.filter_length = data.value<int>("filter_length", 2048);
    m_config_params.mel_bins = data.value<int>("mel_bins", 128);
  }

  std::string version = "default";
  if (m_config->data.contains("model")) {
    auto& model = m_config->data["model"];
    version = model.value<std::string>("version", "v2");
    m_config_params.model_version = version;
  }

  if (m_config->data.contains("sv_embedding")) {
    auto& sv_emb = m_config->data["sv_embedding"];
    m_config_params.sv_dim = sv_emb.value<int>("embedding_size", 20480);
  }

  // 加载 symbol_to_id 到 SymbolManager
  if (m_config->data.contains("symbol_to_id")) {
    auto& sym_mgr = G2P::SymbolManager::Instance();
    std::string sym_json = m_config->data["symbol_to_id"].dump();
    if (sym_mgr.LoadFromJson(sym_json, version)) {
      sym_mgr.SetActiveVersion(version);
      PrintDebug("[PipelineBase] Loaded {} symbols for version: {}",
                 sym_mgr.GetSymbolCount(version), version);
    } else {
      PrintWarn("[PipelineBase] Failed to load symbol_to_id, using default");
    }
  }

  PrintDebug("[PipelineBase] Config initialized: sampling_rate={}, max_len={}, version={}",
             m_config_params.sampling_rate, m_config_params.max_len,
             m_config_params.model_version);
}

std::string PipelineBase::GetModelInfo() const {
  std::ostringstream oss;
  oss << "PipelineBase Info:\n";
  oss << "  Sampling rate: " << m_config_params.sampling_rate << "\n";
  oss << "  Max length: " << m_config_params.max_len << "\n";
  oss << "  Model version: " << m_config_params.model_version << "\n";
  oss << "  Compute precision: "
      << (m_compute_precision == Model::DataType::kFloat16 ? "FP16" : "FP32") << "\n";
  oss << "  SV embedding dim: " << m_config_params.sv_dim << "\n";
  oss << "  Loaded speakers: " << m_speaker_map.size() << "\n";
  return oss.str();
}

}  // namespace GPTSoVITS
