//
// Created by 19254 on 2026/2/8.
//

#include "GPTSoVITS/SpeakerManager.h"

#include <mutex>

#include "GPTSoVITS/AudioTools.h"
#include "GPTSoVITS/Core/DeviceContext.h"
#include "GPTSoVITS/Core/ModelPool.h"
#include "GPTSoVITS/G2P/Pipline.h"
#include "GPTSoVITS/GPTSoVITSCpp.h"
#include "GPTSoVITS/Utils/speaker_serializer.h"
#include "GPTSoVITS/model/bert.h"
#include "GPTSoVITS/plog.h"

namespace GPTSoVITS {

SpeakerManager& SpeakerManager::Instance() {
  static SpeakerManager instance;
  return instance;
}

SpeakerManager::SpeakerManager() = default;

SpeakerManager::~SpeakerManager() = default;

void SpeakerManager::SetDeviceContext(Core::DeviceContext* ctx) {
  std::lock_guard<std::mutex> lock(mutex_);
  device_ctx_ = ctx;
}

void SpeakerManager::SetModelPool(Model::ModelPool* pool) {
  std::lock_guard<std::mutex> lock(mutex_);
  model_pool_ = pool;
}

void SpeakerManager::SetG2PPipeline(std::shared_ptr<G2P::G2PPipline> pipeline) {
  std::lock_guard<std::mutex> lock(mutex_);
  g2p_pipeline_ = std::move(pipeline);
}

Core::DeviceContext* SpeakerManager::GetDeviceContext() const {
  return device_ctx_;
}

// ============ 创建说话人 ============

SpeakerFeatures* SpeakerManager::CreateSpeaker(
    const std::string& name,
    const std::string& lang,
    const std::filesystem::path& ref_audio_path,
    const std::string& ref_text) {
  std::lock_guard<std::mutex> lock(mutex_);

  // 检查是否已存在
  if (speakers_.find(name) != speakers_.end()) {
    PrintWarn("[SpeakerManager] Speaker '{}' already exists, will be overwritten", name);
  }

  PrintInfo("[SpeakerManager] Creating speaker: {}", name);

  // 验证模型池
  if (!model_pool_) {
    PrintError("[SpeakerManager] ModelPool not set, cannot create speaker");
    return nullptr;
  }

  // 验证必要模型
  if (!model_pool_->IsRegistered(Model::ModelType::kSSL) ||
      !model_pool_->IsRegistered(Model::ModelType::kVQ)) {
    PrintError("[SpeakerManager] Required models (SSL, VQ) not registered");
    return nullptr;
  }

  try {
    // 创建说话人特征
    auto features = std::make_unique<SpeakerFeatures>(name, lang);

    // 加载参考音频
    auto ref_audio = AudioTools::FromFile(ref_audio_path.string());
    if (!ref_audio) {
      PrintError("[SpeakerManager] Failed to load reference audio: {}", ref_audio_path.string());
      return nullptr;
    }

    // 准备音频
    auto audio_16k = ref_audio->ReSample(16000);
    auto samples_16k = audio_16k->ReadSamples();

    // 静音填充
    size_t padding_size = static_cast<size_t>(16000 * 0.3);
    samples_16k.insert(samples_16k.end(), padding_size, 0.0f);

    // 获取设备
    Model::Device device = device_ctx_ ? device_ctx_->GetDevice() : Model::Device(Model::DeviceType::kCPU);

    // SSL 提取
    auto ssl_model = model_pool_->GetModel<Model::SSLModel>(Model::ModelType::kSSL);
    if (!ssl_model) {
      PrintError("[SpeakerManager] Failed to get SSL model");
      return nullptr;
    }
    auto ssl_content = ssl_model->GetSSLContent(samples_16k);

    // VQ 提取
    auto vq_model = model_pool_->GetModel<Model::VQModel>(Model::ModelType::kVQ);
    if (!vq_model) {
      PrintError("[SpeakerManager] Failed to get VQ model");
      return nullptr;
    }
    auto vq_codes = vq_model->GetVQCodes(ssl_content.get());

    // 处理 VQ codes 形状
    if (vq_codes && vq_codes->Shape().size() == 3) {
      auto shape = vq_codes->Shape();
      auto sliced = Model::Tensor::Empty({1, shape[2]}, Model::DataType::kInt64, Model::DeviceType::kCPU);
      auto vq_cpu = vq_codes->To(Model::Device(Model::DeviceType::kCPU), Model::DataType::kInt64);
      std::memcpy(sliced->Data<int64_t>(), vq_cpu->Data<int64_t>(), shape[2] * sizeof(int64_t));
      vq_codes = std::move(sliced);
    }

    features->SetVQCodes(std::move(vq_codes));

    // 频谱图
    auto spectrogram_model = model_pool_->GetModel<Model::SpectrogramModel>(Model::ModelType::kSpectrogram);
    if (spectrogram_model) {
      auto audio_32k = ref_audio->ReSample(32000);
      auto refer_spec = spectrogram_model->ComputeSpec(audio_32k->ReadSamples());
      if (refer_spec && refer_spec->Shape().size() == 3) {
        refer_spec->Reshape({refer_spec->Shape()[1], refer_spec->Shape()[2]});
      }
      features->SetReferSpec(std::move(refer_spec));
    }

    // SV Embedding
    auto sv_model = model_pool_->GetModel<Model::SVEmbeddingModel>(Model::ModelType::kSVEmbedding);
    if (sv_model) {
      auto sv_emb = sv_model->ComputeEmbedding(ref_audio->ReSample(16000)->ReadSamples());
      if (sv_emb && sv_emb->Shape().size() == 2) {
        sv_emb->Reshape({sv_emb->Shape()[1]});
      }
      features->SetSVEmbedding(std::move(sv_emb));
    }

    // BERT 特征提取（参考文本的音素和BERT特征）
    if (g2p_pipeline_) {
      auto bert_res = g2p_pipeline_->GetPhoneAndBert(ref_text, lang);
      if (bert_res) {
        features->SetFromBertRes(bert_res);
        PrintDebug("[SpeakerManager] BERT features extracted for reference text");
      } else {
        PrintWarn("[SpeakerManager] Failed to extract BERT features for reference text");
      }
    } else {
      PrintWarn("[SpeakerManager] G2P Pipeline not set, skipping BERT feature extraction");
    }

    // 存储
    auto* ptr = features.get();
    speakers_[name] = std::move(features);

    PrintInfo("[SpeakerManager] Successfully created speaker: {}", name);
    return ptr;

  } catch (const std::exception& e) {
    PrintError("[SpeakerManager] Failed to create speaker: {}", e.what());
    return nullptr;
  }
}

// ============ 导入/导出 ============

SpeakerFeatures* SpeakerManager::ImportFromPackage(
    const std::string& path,
    const SpeakerImportOptions& options) {
  std::lock_guard<std::mutex> lock(mutex_);

  PrintInfo("[SpeakerManager] Importing speaker from: {}", path);

  // 验证数据包
  if (options.validate && !ValidatePackage(path)) {
    PrintError("[SpeakerManager] Invalid speaker package: {}", path);
    return nullptr;
  }

  // 获取数据包信息
  auto package_info = Utils::SpeakerSerializer::GetPackageInfo(path);
  if (!package_info) {
    PrintError("[SpeakerManager] Failed to read package info: {}", path);
    return nullptr;
  }

  // 确定说话人名称
  std::string name = options.rename.empty() ? package_info->speaker_name : options.rename;

  // 检查是否已存在
  if (speakers_.find(name) != speakers_.end()) {
    PrintWarn("[SpeakerManager] Speaker '{}' already exists, will be overwritten", name);
  }

  // 反序列化
  auto speaker_info = Utils::SpeakerSerializer::DeserializeFromFile(path);
  if (!speaker_info) {
    PrintError("[SpeakerManager] Failed to deserialize speaker from: {}", path);
    return nullptr;
  }

  // 转换为 SpeakerFeatures
  auto features = std::make_unique<SpeakerFeatures>(name, speaker_info->m_speaker_lang);

  // 设置元数据
  SpeakerMetadata metadata;
  metadata.name = name;
  metadata.lang = speaker_info->m_speaker_lang;
  metadata.model_version = package_info->metadata.model_version;
  metadata.sv_dim = package_info->metadata.sv_dim;
  features->SetMetadata(metadata);

  // 设置特征
  if (speaker_info->m_bert_res) {
    features->SetFromBertRes(speaker_info->m_bert_res);
  }
  if (speaker_info->m_vq_codes) {
    features->SetVQCodes(std::move(speaker_info->m_vq_codes));
  }
  if (speaker_info->m_refer_spec) {
    features->SetReferSpec(std::move(speaker_info->m_refer_spec));
  }
  if (speaker_info->m_sv_emb) {
    features->SetSVEmbedding(std::move(speaker_info->m_sv_emb));
  }

  // 存储
  auto* ptr = features.get();
  speakers_[name] = std::move(features);

  PrintInfo("[SpeakerManager] Successfully imported speaker: {}", name);
  return ptr;
}

SpeakerFeatures* SpeakerManager::ImportFromPackage(
    const std::string& path,
    const std::string& rename) {
  SpeakerImportOptions options;
  options.rename = rename;
  return ImportFromPackage(path, options);
}

bool SpeakerManager::ExportToPackage(
    const std::string& name,
    const std::string& path,
    const SpeakerExportOptions& options) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = speakers_.find(name);
  if (it == speakers_.end()) {
    PrintError("[SpeakerManager] Speaker not found: {}", name);
    return false;
  }

  PrintInfo("[SpeakerManager] Exporting speaker '{}' to: {}", name, path);

  auto& features = it->second;

  // 创建 SpeakerInfo 用于序列化
  SpeakerInfo info;
  info.m_speaker_name = features->Name();
  info.m_speaker_lang = features->Lang();

  // 初始化 BertRes
  info.m_bert_res = std::make_shared<Bert::BertRes>();

  // 获取 CPU 上的数据（用于序列化）
  auto cpu_device = Model::Device(Model::DeviceType::kCPU);

  auto* phone_seq = features->GetPhoneSeq(cpu_device);
  auto* bert_seq = features->GetBertSeq(cpu_device);
  auto* vq_codes = features->GetVQCodes(cpu_device);
  auto* refer_spec = features->GetReferSpec(cpu_device);
  auto* sv_emb = features->GetSVEmbedding(cpu_device);

  // 复制数据到 SpeakerInfo
  if (phone_seq) info.m_bert_res->PhoneSeq = phone_seq->Clone();
  if (bert_seq) info.m_bert_res->BertSeq = bert_seq->Clone();
  if (vq_codes) info.m_vq_codes = vq_codes->Clone();
  if (refer_spec) info.m_refer_spec = refer_spec->Clone();
  if (sv_emb) info.m_sv_emb = sv_emb->Clone();

  // 调用序列化
  bool success = Utils::SpeakerSerializer::SerializeToFile(info, path, options.include_audio);

  if (success) {
    PrintInfo("[SpeakerManager] Successfully exported speaker '{}' to: {}", name, path);
  } else {
    PrintError("[SpeakerManager] Failed to serialize speaker to: {}", path);
  }

  return success;
}

// ============ 管理 ============

SpeakerFeatures* SpeakerManager::GetSpeaker(const std::string& name) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = speakers_.find(name);
  return it != speakers_.end() ? it->second.get() : nullptr;
}

bool SpeakerManager::HasSpeaker(const std::string& name) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return speakers_.find(name) != speakers_.end();
}

bool SpeakerManager::RemoveSpeaker(const std::string& name) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = speakers_.find(name);
  if (it == speakers_.end()) {
    return false;
  }
  speakers_.erase(it);
  PrintInfo("[SpeakerManager] Removed speaker: {}", name);
  return true;
}

std::vector<std::string> SpeakerManager::ListSpeakers() const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<std::string> names;
  names.reserve(speakers_.size());
  for (const auto& [name, _] : speakers_) {
    names.push_back(name);
  }
  return names;
}

size_t SpeakerManager::SpeakerCount() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return speakers_.size();
}

// ============ 批量操作 ============

void SpeakerManager::PreloadToDevice(const std::string& name, Model::Device device) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = speakers_.find(name);
  if (it != speakers_.end()) {
    it->second->EnsureOnDevice(device);
  }
}

void SpeakerManager::PreloadAllToDevice(Model::Device device) {
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto& [_, features] : speakers_) {
    features->EnsureOnDevice(device);
  }
}

void SpeakerManager::ReleaseAllDeviceCaches() {
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto& [_, features] : speakers_) {
    features->ReleaseAllDeviceCaches();
  }
}

// ============ 遍历 ============

void SpeakerManager::ForEach(
    const std::function<void(const std::string&, SpeakerFeatures&)>& callback) {
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto& [name, features] : speakers_) {
    callback(name, *features);
  }
}

// ============ 验证 ============

bool SpeakerManager::ValidatePackage(const std::string& path) {
  return Utils::SpeakerSerializer::ValidatePackage(path);
}

std::unique_ptr<SpeakerMetadata> SpeakerManager::GetPackageInfo(const std::string& path) {
  auto info = Utils::SpeakerSerializer::GetPackageInfo(path);
  if (!info) return nullptr;

  auto metadata = std::make_unique<SpeakerMetadata>();
  metadata->name = info->speaker_name;
  metadata->lang = info->speaker_lang;
  metadata->created_at = info->metadata.created_at;
  metadata->model_version = info->metadata.model_version;
  metadata->sv_dim = info->metadata.sv_dim;
  metadata->max_seq_len = info->metadata.max_seq_len;

  return metadata;
}

}  // namespace GPTSoVITS
