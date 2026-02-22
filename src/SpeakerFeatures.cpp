//
// Created by 19254 on 2026/2/8.
//

#include "GPTSoVITS/SpeakerFeatures.h"

#include <mutex>
#include <unordered_map>

#include "GPTSoVITS/model/bert.h"
#include "GPTSoVITS/plog.h"

namespace GPTSoVITS {

// 设备缓存键
struct DeviceCacheKey {
  Model::DeviceType type;
  int device_id;

  bool operator==(const DeviceCacheKey& other) const {
    return type == other.type && device_id == other.device_id;
  }
};

}  // namespace GPTSoVITS

// 哈希特化
namespace std {
template <>
struct hash<GPTSoVITS::DeviceCacheKey> {
  size_t operator()(const GPTSoVITS::DeviceCacheKey& k) const {
    return hash<int>()(static_cast<int>(k.type)) ^ hash<int>()(k.device_id);
  }
};
}  // namespace std

namespace GPTSoVITS {

class SpeakerFeatures::Impl {
public:
  SpeakerMetadata metadata;

  // CPU 原始数据
  std::unique_ptr<Model::Tensor> phone_seq_cpu;
  std::unique_ptr<Model::Tensor> bert_seq_cpu;
  std::unique_ptr<Model::Tensor> vq_codes_cpu;
  std::unique_ptr<Model::Tensor> refer_spec_cpu;
  std::unique_ptr<Model::Tensor> sv_emb_cpu;

  // 设备缓存：按设备类型和ID索引
  std::unordered_map<DeviceCacheKey, std::unique_ptr<Model::Tensor>> phone_seq_cache;
  std::unordered_map<DeviceCacheKey, std::unique_ptr<Model::Tensor>> bert_seq_cache;
  std::unordered_map<DeviceCacheKey, std::unique_ptr<Model::Tensor>> vq_codes_cache;
  std::unordered_map<DeviceCacheKey, std::unique_ptr<Model::Tensor>> refer_spec_cache;
  std::unordered_map<DeviceCacheKey, std::unique_ptr<Model::Tensor>> sv_emb_cache;

  std::shared_ptr<Bert::BertRes> bert_res;

  mutable std::mutex mutex;

  Impl(const std::string& name, const std::string& lang) {
    metadata.name = name;
    metadata.lang = lang;
  }

  // 获取设备缓存键
  static DeviceCacheKey MakeKey(const Model::Device& device) {
    return {device.type, device.device_id};
  }

  // 获取或创建设备缓存
  Model::Tensor* GetOrCreateCache(
      std::unique_ptr<Model::Tensor>& cpu_data,
      std::unordered_map<DeviceCacheKey, std::unique_ptr<Model::Tensor>>& cache,
      const Model::Device& device) {
    if (!cpu_data) return nullptr;

    // CPU 设备直接返回原始数据，不使用缓存
    if (device.type == Model::DeviceType::kCPU) {
      return cpu_data.get();
    }

    // 非 CPU 设备检查缓存
    auto key = MakeKey(device);
    auto it = cache.find(key);
    if (it != cache.end()) {
      return it->second.get();
    }

    // 创建设备缓存
    auto device_tensor = cpu_data->ToDevice(device);
    if (!device_tensor) {
      PrintError("[SpeakerFeatures] Failed to move tensor to device");
      return nullptr;
    }

    cache[key] = std::move(device_tensor);
    return cache[key].get();
  }
};

SpeakerFeatures::SpeakerFeatures(const std::string& name, const std::string& lang)
    : impl_(std::make_unique<Impl>(name, lang)) {}

SpeakerFeatures::SpeakerFeatures(const SpeakerMetadata& metadata)
    : impl_(std::make_unique<Impl>(metadata.name, metadata.lang)) {
  impl_->metadata = metadata;
}

SpeakerFeatures::~SpeakerFeatures() = default;

SpeakerFeatures::SpeakerFeatures(SpeakerFeatures&&) noexcept = default;

SpeakerFeatures& SpeakerFeatures::operator=(SpeakerFeatures&&) noexcept = default;

const std::string& SpeakerFeatures::Name() const {
  return impl_->metadata.name;
}

const std::string& SpeakerFeatures::Lang() const {
  return impl_->metadata.lang;
}

const SpeakerMetadata& SpeakerFeatures::GetMetadata() const {
  return impl_->metadata;
}

void SpeakerFeatures::SetMetadata(const SpeakerMetadata& metadata) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->metadata = metadata;
}

// ============ 特征访问 ============

Model::Tensor* SpeakerFeatures::GetPhoneSeq(Model::Device target_device) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->GetOrCreateCache(impl_->phone_seq_cpu, impl_->phone_seq_cache, target_device);
}

Model::Tensor* SpeakerFeatures::GetBertSeq(Model::Device target_device) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->GetOrCreateCache(impl_->bert_seq_cpu, impl_->bert_seq_cache, target_device);
}

Model::Tensor* SpeakerFeatures::GetVQCodes(Model::Device target_device) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->GetOrCreateCache(impl_->vq_codes_cpu, impl_->vq_codes_cache, target_device);
}

Model::Tensor* SpeakerFeatures::GetReferSpec(Model::Device target_device) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->GetOrCreateCache(impl_->refer_spec_cpu, impl_->refer_spec_cache, target_device);
}

Model::Tensor* SpeakerFeatures::GetSVEmbedding(Model::Device target_device) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->GetOrCreateCache(impl_->sv_emb_cpu, impl_->sv_emb_cache, target_device);
}

// ============ 特征设置 ============

void SpeakerFeatures::SetPhoneSeq(std::unique_ptr<Model::Tensor> tensor) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  // 确保数据在 CPU 上
  if (tensor && !tensor->IsCPU()) {
    impl_->phone_seq_cpu = tensor->ToCPU();
  } else {
    impl_->phone_seq_cpu = std::move(tensor);
  }
  impl_->phone_seq_cache.clear();
}

void SpeakerFeatures::SetBertSeq(std::unique_ptr<Model::Tensor> tensor) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (tensor && !tensor->IsCPU()) {
    impl_->bert_seq_cpu = tensor->ToCPU();
  } else {
    impl_->bert_seq_cpu = std::move(tensor);
  }
  impl_->bert_seq_cache.clear();
}

void SpeakerFeatures::SetVQCodes(std::unique_ptr<Model::Tensor> tensor) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (tensor && !tensor->IsCPU()) {
    impl_->vq_codes_cpu = tensor->ToCPU();
  } else {
    impl_->vq_codes_cpu = std::move(tensor);
  }
  impl_->vq_codes_cache.clear();
}

void SpeakerFeatures::SetReferSpec(std::unique_ptr<Model::Tensor> tensor) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (tensor && !tensor->IsCPU()) {
    impl_->refer_spec_cpu = tensor->ToCPU();
  } else {
    impl_->refer_spec_cpu = std::move(tensor);
  }
  impl_->refer_spec_cache.clear();
}

void SpeakerFeatures::SetSVEmbedding(std::unique_ptr<Model::Tensor> tensor) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (tensor && !tensor->IsCPU()) {
    impl_->sv_emb_cpu = tensor->ToCPU();
  } else {
    impl_->sv_emb_cpu = std::move(tensor);
  }
  impl_->sv_emb_cache.clear();
}

void SpeakerFeatures::SetFromBertRes(std::shared_ptr<Bert::BertRes> bert_res) {
  if (!bert_res) return;

  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->bert_res = bert_res;

  // 设置 PhoneSeq 和 BertSeq
  if (bert_res->PhoneSeq) {
    if (!bert_res->PhoneSeq->IsCPU()) {
      impl_->phone_seq_cpu = bert_res->PhoneSeq->ToCPU();
    } else {
      impl_->phone_seq_cpu = bert_res->PhoneSeq->Clone();
    }
  }

  if (bert_res->BertSeq) {
    if (!bert_res->BertSeq->IsCPU()) {
      impl_->bert_seq_cpu = bert_res->BertSeq->ToCPU();
    } else {
      impl_->bert_seq_cpu = bert_res->BertSeq->Clone();
    }
  }

  impl_->phone_seq_cache.clear();
  impl_->bert_seq_cache.clear();
}

// ============ 设备管理 ============

void SpeakerFeatures::EnsureOnDevice(Model::Device device) {
  std::lock_guard<std::mutex> lock(impl_->mutex);

  // 预热所有缓存
  if (device.type != Model::DeviceType::kCPU) {
    if (impl_->phone_seq_cpu && impl_->phone_seq_cache.find(Impl::MakeKey(device)) == impl_->phone_seq_cache.end()) {
      impl_->phone_seq_cache[Impl::MakeKey(device)] = impl_->phone_seq_cpu->ToDevice(device);
    }
    if (impl_->bert_seq_cpu && impl_->bert_seq_cache.find(Impl::MakeKey(device)) == impl_->bert_seq_cache.end()) {
      impl_->bert_seq_cache[Impl::MakeKey(device)] = impl_->bert_seq_cpu->ToDevice(device);
    }
    if (impl_->vq_codes_cpu && impl_->vq_codes_cache.find(Impl::MakeKey(device)) == impl_->vq_codes_cache.end()) {
      impl_->vq_codes_cache[Impl::MakeKey(device)] = impl_->vq_codes_cpu->ToDevice(device);
    }
    if (impl_->refer_spec_cpu && impl_->refer_spec_cache.find(Impl::MakeKey(device)) == impl_->refer_spec_cache.end()) {
      impl_->refer_spec_cache[Impl::MakeKey(device)] = impl_->refer_spec_cpu->ToDevice(device);
    }
    if (impl_->sv_emb_cpu && impl_->sv_emb_cache.find(Impl::MakeKey(device)) == impl_->sv_emb_cache.end()) {
      impl_->sv_emb_cache[Impl::MakeKey(device)] = impl_->sv_emb_cpu->ToDevice(device);
    }
  }

  PrintDebug("[SpeakerFeatures] Ensured features on device for speaker: {}", impl_->metadata.name);
}

void SpeakerFeatures::ReleaseDeviceCache(Model::Device device) {
  std::lock_guard<std::mutex> lock(impl_->mutex);

  auto key = Impl::MakeKey(device);
  impl_->phone_seq_cache.erase(key);
  impl_->bert_seq_cache.erase(key);
  impl_->vq_codes_cache.erase(key);
  impl_->refer_spec_cache.erase(key);
  impl_->sv_emb_cache.erase(key);

  PrintDebug("[SpeakerFeatures] Released device cache for speaker: {}", impl_->metadata.name);
}

void SpeakerFeatures::ReleaseAllDeviceCaches() {
  std::lock_guard<std::mutex> lock(impl_->mutex);

  impl_->phone_seq_cache.clear();
  impl_->bert_seq_cache.clear();
  impl_->vq_codes_cache.clear();
  impl_->refer_spec_cache.clear();
  impl_->sv_emb_cache.clear();

  PrintDebug("[SpeakerFeatures] Released all device caches for speaker: {}", impl_->metadata.name);
}

bool SpeakerFeatures::IsAvailableOnDevice(Model::Device device) const {
  std::lock_guard<std::mutex> lock(impl_->mutex);

  if (device.type == Model::DeviceType::kCPU) {
    return impl_->phone_seq_cpu && impl_->bert_seq_cpu && impl_->vq_codes_cpu &&
           impl_->refer_spec_cpu && impl_->sv_emb_cpu;
  }

  auto key = Impl::MakeKey(device);
  return impl_->phone_seq_cache.find(key) != impl_->phone_seq_cache.end() &&
         impl_->bert_seq_cache.find(key) != impl_->bert_seq_cache.end() &&
         impl_->vq_codes_cache.find(key) != impl_->vq_codes_cache.end() &&
         impl_->refer_spec_cache.find(key) != impl_->refer_spec_cache.end() &&
         impl_->sv_emb_cache.find(key) != impl_->sv_emb_cache.end();
}

// ============ 内存管理 ============

size_t SpeakerFeatures::GetMemoryUsage() const {
  std::lock_guard<std::mutex> lock(impl_->mutex);

  size_t total = 0;

  auto add_tensor_size = [&total](const std::unique_ptr<Model::Tensor>& t) {
    if (t) total += t->ByteSize();
  };

  add_tensor_size(impl_->phone_seq_cpu);
  add_tensor_size(impl_->bert_seq_cpu);
  add_tensor_size(impl_->vq_codes_cpu);
  add_tensor_size(impl_->refer_spec_cpu);
  add_tensor_size(impl_->sv_emb_cpu);

  // 加上设备缓存
  for (const auto& [_, t] : impl_->phone_seq_cache) add_tensor_size(t);
  for (const auto& [_, t] : impl_->bert_seq_cache) add_tensor_size(t);
  for (const auto& [_, t] : impl_->vq_codes_cache) add_tensor_size(t);
  for (const auto& [_, t] : impl_->refer_spec_cache) add_tensor_size(t);
  for (const auto& [_, t] : impl_->sv_emb_cache) add_tensor_size(t);

  return total;
}

size_t SpeakerFeatures::GetDeviceCacheMemoryUsage(Model::Device device) const {
  std::lock_guard<std::mutex> lock(impl_->mutex);

  size_t total = 0;
  auto key = Impl::MakeKey(device);

  auto add_cache_size = [&total, &key](const std::unordered_map<DeviceCacheKey, std::unique_ptr<Model::Tensor>>& cache) {
    auto it = cache.find(key);
    if (it != cache.end() && it->second) {
      total += it->second->ByteSize();
    }
  };

  add_cache_size(impl_->phone_seq_cache);
  add_cache_size(impl_->bert_seq_cache);
  add_cache_size(impl_->vq_codes_cache);
  add_cache_size(impl_->refer_spec_cache);
  add_cache_size(impl_->sv_emb_cache);

  return total;
}

// ============ 序列化 ============

bool SpeakerFeatures::SerializeToBuffer(std::vector<uint8_t>& buffer) const {
  // TODO: 实现序列化,后面再做
  // 使用现有的 SpeakerSerializer
  return false;
}

std::unique_ptr<SpeakerFeatures> SpeakerFeatures::DeserializeFromBuffer(
    const std::vector<uint8_t>& buffer) {
  // TODO: 实现反序列化,后面再做
  return nullptr;
}

// ============ 兼容旧接口 ============

std::shared_ptr<Bert::BertRes> SpeakerFeatures::GetBertRes() const {
  std::lock_guard<std::mutex> lock(impl_->mutex);

  if (!impl_->bert_res) {
    // 创建 BertRes 并填充数据
    impl_->bert_res = std::make_shared<Bert::BertRes>();
    if (impl_->phone_seq_cpu) {
      impl_->bert_res->PhoneSeq = impl_->phone_seq_cpu->Clone();
    }
    if (impl_->bert_seq_cpu) {
      impl_->bert_res->BertSeq = impl_->bert_seq_cpu->Clone();
    }
  }

  return impl_->bert_res;
}

}  // namespace GPTSoVITS
