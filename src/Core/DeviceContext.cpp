//
// Created by 19254 on 2026/2/8.
//

#include "GPTSoVITS/Core/DeviceContext.h"

#include <cstdlib>
#include <cstring>

#include "GPTSoVITS/plog.h"

#ifdef WITH_CUDA
#include <cuda_runtime.h>
#endif

namespace GPTSoVITS::Core {

// 静态默认设备上下文
static DeviceContext* g_default_device_context = nullptr;

class DeviceContext::Impl {
public:
  DeviceConfig config;
  Model::Device current_device;
  DeviceCapabilities capabilities;
  void* stream = nullptr;
  size_t allocated_size = 0;

  Impl(const DeviceConfig& cfg) : config(cfg) {
    current_device = Model::Device(config.preferred_device, config.device_id);
    capabilities = DetectCapabilities(config.preferred_device, config.device_id);

    if (config.verbose) {
      PrintInfo("[DeviceContext] Initialized device: {} (id={})",
                config.preferred_device == Model::DeviceType::kCUDA ? "CUDA" : "CPU",
                config.device_id);
      PrintInfo("[DeviceContext] Device name: {}", capabilities.device_name);
      PrintInfo("[DeviceContext] Supports FP16: {}", capabilities.supports_fp16);
      PrintInfo("[DeviceContext] Total memory: {} MB", capabilities.total_memory / (1024 * 1024));
      PrintInfo("[DeviceContext] Compute precision: {}",
                config.compute_precision == Model::DataType::kFloat16 ? "FP16" : "FP32");
    }

#ifdef WITH_CUDA
    if (config.preferred_device == Model::DeviceType::kCUDA && config.enable_stream) {
      cudaSetDevice(config.device_id);
      cudaStreamCreate(reinterpret_cast<cudaStream_t*>(&stream));
      PrintDebug("[DeviceContext] Created CUDA stream: {}", stream);
    }
#endif
  }

  ~Impl() {
#ifdef WITH_CUDA
    if (stream) {
      cudaStreamDestroy(reinterpret_cast<cudaStream_t>(stream));
      PrintDebug("[DeviceContext] Destroyed CUDA stream");
    }
#endif
  }
};

DeviceContext::DeviceContext(const DeviceConfig& config)
    : impl_(std::make_unique<Impl>(config)) {}

DeviceContext::~DeviceContext() = default;

DeviceContext::DeviceContext(DeviceContext&&) noexcept = default;

DeviceContext& DeviceContext::operator=(DeviceContext&&) noexcept = default;

Model::Device DeviceContext::GetDevice() const {
  Model::Device dev = impl_->current_device;
  dev.stream = impl_->stream;
  return dev;
}

Model::DeviceType DeviceContext::GetDeviceType() const {
  return impl_->current_device.type;
}

int DeviceContext::GetDeviceId() const {
  return impl_->current_device.device_id;
}

Model::DataType DeviceContext::GetComputePrecision() const {
  return impl_->config.compute_precision;
}

void DeviceContext::SetComputePrecision(Model::DataType precision) {
  impl_->config.compute_precision = precision;
}

const DeviceCapabilities& DeviceContext::GetCapabilities() const {
  return impl_->capabilities;
}

const DeviceConfig& DeviceContext::GetConfig() const {
  return impl_->config;
}

void DeviceContext::Synchronize() {
#ifdef WITH_CUDA
  if (impl_->current_device.type == Model::DeviceType::kCUDA) {
    cudaSetDevice(impl_->current_device.device_id);
    if (impl_->stream) {
      cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(impl_->stream));
    } else {
      cudaDeviceSynchronize();
    }
    PrintDebug("[DeviceContext] Synchronized device");
  }
#endif
}

void* DeviceContext::GetOrCreateStream() {
#ifdef WITH_CUDA
  if (impl_->current_device.type == Model::DeviceType::kCUDA && !impl_->stream && impl_->config.enable_stream) {
    cudaSetDevice(impl_->current_device.device_id);
    cudaStreamCreate(reinterpret_cast<cudaStream_t*>(&impl_->stream));
  }
#endif
  return impl_->stream;
}

void* DeviceContext::Allocate(size_t size) {
  if (size == 0) return nullptr;

  void* ptr = nullptr;

  if (impl_->current_device.type == Model::DeviceType::kCPU) {
    ptr = std::malloc(size);
    if (!ptr) {
      PrintError("[DeviceContext] CPU allocation failed for {} bytes", size);
      return nullptr;
    }
  }
#ifdef WITH_CUDA
  else if (impl_->current_device.type == Model::DeviceType::kCUDA) {
    cudaSetDevice(impl_->current_device.device_id);
    cudaError_t err = cudaMalloc(&ptr, size);
    if (err != cudaSuccess) {
      PrintError("[DeviceContext] CUDA allocation failed: {}", cudaGetErrorString(err));
      return nullptr;
    }
  }
#endif

  impl_->allocated_size += size;
  PrintDebug("[DeviceContext] Allocated {} bytes at {}, total: {}", size, ptr, impl_->allocated_size);
  return ptr;
}

void DeviceContext::Deallocate(void* ptr) {
  if (!ptr) return;

  if (impl_->current_device.type == Model::DeviceType::kCPU) {
    std::free(ptr);
  }
#ifdef WITH_CUDA
  else if (impl_->current_device.type == Model::DeviceType::kCUDA) {
    cudaFree(ptr);
  }
#endif

  PrintDebug("[DeviceContext] Deallocated memory at {}", ptr);
}

size_t DeviceContext::GetAllocatedSize() const {
  return impl_->allocated_size;
}

std::unique_ptr<Model::Tensor> DeviceContext::CreateTensor(
    const std::vector<int64_t>& shape,
    Model::DataType dtype) {
  return Model::Tensor::Empty(shape, dtype, impl_->current_device);
}

std::unique_ptr<Model::Tensor> DeviceContext::ToDevice(const Model::Tensor* tensor) {
  if (!tensor) return nullptr;
  return tensor->ToDevice(impl_->current_device);
}

std::unique_ptr<Model::Tensor> DeviceContext::ToPrecision(const Model::Tensor* tensor) {
  if (!tensor) return nullptr;
  return tensor->ToType(impl_->config.compute_precision);
}

// ============ 静态方法 ============

DeviceCapabilities DeviceContext::DetectCapabilities(Model::DeviceType type, int device_id) {
  DeviceCapabilities caps;

  if (type == Model::DeviceType::kCPU) {
    caps.supports_fp16 = false;  // CPU 不原生支持 FP16 计算
    caps.supports_int8 = true;
    caps.supports_tensorrt = false;
    caps.device_name = "CPU";
    return caps;
  }

#ifdef WITH_CUDA
  if (type == Model::DeviceType::kCUDA) {
    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
      PrintError("[DeviceContext] Failed to set CUDA device {}: {}", device_id, cudaGetErrorString(err));
      return caps;
    }

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device_id);
    if (err == cudaSuccess) {
      caps.device_name = prop.name;
      caps.compute_capability_major = prop.major;
      caps.compute_capability_minor = prop.minor;
      caps.total_memory = prop.totalGlobalMem;
      caps.supports_fp16 = (prop.major >= 6);  // Pascal 及以上支持 FP16
      caps.supports_int8 = (prop.major >= 6);
      caps.supports_tensorrt = true;
    }

    // 获取可用内存
    size_t free_mem, total_mem;
    if (cudaMemGetInfo(&free_mem, &total_mem) == cudaSuccess) {
      caps.available_memory = free_mem;
    }
  }
#else
  (void)device_id;  // 避免未使用警告
#endif

  return caps;
}

int DeviceContext::GetDeviceCount(Model::DeviceType type) {
  if (type == Model::DeviceType::kCPU) {
    return 1;
  }

#ifdef WITH_CUDA
  if (type == Model::DeviceType::kCUDA) {
    int count = 0;
    if (cudaGetDeviceCount(&count) == cudaSuccess) {
      return count;
    }
  }
#endif

  return 0;
}

void DeviceContext::SetDefault(DeviceContext* ctx) {
  g_default_device_context = ctx;
}

DeviceContext* DeviceContext::GetDefault() {
  return g_default_device_context;
}

}  // namespace GPTSoVITS::Core
