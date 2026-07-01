//
// TensorRT 推理后端实现
// TensorRT 10.x API
//

#include "GPTSoVITS/model/backend/tensorrt_backend.h"

#include "GPTSoVITS/Utils/exception.h"
#include "GPTSoVITS/plog.h"

#ifdef WITH_TENSORRT

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <unordered_map>

#include "GPTSoVITS/Utils/exception.h"

namespace fs = std::filesystem;

namespace GPTSoVITS::Model {

struct TRTDestroy {
  template <typename T>
  void operator()(T* p) const { delete p; }
};
template <typename T>
using TRTPtr = std::unique_ptr<T, TRTDestroy>;

// TRT 日志适配器
class TRTLogger : public nvinfer1::ILogger {
public:
  void log(Severity sev, const char* msg) noexcept override {
    switch (sev) {
      case Severity::kERROR:   PrintError("[TRT] {}", msg); break;
      case Severity::kWARNING: PrintWarn("[TRT] {}",  msg); break;
      case Severity::kINFO:    PrintInfo("[TRT] {}",  msg); break;
      default: break;
    }
  }
};
static TRTLogger g_trt_logger;

// DataType <-> nvinfer1::DataType
static nvinfer1::DataType ToTRTType(DataType dt) {
  switch (dt) {
    case DataType::kFloat32: return nvinfer1::DataType::kFLOAT;
    case DataType::kFloat16: return nvinfer1::DataType::kHALF;
    case DataType::kFloat8:  return nvinfer1::DataType::kFP8;
    case DataType::kInt32:   return nvinfer1::DataType::kINT32;
    case DataType::kInt64:   return nvinfer1::DataType::kINT64;
    case DataType::kInt8:    return nvinfer1::DataType::kINT8;
    case DataType::kUInt8:   return nvinfer1::DataType::kUINT8;
    default: THROW_ERRORN("Unsupported DataType for TRT: {}", static_cast<int>(dt));
  }
}

static DataType FromTRTType(nvinfer1::DataType dt) {
  switch (dt) {
    case nvinfer1::DataType::kFLOAT: return DataType::kFloat32;
    case nvinfer1::DataType::kHALF:  return DataType::kFloat16;
    case nvinfer1::DataType::kFP8:   return DataType::kFloat8;
    case nvinfer1::DataType::kINT32: return DataType::kInt32;
    case nvinfer1::DataType::kINT64: return DataType::kInt64;
    case nvinfer1::DataType::kINT8:  return DataType::kInt8;
    case nvinfer1::DataType::kUINT8: return DataType::kUInt8;
    default: THROW_ERROR("Unsupported TRT DataType");
  }
}

// ============ Impl ============

struct TensorRTBackend::ContextSlot {
  TRTPtr<nvinfer1::IExecutionContext> context;
  cudaStream_t stream = nullptr;

  // name -> device ptr, scoped to this execution context.
  std::unordered_map<std::string, void*>   gpu_buffers;
  std::unordered_map<std::string, size_t>  gpu_buffer_sizes;  // 追踪每个buffer的已分配大小

  bool busy = false;
  int index = -1;
};

struct TensorRTBackend::Impl {
  struct SlotLease {
    Impl* owner = nullptr;
    ContextSlot* slot = nullptr;

    SlotLease() = default;
    SlotLease(Impl* owner_, ContextSlot* slot_) : owner(owner_), slot(slot_) {}
    SlotLease(const SlotLease&) = delete;
    SlotLease& operator=(const SlotLease&) = delete;

    SlotLease(SlotLease&& other) noexcept
        : owner(other.owner), slot(other.slot) {
      other.owner = nullptr;
      other.slot = nullptr;
    }

    SlotLease& operator=(SlotLease&& other) noexcept {
      if (this != &other) {
        Release();
        owner = other.owner;
        slot = other.slot;
        other.owner = nullptr;
        other.slot = nullptr;
      }
      return *this;
    }

    ~SlotLease() { Release(); }

    ContextSlot& get() const { return *slot; }
    ContextSlot* operator->() const { return slot; }
    ContextSlot& operator*() const { return *slot; }

  private:
    void Release() {
      if (owner && slot) {
        owner->ReleaseSlot(*slot);
        owner = nullptr;
        slot = nullptr;
      }
    }
  };

  TRTPtr<nvinfer1::IRuntime>          runtime;
  TRTPtr<nvinfer1::ICudaEngine>       engine;

  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
  std::unordered_map<std::string, DataType> input_types;
  std::unordered_map<std::string, DataType> output_types;
  std::unordered_map<std::string, std::vector<int64_t>> output_shapes;

  std::vector<std::unique_ptr<ContextSlot>> slots;
  std::mutex slot_mutex;
  std::condition_variable slot_cv;
  bool shutting_down = false;

  cudaMemPool_t mem_pool = nullptr;  // CUDA 内存池

  TensorRTConfig trt_config;
  int device_id = 0;

  ~Impl() {
    cudaSetDevice(device_id);
    {
      std::lock_guard<std::mutex> lock(slot_mutex);
      shutting_down = true;
    }
    slot_cv.notify_all();

    SynchronizeSlots();
    cudaDeviceSynchronize();
    FreeAllSlotBuffers();

    // TensorRT contexts/engines may destroy CUDA modules, events, and internal
    // streams. Release contexts before tearing down our CUDA streams and mem pool.
    for (auto& slot : slots) {
      if (slot) {
        slot->context.reset();
      }
    }
    for (auto& slot : slots) {
      if (slot && slot->stream) {
        cudaStreamDestroy(slot->stream);
        slot->stream = nullptr;
      }
    }
    slots.clear();

    engine.reset();
    runtime.reset();

    if (mem_pool) { cudaMemPoolDestroy(mem_pool); mem_pool = nullptr; }
  }

  void EnsureMemoryPool() {
    if (mem_pool) return;

    cudaSetDevice(device_id);

    cudaMemPoolProps pool_props = {};
    pool_props.allocType = cudaMemAllocationTypePinned;
    pool_props.handleTypes = cudaMemHandleTypeNone;
    pool_props.location.type = cudaMemLocationTypeDevice;
    pool_props.location.id = device_id;

    cudaError_t err = cudaMemPoolCreate(&mem_pool, &pool_props);
    if (err != cudaSuccess) {
      PrintWarn("[TRTBackend] cudaMemPoolCreate failed: {}, fallback to default async pool",
                cudaGetErrorString(err));
      mem_pool = nullptr;
    }
  }

  bool InitializeSlots(int requested_count) {
    if (!engine) {
      PrintError("[TRTBackend] Cannot initialize slots before engine is ready");
      return false;
    }

    ClearSlots();
    EnsureMemoryPool();

    int slot_count = std::max(1, requested_count);
    slots.reserve(static_cast<size_t>(slot_count));

    for (int i = 0; i < slot_count; ++i) {
      auto slot = std::make_unique<ContextSlot>();
      slot->index = i;

      cudaError_t stream_err = cudaStreamCreate(&slot->stream);
      if (stream_err != cudaSuccess) {
        PrintError("[TRTBackend] cudaStreamCreate failed for slot {}: {}",
                   i, cudaGetErrorString(stream_err));
        ClearSlots();
        return false;
      }

      slot->context.reset(engine->createExecutionContext());
      if (!slot->context) {
        PrintError("[TRTBackend] createExecutionContext failed for slot {}", i);
        cudaStreamDestroy(slot->stream);
        slot->stream = nullptr;
        ClearSlots();
        return false;
      }
      slot->context->setAuxStreams(nullptr, 0);
      slots.push_back(std::move(slot));
    }

    PrintInfo("[TRTBackend] Initialized {} context slot(s)", slot_count);
    return true;
  }

  void ClearSlots() {
    SynchronizeSlots();
    FreeAllSlotBuffers();

    for (auto& slot : slots) {
      if (!slot) continue;
      slot->context.reset();
      if (slot->stream) {
        cudaStreamDestroy(slot->stream);
        slot->stream = nullptr;
      }
    }
    slots.clear();
  }

  void SynchronizeSlots() {
    for (auto& slot : slots) {
      if (slot && slot->stream) {
        cudaStreamSynchronize(slot->stream);
      }
    }
  }

  void FreeAllSlotBuffers() {
    for (auto& slot : slots) {
      if (slot) {
        FreeBuffers(*slot);
      }
    }
  }

  void FreeBuffers(ContextSlot& slot) {
    if (slot.stream) {
      cudaStreamSynchronize(slot.stream);
    }

    for (auto& [name, ptr] : slot.gpu_buffers) {
      if (ptr) {
        if (slot.stream) {
          cudaFreeAsync(ptr, slot.stream);
        } else {
          cudaFree(ptr);
        }
        ptr = nullptr;
      }
    }

    if (slot.stream) {
      cudaStreamSynchronize(slot.stream);
    }

    slot.gpu_buffers.clear();
    slot.gpu_buffer_sizes.clear();
  }

  // 异步分配GPU内存
  void* AllocateGPUAsync(size_t bytes, cudaStream_t stream, const std::string& debug_name) {
    void* ptr = nullptr;
    cudaError_t err;

    if (mem_pool) {
      err = cudaMallocFromPoolAsync(&ptr, bytes, mem_pool, stream);
    } else {
      err = cudaMallocAsync(&ptr, bytes, stream);
    }

    if (err != cudaSuccess) {
      THROW_ERRORN("[TRTBackend] GPU allocation failed for '{}' ({} bytes): {}",
                   debug_name, bytes, cudaGetErrorString(err));
    }
    return ptr;
  }

  // 异步释放GPU内存
  void FreeGPUAsync(void* ptr, cudaStream_t stream) {
    if (!ptr) return;
    cudaFreeAsync(ptr, stream);
  }

  SlotLease BorrowSlot() {
    std::unique_lock<std::mutex> lock(slot_mutex);
    slot_cv.wait(lock, [this] {
      if (shutting_down) return true;
      return std::any_of(slots.begin(), slots.end(), [](const auto& slot) {
        return slot && !slot->busy;
      });
    });

    if (shutting_down) {
      THROW_ERROR("[TRTBackend] Cannot borrow slot while backend is shutting down");
    }

    for (auto& slot : slots) {
      if (slot && !slot->busy) {
        slot->busy = true;
        return SlotLease(this, slot.get());
      }
    }

    THROW_ERROR("[TRTBackend] No available context slot");
  }

  void ReleaseSlot(ContextSlot& slot) {
    {
      std::lock_guard<std::mutex> lock(slot_mutex);
      slot.busy = false;
    }
    slot_cv.notify_one();
  }
};

struct TensorRTBackend::ExecutionLease : public BaseModel::InferenceLease {
  explicit ExecutionLease(Impl::SlotLease&& lease_)
      : lease(std::make_unique<Impl::SlotLease>(std::move(lease_))) {}

  ContextSlot& Slot() { return lease->get(); }
  const ContextSlot& Slot() const { return lease->get(); }

  std::unique_ptr<Impl::SlotLease> lease;
};

namespace {
TensorRTBackend::ExecutionLease* AsTRTLease(BaseModel::InferenceLease* lease) {
  if (!lease) {
    THROW_ERROR("[TRTBackend] Inference lease is null");
  }
  auto* trt_lease = dynamic_cast<TensorRTBackend::ExecutionLease*>(lease);
  if (!trt_lease) {
    THROW_ERROR("[TRTBackend] Inference lease belongs to a different backend");
  }
  return trt_lease;
}
}  // namespace

// ============ 每个模型的 optimization profile 参数 ============

struct ProfileDef {
  std::string input_name;
  std::vector<int64_t> min_shape;
  std::vector<int64_t> opt_shape;
  std::vector<int64_t> max_shape;
};

static std::vector<ProfileDef> GetProfileDefs(const std::string& model_stem) {
  if (model_stem == "bert") {
    return {
      {"input_ids",      {1,1},   {1,100},  {1,256}},
      {"attention_mask", {1,1},   {1,100},  {1,256}},
      {"token_type_ids", {1,1},   {1,100},  {1,256}},
    };
  }
  if (model_stem == "gpt_encoder") {
    return {
      {"phoneme_ids",  {1,1},       {1,100},       {1,256}},
      {"prompts",      {1,1},       {1,150},       {1,300}},
      {"bert_feature", {1,1024,1},  {1,1024,100},  {1,1024,256}},
    };
  }
  if (model_stem == "gpt_step") {
    return {
      {"samples",  {1,1},                {1,1},                {1,1}},
      {"k_cache",  {24,1,1000,512},      {24,1,1000,512},      {24,1,1000,512}},
      {"v_cache",  {24,1,1000,512},      {24,1,1000,512},      {24,1,1000,512}},
      {"x_len",    {1},                  {1},                  {1}},
      {"y_len",    {1},                  {1},                  {1}},
      {"idx",      {1},                  {1},                  {1}},
    };
  }
  if (model_stem == "sovits") {
    return {
      {"pred_semantic", {1,1,1},    {1,1,120},    {1,1,250}},
      {"text_seq",      {1,1},      {1,50},       {1,100}},
      {"refer_spec",    {1,1025,1}, {1,1025,280}, {1,1025,400}},
    };
  }
  if (model_stem == "ssl") {
    return {{"audio", {1,16000}, {1,96000}, {1,200000}}};
  }
  if (model_stem == "vq_encoder") {
    return {{"ssl_content", {1,768,50}, {1,768,300}, {1,768,700}}};
  }
  if (model_stem == "spectrogram") {
    return {{"audio", {1,1}, {1,180000}, {1,400000}}};
  }
  if (model_stem == "sv_embedding") {
    return {{"audio", {1,16000}, {1,90000}, {1,180000}}};
  }
  return {};
}

// ============ TensorRTBackend 构造/析构 ============

TensorRTBackend::TensorRTBackend() : impl_(std::make_unique<Impl>()) {}
TensorRTBackend::~TensorRTBackend() = default;

// ============ Load 接口 ============

bool TensorRTBackend::Load(const std::string& model_path,
                           const Device& device, int work_thread_num) {
  BackendConfig cfg;
  cfg.device         = device;
  cfg.work_thread_num = work_thread_num;
  cfg.precision      = PrecisionMode::kAuto;
  return Load(model_path, cfg);
}

bool TensorRTBackend::Load(const std::string& model_path,
                           const BackendConfig& config) {
  TensorRTConfig trt;
  trt.base_config      = config;
  trt.engine_cache_dir = config.engine_cache_dir;
  if (config.precision == PrecisionMode::kFP32) {
    trt.enable_fp16 = false;
  } else if (config.precision == PrecisionMode::kFP16 ||
             config.precision == PrecisionMode::kMixed) {
    trt.enable_fp16 = true;
  }
  // kAuto 保持 默认值
  return Load(model_path, trt);
}

bool TensorRTBackend::Load(const std::string& model_path,
                           const TensorRTConfig& trt_config) {
  impl_->trt_config = trt_config;
  impl_->trt_config.base_config.Validate();
  impl_->device_id  = impl_->trt_config.base_config.device.device_id;
  device_           = impl_->trt_config.base_config.device;
  device_.stream    = nullptr;  // slot pool owns per-request streams
  config_           = impl_->trt_config.base_config;
  config_.device.stream = nullptr;

  {
    static bool printed = false;
    if (!printed) {
      printed = true;
      PrintInfo("[TRTBackend] TensorRT runtime version: {}.{}.{}",
                NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR, NV_TENSORRT_PATCH);
    }
  }

  cudaSetDevice(impl_->device_id);

  // 检测硬件 FP8 支持（Hopper sm_90 / Ada sm_89）
  {
    int cc_major = 0, cc_minor = 0;
    cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, impl_->device_id);
    cudaDeviceGetAttribute(&cc_minor, cudaDevAttrComputeCapabilityMinor, impl_->device_id);
    int sm = cc_major * 10 + cc_minor;
    impl_->trt_config.enable_fp8 = (sm >= 89);
    if (impl_->trt_config.enable_fp8) {
      PrintInfo("[TRTBackend] FP8 enabled (sm_{}{})", cc_major, cc_minor);
    }
  }

  fs::path p(model_path);
  std::string ext = p.extension().string();

  // 直接加载 .engine
  if (ext == ".engine" || ext == ".trt") {
    return LoadEngine(model_path);
  }

  // 从 .onnx 构建
  if (ext == ".onnx") {
    if (!impl_->trt_config.force_rebuild) {
      // 优先检查同目录下同名 .engine 文件
      fs::path sibling_engine = p.parent_path() / (p.stem().string() + ".engine");
      if (fs::exists(sibling_engine)) {
        PrintInfo("[TRTBackend] Found sibling engine, loading: {}", sibling_engine.string());
        if (LoadEngine(sibling_engine.string())) return true;
        PrintWarn("[TRTBackend] Sibling engine load failed, trying cache...");
      }

      // 再检查 engine_cache_dir
      if (!impl_->trt_config.engine_cache_dir.empty()) {
        std::string stem = p.stem().string();
        std::string prec = impl_->trt_config.enable_fp8  ? "fp8"  :
                           impl_->trt_config.enable_fp16 ? "fp16" : "fp32";
        fs::path cache = fs::path(impl_->trt_config.engine_cache_dir) /
                         (stem + "_" + std::to_string(impl_->device_id) +
                          "_" + prec + ".engine");
        if (fs::exists(cache)) {
          PrintInfo("[TRTBackend] Loading cached engine: {}", cache.string());
          if (LoadEngine(cache.string())) return true;
          PrintWarn("[TRTBackend] Cache load failed, rebuilding...");
        }
      }
    }
    return BuildFromONNX(model_path, impl_->trt_config);
  }

  PrintError("[TRTBackend] Unsupported file extension: {}", ext);
  return false;
}

// ============ BuildFromONNX ============

bool TensorRTBackend::BuildFromONNX(const std::string& onnx_path,
                                    const TensorRTConfig& cfg) {
  // 将路径转为原生分隔符，避免 Windows 上 TRT parseFromFile 不识别正斜杠
  std::string native_path = fs::path(onnx_path).make_preferred().string();
  PrintInfo("[TRTBackend] Building engine from ONNX: {}", native_path);

  TRTPtr<nvinfer1::IBuilder> builder(
      nvinfer1::createInferBuilder(g_trt_logger));
  if (!builder) { PrintError("[TRTBackend] createInferBuilder failed"); return false; }

  TRTPtr<nvinfer1::INetworkDefinition> network(
      builder->createNetworkV2(
          1U << static_cast<uint32_t>(
              nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)));
  if (!network) { PrintError("[TRTBackend] createNetworkV2 failed"); return false; }

  TRTPtr<nvonnxparser::IParser> parser(
      nvonnxparser::createParser(*network, g_trt_logger));
  if (!parser) { PrintError("[TRTBackend] createParser failed"); return false; }

  if (!parser->parseFromFile(native_path.c_str(),
        static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
    PrintError("[TRTBackend] ONNX parse failed: {}", native_path);
    return false;
  }

  TRTPtr<nvinfer1::IBuilderConfig> build_cfg(builder->createBuilderConfig());
  if (!build_cfg) { PrintError("[TRTBackend] createBuilderConfig failed"); return false; }

  build_cfg->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE,
                                cfg.max_workspace_size);
  build_cfg->setBuilderOptimizationLevel(cfg.optimization_level);

  if (cfg.enable_fp16 && builder->platformHasFastFp16()) {
    build_cfg->setFlag(nvinfer1::BuilderFlag::kFP16);
    PrintInfo("[TRTBackend] FP16 enabled");
  }
  if (cfg.enable_fp8 && builder->platformHasTf32()) {
    build_cfg->setFlag(nvinfer1::BuilderFlag::kFP8);
    PrintInfo("[TRTBackend] FP8 enabled");
  }
  if (cfg.enable_tf32) {
    build_cfg->setFlag(nvinfer1::BuilderFlag::kTF32);
  }

  // optimization profile
  fs::path p(native_path);
  std::string stem = p.stem().string();
  auto profiles = GetProfileDefs(stem);

  if (!profiles.empty()) {
    auto* opt_profile = builder->createOptimizationProfile();
    for (const auto& pd : profiles) {
      auto to_dims = [](const std::vector<int64_t>& v) {
        nvinfer1::Dims d;
        d.nbDims = static_cast<int>(v.size());
        for (int i = 0; i < d.nbDims; ++i) d.d[i] = static_cast<int32_t>(v[i]);
        return d;
      };
      opt_profile->setDimensions(pd.input_name.c_str(),
          nvinfer1::OptProfileSelector::kMIN, to_dims(pd.min_shape));
      opt_profile->setDimensions(pd.input_name.c_str(),
          nvinfer1::OptProfileSelector::kOPT, to_dims(pd.opt_shape));
      opt_profile->setDimensions(pd.input_name.c_str(),
          nvinfer1::OptProfileSelector::kMAX, to_dims(pd.max_shape));
    }
    build_cfg->addOptimizationProfile(opt_profile);
  }

  PrintInfo("[TRTBackend] Building engine (this may take a while)...");
  TRTPtr<nvinfer1::IHostMemory> serialized(
      builder->buildSerializedNetwork(*network, *build_cfg));
  if (!serialized) {
    PrintError("[TRTBackend] buildSerializedNetwork failed");
    return false;
  }

  // 保存缓存
  if (!cfg.engine_cache_dir.empty()) {
    std::string prec = cfg.enable_fp8  ? "fp8"  :
                       cfg.enable_fp16 ? "fp16" : "fp32";
    fs::path cache_dir(cfg.engine_cache_dir);
    fs::create_directories(cache_dir);
    fs::path cache = cache_dir /
                     (stem + "_" + std::to_string(impl_->device_id) +
                      "_" + prec + ".engine");
    std::ofstream ofs(cache.string(), std::ios::binary);
    if (ofs) {
      ofs.write(static_cast<const char*>(serialized->data()), serialized->size());
      PrintInfo("[TRTBackend] Engine cached to: {}", cache.string());
    }
  }

  // 反序列化
  impl_->ClearSlots();
  impl_->runtime.reset(nvinfer1::createInferRuntime(g_trt_logger));
  impl_->engine.reset(impl_->runtime->deserializeCudaEngine(
      serialized->data(), serialized->size()));
  if (!impl_->engine) {
    PrintError("[TRTBackend] deserializeCudaEngine failed");
    return false;
  }

  CollectIOMetadata();
  if (!impl_->InitializeSlots(impl_->trt_config.base_config.work_thread_num)) {
    return false;
  }
  PrintInfo("[TRTBackend] Engine built successfully from: {}", native_path);
  return true;
}

// ============ LoadEngine / SaveEngine ============

bool TensorRTBackend::LoadEngine(const std::string& engine_path) {
  std::ifstream ifs(engine_path, std::ios::binary | std::ios::ate);
  if (!ifs) {
    PrintError("[TRTBackend] Cannot open engine file: {}", engine_path);
    return false;
  }
  std::streamsize size = ifs.tellg();
  ifs.seekg(0, std::ios::beg);
  std::vector<char> buf(size);
  if (!ifs.read(buf.data(), size)) {
    PrintError("[TRTBackend] Failed to read engine file: {}", engine_path);
    return false;
  }

  impl_->ClearSlots();
  impl_->runtime.reset(nvinfer1::createInferRuntime(g_trt_logger));
  impl_->engine.reset(
      impl_->runtime->deserializeCudaEngine(buf.data(), buf.size()));
  if (!impl_->engine) {
    PrintError("[TRTBackend] deserializeCudaEngine failed");
    return false;
  }

  CollectIOMetadata();
  if (!impl_->InitializeSlots(impl_->trt_config.base_config.work_thread_num)) {
    return false;
  }
  PrintInfo("[TRTBackend] Engine loaded from: {}", engine_path);
  return true;
}

// ============ CollectIOMetadata ============

void TensorRTBackend::CollectIOMetadata() {
  impl_->input_names.clear();
  impl_->output_names.clear();
  impl_->input_types.clear();
  impl_->output_types.clear();

  int n = impl_->engine->getNbIOTensors();
  for (int i = 0; i < n; ++i) {
    const char* name = impl_->engine->getIOTensorName(i);
    auto mode = impl_->engine->getTensorIOMode(name);
    DataType dt = FromTRTType(impl_->engine->getTensorDataType(name));

    if (mode == nvinfer1::TensorIOMode::kINPUT) {
      impl_->input_names.push_back(name);
      impl_->input_types[name] = dt;
    } else {
      impl_->output_names.push_back(name);
      impl_->output_types[name] = dt;
    }
  }
}

// ============ DetermineInputType ============

DataType TensorRTBackend::DetermineInputType(DataType model_type) const {
  // 整数类型不做精度转换
  if (model_type == DataType::kInt32 || model_type == DataType::kInt64 ||
      model_type == DataType::kInt8  || model_type == DataType::kUInt8) {
    return model_type;
  }
  auto prec = impl_->trt_config.base_config.precision;
  if (prec == PrecisionMode::kFP16 || prec == PrecisionMode::kMixed) {
    return DataType::kFloat16;
  }
  if (prec == PrecisionMode::kFP32) {
    return DataType::kFloat32;
  }
  // kAuto: 跟随模型类型
  return model_type;
}

// ============ InferCore ============

bool TensorRTBackend::InferCore(
    ContextSlot& slot,
    const std::unordered_map<std::string, Tensor*>& inputs,
    const std::unordered_map<std::string, Tensor*>& outputs) {

  auto* ctx = slot.context.get();
  cudaStream_t stream = slot.stream;

  bool has_cpu_io = false;

  // 设置所有输入的动态shape
  for (const auto& [name, tensor] : inputs) {
    nvinfer1::Dims dims;
    dims.nbDims = static_cast<int>(tensor->Shape().size());
    for (int i = 0; i < dims.nbDims; ++i)
      dims.d[i] = static_cast<int32_t>(tensor->Shape()[i]);
    ctx->setInputShape(name.c_str(), dims);
    if (tensor->IsCPU()) has_cpu_io = true;
  }

  // 异步H2D拷贝
  for (const auto& [name, tensor] : inputs) {
    void* ptr = tensor->Data();

    if (tensor->IsCPU()) {
      // CPU输入需要异步拷贝到GPU
      const std::string buf_key = name + "_in";
      auto& gpu_buf  = slot.gpu_buffers[buf_key];
      auto& buf_size = slot.gpu_buffer_sizes[buf_key];
      size_t bytes   = tensor->ByteSize();

      // 仅在buffer不存在或不够大时才重新分配
      if (!gpu_buf || buf_size < bytes) {
        if (gpu_buf) {
          impl_->FreeGPUAsync(gpu_buf, stream);
        }
        gpu_buf = impl_->AllocateGPUAsync(bytes, stream, buf_key);
        buf_size = bytes;
      }

      // 异步H2D拷贝
      cudaError_t err = cudaMemcpyAsync(gpu_buf, ptr, bytes,
                                        cudaMemcpyHostToDevice, stream);
      if (err != cudaSuccess) {
        THROW_ERRORN("[TRTBackend] cudaMemcpyAsync H2D failed for input '{}': {}",
                     name, cudaGetErrorString(err));
      }
      ptr = gpu_buf;
    }

    ctx->setTensorAddress(name.c_str(), ptr);
  }

  // 绑定输出
  for (const auto& [name, tensor] : outputs) {
    void* ptr = tensor->Data();

    if (tensor->IsCPU()) {
      has_cpu_io = true;
      // CPU输出需要临时GPU buffer
      const std::string buf_key = name + "_out";
      auto& gpu_buf  = slot.gpu_buffers[buf_key];
      auto& buf_size = slot.gpu_buffer_sizes[buf_key];
      size_t bytes   = tensor->ByteSize();

      if (!gpu_buf || buf_size < bytes) {
        if (gpu_buf) {
          impl_->FreeGPUAsync(gpu_buf, stream);
        }
        gpu_buf = impl_->AllocateGPUAsync(bytes, stream, buf_key);
        buf_size = bytes;
      }
      ptr = gpu_buf;
    }

    ctx->setTensorAddress(name.c_str(), ptr);
  }

  // 异步推理
  bool ok = ctx->enqueueV3(stream);
  if (!ok) {
    PrintError("[TRTBackend] enqueueV3 failed");
    return false;
  }

  // 异步D2H拷贝输出
  for (const auto& [name, tensor] : outputs) {
    if (tensor->IsCPU()) {
      auto it = slot.gpu_buffers.find(name + "_out");
      if (it != slot.gpu_buffers.end() && it->second) {
        cudaError_t err = cudaMemcpyAsync(tensor->Data(), it->second,
                                          tensor->ByteSize(),
                                          cudaMemcpyDeviceToHost, stream);
        if (err != cudaSuccess) {
          THROW_ERRORN("[TRTBackend] cudaMemcpyAsync D2H failed for output '{}': {}",
                       name, cudaGetErrorString(err));
        }
      }
    }
  }

  // 第一版 slot 池采用同步归还，避免 context/buffer 被下个请求提前复用。
  cudaError_t sync_err = cudaStreamSynchronize(stream);
  if (sync_err != cudaSuccess) {
    PrintError("[TRTBackend] cudaStreamSynchronize failed: {}",
               cudaGetErrorString(sync_err));
    return false;
  }

  (void)has_cpu_io;
  return true;
}

// ============ Forward ============

void TensorRTBackend::Forward(
    const std::unordered_map<std::string, Tensor*>& inputs,
    std::unordered_map<std::string, std::unique_ptr<Tensor>>& outputs) {

  auto slot_lease = impl_->BorrowSlot();
  auto& slot = slot_lease.get();
  auto* ctx = slot.context.get();

  // 确定需要输出的名称
  std::vector<std::string> out_names;
  if (outputs.empty()) {
    out_names = impl_->output_names;
  } else {
    for (const auto& [name, _] : outputs) out_names.push_back(name);
  }

  // 先 setInputShape，才能正确查询动态输出 shape
  for (const auto& [name, tensor] : inputs) {
    nvinfer1::Dims dims;
    dims.nbDims = static_cast<int>(tensor->Shape().size());
    for (int i = 0; i < dims.nbDims; ++i)
      dims.d[i] = static_cast<int32_t>(tensor->Shape()[i]);
    ctx->setInputShape(name.c_str(), dims);
  }

  // 为每个输出分配 GPU Tensor
  std::unordered_map<std::string, Tensor*> out_ptrs;
  std::unordered_map<std::string, std::unique_ptr<Tensor>> tmp_tensors;

  for (const auto& name : out_names) {
    nvinfer1::Dims dims = ctx->getTensorShape(name.c_str());
    if (dims.nbDims <= 0)
      THROW_ERRORN("[TRTBackend] getTensorShape failed for output '{}' (nbDims={}), "
                   "check that all input shapes were set correctly", name, dims.nbDims);
    for (int i = 0; i < dims.nbDims; ++i) {
      if (dims.d[i] < 0)
        THROW_ERRORN("[TRTBackend] output '{}' has dynamic dim[{}]=-1 after shape inference, "
                     "input shapes may be incomplete", name, i);
    }
    std::vector<int64_t> shape(dims.d, dims.d + dims.nbDims);
    DataType dt = impl_->output_types.at(name);
    auto t = Tensor::Empty(shape, dt, Device(DeviceType::kCUDA, impl_->device_id));
    out_ptrs[name] = t.get();
    tmp_tensors[name] = std::move(t);
  }

  if (!InferCore(slot, inputs, out_ptrs)) {
    THROW_ERROR("[TRTBackend] Forward failed");
  }

  // 将 GPU tensor 深拷贝到 CPU
  for (const auto& name : out_names) {
    auto& gpu_t = tmp_tensors[name];
    auto cpu_t = gpu_t->ToDevice(Device(DeviceType::kCPU));
    outputs[name] = std::move(cpu_t);
  }
}

void TensorRTBackend::Forward(
    const std::unordered_map<std::string, Tensor*>& inputs,
    std::vector<std::unique_ptr<Tensor>>& outputs) {

  std::unordered_map<std::string, std::unique_ptr<Tensor>> out_map;
  Forward(inputs, out_map);

  outputs.clear();
  for (const auto& name : impl_->output_names) {
    outputs.push_back(std::move(out_map[name]));
  }
}

bool TensorRTBackend::ForwardWithPreallocatedOutput(
    const std::unordered_map<std::string, Tensor*>& inputs,
    std::unordered_map<std::string, Tensor*>& outputs) {
  try {
    auto slot_lease = impl_->BorrowSlot();
    return InferCore(slot_lease.get(), inputs, outputs);
  } catch (const std::exception& e) {
    PrintError("[TRTBackend] ForwardWithPreallocatedOutput failed: {}", e.what());
    return false;
  }
}

std::unique_ptr<BaseModel::InferenceLease> TensorRTBackend::AcquireInferenceLease() {
  try {
    return std::make_unique<ExecutionLease>(impl_->BorrowSlot());
  } catch (const std::exception& e) {
    PrintError("[TRTBackend] AcquireInferenceLease failed: {}", e.what());
    return nullptr;
  }
}

void TensorRTBackend::ForwardWithLease(
    InferenceLease* lease,
    const std::unordered_map<std::string, Tensor*>& inputs,
    std::unordered_map<std::string, std::unique_ptr<Tensor>>& outputs) {
  if (!lease) {
    Forward(inputs, outputs);
    return;
  }

  auto* trt_lease = AsTRTLease(lease);
  auto& slot = trt_lease->Slot();
  auto* ctx = slot.context.get();

  std::vector<std::string> out_names;
  if (outputs.empty()) {
    out_names = impl_->output_names;
  } else {
    for (const auto& [name, _] : outputs) out_names.push_back(name);
  }

  for (const auto& [name, tensor] : inputs) {
    nvinfer1::Dims dims;
    dims.nbDims = static_cast<int>(tensor->Shape().size());
    for (int i = 0; i < dims.nbDims; ++i)
      dims.d[i] = static_cast<int32_t>(tensor->Shape()[i]);
    ctx->setInputShape(name.c_str(), dims);
  }

  std::unordered_map<std::string, Tensor*> out_ptrs;
  std::unordered_map<std::string, std::unique_ptr<Tensor>> tmp_tensors;

  for (const auto& name : out_names) {
    nvinfer1::Dims dims = ctx->getTensorShape(name.c_str());
    if (dims.nbDims <= 0)
      THROW_ERRORN("[TRTBackend] getTensorShape failed for output '{}' (nbDims={}), "
                   "check that all input shapes were set correctly", name, dims.nbDims);
    for (int i = 0; i < dims.nbDims; ++i) {
      if (dims.d[i] < 0)
        THROW_ERRORN("[TRTBackend] output '{}' has dynamic dim[{}]=-1 after shape inference, "
                     "input shapes may be incomplete", name, i);
    }
    std::vector<int64_t> shape(dims.d, dims.d + dims.nbDims);
    DataType dt = impl_->output_types.at(name);
    auto t = Tensor::Empty(shape, dt, Device(DeviceType::kCUDA, impl_->device_id));
    out_ptrs[name] = t.get();
    tmp_tensors[name] = std::move(t);
  }

  if (!InferCore(slot, inputs, out_ptrs)) {
    THROW_ERROR("[TRTBackend] ForwardWithLease failed");
  }

  for (const auto& name : out_names) {
    auto& gpu_t = tmp_tensors[name];
    auto cpu_t = gpu_t->ToDevice(Device(DeviceType::kCPU));
    outputs[name] = std::move(cpu_t);
  }
}

bool TensorRTBackend::ForwardWithPreallocatedOutputWithLease(
    InferenceLease* lease,
    const std::unordered_map<std::string, Tensor*>& inputs,
    std::unordered_map<std::string, Tensor*>& outputs) {
  if (!lease) {
    return ForwardWithPreallocatedOutput(inputs, outputs);
  }

  try {
    auto* trt_lease = AsTRTLease(lease);
    return InferCore(trt_lease->Slot(), inputs, outputs);
  } catch (const std::exception& e) {
    PrintError("[TRTBackend] ForwardWithPreallocatedOutputWithLease failed: {}", e.what());
    return false;
  }
}

void TensorRTBackend::SynchronizeLease(InferenceLease* lease) {
  if (!lease) {
    Synchronize();
    return;
  }
  auto* trt_lease = AsTRTLease(lease);
  cudaStreamSynchronize(trt_lease->Slot().stream);
}

void* TensorRTBackend::GetLeaseStream(InferenceLease* lease) const {
  if (!lease) {
    return nullptr;
  }
  auto* trt_lease = AsTRTLease(lease);
  return static_cast<void*>(trt_lease->Slot().stream);
}

// ============ Getters ============

const std::vector<std::string>& TensorRTBackend::GetInputNames() const {
  return impl_->input_names;
}
const std::vector<std::string>& TensorRTBackend::GetOutputNames() const {
  return impl_->output_names;
}
DataType TensorRTBackend::GetInputDataType(const std::string& name) const {
  auto it = impl_->input_types.find(name);
  if (it == impl_->input_types.end()) THROW_ERRORN("TRT input not found: {}", name);
  return it->second;
}
DataType TensorRTBackend::GetOutputDataType(const std::string& name) const {
  auto it = impl_->output_types.find(name);
  if (it == impl_->output_types.end()) THROW_ERRORN("TRT output not found: {}", name);
  return it->second;
}

void TensorRTBackend::Synchronize() {
  impl_->SynchronizeSlots();
}

cudaStream_t TensorRTBackend::GetStream() const {
  // In slot-pool mode there is no single "current" request stream.
  // Keep this as a compatibility helper for callers that only need a stable
  // stream handle in single-slot usage.
  if (!impl_->slots.empty() && impl_->slots.front()) {
    return impl_->slots.front()->stream;
  }
  return nullptr;
}

// ============ BackendFactory ============

std::unique_ptr<BaseModel> BackendFactory::CreateBackend(BackendType type) {
  switch (type) {
    case BackendType::kTensorRT:
#ifdef WITH_TENSORRT
      return std::make_unique<TensorRTBackend>();
#else
      PrintWarn("[BackendFactory] TensorRT not compiled in");
      return nullptr;
#endif
    default:
      PrintWarn("[BackendFactory] Unknown backend type");
      return nullptr;
  }
}

bool BackendFactory::IsBackendAvailable(BackendType type) {
  switch (type) {
#ifdef WITH_ONNX
    case BackendType::kONNX:
      return true;
#endif
#ifdef WITH_TENSORRT
    case BackendType::kTensorRT:
      return true;
#endif
    default:
      return false;
  }
}

std::vector<BackendType> BackendFactory::GetAvailableBackends() {
  std::vector<BackendType> v;
#ifdef WITH_ONNX
  if (IsBackendAvailable(BackendType::kONNX))     v.push_back(BackendType::kONNX);
#endif
#ifdef WITH_TENSORRT
  if (IsBackendAvailable(BackendType::kTensorRT)) v.push_back(BackendType::kTensorRT);
#endif
  return v;
}

}  // namespace GPTSoVITS::Model

#else  // !WITH_TENSORRT

// ============ 无 TRT 编译时的存根 ============

namespace GPTSoVITS::Model {

struct TensorRTBackend::ContextSlot {};
struct TensorRTBackend::ExecutionLease : public BaseModel::InferenceLease {};
struct TensorRTBackend::Impl {};  // 空 Impl，满足 unique_ptr 析构

TensorRTBackend::TensorRTBackend()  : impl_(std::make_unique<Impl>()) {}
TensorRTBackend::~TensorRTBackend() = default;

bool TensorRTBackend::Load(const std::string&, const Device&, int) {
  PrintError("[TRTBackend] Not compiled with TensorRT support");
  return false;
}
bool TensorRTBackend::Load(const std::string&, const BackendConfig&) {
  PrintError("[TRTBackend] Not compiled with TensorRT support");
  return false;
}
bool TensorRTBackend::Load(const std::string&, const TensorRTConfig&) {
  PrintError("[TRTBackend] Not compiled with TensorRT support");
  return false;
}
void TensorRTBackend::Forward(const std::unordered_map<std::string, Tensor*>&,
                               std::unordered_map<std::string, std::unique_ptr<Tensor>>&) {
  THROW_ERROR("[TRTBackend] Not compiled with TensorRT support");
}
void TensorRTBackend::Forward(const std::unordered_map<std::string, Tensor*>&,
                               std::vector<std::unique_ptr<Tensor>>&) {
  THROW_ERROR("[TRTBackend] Not compiled with TensorRT support");
}
bool TensorRTBackend::ForwardWithPreallocatedOutput(
    const std::unordered_map<std::string, Tensor*>&,
    std::unordered_map<std::string, Tensor*>&) {
  PrintError("[TRTBackend] Not compiled with TensorRT support");
  return false;
}
std::unique_ptr<BaseModel::InferenceLease> TensorRTBackend::AcquireInferenceLease() {
  PrintError("[TRTBackend] Not compiled with TensorRT support");
  return nullptr;
}
void TensorRTBackend::ForwardWithLease(
    InferenceLease*,
    const std::unordered_map<std::string, Tensor*>&,
    std::unordered_map<std::string, std::unique_ptr<Tensor>>&) {
  THROW_ERROR("[TRTBackend] Not compiled with TensorRT support");
}
bool TensorRTBackend::ForwardWithPreallocatedOutputWithLease(
    InferenceLease*,
    const std::unordered_map<std::string, Tensor*>&,
    std::unordered_map<std::string, Tensor*>&) {
  PrintError("[TRTBackend] Not compiled with TensorRT support");
  return false;
}
void TensorRTBackend::SynchronizeLease(InferenceLease*) {}
void* TensorRTBackend::GetLeaseStream(InferenceLease*) const { return nullptr; }
const std::vector<std::string>& TensorRTBackend::GetInputNames() const {
  static std::vector<std::string> empty;
  return empty;
}
const std::vector<std::string>& TensorRTBackend::GetOutputNames() const {
  static std::vector<std::string> empty;
  return empty;
}
DataType TensorRTBackend::GetInputDataType(const std::string&) const {
  return DataType::kFloat32;
}
DataType TensorRTBackend::GetOutputDataType(const std::string&) const {
  return DataType::kFloat32;
}
void TensorRTBackend::Synchronize() {}
cudaStream_t TensorRTBackend::GetStream() const { return nullptr; }
bool TensorRTBackend::BuildFromONNX(const std::string&, const TensorRTConfig&) { return false; }
bool TensorRTBackend::LoadEngine(const std::string&) { return false; }
bool TensorRTBackend::SaveEngine(const std::string&) { return false; }
void TensorRTBackend::CollectIOMetadata() {}
DataType TensorRTBackend::DetermineInputType(DataType t) const { return t; }
bool TensorRTBackend::InferCore(ContextSlot&,
                                 const std::unordered_map<std::string, Tensor*>&,
                                 const std::unordered_map<std::string, Tensor*>&) { return false; }

std::unique_ptr<BaseModel> BackendFactory::CreateBackend(BackendType type) {
#ifdef WITH_ONNX
  if (type == BackendType::kONNX) {
    PrintWarn("[BackendFactory] Use ONNXBackend directly in model Init<>");
  } else {
#endif
    PrintWarn("[BackendFactory] Backend not available");
#ifdef WITH_ONNX
  }
#endif
  return nullptr;
}
bool BackendFactory::IsBackendAvailable(BackendType type) {
#ifdef WITH_ONNX
  return type == BackendType::kONNX;
#else
  (void)type;
  return false;
#endif
}
std::vector<BackendType> BackendFactory::GetAvailableBackends() {
#ifdef WITH_ONNX
  return {BackendType::kONNX};
#else
  return {};
#endif
}

}  // namespace GPTSoVITS::Model

#endif  // WITH_TENSORRT







