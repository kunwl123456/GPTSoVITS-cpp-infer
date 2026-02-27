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

#include <filesystem>
#include <fstream>
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

struct TensorRTBackend::Impl {
  TRTPtr<nvinfer1::IRuntime>          runtime;
  TRTPtr<nvinfer1::ICudaEngine>       engine;
  TRTPtr<nvinfer1::IExecutionContext> context;

  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
  std::unordered_map<std::string, DataType> input_types;
  std::unordered_map<std::string, DataType> output_types;
  std::unordered_map<std::string, std::vector<int64_t>> output_shapes;

  // name -> device ptr
  std::unordered_map<std::string, void*>   gpu_buffers;
  std::unordered_map<std::string, size_t>  gpu_buffer_sizes;  // 追踪每个buffer的已分配大小

  std::unordered_map<std::string, nvinfer1::Dims> cached_input_shapes;

  cudaStream_t own_stream = nullptr;

  TensorRTConfig trt_config;
  int device_id = 0;

  ~Impl() {
    FreeBuffers();
    if (own_stream) { cudaStreamDestroy(own_stream); own_stream = nullptr; }
  }

  void EnsureStream() {
    if (!own_stream) {
      cudaSetDevice(device_id);
      cudaStreamCreate(&own_stream);
    }
  }

  void FreeBuffers() {
    for (auto& [name, ptr] : gpu_buffers) {
      if (ptr) { cudaFree(ptr); ptr = nullptr; }
    }
    gpu_buffers.clear();
    gpu_buffer_sizes.clear();
  }
};

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
      {"input_ids",      {1,1},   {1,128},  {1,512}},
      {"attention_mask", {1,1},   {1,128},  {1,512}},
      {"token_type_ids", {1,1},   {1,128},  {1,512}},
    };
  }
  if (model_stem == "gpt_encoder") {
    return {
      {"phoneme_ids",  {1,1},       {1,100},       {1,512}},
      {"prompts",      {1,1},       {1,50},        {1,512}},
      {"bert_feature", {1,1024,1},  {1,1024,100},  {1,1024,512}},
    };
  }
  if (model_stem == "gpt_step") {
    return {
      {"samples",  {1,1},                {1,1},                {1,1}},
      {"k_cache",  {24,1,1,512},         {24,1,500,512},       {24,1,1000,512}},
      {"v_cache",  {24,1,1,512},         {24,1,500,512},       {24,1,1000,512}},
      {"x_len",    {1},                  {1},                  {1}},
      {"y_len",    {1},                  {1},                  {1}},
      {"idx",      {1},                  {1},                  {1}},
    };
  }
  if (model_stem == "sovits") {
    return {
      {"pred_semantic", {1,1,1},    {1,1,200},    {1,1,1000}},
      {"text_seq",      {1,1},      {1,100},      {1,512}},
      {"refer_spec",    {1,1025,1}, {1,1025,200}, {1,1025,1000}},
    };
  }
  if (model_stem == "ssl") {
    return {{"audio", {1,16000}, {1,160000}, {1,800000}}};
  }
  if (model_stem == "vq_encoder") {
    return {{"ssl_content", {1,768,50}, {1,768,500}, {1,768,5000}}};
  }
  if (model_stem == "spectrogram") {
    return {{"audio", {1,1}, {1,48000}, {1,960000}}};
  }
  if (model_stem == "sv_embedding") {
    return {{"audio", {1,16000}, {1,48000}, {1,160000}}};
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
  impl_->device_id  = trt_config.base_config.device.device_id;
  device_           = trt_config.base_config.device;
  config_           = trt_config.base_config;

  {
    static bool printed = false;
    if (!printed) {
      printed = true;
      PrintInfo("[TRTBackend] TensorRT runtime version: {}.{}.{}",
                NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR, NV_TENSORRT_PATCH);
    }
  }

  cudaSetDevice(impl_->device_id);

  // Create CUDA stream immediately to avoid first-inference latency
  if (!impl_->own_stream) {
    cudaError_t err = cudaStreamCreate(&impl_->own_stream);
    if (err != cudaSuccess) {
      PrintError("[TRTBackend] cudaStreamCreate failed: {}", cudaGetErrorString(err));
      return false;
    }
  }

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
  impl_->runtime.reset(nvinfer1::createInferRuntime(g_trt_logger));
  impl_->engine.reset(impl_->runtime->deserializeCudaEngine(
      serialized->data(), serialized->size()));
  if (!impl_->engine) {
    PrintError("[TRTBackend] deserializeCudaEngine failed");
    return false;
  }

  impl_->context.reset(impl_->engine->createExecutionContext());
  if (!impl_->context) {
    PrintError("[TRTBackend] createExecutionContext failed");
    return false;
  }

  CollectIOMetadata();
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

  impl_->runtime.reset(nvinfer1::createInferRuntime(g_trt_logger));
  impl_->engine.reset(
      impl_->runtime->deserializeCudaEngine(buf.data(), buf.size()));
  if (!impl_->engine) {
    PrintError("[TRTBackend] deserializeCudaEngine failed");
    return false;
  }

  impl_->context.reset(impl_->engine->createExecutionContext());
  if (!impl_->context) {
    PrintError("[TRTBackend] createExecutionContext failed");
    return false;
  }

  CollectIOMetadata();
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
    const std::unordered_map<std::string, Tensor*>& inputs,
    const std::unordered_map<std::string, Tensor*>& outputs) {

  auto* ctx = impl_->context.get();

  // 绑定输入，仅在 shape 变化时设置动态 shape
  for (const auto& [name, tensor] : inputs) {
    // 构建当前 shape
    nvinfer1::Dims dims;
    dims.nbDims = static_cast<int>(tensor->Shape().size());
    for (int i = 0; i < dims.nbDims; ++i)
      dims.d[i] = static_cast<int32_t>(tensor->Shape()[i]);

    auto it = impl_->cached_input_shapes.find(name);
    bool shape_changed = false;
    if (it == impl_->cached_input_shapes.end()) {
      shape_changed = true;
    } else {
      const auto& cached = it->second;
      if (cached.nbDims != dims.nbDims) {
        shape_changed = true;
      } else {
        for (int i = 0; i < dims.nbDims; ++i) {
          if (cached.d[i] != dims.d[i]) {
            shape_changed = true;
            break;
          }
        }
      }
    }

    if (shape_changed) {
      ctx->setInputShape(name.c_str(), dims);
      impl_->cached_input_shapes[name] = dims;
    }

    void* ptr = tensor->Data();
    // 若输入在 CPU，需先拷贝到 GPU
    if (tensor->IsCPU()) {
      const std::string buf_key = name + "_in";
      auto& gpu_buf  = impl_->gpu_buffers[buf_key];
      auto& buf_size = impl_->gpu_buffer_sizes[buf_key];
      size_t bytes   = tensor->ByteSize();
      // 仅在 buffer 不存在或不够大时才重新分配
      if (!gpu_buf || buf_size < bytes) {
        if (gpu_buf) cudaFree(gpu_buf);
        cudaError_t err = cudaMalloc(&gpu_buf, bytes);
        if (err != cudaSuccess)
          THROW_ERRORN("[TRTBackend] cudaMalloc failed for input '{}': {}", name, cudaGetErrorString(err));
        buf_size = bytes;
      }
      // 使用异步拷贝提升性能
      cudaMemcpyAsync(gpu_buf, ptr, bytes, cudaMemcpyHostToDevice, impl_->own_stream);
      ptr = gpu_buf;
    } else {
      // GPU tensor: 直接使用指针,无需拷贝
    }
    ctx->setTensorAddress(name.c_str(), ptr);
  }

  // 绑定输出
  for (const auto& [name, tensor] : outputs) {
    void* ptr = tensor->Data();
    if (tensor->IsCPU()) {
      // 输出预分配在 CPU：先用临时 GPU buffer，推理后拷回
      const std::string buf_key = name + "_out";
      auto& gpu_buf  = impl_->gpu_buffers[buf_key];
      auto& buf_size = impl_->gpu_buffer_sizes[buf_key];
      size_t bytes   = tensor->ByteSize();
      if (!gpu_buf || buf_size < bytes) {
        if (gpu_buf) cudaFree(gpu_buf);
        cudaError_t err = cudaMalloc(&gpu_buf, bytes);
        if (err != cudaSuccess)
          THROW_ERRORN("[TRTBackend] cudaMalloc failed for output '{}': {}", name, cudaGetErrorString(err));
        buf_size = bytes;
      }
      ptr = gpu_buf;
    }
    ctx->setTensorAddress(name.c_str(), ptr);
  }

  // 执行推理
  cudaStream_t stream = impl_->own_stream;
  bool ok = ctx->enqueueV3(stream);
  if (!ok) {
    PrintError("[TRTBackend] enqueueV3 failed");
    return false;
  }

  // 若输出在 CPU，拷回
  for (const auto& [name, tensor] : outputs) {
    if (tensor->IsCPU()) {
      auto it = impl_->gpu_buffers.find(name + "_out");
      if (it != impl_->gpu_buffers.end() && it->second) {
        cudaMemcpyAsync(tensor->Data(), it->second, tensor->ByteSize(),
                        cudaMemcpyDeviceToHost, stream);
      }
    }
  }

  // 注意: 不在这里同步!
  // 对于 ForwardWithPreallocatedOutput (IO Binding 路径),
  // 调用者负责在需要时同步 (例如访问 CPU 输出前)
  // 这样可以让多个推理调用流水线化,减少同步开销

  return true;
}

// ============ Forward ============

void TensorRTBackend::Forward(
    const std::unordered_map<std::string, Tensor*>& inputs,
    std::unordered_map<std::string, std::unique_ptr<Tensor>>& outputs) {

  // 确定需要输出的名称
  std::vector<std::string> out_names;
  if (outputs.empty()) {
    out_names = impl_->output_names;
  } else {
    for (const auto& [name, _] : outputs) out_names.push_back(name);
  }

  // 为每个输出分配 GPU Tensor
  // Note: InferCore will handle setInputShape with caching, so we need to call it first
  // to get the correct output shapes after shape inference
  std::unordered_map<std::string, Tensor*> out_ptrs;
  std::unordered_map<std::string, std::unique_ptr<Tensor>> tmp_tensors;

  // Temporarily set input shapes to infer output shapes
  auto* ctx = impl_->context.get();
  for (const auto& [name, tensor] : inputs) {
    nvinfer1::Dims dims;
    dims.nbDims = static_cast<int>(tensor->Shape().size());
    for (int i = 0; i < dims.nbDims; ++i)
      dims.d[i] = static_cast<int32_t>(tensor->Shape()[i]);

    auto it = impl_->cached_input_shapes.find(name);
    bool need_set = (it == impl_->cached_input_shapes.end() ||
                     it->second.nbDims != dims.nbDims);
    if (!need_set && it != impl_->cached_input_shapes.end()) {
      for (int i = 0; i < dims.nbDims; ++i) {
        if (it->second.d[i] != dims.d[i]) {
          need_set = true;
          break;
        }
      }
    }
    if (need_set) {
      ctx->setInputShape(name.c_str(), dims);
      impl_->cached_input_shapes[name] = dims;
    }
  }

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

  if (!InferCore(inputs, out_ptrs)) {
    THROW_ERROR("[TRTBackend] Forward failed");
  }

  // 保持 GPU tensor 在 GPU 上，与 ONNX 后端行为一致
  // 调用者可以根据需要决定是否拷贝到 CPU
  for (const auto& name : out_names) {
    outputs[name] = std::move(tmp_tensors[name]);
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
    return InferCore(inputs, outputs);
  } catch (const std::exception& e) {
    PrintError("[TRTBackend] ForwardWithPreallocatedOutput failed: {}", e.what());
    return false;
  }
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
  if (impl_->own_stream) {
    cudaStreamSynchronize(impl_->own_stream);
  }
}

cudaStream_t TensorRTBackend::GetStream() const {
  return impl_->own_stream;
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
    case BackendType::kONNX:
#ifdef WITH_ONNX
      return true;
#else
      return false;
#endif
    case BackendType::kTensorRT:
#ifdef WITH_TENSORRT
      return true;
#else
      return false;
#endif
    default:
      return false;
  }
}

std::vector<BackendType> BackendFactory::GetAvailableBackends() {
  std::vector<BackendType> v;
  if (IsBackendAvailable(BackendType::kONNX))     v.push_back(BackendType::kONNX);
  if (IsBackendAvailable(BackendType::kTensorRT)) v.push_back(BackendType::kTensorRT);
  return v;
}

}  // namespace GPTSoVITS::Model

#else  // !WITH_TENSORRT

// ============ 无 TRT 编译时的存根 ============

namespace GPTSoVITS::Model {

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
bool TensorRTBackend::BuildFromONNX(const std::string&, const TensorRTConfig&) { return false; }
bool TensorRTBackend::LoadEngine(const std::string&) { return false; }
bool TensorRTBackend::SaveEngine(const std::string&) { return false; }
void TensorRTBackend::CollectIOMetadata() {}
DataType TensorRTBackend::DetermineInputType(DataType t) const { return t; }
bool TensorRTBackend::InferCore(const std::unordered_map<std::string, Tensor*>&,
                                 const std::unordered_map<std::string, Tensor*>&) { return false; }

std::unique_ptr<BaseModel> BackendFactory::CreateBackend(BackendType type) {
  if (type == BackendType::kONNX) {
    PrintWarn("[BackendFactory] Use ONNXBackend directly in model Init<>");
  } else {
    PrintWarn("[BackendFactory] Backend not available");
  }
  return nullptr;
}
bool BackendFactory::IsBackendAvailable(BackendType type) {
  return type == BackendType::kONNX;
}
std::vector<BackendType> BackendFactory::GetAvailableBackends() {
  return {BackendType::kONNX};
}

}  // namespace GPTSoVITS::Model

#endif  // WITH_TENSORRT







