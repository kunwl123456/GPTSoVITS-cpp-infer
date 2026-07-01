//
// Created by 19254 on 2026/2/8.
//
#include "GPTSoVITS/InferencePipeline.h"

#include <algorithm>
#include <chrono>
#include <deque>
#include <numeric>
#include <random>

#include "GPTSoVITS/AudioTools.h"
#include "GPTSoVITS/Core/DeviceContext.h"
#include "GPTSoVITS/Core/ModelPool.h"
#include "GPTSoVITS/G2P/G2P_EN.h"
#include "GPTSoVITS/G2P/G2P_JA.h"
#include "GPTSoVITS/G2P/G2P_Zh.h"
#include "GPTSoVITS/G2P/Pipline.h"
#include "GPTSoVITS/Text/Sentence.h"
#include "GPTSoVITS/Utils/LoudnessNormalizer.h"
#include "GPTSoVITS/Utils/Precision.h"
#include "GPTSoVITS/Utils/Sampling.h"
#include "GPTSoVITS/Utils/speaker_serializer.h"
#include "GPTSoVITS/model/CNBertModel.h"
#ifdef WITH_ONNX
#include "GPTSoVITS/model/backend/onnx_backend.h"
#endif
#include "GPTSoVITS/model/backend/tensorrt_backend.h"
#ifdef WITH_CUDA
#include <cuda_runtime_api.h>
#endif
#include "GPTSoVITS/plog.h"
#include "fmt/xchar.h"
#include "nlohmann/json.hpp"
namespace GPTSoVITS {
// JSON 配置实现
class JsonConfig {
public:
  nlohmann::json data;
  void Parse(const std::string& content) {
    data = nlohmann::json::parse(content);
  }
  int GetSamplingRate() const {
    return data.value("/data/sampling_rate"_json_pointer, 32000);
  }
  int GetMaxLen() const {
    return data.value("/data/max_len"_json_pointer, 1000);
  }
  std::string GetModelVersion() const {
    return data.value("/model/version"_json_pointer, "v2");
  }
  int GetSVDim() const {
    return data.value("/sv_embedding/embedding_size"_json_pointer, 20480);
  }
};
class InferencePipeline::Impl {
public:
  PipelineConfig config;
  std::unique_ptr<Core::DeviceContext> device_ctx;
  Model::ModelPool& model_pool;
  SpeakerManager& speaker_manager;
  std::shared_ptr<G2P::G2PPipline> g2p_pipeline;
  JsonConfig json_config;
  // 配置参数
  int sampling_rate = 32000;
  int max_len = 1000;
  std::string model_version = "v2";
  int sv_dim = 20480;
  Model::DataType compute_precision = Model::DataType::kFloat32;
  Impl(const PipelineConfig& cfg)
      : config(cfg),
        model_pool(Model::ModelPool::Instance()),
        speaker_manager(SpeakerManager::Instance()) {
    Initialize();
  }
  ~Impl() {
    if (device_ctx) {
      device_ctx->Synchronize();
    }
    g2p_pipeline.reset();
    model_pool.UnloadAll();
  }
  void Initialize() {
    // 创建设备上下文
    Core::DeviceConfig device_config;
    device_config.preferred_device = config.device_type;
    device_config.device_id = config.device_id;
    device_config.compute_precision = config.compute_precision;
    device_config.thread_num = config.thread_num;
    device_config.verbose = config.verbose;
    device_ctx = std::make_unique<Core::DeviceContext>(device_config);
    // 设置默认设备上下文
    Core::DeviceContext::SetDefault(device_ctx.get());
    // 设置模型池和说话人管理器
    model_pool.SetDeviceContext(device_ctx.get());
    speaker_manager.SetDeviceContext(device_ctx.get());
    speaker_manager.SetModelPool(&model_pool);
    // 加载配置
    LoadConfig();
    // 加载模型
    LoadModels();
    // 加载 G2P
    LoadG2P();
    // 检测模型精度
    DetectModelPrecision();
    // 将 G2P Pipeline 传递给 SpeakerManager（用于创建说话人）
    speaker_manager.SetG2PPipeline(g2p_pipeline);
    PrintInfo("[InferencePipeline] Initialized {} mode pipeline",
              config.mode == PipelineMode::kFull   ? "Full"
              : config.mode == PipelineMode::kEdge ? "Edge"
                                                   : "Stream");
  }
  void DetectModelPrecision() {
    // 通过检查 GPT Encoder 的输入数据类型来检测模型精度
    auto gpt_encoder = model_pool.GetModel<Model::GPTEncoderModel>(
        Model::ModelType::kGPTEncoder);
    if (gpt_encoder && gpt_encoder->GetModel()) {
      // 注入 max_len，使 ONNX 动态形状后端能正确填充 KVCacheDesc::max_seq_len
      gpt_encoder->SetMaxLen(static_cast<int64_t>(max_len));

      auto dtype = gpt_encoder->GetModel()->GetInputDataType("bert_feature");
      if (dtype == Model::DataType::kFloat16) {
        compute_precision = Model::DataType::kFloat16;
        PrintInfo(
            "[InferencePipeline] Detected FP16 model from GPT Encoder input");
      } else {
        compute_precision = Model::DataType::kFloat32;
        PrintInfo(
            "[InferencePipeline] Detected FP32 model from GPT Encoder input");
      }
    } else {
      PrintWarn(
          "[InferencePipeline] Failed to detect model precision, defaulting to "
          "FP32");
      compute_precision = Model::DataType::kFloat32;
    }
  }
  void LoadConfig() {
    std::string config_path = config.config_path;
    if (config_path.empty()) {
      config_path = config.model_path + "/config.json";
    }
    try {
      std::ifstream file(config_path);
      if (file) {
        std::stringstream buffer;
        buffer << file.rdbuf();
        json_config.Parse(buffer.str());
        sampling_rate = json_config.GetSamplingRate();
        max_len = json_config.GetMaxLen();
        model_version = json_config.GetModelVersion();
        sv_dim = json_config.GetSVDim();
      }
    } catch (const std::exception& e) {
      PrintWarn("[InferencePipeline] Failed to load config: {}, using defaults",
                e.what());
    }
  }
  void LoadModels() {
    Model::Device device = device_ctx->GetDevice();
    // 根据模式注册模型
    if (config.mode == PipelineMode::kFull) {
      model_pool.RegisterModelGroup(Model::ModelGroup::kAll, config.model_path,
                                    device, config.compute_precision);
    } else {
      model_pool.RegisterModelGroup(Model::ModelGroup::kInference,
                                    config.model_path, device,
                                    config.compute_precision);
    }
    auto group = config.mode == PipelineMode::kFull
                     ? Model::ModelGroup::kAll
                     : Model::ModelGroup::kInference;
    for (auto type : Model::GetModelTypesInGroup(group)) {
      model_pool.UpdateConfig(type, [&](Model::ModelConfig& cfg) {
        cfg.thread_num = config.thread_num;
        if (config.backend != Model::BackendType::kAuto)
          cfg.backend = config.backend;
        if (!config.engine_cache_dir.empty())
          cfg.engine_cache_dir = config.engine_cache_dir;
      });
    }
    // 预加载模型
    if (config.mode == PipelineMode::kFull) {
      model_pool.PreloadModelGroup(Model::ModelGroup::kAll);
    } else {
      model_pool.PreloadModelGroup(Model::ModelGroup::kInference);
    }
  }
  void LoadG2P() {
    g2p_pipeline = std::make_shared<G2P::G2PPipline>();
    std::filesystem::path resources_path =
        config.resources_path.empty()
            ? std::filesystem::current_path() / "res"
            : std::filesystem::path(config.resources_path);
    // BERT
    auto bert_model = std::make_unique<Model::CNBertModel>();
    auto bert_onnx_path   = std::filesystem::path(config.model_path) / "bert.onnx";
    auto bert_engine_path = std::filesystem::path(config.model_path) / "bert.engine";
    auto tokenizer_path   = resources_path / "bert_tokenizer.json";
    // TRT backend 优先找 .engine，再找 .onnx
    bool use_trt = config.backend == Model::BackendType::kTensorRT;
#ifdef WITH_TENSORRT
    if (config.backend == Model::BackendType::kAuto &&
        device_ctx->GetDevice().type == Model::DeviceType::kCUDA) {
      use_trt = true;
    }
#endif
    auto bert_path = (use_trt && std::filesystem::exists(bert_engine_path))
                         ? bert_engine_path
                         : bert_onnx_path;
    if (std::filesystem::exists(bert_path)) {
      if (use_trt) {
        Model::BackendConfig bert_cfg;
        bert_cfg.device = device_ctx->GetDevice();
        bert_cfg.engine_cache_dir = config.engine_cache_dir;
        bert_model->Init<Model::TensorRTBackend>(
            bert_path.string(), tokenizer_path.string(), bert_cfg);
#ifdef WITH_ONNX
      } else {
        bert_model->Init<Model::ONNXBackend>(
            bert_onnx_path.string(), tokenizer_path.string(), device_ctx->GetDevice());
#else
      } else {
        PrintError("[InferencePipeline] ONNX backend not compiled in");
        return;
#endif
      }
      g2p_pipeline->RegisterLangProcess("zh", std::make_unique<G2P::G2PZH>(),
                                        std::move(bert_model), true);
    }
    g2p_pipeline->RegisterLangProcess("en", std::make_unique<G2P::G2PEN>(),
                                      nullptr, true);
    g2p_pipeline->RegisterLangProcess("ja", std::make_unique<G2P::G2PJA>(),
                                      nullptr, true);
    g2p_pipeline->SetDefaultLang(config.default_lang);
  }
  bool CanCreateSpeaker() const { return config.mode == PipelineMode::kFull; }
  std::unique_ptr<AudioTools> InferSegment(
      SpeakerFeatures& speaker, const std::string& text,
      const std::string& lang, const Model::SampleConfig& sample_config,
      float noise_scale, float speed, Model::InferStats* stats = nullptr) {
    PrintDebug("[InferencePipeline] Inferring segment: {}", text);
    auto device = device_ctx->GetDevice();

    // G2P 处理
    auto target_bert_res = g2p_pipeline->GetPhoneAndBert(text, lang);
    if (!target_bert_res || !target_bert_res->PhoneSeq ||
        !target_bert_res->BertSeq) {
      PrintError("[InferencePipeline] G2P processing failed");
      return nullptr;
    }

    // 获取模型
    auto gpt_encoder = model_pool.GetModel<Model::GPTEncoderModel>(
        Model::ModelType::kGPTEncoder);
    auto gpt_step =
        model_pool.GetModel<Model::GPTStepModel>(Model::ModelType::kGPTStep);
    auto sovits =
        model_pool.GetModel<Model::SoVITSModel>(Model::ModelType::kSoVITS);
    if (!gpt_encoder || !gpt_step || !sovits) {
      PrintError("[InferencePipeline] Models not loaded");
      return nullptr;
    }

    // 获取模型期望的数据类型和设备
    auto target_device = gpt_encoder->GetModel()->GetDevice();
    auto phone_dtype = gpt_encoder->GetModel()->GetInputDataType("phoneme_ids");
    auto bert_dtype = gpt_encoder->GetModel()->GetInputDataType("bert_feature");

    // 获取说话人特征
    auto ref_phones = speaker.GetPhoneSeq(device, phone_dtype);
    auto ref_bert = speaker.GetBertSeq(device, bert_dtype);
    auto vq_codes = speaker.GetVQCodes(device, phone_dtype);  // VQ codes使用phone_dtype
    auto refer_spec = speaker.GetReferSpec(device, Model::DataType::kFloat32);  // refer_spec通常为FP32
    auto sv_emb = speaker.GetSVEmbedding(device, Model::DataType::kFloat32);    // sv_emb通常为FP32

    if (!vq_codes || !refer_spec || !sv_emb) {
      PrintError("[InferencePipeline] Speaker features incomplete");
      return nullptr;
    }

    // 缓存命中
    if (device.type == Model::DeviceType::kCUDA) {
      PrintDebug("[InferencePipeline] Using GPU-cached speaker features (zero-copy)");
    }

    PrintDebug("[InferencePipeline] ref_phones={}, ref_bert={}",
               ref_phones ? fmt::format("[{}]", fmt::join(ref_phones->Shape(), ",")) : "null",
               ref_bert   ? fmt::format("[{}]", fmt::join(ref_bert->Shape(),   ",")) : "null");

    // ================================================================
    // 准备 GPT Encoder 输入
    // ================================================================

    // 一次性转换音素到目标设备和类型
    std::unique_ptr<Model::Tensor> ref_phones_final;
    std::unique_ptr<Model::Tensor> target_phones_final;

    if (ref_phones) {
      ref_phones_final = ref_phones->To(target_device, phone_dtype);
    }
    target_phones_final = target_bert_res->PhoneSeq->To(target_device, phone_dtype);

    // 拼接 phones (axis=0)
    std::vector<Model::Tensor*> phones_to_concat;
    if (ref_phones_final) phones_to_concat.push_back(ref_phones_final.get());
    phones_to_concat.push_back(target_phones_final.get());
    auto all_phones = Model::Tensor::Concat(phones_to_concat, 0);

    // 一次性转换 BERT 特征到目标设备和类型
    std::unique_ptr<Model::Tensor> ref_bert_final;
    std::unique_ptr<Model::Tensor> target_bert_final;

    if (ref_bert) {
      ref_bert_final = ref_bert->To(target_device, bert_dtype);
    }
    target_bert_final = target_bert_res->BertSeq->To(target_device, bert_dtype);

    // 拼接 bert (axis=1)
    std::vector<Model::Tensor*> bert_to_concat;
    if (ref_bert_final) bert_to_concat.push_back(ref_bert_final.get());
    bert_to_concat.push_back(target_bert_final.get());
    auto all_bert = Model::Tensor::Concat(bert_to_concat, 1);

    // 扩展维度 (2D -> 3D)，使用 View 避免拷贝
    if (all_bert->Shape().size() == 2) {
      all_bert->Reshape({1, all_bert->Shape()[0], all_bert->Shape()[1]});
    }

    // 准备 phoneme_ids
    auto phoneme_ids = std::move(all_phones);
    if (phoneme_ids->Shape().size() == 1) {
      phoneme_ids->Reshape({1, phoneme_ids->Shape()[0]});
    }

    // 准备 prompts
    auto prompt_dtype = gpt_encoder->GetModel()->GetInputDataType("prompts");
    auto prompts = vq_codes->To(target_device, prompt_dtype);
    if (prompts->Shape().size() == 1) {
      prompts->Reshape({1, prompts->Shape()[0]});
    }

    PrintDebug("[InferencePipeline] GPT Encoder inputs:");
    PrintDebug("  phoneme_ids: [{}, {}], bert_feature: [{}, {}, {}], prompts: [{}, {}]",
               phoneme_ids->Shape()[0], phoneme_ids->Shape()[1],
               all_bert->Shape()[0], all_bert->Shape()[1], all_bert->Shape()[2],
               prompts->Shape()[0], prompts->Shape()[1]);

    // ================================================================
    // GPT Encoder
    // ================================================================
    auto encoder_output =
        gpt_encoder->Encode(phoneme_ids.get(), prompts.get(), all_bert.get());

    // 采样第一个 token
    if (encoder_output.topk_indices && encoder_output.topk_values) {
      auto idx_cpu = encoder_output.topk_indices->ToCPU();
      auto val_cpu = encoder_output.topk_values->To(
          Model::Device(Model::DeviceType::kCPU), Model::DataType::kFloat32);
      std::string idx_str, val_str;
      int n = std::min((int)idx_cpu->ElementCount(), 5);
      for (int i = 0; i < n; ++i) {
        idx_str += std::to_string(idx_cpu->At<int64_t>(i)) + " ";
        val_str += std::to_string(val_cpu->Data<float>()[i]) + " ";
      }
      PrintDebug("[InferencePipeline] Encoder topk_indices dtype={} (first {}): {}",
                 static_cast<int>(encoder_output.topk_indices->Type()), n, idx_str);
      PrintDebug("[InferencePipeline] Encoder topk_values  dtype={} (first {}): {}",
                 static_cast<int>(encoder_output.topk_values->Type()), n, val_str);
    }
    int64_t first_token = Utils::SampleTopK(encoder_output.topk_values.get(),
                                            encoder_output.topk_indices.get(),
                                            sample_config.temperature);
    PrintDebug("[InferencePipeline] First sampled token: {}", first_token);

    // ================================================================
    // GPT Step 自回归生成
    // ================================================================
    const int64_t kv_max_len  = encoder_output.kv_cache->MaxSeqLen();
    const int64_t x_len_val   = encoder_output.x_len;
    const int64_t y_len_val   = encoder_output.y_len;
    const int64_t max_gen_len = kv_max_len - x_len_val - y_len_val - 1;
    const int max_steps = static_cast<int>(
        std::min((int64_t)1000, std::max((int64_t)1, max_gen_len)));
    PrintDebug("[InferencePipeline] KV cache: max={}, base={}+{}={}, max_gen_len={}, max_steps={}",
               kv_max_len, x_len_val, y_len_val, x_len_val + y_len_val, max_gen_len, max_steps);

    int64_t current_token = first_token;

    std::vector<int64_t> generated_tokens;
    const int64_t eos_token = 1024;

    // 始终将第一个 token 加入（即使是 EOS）
    generated_tokens.push_back(first_token);

    int consecutive_invalid_count = 0;
    const int max_consecutive_invalid = 10;

    // 记录 GPT 生成开始时间（用于计算 tokens/s）
    auto gpt_gen_start = std::chrono::steady_clock::now();

    // 如果第一个 token 就是 EOS，跳过循环
    if (first_token == eos_token) {
      PrintDebug("[InferencePipeline] First token is EOS, skipping generation loop");
    }

    // 双缓冲上下文 — CreateContext(KVCacheBuffer) 内部处理初始 k/v 拷贝
    int top_k = static_cast<int>(encoder_output.topk_values->Shape().back());
    auto gpt_ctx = gpt_step->CreateContext(
        *encoder_output.kv_cache,
        max_steps,
        top_k
    );

    bool use_gpu_sampling = false;
#ifdef WITH_CUDA
    if (target_device.type == Model::DeviceType::kCUDA) {
      use_gpu_sampling = gpt_step->EnableGPUSampling(gpt_ctx.get());
      if (use_gpu_sampling) {
        PrintInfo("[InferencePipeline] GPU sampling enabled for GPT generation");
      }
    }
#endif

    // GPU 采样路径
    if (use_gpu_sampling) {
#ifdef WITH_CUDA
      for (int step = 0; step < max_steps && first_token != eos_token; ++step) {
        bool step_ok = gpt_step->StepWithGPUSampling(
            gpt_ctx.get(),
            current_token,
            step,
            x_len_val,
            y_len_val,
            sample_config.temperature
        );

        if (!step_ok) {
          consecutive_invalid_count++;
          if (consecutive_invalid_count >= max_consecutive_invalid) {
            PrintError("[InferencePipeline] GPT Step failed {} times, stopping",
                       consecutive_invalid_count);
            break;
          }
          generated_tokens.push_back(current_token);
          continue;
        }

        consecutive_invalid_count = 0;

        // 从 GPU 获取采样的 token
        int64_t next_token = gpt_step->GetSampledTokenGPU(gpt_ctx.get());

        // Token 有效性检查
        if (next_token < 0 || next_token > eos_token) {
          next_token = std::clamp(next_token, (int64_t)0, eos_token);
        }

        current_token = next_token;
        generated_tokens.push_back(next_token);

        if (next_token == eos_token) {
          PrintDebug("[InferencePipeline] EOS at step {}", step);
          break;
        }
      }
#endif
    } else {
      // CPU 采样路径 (原有逻辑，优化了 D2H)
      std::unique_ptr<Model::Tensor> topk_values_cpu, topk_indices_cpu;
#ifdef WITH_CUDA
      if (target_device.type == Model::DeviceType::kCUDA) {
        void* pv = nullptr; void* pi = nullptr;
        cudaMallocHost(&pv, top_k * sizeof(float));
        cudaMallocHost(&pi, top_k * sizeof(int64_t));
        topk_values_cpu = Model::Tensor::CreateFromHost(
            pv, {1, top_k}, Model::DataType::kFloat32, [](void* p){ cudaFreeHost(p); });
        topk_indices_cpu = Model::Tensor::CreateFromHost(
            pi, {1, top_k}, Model::DataType::kInt64, [](void* p){ cudaFreeHost(p); });
      } else
#endif
      {
        topk_values_cpu = Model::Tensor::Empty(
            {1, top_k}, Model::DataType::kFloat32, Model::DeviceType::kCPU);
        topk_indices_cpu = Model::Tensor::Empty(
            {1, top_k}, Model::DataType::kInt64, Model::DeviceType::kCPU);
      }

      for (int step = 0; step < max_steps && first_token != eos_token; ++step) {

        bool step_ok = gpt_step->StepWithContext(
            gpt_ctx.get(),
            current_token,
            step,
            x_len_val,
            y_len_val
        );

        if (!step_ok) {
          consecutive_invalid_count++;
          if (consecutive_invalid_count >= max_consecutive_invalid) {
            PrintError("[InferencePipeline] GPT Step failed {} times, stopping",
                       consecutive_invalid_count);
            break;
          }
          generated_tokens.push_back(current_token);
          continue;
        }

        consecutive_invalid_count = 0;

        {
          auto tv = gpt_ctx->topk_values->To(Model::Device(Model::DeviceType::kCPU), Model::DataType::kFloat32);
          std::memcpy(topk_values_cpu->Data(), tv->Data(), topk_values_cpu->ByteSize());
        }
        {
          auto ti = gpt_ctx->topk_indices->To(Model::Device(Model::DeviceType::kCPU), Model::DataType::kInt64);
          std::memcpy(topk_indices_cpu->Data(), ti->Data(), topk_indices_cpu->ByteSize());
        }

        // 采样下一个 token
        int64_t next_token = Utils::SampleTopK(topk_values_cpu.get(),
                                               topk_indices_cpu.get(),
                                               sample_config.temperature);

        // Token 有效性检查
        if (next_token < 0 || next_token > eos_token) {
          next_token = std::clamp(next_token, (int64_t)0, eos_token);
        }

        // 更新 current_token
        current_token = next_token;
        generated_tokens.push_back(next_token);

        if (next_token == eos_token) {
          PrintDebug("[InferencePipeline] EOS at step {}", step);
          break;
        }
      }
    }

    // 计算 GPT 生成性能
    auto gpt_gen_end = std::chrono::steady_clock::now();
    double gpt_gen_time_s = std::chrono::duration<double>(gpt_gen_end - gpt_gen_start).count();
    double tokens_per_sec = generated_tokens.size() / gpt_gen_time_s;
    PrintInfo("[InferencePipeline] GPT generation: {} tokens in {:.3f}s ({:.2f} tokens/s)",
              generated_tokens.size(), gpt_gen_time_s, tokens_per_sec);
    if (stats) {
      stats->gpt_tokens += static_cast<int>(generated_tokens.size());
      stats->gpt_time_s += gpt_gen_time_s;
    }

    if (generated_tokens.empty()) {
      PrintWarn("[InferencePipeline] No tokens generated");
      return AudioTools::FromByte({}, sampling_rate);
    }

    PrintDebug("[InferencePipeline] Generated {} tokens", generated_tokens.size());

    // ================================================================
    // 构建 generated_sem
    // ================================================================
    int generated_len = static_cast<int>(generated_tokens.size());

    // 移除末尾的 EOS token
    if (!generated_tokens.empty() && generated_tokens.back() == eos_token) {
      generated_tokens.pop_back();
      generated_len--;
    }

    if (generated_len <= 0) {
      PrintWarn("[InferencePipeline] Generated semantic length is non-positive (text too short or model output EOS immediately), returning silence");
      std::vector<float> silence(sampling_rate / 10, 0.0f);  // 0.1秒静音
      return AudioTools::FromByte(silence, sampling_rate);
    }

    // 创建 generated_sem tensor (1, 1, generated_len)
    auto generated_sem = Model::Tensor::Empty({1, 1, generated_len},
                                              Model::DataType::kInt64,
                                              Model::DeviceType::kCPU);
    std::memcpy(generated_sem->Data<int64_t>(), generated_tokens.data(),
                generated_len * sizeof(int64_t));

    // ================================================================
    // SoVITS 解码
    // ================================================================
    auto sovits_device = sovits->GetModel()->GetDevice();
    auto pred_dtype = sovits->GetModel()->GetInputDataType("pred_semantic");
    auto text_dtype = sovits->GetModel()->GetInputDataType("text_seq");
    auto spec_dtype = sovits->GetModel()->GetInputDataType("refer_spec");
    auto sv_dtype = sovits->GetModel()->GetInputDataType("sv_emb");

    // 直接构建正确形状的输入，避免 reshape
    auto pred_semantic_final = generated_sem->To(sovits_device, pred_dtype);

    // text_seq: 直接从 PhoneSeq 构建 (1, seq_len)
    auto text_seq = target_bert_res->PhoneSeq->To(
        Model::Device(Model::DeviceType::kCPU), Model::DataType::kInt64);
    if (text_seq->Shape().size() == 1) {
      text_seq->Reshape({1, text_seq->Shape()[0]});
    }
    auto text_seq_final = text_seq->To(sovits_device, text_dtype);

    // refer_spec: 确保 3D
    Model::Tensor* refer_spec_ptr = refer_spec;
    std::unique_ptr<Model::Tensor> refer_spec_reshaped;
    if (refer_spec->Shape().size() == 2) {
      refer_spec_reshaped = refer_spec->View({1, refer_spec->Shape()[0],
                                              refer_spec->Shape()[1]});
      refer_spec_ptr = refer_spec_reshaped.get();
    }
    auto refer_spec_final = refer_spec_ptr->To(sovits_device, spec_dtype);

    // sv_emb: 确保 2D
    Model::Tensor* sv_emb_ptr = sv_emb;
    std::unique_ptr<Model::Tensor> sv_emb_reshaped;
    if (sv_emb->Shape().size() == 1) {
      sv_emb_reshaped = sv_emb->View({1, sv_emb->Shape()[0]});
      sv_emb_ptr = sv_emb_reshaped.get();
    }
    auto sv_emb_final = sv_emb_ptr->To(sovits_device, sv_dtype);

    // SoVITS 推理
    auto t_sovits_start = std::chrono::steady_clock::now();
    auto audio_tensor = sovits->GenerateTensor(
        pred_semantic_final.get(), text_seq_final.get(), refer_spec_final.get(),
        sv_emb_final.get(), noise_scale, speed);
    if (stats) {
      stats->sovits_time_s += std::chrono::duration<double>(
          std::chrono::steady_clock::now() - t_sovits_start).count();
    }

    if (!audio_tensor) {
      PrintError("[InferencePipeline] SoVITS generation failed");
      return nullptr;
    }

    // ================================================================
    // 音频后处理
    // ================================================================
    auto audio_cpu = audio_tensor->To(Model::Device(Model::DeviceType::kCPU),
                                      Model::DataType::kFloat32);
    if (!audio_cpu || audio_cpu->ElementCount() == 0) {
      PrintError("[InferencePipeline] Failed to convert audio to CPU");
      return AudioTools::FromByte({}, sampling_rate);
    }

    size_t audio_size = audio_cpu->ElementCount();
    const float* audio_ptr = audio_cpu->Data<float>();

    // 直接从指针构造 vector
    std::vector<float> audio_data(audio_ptr, audio_ptr + audio_size);

    // DC offset 去除 - 与 Python 一致
    // Python: audio_np = audio_np - np.mean(audio_np)
    float mean = 0.0f;
    for (const auto& s : audio_data) mean += s;
    mean /= static_cast<float>(audio_size);
    for (auto& s : audio_data) s -= mean;

    PrintDebug("[InferencePipeline] Audio: {} samples, DC offset: {:.6f}",
               audio_size, mean);

    return AudioTools::FromByte(audio_data, sampling_rate);
  }
};
InferencePipeline::InferencePipeline(const PipelineConfig& config)
    : impl_(std::make_unique<Impl>(config)) {}
InferencePipeline::~InferencePipeline() = default;
InferencePipeline::InferencePipeline(InferencePipeline&&) noexcept = default;
InferencePipeline& InferencePipeline::operator=(InferencePipeline&&) noexcept =
    default;
PipelineMode InferencePipeline::GetMode() const { return impl_->config.mode; }
const PipelineConfig& InferencePipeline::GetConfig() const {
  return impl_->config;
}
std::string InferencePipeline::GetModelInfo() const {
  std::ostringstream oss;
  oss << "InferencePipeline Info:\n";
  oss << "  Mode: "
      << (impl_->config.mode == PipelineMode::kFull   ? "Full"
          : impl_->config.mode == PipelineMode::kEdge ? "Edge"
                                                      : "Stream")
      << "\n";
  oss << "  Model path: " << impl_->config.model_path << "\n";
  oss << "  Device: "
      << (impl_->config.device_type == Model::DeviceType::kCUDA ? "CUDA"
                                                                : "CPU")
      << "\n";
  oss << "  Precision: "
      << (impl_->compute_precision == Model::DataType::kFloat16 ? "FP16"
                                                                : "FP32")
      << "\n";
  oss << "  Sampling rate: " << impl_->sampling_rate << " Hz\n";
  oss << "  Speakers: " << impl_->speaker_manager.SpeakerCount() << "\n";
  return oss.str();
}
bool InferencePipeline::CanCreateSpeaker() const {
  return impl_->CanCreateSpeaker();
}
// ============ 说话人管理 ============
SpeakerFeatures* InferencePipeline::CreateSpeaker(
    const std::string& name, const std::string& lang,
    const std::filesystem::path& ref_audio_path, const std::string& ref_text) {
  if (!CanCreateSpeaker()) {
    PrintError("[InferencePipeline] Cannot create speaker in Edge mode");
    return nullptr;
  }
  // TODO: 完整实现需要 G2P，后面再处理
  return impl_->speaker_manager.CreateSpeaker(name, lang, ref_audio_path,
                                              ref_text);
}
bool InferencePipeline::ImportSpeaker(const std::string& package_path,
                                      const std::string& rename) {
  return impl_->speaker_manager.ImportFromPackage(package_path, rename) !=
         nullptr;
}
bool InferencePipeline::ExportSpeaker(const std::string& name,
                                      const std::string& output_path,
                                      bool include_audio) {
  SpeakerExportOptions options;
  options.include_audio = include_audio;
  return impl_->speaker_manager.ExportToPackage(name, output_path, options);
}
SpeakerFeatures* InferencePipeline::GetSpeaker(const std::string& name) {
  return impl_->speaker_manager.GetSpeaker(name);
}
bool InferencePipeline::HasSpeaker(const std::string& name) const {
  return impl_->speaker_manager.HasSpeaker(name);
}
bool InferencePipeline::RemoveSpeaker(const std::string& name) {
  return impl_->speaker_manager.RemoveSpeaker(name);
}
std::vector<std::string> InferencePipeline::ListSpeakers() const {
  return impl_->speaker_manager.ListSpeakers();
}
// ============ 推理 ============
std::unique_ptr<AudioTools> InferencePipeline::Infer(
    const std::string& speaker_name, const std::string& text,
    const std::string& lang, const Model::SampleConfig& sample_config,
    float noise_scale, float speed, Model::InferStats* stats,
    std::function<void()> on_first_segment) {
  auto* speaker = GetSpeaker(speaker_name);
  if (!speaker) {
    PrintError("[InferencePipeline] Speaker not found: {}", speaker_name);
    return nullptr;
  }
  std::string use_lang = lang.empty() ? impl_->config.default_lang : lang;

  // 文本分句
  Text::Sentence sentence(Text::Sentence::SentenceSplitMethod::Punctuation);
  std::vector<std::string> segments;
  sentence.AppendCallBack([&segments](const std::string& s) -> bool {
    segments.push_back(s);
    return true;
  });

  sentence.Append(text);
  sentence.Flush();

  if (segments.empty()) {
    PrintWarn("[InferencePipeline] No text segments to process");
    return nullptr;
  }

  // len(current) + len(s) < 20 则合并
  std::vector<std::string> merged_segments;
  std::string current;
  for (const auto& s : segments) {
    if (current.size() + s.size() < 20) {
      current += s;
    } else {
      if (!current.empty()) {
        merged_segments.push_back(current);
      }
      current = s;
    }
  }
  if (!current.empty()) {
    merged_segments.push_back(current);
  }
  segments = std::move(merged_segments);

  PrintInfo("[InferencePipeline] Processing {} segments", segments.size());

  // 预加载说话人特征到设备（使用模型检测的精度）
  speaker->EnsureOnDevice(impl_->device_ctx->GetDevice(), impl_->compute_precision);

  // 处理每个段落
  std::vector<float> final_audio;
  for (size_t i = 0; i < segments.size(); ++i) {
    PrintDebug("[InferencePipeline] Processing segment {}/{}: {}",
               i + 1, segments.size(), segments[i]);

    auto audio = impl_->InferSegment(*speaker, segments[i], use_lang,
                                     sample_config, noise_scale, speed, stats);
    if (i == 0 && on_first_segment) {
      on_first_segment();
    }
    if (audio) {
      auto samples = audio->ReadSamples();

      // 每个 segment 进行 DC offset 去除
      final_audio.insert(final_audio.end(), samples.begin(), samples.end());

      // 添加段落间停顿
      if (i < segments.size() - 1) {
        int pause_samples = static_cast<int>(impl_->sampling_rate * 0.3f);
        final_audio.insert(final_audio.end(), pause_samples, 0.0f);
      }
    }
  }

  // ================================================================
  // RMS + Peak 组合归一化
  // 先进行 RMS 归一化到目标响度，再进行峰值限制防止削波
  // ================================================================
  if (!final_audio.empty()) {
    LoudnessConfig loudness_config;
    loudness_config.target_rms = 0.18f;      // 目标 RMS (~-15dBFS)
    loudness_config.max_gain = 10.0f;        // 最大增益
    loudness_config.min_gain = 0.1f;         // 最小增益
    loudness_config.enable_peak_limiting = true;
    loudness_config.peak_threshold = 0.9f;   // 峰值限制阈值

    LoudnessNormalizer normalizer(loudness_config);

    float gain = normalizer.NormalizeCombined(final_audio);
    float rms = normalizer.CalculateRMS(final_audio);
    float peak = normalizer.CalculatePeak(final_audio);

    PrintDebug("[InferencePipeline] Applied RMS normalization, gain: {:.4f}, "
               "final RMS: {:.4f}, peak: {:.4f}", gain, rms, peak);
  }

  PrintInfo("[InferencePipeline] Generated audio: {} samples ({:.2f}s)",
            final_audio.size(),
            static_cast<float>(final_audio.size()) / impl_->sampling_rate);

  return AudioTools::FromByte(final_audio, impl_->sampling_rate);
}
std::unique_ptr<AudioTools> InferencePipeline::Infer(
    const std::string& speaker_name, const std::string& text,
    float temperature) {
  Model::SampleConfig config;
  config.temperature = temperature;
  return Infer(speaker_name, text, "", config);
}
// ============ 流式推理 ============
namespace {
// 流推理配置
struct StreamingConfigInternal {
  int chunk_length = 24;      // 分块长度（token数）
  float pause_length = 0.3f;  // 段落间停顿（秒）
  int fade_length = 1280;     // 淡入淡出长度（采样点）
  int h_len = 512;            // 历史token长度
  int l_len = 16;             // 前瞻token长度
  bool enable_fade = true;    // 是否启用淡入淡出
};
// 流式解码单个分块
std::vector<float> DecodeChunkStreaming(
    Model::SoVITSModel* sovits_model, const std::vector<int64_t>& chunk_tokens,
    const std::vector<int64_t>& history_tokens,
    const std::vector<int64_t>& lookahead_tokens,
    const std::vector<int64_t>& target_phones, Model::Tensor* refer_spec,
    Model::Tensor* sv_emb, const StreamingConfigInternal& stream_config,
    float noise_scale, float speed, int sampling_rate) {
  if (chunk_tokens.empty()) {
    return {};
  }
  // 构建输入tokens：历史 + 当前 + 前瞻
  std::vector<int64_t> input_tokens;
  // 添加历史tokens（最后h_len个）
  if (!history_tokens.empty()) {
    size_t h_start =
        history_tokens.size() > static_cast<size_t>(stream_config.h_len)
            ? history_tokens.size() - stream_config.h_len
            : 0;
    input_tokens.insert(input_tokens.end(), history_tokens.begin() + h_start,
                        history_tokens.end());
  }
  // 添加当前分块
  input_tokens.insert(input_tokens.end(), chunk_tokens.begin(),
                      chunk_tokens.end());
  // 添加前瞻tokens（前l_len个）
  if (!lookahead_tokens.empty()) {
    input_tokens.insert(input_tokens.end(), lookahead_tokens.begin(),
                        lookahead_tokens.end());
  }
  // 准备 pred_semantic 张量 (1, 1, seq_len)
  auto pred_semantic =
      Model::Tensor::Empty({1, 1, static_cast<int64_t>(input_tokens.size())},
                           Model::DataType::kInt64, Model::DeviceType::kCPU);
  int64_t* semantic_ptr = pred_semantic->Data<int64_t>();
  std::memcpy(semantic_ptr, input_tokens.data(),
              input_tokens.size() * sizeof(int64_t));
  // 准备 text_seq 张量 (1, phone_len)
  auto text_seq =
      Model::Tensor::Empty({1, static_cast<int64_t>(target_phones.size())},
                           Model::DataType::kInt64, Model::DeviceType::kCPU);
  int64_t* text_ptr = text_seq->Data<int64_t>();
  std::memcpy(text_ptr, target_phones.data(),
              target_phones.size() * sizeof(int64_t));
  // 准备 refer_spec
  auto refer_spec_input = refer_spec->Clone();
  auto refer_shape = refer_spec_input->Shape();
  if (refer_shape.size() == 2) {
    refer_spec_input->Reshape({1, refer_shape[0], refer_shape[1]});
  }
  // 准备 sv_emb
  auto sv_emb_input = sv_emb->Clone();
  auto sv_shape = sv_emb_input->Shape();
  if (sv_shape.size() == 1) {
    sv_emb_input->Reshape({1, sv_shape[0]});
  }
  auto target_device = sovits_model->GetModel()->GetDevice();
  auto pred_dtype = sovits_model->GetModel()->GetInputDataType("pred_semantic");
  auto text_dtype = sovits_model->GetModel()->GetInputDataType("text_seq");
  auto spec_dtype = sovits_model->GetModel()->GetInputDataType("refer_spec");
  auto sv_dtype = sovits_model->GetModel()->GetInputDataType("sv_emb");
  // 类型转换到目标设备
  auto pred_semantic_final = pred_semantic->To(target_device, pred_dtype);
  auto text_seq_final = text_seq->To(target_device, text_dtype);
  auto refer_spec_final = refer_spec_input->To(target_device, spec_dtype);
  auto sv_emb_final = sv_emb_input->To(target_device, sv_dtype);
  // SoVITS 推理
  auto audio_tensor = sovits_model->GenerateTensor(
      pred_semantic_final.get(), text_seq_final.get(), refer_spec_final.get(),
      sv_emb_final.get(), noise_scale, speed);
  if (!audio_tensor || audio_tensor->ElementCount() == 0) {
    return {};
  }
  // 提取音频数据
  auto audio_cpu = audio_tensor->ToCPU();
  size_t audio_size = audio_cpu->ElementCount();
  const float* audio_ptr = audio_cpu->Data<float>();
  // 立即将音频数据拷贝到 vector
  std::vector<float> audio_vec(audio_ptr, audio_ptr + audio_size);
  // 计算应该返回的样本数（仅当前分块的部分）
  int samples_per_token = static_cast<int>((sampling_rate / 25) / speed);
  size_t effective_history =
      std::min(history_tokens.size(), static_cast<size_t>(stream_config.h_len));
  int h_samples = static_cast<int>(effective_history) * samples_per_token;
  int chunk_samples = static_cast<int>(chunk_tokens.size()) * samples_per_token;
  std::vector<float> result;
  audio_size = audio_vec.size();
  if (static_cast<size_t>(h_samples) >= audio_size) {
    size_t return_size =
        std::min(audio_size, static_cast<size_t>(chunk_samples));
    result.assign(audio_vec.begin(), audio_vec.begin() + return_size);
  } else {
    size_t available = audio_size - h_samples;
    size_t return_size =
        std::min(static_cast<size_t>(chunk_samples), available);
    if (h_samples + return_size > audio_size) {
      return_size = audio_size > static_cast<size_t>(h_samples)
                        ? audio_size - h_samples
                        : 0;
    }
    if (return_size > 0) {
      result.assign(audio_vec.begin() + h_samples,
                    audio_vec.begin() + h_samples + return_size);
    }
  }
  return result;
}
}  // anonymous namespace
bool InferencePipeline::InferStreaming(const std::string& speaker_name,
                                       const std::string& text,
                                       const std::string& lang,
                                       AudioChunkCallback callback,
                                       const Model::SampleConfig& sample_config,
                                       float noise_scale, float speed) {
  auto* speaker = impl_->speaker_manager.GetSpeaker(speaker_name);
  if (!speaker) {
    PrintError("[InferencePipeline::InferStreaming] Speaker not found: {}",
               speaker_name);
    return false;
  }
  if (!callback) {
    PrintError("[InferencePipeline::InferStreaming] Callback is null");
    return false;
  }
  // 验证采样配置
  if (!sample_config.Validate()) {
    PrintError("[InferencePipeline::InferStreaming] Invalid sample config");
    return false;
  }
  std::string use_lang = lang.empty() ? impl_->config.default_lang : lang;
  // 文本分句
  Text::Sentence sentence(Text::Sentence::SentenceSplitMethod::Punctuation);
  std::vector<std::string> segments;
  sentence.AppendCallBack([&segments](const std::string& s) -> bool {
    segments.push_back(s);
    return true;
  });
  int chunk_size = 11;
  int index = 0;
  while (index < static_cast<int>(text.size())) {
    std::string chunk = text.substr(index, chunk_size);
    sentence.Append(chunk);
    index += chunk_size;
  }
  sentence.Flush();
  if (segments.empty()) {
    PrintWarn(
        "[InferencePipeline::InferStreaming] No text segments to process");
    return false;
  }

  // 与 Python 对齐：短句合并逻辑 (len(current) + len(s) < 20 则合并)
  std::vector<std::string> merged_segments;
  std::string current;
  for (const auto& s : segments) {
    if (current.size() + s.size() < 20) {
      current += s;
    } else {
      if (!current.empty()) {
        merged_segments.push_back(current);
      }
      current = s;
    }
  }
  if (!current.empty()) {
    merged_segments.push_back(current);
  }
  segments = std::move(merged_segments);

  PrintInfo(
      "[InferencePipeline::InferStreaming] Starting streaming inference for "
      "speaker: {}, segments: {}",
      speaker_name, segments.size());

  // 创建流式响度归一化器
  LoudnessConfig loudness_config;
  loudness_config.target_rms = 0.18f;
  loudness_config.smoothing_factor = 0.9f;
  loudness_config.enable_peak_limiting = true;
  loudness_config.peak_threshold = 0.9f;
  LoudnessNormalizer loudness_normalizer(loudness_config);

  // 预加载说话人特征到设备（使用模型检测的精度）
  auto device = impl_->device_ctx->GetDevice();
  speaker->EnsureOnDevice(device, impl_->compute_precision);
  // 获取模型
  auto gpt_encoder = impl_->model_pool.GetModel<Model::GPTEncoderModel>(
      Model::ModelType::kGPTEncoder);
  auto gpt_step = impl_->model_pool.GetModel<Model::GPTStepModel>(
      Model::ModelType::kGPTStep);
  auto sovits =
      impl_->model_pool.GetModel<Model::SoVITSModel>(Model::ModelType::kSoVITS);
  if (!gpt_encoder || !gpt_step || !sovits) {
    PrintError("[InferencePipeline::InferStreaming] Models not loaded");
    return false;
  }
  // 获取模型期望的数据类型
  auto phone_dtype = gpt_encoder->GetModel()->GetInputDataType("phoneme_ids");
  auto bert_dtype = gpt_encoder->GetModel()->GetInputDataType("bert_feature");

  // 流推理配置
  StreamingConfigInternal stream_config;
  // 获取说话人特征（使用模型期望的数据类型）
  auto ref_phones =
      speaker->GetPhoneSeq(Model::Device(Model::DeviceType::kCPU), phone_dtype);
  auto ref_bert = speaker->GetBertSeq(Model::Device(Model::DeviceType::kCPU), bert_dtype);
  auto vq_codes = speaker->GetVQCodes(device, phone_dtype);  // VQ codes使用phone_dtype
  auto refer_spec = speaker->GetReferSpec(device, Model::DataType::kFloat32);  // refer_spec通常为FP32
  auto sv_emb = speaker->GetSVEmbedding(device, Model::DataType::kFloat32);    // sv_emb通常为FP32
  if (!vq_codes || !refer_spec || !sv_emb) {
    PrintError(
        "[InferencePipeline::InferStreaming] Speaker features incomplete");
    return false;
  }
  std::vector<float> prev_fade_out;
  int sampling_rate = impl_->sampling_rate;
  int segment_index = 0;
  // 处理每个段落
  for (size_t seg_idx = 0; seg_idx < segments.size(); ++seg_idx) {
    const std::string& segment = segments[seg_idx];
    PrintDebug(
        "[InferencePipeline::InferStreaming] Processing segment {}/{}: {}",
        seg_idx + 1, segments.size(), segment);
    // G2P 处理
    auto target_bert_res =
        impl_->g2p_pipeline->GetPhoneAndBert(segment, use_lang);
    if (!target_bert_res || !target_bert_res->PhoneSeq ||
        !target_bert_res->BertSeq) {
      PrintError(
          "[InferencePipeline::InferStreaming] G2P processing failed for "
          "segment: {}",
          segment);
      continue;
    }
    // 获取目标音素序列
    auto target_phones_tensor = target_bert_res->PhoneSeq;
    std::vector<int64_t> target_phones;
    {
      auto cpu_phones = target_phones_tensor->ToCPU();
      const int64_t* ptr = cpu_phones->Data<int64_t>();
      size_t count = cpu_phones->ElementCount();
      target_phones.assign(ptr, ptr + count);
    }
    // 获取模型期望的数据类型和设备
    auto target_device = gpt_encoder->GetModel()->GetDevice();
    auto phone_dtype = gpt_encoder->GetModel()->GetInputDataType("phoneme_ids");
    auto bert_dtype = gpt_encoder->GetModel()->GetInputDataType("bert_feature");
    auto prompt_dtype = gpt_encoder->GetModel()->GetInputDataType("prompts");
    // 拼接参考和目标的音素
    std::vector<Model::Tensor*> phones_to_concat;
    auto ref_phones_final =
        ref_phones ? ref_phones->To(target_device, phone_dtype) : nullptr;
    auto target_phones_final =
        target_phones_tensor->To(target_device, phone_dtype);
    if (ref_phones_final) phones_to_concat.push_back(ref_phones_final.get());
    phones_to_concat.push_back(target_phones_final.get());
    auto all_phones = Model::Tensor::Concat(phones_to_concat, 0);
    // 拼接参考和目标的 BERT 特征
    std::vector<Model::Tensor*> bert_to_concat;
    auto ref_bert_final =
        ref_bert ? ref_bert->To(target_device, bert_dtype) : nullptr;
    auto target_bert_final =
        target_bert_res->BertSeq->To(target_device, bert_dtype);
    if (ref_bert_final) bert_to_concat.push_back(ref_bert_final.get());
    bert_to_concat.push_back(target_bert_final.get());
    auto all_bert = Model::Tensor::Concat(bert_to_concat, 1);
    // 扩展维度
    if (all_bert->Shape().size() == 2) {
      all_bert =
          all_bert->View({1, all_bert->Shape()[0], all_bert->Shape()[1]});
    }
    // 准备 phoneme_ids 和 prompts
    auto phoneme_ids = all_phones->To(Model::Device(Model::DeviceType::kCPU),
                                      Model::DataType::kInt64);
    if (phoneme_ids->Shape().size() == 1) {
      phoneme_ids->Reshape({1, phoneme_ids->Shape()[0]});
    }
    auto prompts = vq_codes->To(Model::Device(Model::DeviceType::kCPU),
                                Model::DataType::kInt64);
    if (prompts->Shape().size() == 1) {
      prompts->Reshape({1, prompts->Shape()[0]});
    }
    // GPT Encoder
    auto encoder_output =
        gpt_encoder->Encode(phoneme_ids.get(), prompts.get(), all_bert.get());
    // 采样第一个 token
    int64_t first_token = Utils::SampleTopK(encoder_output.topk_values.get(),
                                            encoder_output.topk_indices.get(),
                                            sample_config.temperature);

    // 创建 GPT Step 上下文（内部处理 dtype 转换和双缓冲）
    const int max_steps = 1500;
    int s_top_k = static_cast<int>(encoder_output.topk_values->Shape().back());
    auto s_ctx = gpt_step->CreateContext(*encoder_output.kv_cache, max_steps, s_top_k);

    std::vector<int64_t> generated_tokens;
    std::deque<std::vector<int64_t>> chunk_queue;
    std::vector<int64_t> history_tokens;
    const int64_t eos_token = 1024;
    if (first_token != eos_token) {
      generated_tokens.push_back(first_token);
    }
    // 边生成边解码
    int token_counter = 0;
    int chunk_index = 0;
    std::vector<float> last_fade_out = prev_fade_out;
    auto decode_and_yield = [&](std::vector<int64_t>& chunk_tokens,
                                const std::vector<int64_t>* lookahead_ptr,
                                bool is_final_chunk) {
      // 获取前瞻tokens
      std::vector<int64_t> lookahead_tokens;
      if (lookahead_ptr && !lookahead_ptr->empty()) {
        size_t l_len = std::min(lookahead_ptr->size(),
                                static_cast<size_t>(stream_config.l_len));
        lookahead_tokens.assign(lookahead_ptr->begin(),
                                lookahead_ptr->begin() + l_len);
      }
      // 解码
      auto audio_data = DecodeChunkStreaming(
          sovits.get(), chunk_tokens, history_tokens, lookahead_tokens,
          target_phones, refer_spec, sv_emb, stream_config, noise_scale, speed,
          sampling_rate);
      if (audio_data.empty()) {
        return;
      }
      // 应用淡入淡出
      if (!last_fade_out.empty() && stream_config.enable_fade) {
        int fade_len = std::min(static_cast<int>(last_fade_out.size()),
                                static_cast<int>(audio_data.size()));
        for (int i = 0; i < fade_len; ++i) {
          float fade_in = static_cast<float>(i) / fade_len;
          audio_data[i] =
              audio_data[i] * fade_in + last_fade_out[i] * (1.0f - fade_in);
        }
      }
      // 计算淡出
      if (!is_final_chunk && stream_config.enable_fade &&
          audio_data.size() > static_cast<size_t>(stream_config.fade_length)) {
        last_fade_out.assign(audio_data.end() - stream_config.fade_length,
                             audio_data.end());
        for (int i = 0; i < stream_config.fade_length; ++i) {
          float fade_out =
              1.0f - static_cast<float>(i) / stream_config.fade_length;
          audio_data[audio_data.size() - stream_config.fade_length + i] *=
              fade_out;
        }
      } else {
        last_fade_out.clear();
      }
      // 更新历史tokens
      history_tokens.insert(history_tokens.end(), chunk_tokens.begin(),
                            chunk_tokens.end());
      if (history_tokens.size() > static_cast<size_t>(stream_config.h_len)) {
        history_tokens.erase(history_tokens.begin(),
                             history_tokens.begin() +
                                 (history_tokens.size() - stream_config.h_len));
      }
      // 创建音频分块
      AudioChunk chunk;
      chunk.audio_data = audio_data;
      chunk.is_first = (segment_index == 0 && chunk_index == 0);
      chunk.is_last = is_final_chunk && (seg_idx == segments.size() - 1);
      chunk.segment_index = segment_index;
      chunk.chunk_index = chunk_index;
      chunk.duration = static_cast<float>(audio_data.size()) / sampling_rate;

      // 应用流式响度归一化
      loudness_normalizer.NormalizeStreaming(chunk.audio_data);

      callback(chunk);
      chunk_index++;
    };
    // GPT Step 自回归生成
    int64_t s_current_token = first_token;
    int consecutive_invalid_count = 0;
    const int max_consecutive_invalid = 10;
    for (int step = 0; step < max_steps; ++step) {
      bool step_ok = gpt_step->StepWithContext(
          s_ctx.get(), s_current_token, step,
          encoder_output.x_len, encoder_output.y_len);

      if (!step_ok) {
        consecutive_invalid_count++;
        if (consecutive_invalid_count >= max_consecutive_invalid) {
          PrintError(
              "[InferencePipeline::InferStreaming] GPT Step failed {} times "
              "consecutively, terminating at step {}",
              consecutive_invalid_count, step);
          break;
        }
        generated_tokens.push_back(s_current_token);
        token_counter++;
        continue;
      }

      consecutive_invalid_count = 0;

      // 采样下一个 token
      int64_t next_token = Utils::SampleTopK(s_ctx->topk_values.get(),
                                             s_ctx->topk_indices.get(),
                                             sample_config.temperature);
      // 检查 EOS
      if (next_token == eos_token) {
        break;
      }
      generated_tokens.push_back(next_token);
      token_counter++;
      s_current_token = next_token;

      // 当token数量达到chunk_length时进行分割
      bool is_split = false;
      if (token_counter >= stream_config.chunk_length) {
        std::vector<int64_t> chunk_tokens(
            generated_tokens.end() - token_counter, generated_tokens.end());
        chunk_queue.push_back(std::move(chunk_tokens));
        token_counter = 0;
        is_split = true;
      }
      // 如果有多个待解码块，解码第一个
      if (is_split && chunk_queue.size() > 1) {
        auto chunk_to_decode = std::move(chunk_queue.front());
        chunk_queue.pop_front();
        const std::vector<int64_t>* lookahead_ptr =
            chunk_queue.empty() ? nullptr : &chunk_queue.front();
        decode_and_yield(chunk_to_decode, lookahead_ptr, false);
      }
    }
    // 处理剩余的tokens
    if (token_counter > 0) {
      std::vector<int64_t> remaining_tokens(
          generated_tokens.end() - token_counter, generated_tokens.end());
      chunk_queue.push_back(std::move(remaining_tokens));
    }
    // 解码所有剩余的分块
    while (!chunk_queue.empty()) {
      auto chunk_to_decode = std::move(chunk_queue.front());
      chunk_queue.pop_front();
      bool is_final = chunk_queue.empty();
      const std::vector<int64_t>* lookahead_ptr =
          chunk_queue.empty() ? nullptr : &chunk_queue.front();
      decode_and_yield(chunk_to_decode, lookahead_ptr, is_final);
    }
    prev_fade_out = last_fade_out;
    segment_index++;
    // 添加段落间停顿
    if (seg_idx < segments.size() - 1 && stream_config.pause_length > 0) {
      int pause_samples =
          static_cast<int>(sampling_rate * stream_config.pause_length);
      std::vector<float> pause_audio(pause_samples, 0.0f);
      AudioChunk pause_chunk;
      pause_chunk.audio_data = pause_audio;
      pause_chunk.is_first = false;
      pause_chunk.is_last = false;
      pause_chunk.segment_index = segment_index - 1;
      pause_chunk.chunk_index = -1;
      pause_chunk.duration = stream_config.pause_length;
      callback(pause_chunk);
      prev_fade_out.clear();
    }
  }
  PrintInfo(
      "[InferencePipeline::InferStreaming] Streaming inference completed");
  return true;
}
// ============ 底层组件 ============
Core::DeviceContext& InferencePipeline::GetDeviceContext() {
  return *impl_->device_ctx;
}
Model::ModelPool& InferencePipeline::GetModelPool() {
  return impl_->model_pool;
}
SpeakerManager& InferencePipeline::GetSpeakerManager() {
  return impl_->speaker_manager;
}
}  // namespace GPTSoVITS
