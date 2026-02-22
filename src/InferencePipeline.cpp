//
// Created by 19254 on 2026/2/8.
//

#include "GPTSoVITS/InferencePipeline.h"

#include <algorithm>
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
#include "GPTSoVITS/Utils/speaker_serializer.h"
#include "GPTSoVITS/model/CNBertModel.h"
#include "GPTSoVITS/model/backend/onnx_backend.h"
#include "GPTSoVITS/plog.h"
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
      auto dtype = gpt_encoder->GetModel()->GetInputDataType("bert_feature");
      if (dtype == Model::DataType::kFloat16) {
        compute_precision = Model::DataType::kFloat16;
        PrintInfo("[InferencePipeline] Detected FP16 model from GPT Encoder input");
      } else {
        compute_precision = Model::DataType::kFloat32;
        PrintInfo("[InferencePipeline] Detected FP32 model from GPT Encoder input");
      }
    } else {
      PrintWarn("[InferencePipeline] Failed to detect model precision, defaulting to FP32");
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

    // BERT 模型
    auto bert_model = std::make_unique<Model::CNBertModel>();
    auto bert_path = std::filesystem::path(config.model_path) / "bert.onnx";
    auto tokenizer_path = resources_path / "bert_tokenizer.json";

    if (std::filesystem::exists(bert_path)) {
      bert_model->Init<Model::ONNXBackend>(
          bert_path.string(), tokenizer_path.string(), device_ctx->GetDevice());

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
      float noise_scale, float speed) {
    PrintDebug("[InferencePipeline] Inferring segment: {}", text);

    auto device = device_ctx->GetDevice();

    // G2P 处理
    auto target_bert_res = g2p_pipeline->GetPhoneAndBert(text, lang);
    if (!target_bert_res || !target_bert_res->PhoneSeq ||
        !target_bert_res->BertSeq) {
      PrintError("[InferencePipeline] G2P processing failed");
      return nullptr;
    }

    // 获取说话人特征
    auto ref_phones =
        speaker.GetPhoneSeq(Model::Device(Model::DeviceType::kCPU));
    auto ref_bert = speaker.GetBertSeq(Model::Device(Model::DeviceType::kCPU));
    auto vq_codes = speaker.GetVQCodes(device);
    auto refer_spec = speaker.GetReferSpec(device);
    auto sv_emb = speaker.GetSVEmbedding(device);

    if (!vq_codes || !refer_spec || !sv_emb) {
      PrintError("[InferencePipeline] Speaker features incomplete");
      return nullptr;
    }

    // 参考特征
    if (ref_phones) {
      PrintDebug("[InferencePipeline] ref_phones shape: [{}]", ref_phones->Shape()[0]);
      if (ref_phones->ElementCount() > 5) {
        auto* p = ref_phones->Data<int64_t>();
        PrintDebug("[InferencePipeline] ref_phones[0..4]: {}, {}, {}, {}, {}", p[0], p[1], p[2], p[3], p[4]);
      }
    } else {
      PrintWarn("[InferencePipeline] ref_phones is NULL!");
    }
    if (ref_bert) {
      PrintDebug("[InferencePipeline] ref_bert shape: [{}, {}]",
                ref_bert->Shape().size() > 0 ? ref_bert->Shape()[0] : 0,
                ref_bert->Shape().size() > 1 ? ref_bert->Shape()[1] : 0);
    } else {
      PrintWarn("[InferencePipeline] ref_bert is NULL!");
    }
    PrintDebug("[InferencePipeline] vq_codes shape: [{}, {}]",
              vq_codes->Shape().size() > 0 ? vq_codes->Shape()[0] : 0,
              vq_codes->Shape().size() > 1 ? vq_codes->Shape()[1] : 0);

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
    auto prompt_dtype = gpt_encoder->GetModel()->GetInputDataType("prompts");

    // 准备输入
    auto ref_phones_final =
        ref_phones ? ref_phones->To(target_device, phone_dtype) : nullptr;
    auto target_phones_final =
        target_bert_res->PhoneSeq->To(target_device, phone_dtype);

    // 在 axis=0 上拼接 phones
    std::vector<Model::Tensor*> phones_to_concat;
    if (ref_phones_final) phones_to_concat.push_back(ref_phones_final.get());
    phones_to_concat.push_back(target_phones_final.get());
    auto all_phones = Model::Tensor::Concat(phones_to_concat, 0);

    // 在 axis=1 上拼接 bert
    auto ref_bert_final =
        ref_bert ? ref_bert->To(target_device, bert_dtype) : nullptr;
    auto target_bert_final =
        target_bert_res->BertSeq->To(target_device, bert_dtype);

    std::vector<Model::Tensor*> bert_to_concat;
    if (ref_bert_final) bert_to_concat.push_back(ref_bert_final.get());
    bert_to_concat.push_back(target_bert_final.get());
    auto all_bert = Model::Tensor::Concat(bert_to_concat, 1);

    PrintDebug("[InferencePipeline] ref_bert shape: [{}, {}]",
              ref_bert ? (ref_bert->Shape().size() > 0 ? ref_bert->Shape()[0] : 0) : 0,
              ref_bert ? (ref_bert->Shape().size() > 1 ? ref_bert->Shape()[1] : 0) : 0);
    PrintDebug("[InferencePipeline] target_bert shape: [{}, {}]",
              target_bert_res->BertSeq->Shape().size() > 0 ? target_bert_res->BertSeq->Shape()[0] : 0,
              target_bert_res->BertSeq->Shape().size() > 1 ? target_bert_res->BertSeq->Shape()[1] : 0);
    PrintDebug("[InferencePipeline] all_bert (before reshape) shape: [{}, {}, {}]",
              all_bert->Shape().size() > 0 ? all_bert->Shape()[0] : 0,
              all_bert->Shape().size() > 1 ? all_bert->Shape()[1] : 0,
              all_bert->Shape().size() > 2 ? all_bert->Shape()[2] : 0);

    // 扩展维度
    if (all_bert->Shape().size() == 2) {
      all_bert->Reshape({1, all_bert->Shape()[0], all_bert->Shape()[1]});
      PrintDebug("[InferencePipeline] Reshaped all_bert to: [{}, {}, {}]",
                all_bert->Shape()[0], all_bert->Shape()[1], all_bert->Shape()[2]);
    }

    // 确保精度一致
    if (all_bert->Type() != compute_precision) {
      PrintDebug("[InferencePipeline] Converting bert_feature from {} to {} for GPT Encoder",
                all_bert->Type() == Model::DataType::kFloat32 ? "float32" : "float16",
                compute_precision == Model::DataType::kFloat32 ? "float32" : "float16");
      all_bert = all_bert->To(all_bert->GetDevice(), compute_precision);
    }

    // 准备 phoneme_ids 和 prompts
    auto phoneme_ids = all_phones->To(Model::Device(Model::DeviceType::kCPU), Model::DataType::kInt64);
    if (phoneme_ids->Shape().size() == 1) {
      phoneme_ids->Reshape({1, phoneme_ids->Shape()[0]});
    }

    auto prompts = vq_codes->To(Model::Device(Model::DeviceType::kCPU), Model::DataType::kInt64);
    if (prompts->Shape().size() == 1) {
      prompts->Reshape({1, prompts->Shape()[0]});
    }

    PrintDebug("[InferencePipeline] GPT Encoder inputs:");
    PrintDebug("  phoneme_ids shape: [{}, {}], dtype: {}",
              phoneme_ids->Shape()[0], phoneme_ids->Shape()[1],
              phoneme_ids->Type() == Model::DataType::kInt64 ? "int64" : "other");
    if (phoneme_ids->ElementCount() > 5) {
      auto* p_ids = phoneme_ids->Data<int64_t>();
      PrintDebug("  phoneme_ids[0..4]: {}, {}, {}, {}, {}", p_ids[0], p_ids[1], p_ids[2], p_ids[3], p_ids[4]);
    }
    PrintDebug("  prompts shape: [{}, {}], dtype: {}",
              prompts->Shape()[0], prompts->Shape()[1],
              prompts->Type() == Model::DataType::kInt64 ? "int64" : "other");
    if (prompts->ElementCount() > 5) {
      auto* p_prompts = prompts->Data<int64_t>();
      PrintDebug("  prompts[0..4]: {}, {}, {}, {}, {}", p_prompts[0], p_prompts[1], p_prompts[2], p_prompts[3], p_prompts[4]);
    }
    PrintDebug("  bert_feature shape: [{}, {}, {}], dtype: {}",
              all_bert->Shape()[0], all_bert->Shape()[1], all_bert->Shape()[2],
              all_bert->Type() == Model::DataType::kFloat16 ? "float16" : "float32");

    // GPT Encoder
    auto encoder_output =
        gpt_encoder->Encode(phoneme_ids.get(), prompts.get(), all_bert.get());

    // 检查 encoder 输出
    if (encoder_output.topk_values && encoder_output.topk_values->ElementCount() > 0) {
      auto topk_values_cpu = encoder_output.topk_values->To(
          Model::Device(Model::DeviceType::kCPU), Model::DataType::kFloat32);
      auto topk_indices_cpu = encoder_output.topk_indices->To(
          Model::Device(Model::DeviceType::kCPU), Model::DataType::kInt64);
      const float* enc_values = topk_values_cpu->Data<float>();
      const int64_t* enc_indices = topk_indices_cpu->Data<int64_t>();
      int enc_k = topk_values_cpu->ElementCount();

      PrintDebug("[InferencePipeline] GPT Encoder topk_values[0..5]: {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}",
                enc_values[0], enc_values[1], enc_values[2], enc_values[3], enc_values[4]);
      PrintDebug("[InferencePipeline] GPT Encoder topk_indices[0..5]: {}, {}, {}, {}, {}",
                enc_indices[0], enc_indices[1], enc_indices[2], enc_indices[3], enc_indices[4]);

      bool enc_has_nan = false;
      for (int j = 0; j < enc_k; ++j) {
        if (!std::isfinite(enc_values[j])) {
          enc_has_nan = true;
          break;
        }
      }
      if (enc_has_nan) {
        PrintError("[InferencePipeline] GPT Encoder topk_values contains NaN/Inf!");
      }
    }

    // 采样第一个 token
    int64_t first_token = SampleTopK(encoder_output.topk_values.get(),
                                     encoder_output.topk_indices.get(),
                                     sample_config.temperature);

    PrintDebug("[InferencePipeline] First sampled token: {}", first_token);

    // 准备 GPT Step 循环
    auto current_samples = Model::Tensor::Empty({1, 1}, Model::DataType::kInt64,
                                                Model::DeviceType::kCPU);
    current_samples->At<int64_t>(0) = first_token;

    auto k_cache = std::move(encoder_output.k_cache);
    auto v_cache = std::move(encoder_output.v_cache);
    auto x_len = std::move(encoder_output.x_len);
    auto y_len = std::move(encoder_output.y_len);

    std::vector<std::unique_ptr<Model::Tensor>> generated_tokens;
    const int64_t eos_token = 1024;
    const int max_steps = 1500;

    if (first_token != eos_token) {
      generated_tokens.push_back(current_samples->Clone());
    }

    // GPT Step 循环
    int consecutive_invalid_count = 0;
    const int max_consecutive_invalid = 10;

    for (int step = 0; step < max_steps; ++step) {
      auto idx = Model::Tensor::Empty({1}, Model::DataType::kInt64,
                                      Model::DeviceType::kCPU);
      idx->At<int64_t>(0) = step;

      auto step_output =
          gpt_step->Step(current_samples.get(), k_cache.get(), v_cache.get(),
                         idx.get(), x_len.get(), y_len.get());

      // 检查输出有效性
      bool output_valid = true;
      if (!step_output.topk_values || step_output.topk_values->ElementCount() == 0) {
        PrintError("[InferencePipeline] Step {} failed: topk_values is empty", step);
        output_valid = false;
      } else {
        // 检查 NaN/Inf
        auto topk_values_cpu = step_output.topk_values->To(
            Model::Device(Model::DeviceType::kCPU), Model::DataType::kFloat32);
        const float* values = topk_values_cpu->Data<float>();
        int k = topk_values_cpu->ElementCount();
        for (int j = 0; j < k; ++j) {
          if (!std::isfinite(values[j])) {
            PrintError("[InferencePipeline] Step {} failed: topk_values contains NaN/Inf", step);
            output_valid = false;
            break;
          }
        }
      }

      if (!output_valid) {
        consecutive_invalid_count++;
        if (consecutive_invalid_count >= max_consecutive_invalid) {
          PrintError("[InferencePipeline] GPT Step failed {} times consecutively, terminating at step {}",
                     consecutive_invalid_count, step);
          break;
        }
        // 使用上一个 token 继续
        auto next_tensor = Model::Tensor::Empty({1, 1}, Model::DataType::kInt64,
                                                Model::DeviceType::kCPU);
        next_tensor->At<int64_t>(0) = current_samples->At<int64_t>(0);
        current_samples = next_tensor->Clone();
        generated_tokens.push_back(std::move(next_tensor));
        continue;
      }

      consecutive_invalid_count = 0;

      // 更新 cache
      k_cache = std::move(step_output.k_cache_new);
      v_cache = std::move(step_output.v_cache_new);

      // 采样
      int64_t next_token =
          SampleTopK(step_output.topk_values.get(),
                     step_output.topk_indices.get(), sample_config.temperature);

      // 检查 token 有效性
      if (next_token < 0 || next_token > eos_token) {
        PrintWarn("[InferencePipeline] Step {} generated invalid token {}, clamping", step, next_token);
        next_token = std::max<int64_t>(0, std::min<int64_t>(next_token, eos_token));
      }

      auto next_tensor = Model::Tensor::Empty({1, 1}, Model::DataType::kInt64,
                                              Model::DeviceType::kCPU);
      next_tensor->At<int64_t>(0) = next_token;
      current_samples = next_tensor->Clone();
      // 先加入列表（包括 EOS），再检查是否终止
      generated_tokens.push_back(std::move(next_tensor));

      if (next_token == eos_token) {
        PrintDebug("[InferencePipeline] EOS at step {}", step);
        break;
      }
    }

    if (generated_tokens.empty()) {
      PrintWarn("[InferencePipeline] No tokens generated");
      return AudioTools::FromByte({}, sampling_rate);
    }

    PrintDebug("[InferencePipeline] Generated {} tokens", generated_tokens.size());

    // 拼接生成的 tokens
    std::vector<std::unique_ptr<Model::Tensor>> semantic_cpu_list;
    std::vector<Model::Tensor*> semantic_ptrs;

    // 先加入 VQ codes (prompts)
    semantic_cpu_list.push_back(vq_codes->ToCPU());
    semantic_ptrs.push_back(semantic_cpu_list.back().get());

    // 再加入生成的 tokens
    for (const auto& t : generated_tokens) {
      semantic_cpu_list.push_back(t->ToCPU());
      semantic_ptrs.push_back(semantic_cpu_list.back().get());
    }

    auto pred_semantic = Model::Tensor::Concat(semantic_ptrs, 1);

    PrintDebug("[InferencePipeline] pred_semantic shape after concat: [{}, {}]",
              pred_semantic->Shape()[0], pred_semantic->Shape()[1]);

    // 移除 prompt 部分（VQ codes）
    int prompt_len = vq_codes->Shape()[1];
    int generated_len = pred_semantic->Shape()[1] - prompt_len;
    if (generated_len <= 0) {
      PrintError("[InferencePipeline] Generated semantic length is non-positive: {}", generated_len);
      return AudioTools::FromByte({}, sampling_rate);
    }

    auto generated_sem = Model::Tensor::Empty(
        {1, 1, generated_len}, Model::DataType::kInt64, Model::DeviceType::kCPU);

    auto pred_semantic_data = pred_semantic->Data<int64_t>();
    auto generated_sem_data = generated_sem->Data<int64_t>();
    std::memcpy(generated_sem_data, pred_semantic_data + prompt_len,
                generated_len * sizeof(int64_t));

    // 移除末尾的 EOS token
    if (generated_sem->Shape()[2] > 0) {
      int64_t last_token = generated_sem->At<int64_t>(generated_sem->Shape()[2] - 1);
      if (last_token == eos_token) {
        auto trimmed_sem = Model::Tensor::Empty(
            {1, 1, generated_sem->Shape()[2] - 1}, Model::DataType::kInt64,
            Model::DeviceType::kCPU);
        std::memcpy(trimmed_sem->Data<int64_t>(),
                    generated_sem->Data<int64_t>(),
                    (generated_sem->Shape()[2] - 1) * sizeof(int64_t));
        generated_sem = std::move(trimmed_sem);
        PrintDebug("[InferencePipeline] Removed trailing EOS token");
      }
    }

    PrintDebug("[InferencePipeline] generated_sem shape for SoVITS: [{}, {}, {}]",
              generated_sem->Shape()[0], generated_sem->Shape()[1], generated_sem->Shape()[2]);

    if (generated_sem->Shape()[2] > 5) {
      auto* sem_data = generated_sem->Data<int64_t>();
      PrintDebug("[InferencePipeline] generated_sem[0..4]: {}, {}, {}, {}, {}",
                sem_data[0], sem_data[1], sem_data[2], sem_data[3], sem_data[4]);
      PrintDebug("[InferencePipeline] generated_sem[last 5]: {}, {}, {}, {}, {}",
                sem_data[generated_sem->Shape()[2] - 5],
                sem_data[generated_sem->Shape()[2] - 4],
                sem_data[generated_sem->Shape()[2] - 3],
                sem_data[generated_sem->Shape()[2] - 2],
                sem_data[generated_sem->Shape()[2] - 1]);
    }

    // SoVITS 解码
    auto pred_semantic_final = generated_sem->To(
        sovits->GetModel()->GetDevice(),
        sovits->GetModel()->GetInputDataType("pred_semantic"));

    // text_seq: (1, seq_len)
    auto text_seq = target_bert_res->PhoneSeq->To(Model::Device(Model::DeviceType::kCPU), Model::DataType::kInt64);
    if (text_seq->Shape().size() == 1) {
      text_seq->Reshape({1, text_seq->Shape()[0]});
    }
    auto text_seq_final = text_seq->To(sovits->GetModel()->GetDevice(),
                            sovits->GetModel()->GetInputDataType("text_seq"));

    PrintDebug("[InferencePipeline] text_seq shape: [{}, {}]",
              text_seq->Shape()[0], text_seq->Shape()[1]);

    if (text_seq->ElementCount() > 5) {
      auto* text_data = text_seq->Data<int64_t>();
      PrintDebug("[InferencePipeline] text_seq[0..4]: {}, {}, {}, {}, {}",
                text_data[0], text_data[1], text_data[2], text_data[3], text_data[4]);
    }

    // refer_spec: (1, n_mels, time)
    auto refer_spec_input = refer_spec;
    std::unique_ptr<Model::Tensor> refer_spec_expanded;
    if (refer_spec_input->Shape().size() == 2) {
      refer_spec_expanded = refer_spec_input->Clone();
      refer_spec_expanded->Reshape({1, refer_spec_input->Shape()[0], refer_spec_input->Shape()[1]});
      refer_spec_input = refer_spec_expanded.get();
    }
    auto refer_spec_final = refer_spec_input->To(
        sovits->GetModel()->GetDevice(),
        sovits->GetModel()->GetInputDataType("refer_spec"));

    PrintDebug("[InferencePipeline] refer_spec shape: [{}, {}, {}]",
              refer_spec_input->Shape()[0],
              refer_spec_input->Shape().size() > 1 ? refer_spec_input->Shape()[1] : 0,
              refer_spec_input->Shape().size() > 2 ? refer_spec_input->Shape()[2] : 0);

    // sv_emb: (1, sv_dim)
    auto sv_emb_input = sv_emb;
    std::unique_ptr<Model::Tensor> sv_emb_expanded;
    if (sv_emb_input->Shape().size() == 1) {
      sv_emb_expanded = sv_emb_input->Clone();
      sv_emb_expanded->Reshape({1, sv_emb_input->Shape()[0]});
      sv_emb_input = sv_emb_expanded.get();
    }
    auto sv_emb_final =
        sv_emb_input->To(sovits->GetModel()->GetDevice(),
                         sovits->GetModel()->GetInputDataType("sv_emb"));

    PrintDebug("[InferencePipeline] sv_emb shape: [{}, {}]",
              sv_emb_input->Shape()[0],
              sv_emb_input->Shape().size() > 1 ? sv_emb_input->Shape()[1] : 0);

    auto audio_tensor = sovits->GenerateTensor(
        pred_semantic_final.get(), text_seq_final.get(), refer_spec_final.get(),
        sv_emb_final.get(), noise_scale, speed);

    if (!audio_tensor) {
      PrintError("[InferencePipeline] SoVITS generation failed");
      return nullptr;
    }

    PrintDebug("[InferencePipeline] SoVITS output audio samples: {}", audio_tensor->ElementCount());

    // 提取音频
    auto audio_cpu = audio_tensor->To(Model::Device(Model::DeviceType::kCPU), Model::DataType::kFloat32);
    if (!audio_cpu || audio_cpu->ElementCount() == 0) {
      PrintError("[InferencePipeline] Failed to convert audio to CPU");
      return AudioTools::FromByte({}, sampling_rate);
    }

    size_t audio_size = audio_cpu->ElementCount();
    const float* audio_ptr = audio_cpu->Data<float>();

    // 验证指针有效性
    if (!audio_ptr) {
      PrintError("[InferencePipeline] Audio data pointer is null");
      return AudioTools::FromByte({}, sampling_rate);
    }

    std::vector<float> audio_data(audio_size);
    std::memcpy(audio_data.data(), audio_ptr, audio_size * sizeof(float));

    if (audio_size > 0) {
      float min_val = audio_data[0];
      float max_val = audio_data[0];
      float sum = 0.0f;
      for (size_t i = 0; i < audio_size; ++i) {
        if (audio_data[i] < min_val) min_val = audio_data[i];
        if (audio_data[i] > max_val) max_val = audio_data[i];
        sum += std::abs(audio_data[i]);
      }
      float avg_amp = sum / static_cast<float>(audio_size);
      PrintDebug("[InferencePipeline] Audio data range: min={:.6f}, max={:.6f}, avg_amp={:.6f}",
                min_val, max_val, avg_amp);
      if (audio_size >= 5) {
        PrintDebug("[InferencePipeline] Audio samples [0..4]: {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}",
                  audio_data[0], audio_data[1], audio_data[2], audio_data[3], audio_data[4]);
      }
    }

    return AudioTools::FromByte(audio_data, sampling_rate);
  }

  // Top-K 采样
  static int64_t SampleTopK(const Model::Tensor* topk_values,
                            const Model::Tensor* topk_indices,
                            float temperature) {
    if (!topk_values || !topk_indices || topk_values->ElementCount() == 0) {
      PrintError("[InferencePipeline] SampleTopK: invalid input");
      return 0;
    }

    auto values_cpu = topk_values->To(Model::Device(Model::DeviceType::kCPU),
                                      Model::DataType::kFloat32);
    auto indices_cpu = topk_indices->To(Model::Device(Model::DeviceType::kCPU),
                                        Model::DataType::kInt64);

    const float* values = values_cpu->Data<float>();
    const int64_t* indices = indices_cpu->Data<int64_t>();
    int k = values_cpu->ElementCount();

    if (k == 0) {
      PrintError("[InferencePipeline] SampleTopK: k is zero!");
      return 0;
    }

    // 应用温度
    std::vector<float> probs(values, values + k);
    if (temperature != 1.0f && temperature > 1e-6f) {
      for (auto& p : probs) {
        p /= temperature;
      }
    }

    // Softmax
    float max_val = *std::max_element(probs.begin(), probs.end());
    float sum = 0.0f;
    for (size_t i = 0; i < probs.size(); ++i) {
      probs[i] -= max_val;  // 减去最大值防止溢出

      // Clamp 防止 exp 溢出/下溢
      if (probs[i] > 50.0f) {
        probs[i] = 50.0f;
      } else if (probs[i] < -50.0f) {
        probs[i] = -50.0f;
      }

      probs[i] = std::exp(probs[i]);

      // 检查 NaN/Inf
      if (!std::isfinite(probs[i])) {
        PrintWarn("[InferencePipeline] SampleTopK: Invalid exp value at index {}, resetting to 0", i);
        probs[i] = 0.0f;
      }

      sum += probs[i];
    }

    // 归一化
    if (sum > 1e-10f) {
      for (auto& p : probs) {
        p /= sum;
      }
    } else {
      // 均匀分布
      PrintWarn("[InferencePipeline] SampleTopK: Sum of probabilities is too small ({:.6f}), using uniform distribution", sum);
      float uniform_prob = 1.0f / static_cast<float>(k);
      for (auto& p : probs) {
        p = uniform_prob;
      }
    }

    // 验证概率
    for (const auto& p : probs) {
      if (p < 0.0f || !std::isfinite(p)) {
        PrintError("[InferencePipeline] SampleTopK: Invalid probability detected, falling back to argmax");
        // Fallback to argmax
        int max_idx = 0;
        float max_prob = probs[0];
        for (int i = 1; i < k; ++i) {
          if (probs[i] > max_prob) {
            max_prob = probs[i];
            max_idx = i;
          }
        }
        return indices[max_idx];
      }
    }

    // 采样
    try {
      std::random_device rd;
      std::mt19937 gen(rd());
      std::discrete_distribution<int> dist(probs.begin(), probs.end());
      return indices[dist(gen)];
    } catch (const std::exception& e) {
      PrintError("[InferencePipeline] SampleTopK: discrete_distribution failed: {}, falling back to argmax", e.what());
      // Fallback to argmax
      int max_idx = 0;
      float max_prob = probs[0];
      for (int i = 1; i < k; ++i) {
        if (probs[i] > max_prob) {
          max_prob = probs[i];
          max_idx = i;
        }
      }
      return indices[max_idx];
    }
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
    float noise_scale, float speed) {
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

  // 逐块添加文本
  int chunk_size = 11;
  int index = 0;
  while (index < text.size()) {
    std::string chunk = text.substr(index, chunk_size);
    sentence.Append(chunk);
    index += chunk_size;
  }
  sentence.Flush();

  if (segments.empty()) {
    PrintWarn("[InferencePipeline] No text segments to process");
    return nullptr;
  }

  // 预加载说话人特征到设备
  speaker->EnsureOnDevice(impl_->device_ctx->GetDevice());

  // 处理每个段落
  std::vector<float> final_audio;
  for (size_t i = 0; i < segments.size(); ++i) {
    auto audio = impl_->InferSegment(*speaker, segments[i], use_lang,
                                     sample_config, noise_scale, speed);

    if (audio) {
      auto samples = audio->ReadSamples();
      final_audio.insert(final_audio.end(), samples.begin(), samples.end());

      // 添加停顿
      if (i < segments.size() - 1) {
        int pause_samples = static_cast<int>(impl_->sampling_rate * 0.3f);
        final_audio.insert(final_audio.end(), pause_samples, 0.0f);
      }
    }
  }

  // 归一化
  if (!final_audio.empty()) {
    float max_amp = *std::max_element(
        final_audio.begin(), final_audio.end(),
        [](float a, float b) { return std::abs(a) < std::abs(b); });
    if (max_amp > 0.9f) {
      float scale = 0.9f / max_amp;
      for (auto& sample : final_audio) {
        sample *= scale;
      }
    }
  }

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

bool InferencePipeline::InferStreaming(const std::string& speaker_name,
                                       const std::string& text,
                                       const std::string& lang,
                                       AudioChunkCallback callback,
                                       const Model::SampleConfig& sample_config,
                                       float noise_scale, float speed) {
  // TODO: 实现流式推理
  PrintError("[InferencePipeline] Streaming inference not yet implemented");
  return false;
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
