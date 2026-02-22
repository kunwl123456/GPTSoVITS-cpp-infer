//
// Created by iFlow CLI on 2026/2/20.
//

#include "GPTSoVITS/EdgePipeline.h"

#include <algorithm>
#include <numeric>
#include <random>

#include "GPTSoVITS/Utils/speaker_serializer.h"
#include "GPTSoVITS/model/sample_config.h"
#include "GPTSoVITS/plog.h"
#include "nlohmann/json.hpp"

namespace GPTSoVITS {

class _JsonImpl {
public:
  nlohmann::json data;
};

EdgePipeline::EdgePipeline(
    const std::string& config,
    const std::string& model_path,
    const std::shared_ptr<G2P::G2PPipline>& g2p_pipline,
    const std::shared_ptr<Model::GPTEncoderModel>& gpt_encoder_model,
    const std::shared_ptr<Model::GPTStepModel>& gpt_step_model,
    const std::shared_ptr<Model::SoVITSModel>& sovits_model)
    : m_g2p_pipline(g2p_pipline),
      m_gpt_encoder_model(gpt_encoder_model),
      m_gpt_step_model(gpt_step_model),
      m_sovits_model(sovits_model) {
  m_config = std::make_shared<_JsonImpl>();
  m_config->data = nlohmann::json::parse(config);

  // 初始化配置参数
  InitializeConfig();

  PrintInfo("[EdgePipeline] Initialized with minimum model set:");
  PrintInfo("  Model path: {}", model_path);
  PrintInfo("  Model version: {}", m_model_version);
  PrintInfo("  Sampling rate: {} Hz", m_sampling_rate);
  PrintInfo("  Max sequence length: {}", m_max_len);
  PrintInfo("  Compute precision: {}",
            m_compute_precision == Model::DataType::kFloat16 ? "FP16" : "FP32");
  PrintInfo("  SV embedding dim: {}", m_sv_dim);
}

bool EdgePipeline::ImportSpeaker(const std::string& input_path,
                                 const std::string& speaker_name) {
  PrintInfo("[EdgePipeline] Importing speaker from: {}", input_path);

  // 验证数据包
  if (!Utils::SpeakerSerializer::ValidatePackage(input_path)) {
    PrintError("[EdgePipeline] Invalid speaker package: {}", input_path);
    return false;
  }

  // 获取数据包信息
  auto package_info = Utils::SpeakerSerializer::GetPackageInfo(input_path);
  if (!package_info) {
    PrintError("[EdgePipeline] Failed to read package info: {}", input_path);
    return false;
  }

  // 确定说话人名称
  std::string final_name = speaker_name.empty() ?
      package_info->speaker_name : speaker_name;

  // 检查是否已存在
  if (m_speaker_map.find(final_name) != m_speaker_map.end()) {
    PrintWarn("[EdgePipeline] Speaker '{}' already exists, will be overwritten",
             final_name);
  }

  // 反序列化
  auto speaker_info = Utils::SpeakerSerializer::DeserializeFromFile(input_path);
  if (!speaker_info) {
    PrintError("[EdgePipeline] Failed to deserialize speaker from: {}", input_path);
    return false;
  }

  // 更新说话人名称
  speaker_info->m_speaker_name = final_name;

  // 存储到映射表
  std::string speaker_lang = speaker_info->m_speaker_lang;
  m_speaker_map[final_name] = std::move(*speaker_info);

  PrintInfo("[EdgePipeline] Successfully imported speaker '{}', lang: {}",
            final_name, speaker_lang);

  return true;
}

std::unique_ptr<AudioTools> EdgePipeline::InferSpeaker(
    const std::string& speaker_name,
    const std::string& text,
    const std::string& text_lang,
    float temperature,
    float noise_scale,
    float speed) {
  // 将旧的接口转换为新的 SampleConfig 接口
  Model::SampleConfig config;
  config.temperature = temperature;
  // 保持默认的 top_k 和 top_p
  return InferSpeaker(speaker_name, text, text_lang, config, noise_scale, speed);
}

std::unique_ptr<AudioTools> EdgePipeline::InferSpeaker(
    const std::string& speaker_name,
    const std::string& text,
    const std::string& text_lang,
    const Model::SampleConfig& sample_config,
    float noise_scale,
    float speed) {
  auto iter = m_speaker_map.find(speaker_name);
  if (iter == m_speaker_map.end()) {
    PrintError("[EdgePipeline] Speaker '{}' not found", speaker_name);
    return nullptr;
  }

  const SpeakerInfo& speaker_info = iter->second;

  PrintDebug("[EdgePipeline] Inferring speaker: {}, text: {}", speaker_name, text);

  std::vector<std::string> segments;
  Text::Sentence sentence(Text::Sentence::SentenceSplitMethod::Punctuation);
  
  sentence.AppendCallBack([&segments](const std::string& s) -> bool {
    segments.push_back(s);
    return true;
  });

  // 逐块添加文本（每次处理 11 个字符）
  int chunk_size = 11;
  int index = 0;
  while (index < text.size()) {
    std::string chunk = text.substr(index, chunk_size);
    sentence.Append(chunk);
    index += chunk_size;
  }
  sentence.Flush();

  if (segments.empty()) {
    PrintWarn("[EdgePipeline] No text segments to process");
    return nullptr;
  }

  std::vector<float> audio_result;

  // 遍历每个句子段落
  for (size_t seg_idx = 0; seg_idx < segments.size(); ++seg_idx) {
    const std::string& segment = segments[seg_idx];

    PrintDebug("[EdgePipeline] Processing segment {}/{}: {}",
              seg_idx + 1, segments.size(), segment);

    // G2P 处理目标文本
    auto target_bert_res = m_g2p_pipline->GetPhoneAndBert(segment, text_lang);

    // 验证 BERT 结果有效性
    if (!target_bert_res || !target_bert_res->PhoneSeq || !target_bert_res->BertSeq) {
      PrintError("[EdgePipeline] Failed to get valid BERT result for segment: {}", segment);
      continue;
    }
    if (!speaker_info.m_bert_res || !speaker_info.m_bert_res->PhoneSeq || !speaker_info.m_bert_res->BertSeq) {
      PrintError("[EdgePipeline] Speaker BERT data is invalid");
      return nullptr;
    }

    // 获取目标数据类型和设备
    auto target_device = m_gpt_encoder_model->GetModel()->GetDevice();
    auto phone_dtype = m_gpt_encoder_model->GetModel()->GetInputDataType("phoneme_ids");
    auto bert_dtype = m_gpt_encoder_model->GetModel()->GetInputDataType("bert_feature");
    auto prompt_dtype = m_gpt_encoder_model->GetModel()->GetInputDataType("prompts");

    // 拼接参考和目标的音素 - 使用引用避免拷贝
    auto& ref_phones = speaker_info.m_bert_res->PhoneSeq;
    auto& target_phones = target_bert_res->PhoneSeq;

    // 转换到统一的类型和设备
    auto ref_phones_final = ref_phones->To(target_device, phone_dtype);
    auto target_phones_final = target_phones->To(target_device, phone_dtype);
    auto all_phones = ConcatTensor(ref_phones_final.get(), target_phones_final.get(), 0);

    // 拼接参考和目标的 BERT 特征 - 确保类型和设备一致
    // BertSeq 形状为 (1024, seq_len)，在 axis=1 上拼接
    auto& ref_bert = speaker_info.m_bert_res->BertSeq;
    auto& target_bert = target_bert_res->BertSeq;

    // 转换到统一的类型和设备
    auto ref_bert_final = ref_bert->To(target_device, bert_dtype);
    auto target_bert_final = target_bert->To(target_device, bert_dtype);
    auto all_bert = ConcatTensor(ref_bert_final.get(), target_bert_final.get(), 1);

    // 扩展维度为 (1, 1024, total_seq_len)
    if (all_bert->Shape().size() == 2) {
      all_bert = all_bert->View({1, all_bert->Shape()[0], all_bert->Shape()[1]});
    }

    // 确保 bert_feature 类型与模型期望的计算精度一致
    auto compute_dtype = GetComputeDataType();
    if (all_bert->Type() != compute_dtype) {
      PrintDebug("[EdgePipeline] Converting bert_feature from {} to {} for GPT Encoder",
                all_bert->Type() == Model::DataType::kFloat32 ? "float32" : "float16",
                compute_dtype == Model::DataType::kFloat32 ? "float32" : "float16");
      all_bert = all_bert->To(all_bert->GetDevice(), compute_dtype);
    }

    auto all_bert_expanded = std::move(all_bert);

    // 验证 VQ codes 和 Refer spec 有效性
    if (!speaker_info.m_vq_codes) {
      PrintError("[EdgePipeline] Speaker VQ codes is null");
      return nullptr;
    }

    // 准备 prompts (VQ codes) - m_vq_codes 已经是 (1, T) 格式，直接使用
    auto prompts_final = speaker_info.m_vq_codes->To(target_device, prompt_dtype);

    // 准备 phoneme_ids - 扩展维度为 (1, seq_len)
    auto phoneme_ids_expanded = all_phones->View({1, all_phones->Shape()[0]});

    // GPT Encoder 编码
    auto encoder_output = m_gpt_encoder_model->Encode(
        phoneme_ids_expanded.get(),
        prompts_final.get(),
        all_bert_expanded.get());

    // 采样第一个 token
    int64_t first_token = SampleTopK(
        encoder_output.topk_values.get(),
        encoder_output.topk_indices.get(),
        sample_config.temperature);

    // 验证 token 有效性 (范围 [0, 1024])
    if (first_token < 0 || first_token > 1024) {
      PrintWarn("[EdgePipeline] Invalid first token {}, clamping to valid range", first_token);
      first_token = std::max<int64_t>(0, std::min<int64_t>(first_token, 1024));
    }

    // 创建第一个 token tensor (CPU)
    auto current_samples = Model::Tensor::Empty(
        {1, 1}, Model::DataType::kInt64, Model::DeviceType::kCPU);
    current_samples->At<int64_t>(0) = first_token;

    // 准备索引 (CPU tensor)
    auto idx = Model::Tensor::Empty(
        {1}, Model::DataType::kInt64, Model::DeviceType::kCPU);
    idx->At<int64_t>(0) = 0;

    // 从 encoder_output 获取 x_len 和 y_len (已在正确设备上)
    auto x_len = std::move(encoder_output.x_len);
    auto y_len = std::move(encoder_output.y_len);

    // KV Cache
    auto k_cache = std::move(encoder_output.k_cache);
    auto v_cache = std::move(encoder_output.v_cache);

    // 生成的语义 tokens 列表（不包含参考部分）
    // 每个 token 是 (1, 1)，最终在 axis=1 上拼接
    std::vector<std::unique_ptr<Model::Tensor>> generated_tokens_list;

    // 添加第一个 token（如果不是 EOS）
    const int64_t eos_token = 1024;
    if (first_token != eos_token) {
      generated_tokens_list.push_back(current_samples->Clone());
    }

    // GPT Step 自回归生成
    const int max_steps = 1500;
    int consecutive_invalid_count = 0;
    const int max_consecutive_invalid = 10;

    for (int step = 0; step < max_steps; ++step) {
      auto step_output = m_gpt_step_model->Step(
          current_samples.get(),
          k_cache.get(),
          v_cache.get(),
          idx.get(),
          x_len.get(),
          y_len.get());

      // 检查输出有效性
      bool output_valid = true;
      if (!step_output.topk_values || step_output.topk_values->ElementCount() == 0) {
        PrintError("[EdgePipeline] Step {}: topk_values is empty!", step);
        output_valid = false;
      } else if (!step_output.k_cache_new || !step_output.v_cache_new) {
        PrintError("[EdgePipeline] Step {}: k_cache_new or v_cache_new is null!", step);
        output_valid = false;
      } else {
        // 检查是否有 NaN/Inf
        auto topk_values_cpu = step_output.topk_values->To(
            Model::Device(Model::DeviceType::kCPU), Model::DataType::kFloat32);
        const float* values = topk_values_cpu->Data<float>();
        int k = topk_values_cpu->ElementCount();
        for (int j = 0; j < k; ++j) {
          if (!std::isfinite(values[j])) {
            PrintError("[EdgePipeline] Step {}: topk_values contains NaN/Inf!", step);
            output_valid = false;
            break;
          }
        }
      }

      if (!output_valid) {
        consecutive_invalid_count++;
        if (consecutive_invalid_count >= max_consecutive_invalid) {
          PrintError("[EdgePipeline] GPT Step failed {} times consecutively, terminating at step {}",
                    consecutive_invalid_count, step);
          break;
        }
        // 使用上一个有效 token 继续
        int64_t last_valid_token = current_samples->At<int64_t>(0);
        auto next_token_tensor = Model::Tensor::Empty(
            {1, 1}, Model::DataType::kInt64, Model::DeviceType::kCPU);
        next_token_tensor->At<int64_t>(0) = last_valid_token;
        current_samples = next_token_tensor->Clone();
        generated_tokens_list.push_back(std::move(next_token_tensor));
        idx->At<int64_t>(0)++;
        continue;
      }

      consecutive_invalid_count = 0;

      // 采样下一个 token
      int64_t next_token = SampleTopK(
          step_output.topk_values.get(),
          step_output.topk_indices.get(),
          sample_config.temperature);

      // 验证 token 有效性
      if (next_token < 0 || next_token > 1024) {
        PrintWarn("[EdgePipeline] Step {}: Invalid token {}, clamping to valid range", step, next_token);
        next_token = std::max<int64_t>(0, std::min<int64_t>(next_token, 1024));
      }

      // 检查 EOS
      if (next_token == eos_token) {
        PrintInfo("[EdgePipeline] Generated {} tokens before EOS", step + 1);
        break;
      }

      // 创建新的 token tensor
      auto next_token_tensor = Model::Tensor::Empty(
          {1, 1}, Model::DataType::kInt64, Model::DeviceType::kCPU);
      next_token_tensor->At<int64_t>(0) = next_token;
      
      // 先克隆用于下一次 step 的输入，再 move 到列表
      current_samples = next_token_tensor->Clone();
      generated_tokens_list.push_back(std::move(next_token_tensor));

      // 更新 cache
      k_cache = std::move(step_output.k_cache_new);
      v_cache = std::move(step_output.v_cache_new);

      // 更新索引 (CPU tensor 可直接访问)
      idx->At<int64_t>(0)++;
    }

    // 检查是否有生成的 tokens
    if (generated_tokens_list.empty()) {
      PrintWarn("[EdgePipeline] No tokens generated, returning empty audio");
      return AudioTools::FromByte({}, m_sampling_rate);
    }

    // 拼接所有生成的语义 tokens (在 axis=1 上拼接)
    // 每个 token 是 (1, 1)，拼接后变成 (1, N)
    std::vector<Model::Tensor*> token_ptrs;
    for (const auto& t : generated_tokens_list) {
      token_ptrs.push_back(t.get());
    }
    auto generated_sem = Model::Tensor::Concat(token_ptrs, 1);  // (1, N)
    // 扩展维度为 (1, 1, N) 供 SoVITS 使用
    generated_sem = generated_sem->View({1, 1, generated_sem->Shape()[1]});

    // SoVITS 音频生成
    // 验证 speaker_info 数据完整性
    if (!speaker_info.m_refer_spec || !speaker_info.m_sv_emb) {
      PrintError("[EdgePipeline] Speaker refer_spec or sv_emb is null");
      return nullptr;
    }

    auto pred_semantic_final = generated_sem->To(
        m_sovits_model->GetModel()->GetDevice(),
        m_sovits_model->GetModel()->GetInputDataType("pred_semantic"));

    // text_seq 需要 (1, seq_len) 格式
    auto text_seq_expanded = target_phones->View({1, target_phones->Shape()[0]});
    auto text_seq = text_seq_expanded->To(
        m_sovits_model->GetModel()->GetDevice(),
        m_sovits_model->GetModel()->GetInputDataType("text_seq"));

    // refer_spec 需要 (1, mel_bins, spec_len) 格式
    Model::Tensor* refer_spec_input = speaker_info.m_refer_spec.get();
    std::unique_ptr<Model::Tensor> refer_spec_expanded;
    if (refer_spec_input->Shape().size() == 2) {
      refer_spec_expanded = refer_spec_input->View({1, refer_spec_input->Shape()[0], refer_spec_input->Shape()[1]});
      refer_spec_input = refer_spec_expanded.get();
    }
    auto refer_spec = refer_spec_input->To(
        m_sovits_model->GetModel()->GetDevice(),
        m_sovits_model->GetModel()->GetInputDataType("refer_spec"));

    // sv_emb 需要 (1, sv_dim) 格式
    Model::Tensor* sv_emb_input = speaker_info.m_sv_emb.get();
    std::unique_ptr<Model::Tensor> sv_emb_expanded;
    if (sv_emb_input->Shape().size() == 1) {
      sv_emb_expanded = sv_emb_input->View({1, sv_emb_input->Shape()[0]});
      sv_emb_input = sv_emb_expanded.get();
    }
    auto sv_emb = sv_emb_input->To(
        m_sovits_model->GetModel()->GetDevice(),
        m_sovits_model->GetModel()->GetInputDataType("sv_emb"));

    auto audio_tensor = m_sovits_model->GenerateTensor(
        pred_semantic_final.get(),
        text_seq.get(),
        refer_spec.get(),
        sv_emb.get(),
        noise_scale,
        speed);

    // 提取音频数据
    auto audio_cpu = audio_tensor->ToCPU();
    const float* audio_ptr = audio_cpu->Data<float>();
    size_t audio_size = audio_cpu->ElementCount();

    audio_result.insert(audio_result.end(), audio_ptr, audio_ptr + audio_size);
  }

  // 音频后处理：归一化
  if (!audio_result.empty()) {
    float max_amp = *std::max_element(audio_result.begin(), audio_result.end());
    if (max_amp > 0.9f) {
      float scale = 0.9f / max_amp;
      for (auto& sample : audio_result) {
        sample *= scale;
      }
    }
  }

  // 创建 AudioTools 对象
  auto result = AudioTools::FromByte(audio_result, m_sampling_rate);

  return result;
}

std::vector<std::string> EdgePipeline::ListSpeakers() const {
  std::vector<std::string> speaker_names;
  speaker_names.reserve(m_speaker_map.size());

  for (const auto& [name, _] : m_speaker_map) {
    speaker_names.push_back(name);
  }

  return speaker_names;
}

bool EdgePipeline::RemoveSpeaker(const std::string& speaker_name) {
  auto iter = m_speaker_map.find(speaker_name);
  if (iter == m_speaker_map.end()) {
    PrintError("[EdgePipeline] Speaker '{}' not found", speaker_name);
    return false;
  }

  m_speaker_map.erase(iter);
  PrintInfo("[EdgePipeline] Removed speaker: {}", speaker_name);

  return true;
}

bool EdgePipeline::HasSpeaker(const std::string& speaker_name) const {
  return m_speaker_map.find(speaker_name) != m_speaker_map.end();
}

const SpeakerInfo* EdgePipeline::GetSpeakerInfo(const std::string& speaker_name) const {
  auto iter = m_speaker_map.find(speaker_name);
  if (iter != m_speaker_map.end()) {
    return &iter->second;
  }
  return nullptr;
}

std::string EdgePipeline::GetModelInfo() const {
  std::ostringstream oss;
  oss << "EdgePipeline Info:\n";
  oss << "  Model version: " << m_model_version << "\n";
  oss << "  Sampling rate: " << m_sampling_rate << " Hz\n";
  oss << "  Max sequence length: " << m_max_len << "\n";
  oss << "  Compute precision: "
      << (m_compute_precision == Model::DataType::kFloat16 ? "FP16" : "FP32") << "\n";
  oss << "  SV embedding dim: " << m_sv_dim << "\n";
  oss << "  Loaded speakers: " << m_speaker_map.size() << "\n";
  return oss.str();
}

// ============ Helper Methods ============

int64_t EdgePipeline::SampleTopK(const Model::Tensor* topk_values,
                                  const Model::Tensor* topk_indices,
                                  float temperature) {
  // 空指针检查
  if (!topk_values || !topk_indices || topk_values->ElementCount() == 0) {
    PrintError("[EdgePipeline] SampleTopK: invalid input (null or empty)");
    return 0;
  }

  // 确保在 CPU 上并转换为正确类型
  auto values_cpu = topk_values->To(Model::Device(Model::DeviceType::kCPU),
                                     Model::DataType::kFloat32);
  auto indices_cpu = topk_indices->To(Model::Device(Model::DeviceType::kCPU),
                                       Model::DataType::kInt64);

  const float* values_ptr = values_cpu->Data<float>();
  const int64_t* indices_ptr = indices_cpu->Data<int64_t>();
  int k = values_cpu->ElementCount();

  if (k == 0) {
    PrintError("[EdgePipeline] SampleTopK: k is zero!");
    return 0;
  }

  // 应用温度
  std::vector<float> probs(values_ptr, values_ptr + k);
  if (temperature != 1.0f && temperature > 1e-6f) {
    for (auto& p : probs) {
      p /= temperature;
    }
  }

  // 检查 NaN/Inf
  bool has_invalid = false;
  for (int i = 0; i < k; ++i) {
    if (!std::isfinite(probs[i])) {
      has_invalid = true;
      break;
    }
  }
  if (has_invalid) {
    PrintError("[EdgePipeline] SampleTopK: input contains NaN/Inf, returning first index");
    return indices_ptr[0];
  }

  // 找最大值用于数值稳定性
  float max_val = *std::max_element(probs.begin(), probs.end());

  // Apply softmax with numerical stability
  float sum = 0.0f;
  for (size_t i = 0; i < probs.size(); ++i) {
    probs[i] -= max_val;

    // Clamp 防止 exp 溢出
    if (probs[i] > 50.0f) {
      probs[i] = 50.0f;
    } else if (probs[i] < -50.0f) {
      probs[i] = -50.0f;
    }

    probs[i] = std::exp(probs[i]);

    if (!std::isfinite(probs[i])) {
      PrintWarn("[EdgePipeline] SampleTopK: Invalid exp value at index {}, resetting to 0", i);
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
    PrintWarn("[EdgePipeline] SampleTopK: Sum too small ({:.6f}), using uniform", sum);
    float uniform_prob = 1.0f / static_cast<float>(k);
    for (auto& p : probs) {
      p = uniform_prob;
    }
  }

  // 多项式采样
  try {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    int choice = dist(gen);
    return indices_ptr[choice];
  } catch (const std::exception& e) {
    PrintError("[EdgePipeline] SampleTopK: discrete_distribution failed: {}, using argmax", e.what());
    // Fallback to argmax
    int max_idx = 0;
    float max_prob = probs[0];
    for (int i = 1; i < k; ++i) {
      if (probs[i] > max_prob) {
        max_prob = probs[i];
        max_idx = i;
      }
    }
    return indices_ptr[max_idx];
  }
}

int64_t EdgePipeline::SampleWithConfig(const Model::Tensor* topk_values,
                                       const Model::Tensor* topk_indices,
                                       const Model::SampleConfig& config) const {
  // 目前只支持基于 topk_values 和 topk_indices 的采样
  return SampleTopK(topk_values, topk_indices, config.temperature);
}

int64_t EdgePipeline::SampleTopKWithConfig(const Model::Tensor* logits,
                                            const Model::SampleConfig& config) {
  // 验证配置
  if (!config.Validate()) {
    PrintError("[EdgePipeline] Invalid sample config");
    return 0;
  }

  // 确保在 CPU 上
  auto logits_cpu = logits->IsCPU() ? logits->Clone() : logits->ToCPU();

  // 获取词汇表大小
  int64_t vocab_size = logits_cpu->Shape()[logits_cpu->Shape().size() - 1];
  const float* logits_ptr = logits_cpu->Data<float>();

  std::vector<float> probs(static_cast<size_t>(vocab_size));
  float max_val = *std::max_element(logits_ptr, logits_ptr + static_cast<size_t>(vocab_size));
  float sum = 0.0f;
  for (int64_t i = 0; i < vocab_size; ++i) {
    probs[static_cast<size_t>(i)] = std::exp((logits_ptr[static_cast<size_t>(i)] - max_val) / config.temperature);
    sum += probs[static_cast<size_t>(i)];
  }

  // 归一化
  for (int64_t i = 0; i < vocab_size; ++i) {
    probs[static_cast<size_t>(i)] /= sum;
  }

  // Top-K 筛选
  if (config.UseTopK()) {
    int64_t k = std::min(static_cast<int64_t>(config.top_k), vocab_size);

    // 创建索引数组并排序
    std::vector<int64_t> indices(static_cast<size_t>(vocab_size));
    for (int64_t i = 0; i < vocab_size; ++i) {
      indices[static_cast<size_t>(i)] = i;
    }

    std::partial_sort(indices.begin(), indices.begin() + static_cast<ptrdiff_t>(k), indices.end(),
                      [&probs](int64_t a, int64_t b) { return probs[static_cast<size_t>(a)] > probs[static_cast<size_t>(b)]; });

    // 只保留 top-k 的概率
    std::vector<float> filtered_probs(static_cast<size_t>(k));
    std::vector<int64_t> filtered_indices(static_cast<size_t>(k));
    float filtered_sum = 0.0f;
    for (int64_t i = 0; i < k; ++i) {
      filtered_probs[static_cast<size_t>(i)] = probs[static_cast<size_t>(indices[static_cast<size_t>(i)])];
      filtered_indices[static_cast<size_t>(i)] = indices[static_cast<size_t>(i)];
      filtered_sum += filtered_probs[static_cast<size_t>(i)];
    }

    // 重新归一化
    for (int64_t i = 0; i < k; ++i) {
      filtered_probs[static_cast<size_t>(i)] /= filtered_sum;
    }

    // Top-P (nucleus) 筛选
    if (config.UseTopP()) {
      // 按概率排序
      std::vector<int64_t> order(static_cast<size_t>(k));
      for (int64_t i = 0; i < k; ++i) {
        order[static_cast<size_t>(i)] = i;
      }
      std::sort(order.begin(), order.end(),
                [&filtered_probs](int64_t a, int64_t b) { return filtered_probs[static_cast<size_t>(a)] > filtered_probs[static_cast<size_t>(b)]; });

      // 找到累积概率达到 top_p 的最小集合
      float cumulative = 0.0f;
      int64_t cutoff = k;
      for (int64_t i = 0; i < k; ++i) {
        cumulative += filtered_probs[static_cast<size_t>(order[static_cast<size_t>(i)])];
        if (cumulative >= config.top_p) {
          cutoff = i + 1;
          break;
        }
      }

      // 只保留前 cutoff 个
      if (cutoff < k) {
        filtered_probs.resize(cutoff);
        filtered_indices.resize(cutoff);
        // 重新归一化
        float new_sum = 0.0f;
        for (int i = 0; i < cutoff; ++i) {
          new_sum += filtered_probs[i];
        }
        for (int i = 0; i < cutoff; ++i) {
          filtered_probs[i] /= new_sum;
        }
      }
    }

    // 多项式采样
    float r = static_cast<float>(rand()) / RAND_MAX;
    float cumulative = 0.0f;
    for (size_t i = 0; i < filtered_probs.size(); ++i) {
      cumulative += filtered_probs[i];
      if (r <= cumulative) {
        return filtered_indices[i];
      }
    }

    return filtered_indices.back();
  } else {
    // 没有使用 Top-K，直接应用 Top-P
    if (config.UseTopP()) {
      // 按概率排序
      std::vector<int64_t> indices(static_cast<size_t>(vocab_size));
      for (int64_t i = 0; i < vocab_size; ++i) {
        indices[static_cast<size_t>(i)] = i;
      }
      std::sort(indices.begin(), indices.end(),
                [&probs](int64_t a, int64_t b) { return probs[static_cast<size_t>(a)] > probs[static_cast<size_t>(b)]; });

      // 找到累积概率达到 top_p 的最小集合
      float cumulative = 0.0f;
      int64_t cutoff = vocab_size;
      for (int64_t i = 0; i < vocab_size; ++i) {
        cumulative += probs[static_cast<size_t>(indices[static_cast<size_t>(i)])];
        if (cumulative >= config.top_p) {
          cutoff = i + 1;
          break;
        }
      }

      // 只采样前 cutoff 个
      std::vector<float> filtered_probs(cutoff);
      for (int i = 0; i < cutoff; ++i) {
        filtered_probs[i] = probs[indices[i]];
      }

      // 重新归一化
      float filtered_sum = 0.0f;
      for (int i = 0; i < cutoff; ++i) {
        filtered_sum += filtered_probs[i];
      }
      for (int i = 0; i < cutoff; ++i) {
        filtered_probs[i] /= filtered_sum;
      }

      // 多项式采样
      float r = static_cast<float>(rand()) / RAND_MAX;
      cumulative = 0.0f;
      for (int i = 0; i < cutoff; ++i) {
        cumulative += filtered_probs[i];
        if (r <= cumulative) {
          return indices[i];
        }
      }

      return indices[cutoff - 1];
    } else {
      // 没有任何筛选，直接采样
      float r = static_cast<float>(rand()) / RAND_MAX;
      float cumulative = 0.0f;
      for (int64_t i = 0; i < vocab_size; ++i) {
        cumulative += probs[static_cast<size_t>(i)];
        if (r <= cumulative) {
          return i;
        }
      }

      return vocab_size - 1;
    }
  }
}

std::unique_ptr<Model::Tensor> EdgePipeline::ConcatTensor(
    const Model::Tensor* a, const Model::Tensor* b, int axis) {
  std::vector<Model::Tensor*> tensors = {
      const_cast<Model::Tensor*>(a),
      const_cast<Model::Tensor*>(b)
  };
  return Model::Tensor::Concat(tensors, axis);
}

void EdgePipeline::InitializeConfig() {
  if (m_config->data.contains("data")) {
    auto& data = m_config->data["data"];
    m_sampling_rate = data.value<int>("sampling_rate", 32000);
    m_max_len = data.value<int>("max_len", 1000);
    m_hop_length = data.value<int>("hop_length", 640);
    m_filter_length = data.value<int>("filter_length", 2048);
  }

  if (m_config->data.contains("model")) {
    auto& model = m_config->data["model"];
    m_model_version = model.value<std::string>("version", "v2");
  }

  if (m_config->data.contains("sv_embedding")) {
    auto& sv_emb = m_config->data["sv_embedding"];
    m_sv_dim = sv_emb.value<int>("embedding_size", 20480);
  }

  if (m_sovits_model) {
    m_sovits_model->SetSVDim(m_sv_dim);
  }
}

Model::DataType EdgePipeline::GetComputeDataType() const {
  return m_compute_precision;
}

}  // namespace GPTSoVITS
