#include "GPTSoVITS/EdgePipeline.h"

#include <algorithm>
#include <chrono>
#include <numeric>

#include "GPTSoVITS/Utils/LoudnessNormalizer.h"
#include "GPTSoVITS/Utils/Precision.h"
#include "GPTSoVITS/Utils/Sampling.h"
#include "GPTSoVITS/Utils/speaker_serializer.h"
#include "GPTSoVITS/Utils/TensorOps.h"
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

  // 初始化配置参数（基类方法）
  InitializeConfig();

  // 设置 SoVITS 模型的 SV 维度
  if (m_sovits_model) {
    m_sovits_model->SetSVDim(m_config_params.sv_dim);
  }

  PrintInfo("[EdgePipeline] Initialized with minimum model set:");
  PrintInfo("  Model path: {}", model_path);
  PrintInfo("  Model version: {}", m_config_params.model_version);
  PrintInfo("  Sampling rate: {} Hz", m_config_params.sampling_rate);
  PrintInfo("  Max sequence length: {}", m_config_params.max_len);
  PrintInfo("  Compute precision: {}",
            m_compute_precision == Model::DataType::kFloat16 ? "FP16" : "FP32");
  PrintInfo("  SV embedding dim: {}", m_config_params.sv_dim);
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
    float speed,
    Model::InferStats* stats) {
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
    auto all_phones = Utils::ConcatTensors(ref_phones_final.get(), target_phones_final.get(), 0);

    // 拼接参考和目标的 BERT 特征 - 确保类型和设备一致
    // BertSeq 形状为 (1024, seq_len)，在 axis=1 上拼接
    auto& ref_bert = speaker_info.m_bert_res->BertSeq;
    auto& target_bert = target_bert_res->BertSeq;

    // 转换到统一的类型和设备
    auto ref_bert_final = ref_bert->To(target_device, bert_dtype);
    auto target_bert_final = target_bert->To(target_device, bert_dtype);
    auto all_bert = Utils::ConcatTensors(ref_bert_final.get(), target_bert_final.get(), 1);

    // 扩展维度为 (1, 1024, total_seq_len)
    if (all_bert->Shape().size() == 2) {
      all_bert = all_bert->View({1, all_bert->Shape()[0], all_bert->Shape()[1]});
    }

    // 确保 bert_feature 类型与模型期望的计算精度一致
    auto compute_dtype = GetComputePrecision();
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
    int64_t first_token = Utils::SampleTopK(
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

    // 从 encoder_output 获取 x_len 和 y_len
    auto x_len = std::move(encoder_output.x_len);
    auto y_len = std::move(encoder_output.y_len);

    // KV Cache + ping-pong 缓冲区
    auto expected_cache_dtype = m_gpt_step_model->GetCacheDType();
    auto k_cache = std::move(encoder_output.k_cache);
    auto v_cache = std::move(encoder_output.v_cache);
    if (k_cache && k_cache->Type() != expected_cache_dtype) {
      k_cache = k_cache->To(k_cache->GetDevice(), expected_cache_dtype);
    }
    if (v_cache && v_cache->Type() != expected_cache_dtype) {
      v_cache = v_cache->To(v_cache->GetDevice(), expected_cache_dtype);
    }
    auto k_cache_out = k_cache->Clone();
    auto v_cache_out = v_cache->Clone();

    // 预分配 topk 输出缓冲区（dtype 匹配模型输出，留在 CPU 供采样使用）
    int top_k = static_cast<int>(encoder_output.topk_values->Shape().back());
    auto topk_val_dtype = m_gpt_step_model->GetModel()->GetOutputDataType("topk_values");
    auto topk_idx_dtype = m_gpt_step_model->GetModel()->GetOutputDataType("topk_indices");
    auto topk_values_buf = Model::Tensor::Empty(
        {1, top_k}, topk_val_dtype, Model::DeviceType::kCPU);
    auto topk_indices_buf = Model::Tensor::Empty(
        {1, top_k}, topk_idx_dtype, Model::DeviceType::kCPU);

    const bool use_iobinding = m_gpt_step_model->SupportsIOBinding();

    // 生成的语义 tokens
    std::vector<int64_t> generated_tokens;

    // 添加第一个 token（如果不是 EOS）
    const int64_t eos_token = 1024;
    if (first_token != eos_token) {
      generated_tokens.push_back(first_token);
    }

    // GPT Step 自回归生成
    const int max_steps = 1500;
    int consecutive_invalid_count = 0;
    const int max_consecutive_invalid = 10;

    auto gpt_gen_start = std::chrono::steady_clock::now();

    for (int step = 0; step < max_steps; ++step) {
      bool step_ok = false;
      if (use_iobinding) {
        step_ok = m_gpt_step_model->StepWithIOBinding(
            current_samples.get(),
            k_cache.get(), v_cache.get(),
            k_cache_out.get(), v_cache_out.get(),
            idx.get(), x_len.get(), y_len.get(),
            topk_values_buf.get(), topk_indices_buf.get());
      } else {
        auto step_output = m_gpt_step_model->Step(
            current_samples.get(),
            k_cache.get(), v_cache.get(),
            idx.get(), x_len.get(), y_len.get());
        step_ok = step_output.topk_values && step_output.k_cache_new && step_output.v_cache_new;
        if (step_ok) {
          std::memcpy(topk_values_buf->Data(),
                      step_output.topk_values->ToCPU()->Data(),
                      topk_values_buf->ByteSize());
          std::memcpy(topk_indices_buf->Data(),
                      step_output.topk_indices->ToCPU()->Data(),
                      topk_indices_buf->ByteSize());
          k_cache_out = std::move(step_output.k_cache_new);
          v_cache_out = std::move(step_output.v_cache_new);
        }
      }

      if (!step_ok) {
        consecutive_invalid_count++;
        if (consecutive_invalid_count >= max_consecutive_invalid) {
          PrintError("[EdgePipeline] GPT Step failed {} times consecutively, terminating at step {}",
                    consecutive_invalid_count, step);
          break;
        }
        generated_tokens.push_back(current_samples->At<int64_t>(0));
        idx->At<int64_t>(0)++;
        continue;
      }

      consecutive_invalid_count = 0;

      // 交换 cache 指针（ping-pong）
      std::swap(k_cache, k_cache_out);
      std::swap(v_cache, v_cache_out);

      // 采样下一个 token
      int64_t next_token = Utils::SampleTopK(
          topk_values_buf.get(),
          topk_indices_buf.get(),
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

      generated_tokens.push_back(next_token);
      current_samples->At<int64_t>(0) = next_token;
      idx->At<int64_t>(0)++;
    }

    auto gpt_gen_end = std::chrono::steady_clock::now();
    double gpt_gen_time_s = std::chrono::duration<double>(gpt_gen_end - gpt_gen_start).count();
    PrintInfo("[EdgePipeline] GPT generation: {} tokens in {:.3f}s ({:.2f} tokens/s)",
              generated_tokens.size(), gpt_gen_time_s,
              generated_tokens.size() / gpt_gen_time_s);
    if (stats) {
      stats->gpt_tokens = static_cast<int>(generated_tokens.size());
      stats->gpt_time_s = gpt_gen_time_s;
    }

    // 检查是否有生成的 tokens
    if (generated_tokens.empty()) {
      PrintWarn("[EdgePipeline] No tokens generated, returning empty audio");
      return AudioTools::FromByte({}, m_config_params.sampling_rate);
    }

    PrintInfo("[EdgePipeline] Generated {} tokens for segment", generated_tokens.size());

    // pred_semantic tensor (1, 1, N)
    auto generated_sem = Model::Tensor::Empty(
        {1, 1, static_cast<int64_t>(generated_tokens.size())},
        Model::DataType::kInt64, Model::DeviceType::kCPU);
    std::memcpy(generated_sem->Data<int64_t>(), generated_tokens.data(),
                generated_tokens.size() * sizeof(int64_t));

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

    auto t_sovits_start = std::chrono::steady_clock::now();
    auto audio_tensor = m_sovits_model->GenerateTensor(
        pred_semantic_final.get(),
        text_seq.get(),
        refer_spec.get(),
        sv_emb.get(),
        noise_scale,
        speed);
    if (stats) {
      stats->sovits_time_s += std::chrono::duration<double>(
          std::chrono::steady_clock::now() - t_sovits_start).count();
    }

    // 提取音频数据
    auto audio_cpu = audio_tensor->ToCPU();
    const float* audio_ptr = audio_cpu->Data<float>();
    size_t audio_size = audio_cpu->ElementCount();

    audio_result.insert(audio_result.end(), audio_ptr, audio_ptr + audio_size);
  }

  // RMS + Peak 组合归一化
  if (!audio_result.empty()) {
    LoudnessConfig loudness_config;
    loudness_config.target_rms = 0.18f;      // 目标 RMS (~-15dBFS)
    loudness_config.max_gain = 10.0f;        // 最大增益
    loudness_config.min_gain = 0.1f;         // 最小增益
    loudness_config.enable_peak_limiting = true;
    loudness_config.peak_threshold = 0.9f;   // 峰值限制阈值
    
    LoudnessNormalizer normalizer(loudness_config);
    normalizer.NormalizeCombined(audio_result);
    
    PrintDebug("[EdgePipeline] Applied loudness normalization, RMS: {:.4f}, peak: {:.4f}",
               normalizer.CalculateRMS(audio_result),
               normalizer.CalculatePeak(audio_result));
  }

  // 创建 AudioTools 对象
  auto result = AudioTools::FromByte(audio_result, m_config_params.sampling_rate);

  return result;
}

std::string EdgePipeline::GetModelInfo() const {
  std::ostringstream oss;
  oss << "EdgePipeline Info:\n";
  oss << "  Model version: " << m_config_params.model_version << "\n";
  oss << "  Sampling rate: " << m_config_params.sampling_rate << " Hz\n";
  oss << "  Max sequence length: " << m_config_params.max_len << "\n";
  oss << "  Compute precision: "
      << (m_compute_precision == Model::DataType::kFloat16 ? "FP16" : "FP32") << "\n";
  oss << "  SV embedding dim: " << m_config_params.sv_dim << "\n";
  oss << "  Loaded speakers: " << m_speaker_map.size() << "\n";
  return oss.str();
}

}  // namespace GPTSoVITS
