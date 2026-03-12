//
// Created by 19254 on 2026/2/8.
//

#include "GPTSoVITS/StreamingPipeline.h"

#include <chrono>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <utility>

#include "GPTSoVITS/GPTSoVITSCpp.h"
#include "GPTSoVITS/Text/Sentence.h"
#include "GPTSoVITS/Utils/LoudnessNormalizer.h"
#include "GPTSoVITS/Utils/Precision.h"
#include "GPTSoVITS/Utils/Sampling.h"
#include "GPTSoVITS/Utils/TensorOps.h"
#include "GPTSoVITS/model/sample_config.h"
#include "GPTSoVITS/model/tensor.h"
#include "GPTSoVITS/plog.h"
#include "boost/algorithm/string.hpp"

namespace GPTSoVITS {

StreamingPipeline::StreamingPipeline(
    std::shared_ptr<EdgePipeline> edge_pipeline,
    const StreamingConfig& config)
    : m_edge_pipeline(std::move(edge_pipeline)), m_config(config) {
  PrintInfo("[StreamingPipeline] Initialized with config:");
  PrintInfo("  chunk_length: {}", m_config.chunk_length);
  PrintInfo("  pause_length: {}s", m_config.pause_length);
  PrintInfo("  fade_length: {}", m_config.fade_length);
  PrintInfo("  h_len: {}, l_len: {}", m_config.h_len, m_config.l_len);
}

bool StreamingPipeline::InferSpeakerStreaming(
    const std::string& speaker_name,
    const std::string& text,
    const std::string& text_lang,
    AudioChunkCallback callback,
    const Model::SampleConfig& sample_config,
    float noise_scale,
    float speed,
    Model::InferStats* stats) {
  if (!m_edge_pipeline->HasSpeaker(speaker_name)) {
    PrintError("[StreamingPipeline] Speaker '{}' not found", speaker_name);
    return false;
  }

  PrintInfo("[StreamingPipeline] Starting streaming inference for speaker: {}, text: {}",
            speaker_name, text);

  // 重置流式响度归一化器
  m_loudness_normalizer.Reset();

  // 获取说话人信息
  const SpeakerInfo* speaker_info = m_edge_pipeline->GetSpeakerInfo(speaker_name);
  if (!speaker_info) {
    PrintError("[StreamingPipeline] Failed to get speaker info for: {}", speaker_name);
    return false;
  }

  // 文本分句
  std::vector<std::string> segments;
  Text::Sentence sentence(Text::Sentence::SentenceSplitMethod::Punctuation);

  sentence.AppendCallBack([&segments](const std::string& s) -> bool {
    segments.push_back(s);
    return true;
  });

  int chunk_size = 11;
  int index = 0;
  while (index < text.size()) {
    std::string chunk = text.substr(index, chunk_size);
    sentence.Append(chunk);
    index += chunk_size;
  }
  sentence.Flush();

  if (segments.empty()) {
    PrintWarn("[StreamingPipeline] No text segments to process");
    return false;
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

  std::vector<float> prev_fade_out;
  int sampling_rate = m_edge_pipeline->GetSamplingRate();

  // 遍历每个句子段落
  for (size_t seg_idx = 0; seg_idx < segments.size(); ++seg_idx) {
    const std::string& segment = segments[seg_idx];

    PrintDebug("[StreamingPipeline] Processing segment {}/{}: {}",
              seg_idx + 1, segments.size(), segment);

    // 流式处理段落
    prev_fade_out = ProcessSegmentStreaming(
        *speaker_info, segment, text_lang, seg_idx, callback,
        sample_config.temperature, noise_scale, speed, prev_fade_out, stats);

    // 添加段落间停顿
    if (seg_idx < segments.size() - 1 && m_config.pause_length > 0) {
      auto pause_audio = GeneratePause(m_config.pause_length, sampling_rate);

      AudioChunk pause_chunk;
      pause_chunk.audio_data = pause_audio;
      pause_chunk.is_first = false;
      pause_chunk.is_last = false;
      pause_chunk.segment_index = static_cast<int>(seg_idx);
      pause_chunk.chunk_index = -1;
      pause_chunk.duration = static_cast<float>(pause_audio.size()) / sampling_rate;

      if (callback) {
        callback(pause_chunk);
      }

      // 重置淡出
      prev_fade_out.clear();
    }
  }

  PrintInfo("[StreamingPipeline] Streaming inference completed");
  return true;
}

std::vector<float> StreamingPipeline::ProcessSegmentStreaming(
    const SpeakerInfo& speaker_info,
    const std::string& segment,
    const std::string& text_lang,
    int segment_index,
    AudioChunkCallback callback,
    float temperature,
    float noise_scale,
    float speed,
    const std::vector<float>& prev_fade_out,
    Model::InferStats* stats) {
  
  auto gpt_encoder_model = m_edge_pipeline->GetGPTEncoderModel();
  auto gpt_step_model = m_edge_pipeline->GetGPTStepModel();
  auto sovits_model = m_edge_pipeline->GetSoVITSModel();
  auto g2p_pipeline = m_edge_pipeline->GetG2PPipeline();
  int sampling_rate = m_edge_pipeline->GetSamplingRate();

  PrintDebug("[StreamingPipeline::ProcessSegmentStreaming] segment: {}", segment);

  // G2P 处理目标文本
  auto target_bert_res = g2p_pipeline->GetPhoneAndBert(segment, text_lang);

  // 获取音素序列
  auto target_phones_tensor = target_bert_res->PhoneSeq;
  std::vector<int64_t> target_phones;
  {
    auto cpu_phones = target_phones_tensor->ToCPU();
    const int64_t* ptr = cpu_phones->Data<int64_t>();
    size_t count = cpu_phones->ElementCount();
    target_phones.assign(ptr, ptr + count);
  }

  // 获取目标数据类型和设备
  auto target_device = gpt_encoder_model->GetModel()->GetDevice();
  auto phone_dtype = gpt_encoder_model->GetModel()->GetInputDataType("phoneme_ids");
  auto bert_dtype = gpt_encoder_model->GetModel()->GetInputDataType("bert_feature");
  auto prompt_dtype = gpt_encoder_model->GetModel()->GetInputDataType("prompts");

  // 拼接参考和目标的音素
  auto ref_phones = speaker_info.m_bert_res->PhoneSeq;
  auto target_phones_final = target_phones_tensor->To(target_device, phone_dtype);
  auto ref_phones_final = ref_phones->To(target_device, phone_dtype);
  auto all_phones = Utils::ConcatTensors(ref_phones_final.get(), target_phones_final.get(), 0);

  // 拼接参考和目标的 BERT 特征
  auto ref_bert = speaker_info.m_bert_res->BertSeq;
  auto target_bert = target_bert_res->BertSeq;
  auto ref_bert_final = ref_bert->To(target_device, bert_dtype);
  auto target_bert_final = target_bert->To(target_device, bert_dtype);
  auto all_bert = Utils::ConcatTensors(ref_bert_final.get(), target_bert_final.get(), 1);

  // 扩展维度(1, 1024, total_seq_len)
  auto all_bert_expanded = all_bert->View({1, all_bert->Shape()[0], all_bert->Shape()[1]});

  // 准备 prompts (VQ codes)
  auto prompts_final = speaker_info.m_vq_codes->To(target_device, prompt_dtype);
  auto phoneme_ids_expanded = all_phones->View({1, all_phones->Shape()[0]});

  // GPT Encoder 编码
  auto encoder_output = gpt_encoder_model->Encode(
      phoneme_ids_expanded.get(),
      prompts_final.get(),
      all_bert_expanded.get());

  // 检查 encoder 输出有效性
  if (!encoder_output.topk_values || !encoder_output.topk_indices) {
    PrintError("[StreamingPipeline] GPT Encoder output is null");
    return prev_fade_out;
  }

  if (encoder_output.kv_cache) {
    PrintDebug("[StreamingPipeline] GPT Encoder cache types: k_cache={}, v_cache={}",
               static_cast<int>(encoder_output.kv_cache->CurrentK()->Type()),
               static_cast<int>(encoder_output.kv_cache->CurrentV()->Type()));
    PrintDebug("[StreamingPipeline] GPT Step expected k_cache type: {}",
               static_cast<int>(gpt_step_model->GetModel()->GetInputDataType("k_cache")));
  }

  // 采样第一个 token
  int64_t first_token = Utils::SampleTopK(
      encoder_output.topk_values.get(),
      encoder_output.topk_indices.get(),
      temperature);

  // 验证 token 有效性
  if (first_token < 0 || first_token > 1024) {
    PrintWarn("[StreamingPipeline] Invalid first token {}, clamping to valid range", first_token);
    first_token = std::max<int64_t>(0, std::min<int64_t>(first_token, 1024));
  }

  // 创建 GPT Step 上下文（内部处理 dtype 转换和双缓冲）
  int top_k = static_cast<int>(encoder_output.topk_values->Shape().back());
  const int max_steps = 1500;
  auto ctx = gpt_step_model->CreateContext(*encoder_output.kv_cache, max_steps, top_k);

  // P0-1: CUDA 设备下启用 GPU 采样，将 softmax+采样移到 GPU，消除每步 D2H 同步
  const bool use_gpu_sampling = (target_device.type == Model::DeviceType::kCUDA);
  if (use_gpu_sampling) {
    if (!gpt_step_model->EnableGPUSampling(ctx.get())) {
      PrintWarn("[StreamingPipeline] Failed to enable GPU sampling, falling back to CPU");
    }
  }

  // P1-3: 在 segment 级别预转换说话人特征，所有 chunk 复用，避免每次 DecodeChunk 重复 ->To()
  auto sovits_model_ptr = m_edge_pipeline->GetSoVITSModel();
  auto sovits_target_device = sovits_model_ptr->GetModel()->GetDevice();
  auto spec_dtype = sovits_model_ptr->GetModel()->GetInputDataType("refer_spec");
  auto sv_dtype   = sovits_model_ptr->GetModel()->GetInputDataType("sv_emb");
  auto text_dtype = sovits_model_ptr->GetModel()->GetInputDataType("text_seq");

  // refer_spec
  auto refer_shape = speaker_info.m_refer_spec->Shape();
  Model::Tensor* refer_spec_raw = speaker_info.m_refer_spec.get();
  std::unique_ptr<Model::Tensor> refer_spec_view;
  if (refer_shape.size() == 2) {
    refer_spec_view = speaker_info.m_refer_spec->View({1, refer_shape[0], refer_shape[1]});
    refer_spec_raw = refer_spec_view.get();
  }
  auto refer_spec_conv = refer_spec_raw->To(sovits_target_device, spec_dtype);

  // sv_emb
  auto sv_shape = speaker_info.m_sv_emb->Shape();
  Model::Tensor* sv_emb_raw = speaker_info.m_sv_emb.get();
  std::unique_ptr<Model::Tensor> sv_emb_view;
  if (sv_shape.size() == 1) {
    sv_emb_view = speaker_info.m_sv_emb->View({1, sv_shape[0]});
    sv_emb_raw = sv_emb_view.get();
  }
  auto sv_emb_conv = sv_emb_raw->To(sovits_target_device, sv_dtype);

  // text_seq（target_phones 在整个 segment 中不变）
  auto text_seq_cpu = Model::Tensor::Empty(
      {1, static_cast<int64_t>(target_phones.size())},
      Model::DataType::kInt64,
      Model::DeviceType::kCPU);
  std::memcpy(text_seq_cpu->Data<int64_t>(), target_phones.data(),
              target_phones.size() * sizeof(int64_t));
  auto text_seq_conv = text_seq_cpu->To(sovits_target_device, text_dtype);

  // 生成的语义 tokens 列表
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
    // 获取前瞻tokens（如果有）
    std::vector<int64_t> lookahead_tokens;
    if (lookahead_ptr && !lookahead_ptr->empty()) {
      size_t l_len = std::min(lookahead_ptr->size(), static_cast<size_t>(m_config.l_len));
      lookahead_tokens.assign(lookahead_ptr->begin(), lookahead_ptr->begin() + l_len);
    }

    // P1-3: 传入 segment 级别预转换好的特征，DecodeChunk 内不再重复 ->To()
    auto audio_data = DecodeChunk(
        chunk_tokens, history_tokens, lookahead_tokens,
        speaker_info, target_phones,
        refer_spec_conv.get(), sv_emb_conv.get(), text_seq_conv.get(),
        noise_scale, speed, stats);

    if (audio_data.empty()) {
      return;
    }

    // 应用交叉淡入淡出 (Cross-fade)
    if (!last_fade_out.empty() && m_config.enable_fade) {
      int fade_len = std::min(static_cast<int>(last_fade_out.size()), 
                              static_cast<int>(audio_data.size()));
      for (int i = 0; i < fade_len; ++i) {
        float fade_in = static_cast<float>(i) / fade_len;
        float fade_out = 1.0f - fade_in;  // 淡出曲线：从1到0
        // 当前音频应用淡入，前一音频末尾应用淡出
        audio_data[i] = audio_data[i] * fade_in + last_fade_out[i] * fade_out;
      }
    }

    // prev_fade_out = audio_data[-fade_len:]
    if (!is_final_chunk && m_config.enable_fade && 
        audio_data.size() > static_cast<size_t>(m_config.fade_length)) {
      last_fade_out.assign(
          audio_data.end() - m_config.fade_length, 
          audio_data.end());
    } else {
      last_fade_out.clear();
    }

    // 更新历史tokens
    history_tokens.insert(history_tokens.end(), chunk_tokens.begin(), chunk_tokens.end());
    if (history_tokens.size() > static_cast<size_t>(m_config.h_len)) {
      history_tokens.erase(
          history_tokens.begin(),
          history_tokens.begin() + (history_tokens.size() - m_config.h_len));
    }

    // 创建音频分块
    // audio_to_yield = audio_data[:-fade_len]
    AudioChunk chunk;
    if (!is_final_chunk && m_config.enable_fade && 
        audio_data.size() > static_cast<size_t>(m_config.fade_length)) {
      // 截断末尾淡出区域，这部分保存在 last_fade_out 中，会在下一块的交叉淡入淡出中处理
      chunk.audio_data.assign(audio_data.begin(), audio_data.end() - m_config.fade_length);
    } else {
      // 最后一块或禁用淡入淡出：输出完整音频
      chunk.audio_data = audio_data;
    }
    chunk.is_first = (segment_index == 0 && chunk_index == 0);
    chunk.is_last = is_final_chunk;
    chunk.segment_index = segment_index;
    chunk.chunk_index = chunk_index;
    chunk.duration = static_cast<float>(chunk.audio_data.size()) / sampling_rate;
    
    // 流式响度归一化
    if (m_config.enable_loudness_normalize) {
      m_loudness_normalizer.NormalizeStreaming(chunk.audio_data);
    }

    // 调用回调
    if (callback) {
      PrintDebug("[StreamingPipeline::decode_and_yield] Calling callback, chunk_index: {}", chunk_index);
      callback(chunk);
      PrintDebug("[StreamingPipeline::decode_and_yield] Callback returned");
    }

    chunk_index++;
  };

  // GPT Step 自回归生成
  int consecutive_invalid_count = 0;
  const int max_consecutive_invalid = 10;

  auto gpt_gen_start = std::chrono::steady_clock::now();

  int64_t current_token = first_token;

  for (int step = 0; step < max_steps; ++step) {
    // CUDA 设备下走 GPU 采样路径
    bool step_ok;
    int64_t next_token;
    if (use_gpu_sampling && ctx->enable_gpu_sampling) {
      step_ok = gpt_step_model->StepWithGPUSampling(
          ctx.get(), current_token, step,
          encoder_output.x_len, encoder_output.y_len,
          temperature);
      if (step_ok) {
        next_token = gpt_step_model->GetSampledTokenGPU(ctx.get());
      }
    } else {
      step_ok = gpt_step_model->StepWithContext(
          ctx.get(), current_token, step,
          encoder_output.x_len, encoder_output.y_len);
      if (step_ok) {
        next_token = Utils::SampleTopK(
            ctx->topk_values.get(),
            ctx->topk_indices.get(),
            temperature);
      }
    }

    if (!step_ok) {
      consecutive_invalid_count++;
      if (consecutive_invalid_count >= max_consecutive_invalid) {
        PrintError("[StreamingPipeline] GPT Step failed {} times consecutively, terminating at step {}",
                   consecutive_invalid_count, step);
        break;
      }
      continue;
    }

    consecutive_invalid_count = 0;

    if (next_token < 0 || next_token > 1024) {
      next_token = std::max<int64_t>(0, std::min<int64_t>(next_token, 1024));
    }

    // 检查 EOS
    if (next_token == eos_token) {
      PrintInfo("[StreamingPipeline] Generated {} tokens before EOS", step + 1);
      break;
    }

    generated_tokens.push_back(next_token);
    token_counter++;

    current_token = next_token;

    // mute_matrix
    bool is_split = false;

    if (m_config.enable_mute_matrix && m_mute_matrix && token_counter >= m_config.chunk_length + 2) {
      // 获取最近的 tokens 用于计算分割点
      std::vector<int64_t> recent_tokens(
          generated_tokens.end() - token_counter,
          generated_tokens.end());

      int split_idx = FindBestSplitPoint(recent_tokens, m_config.chunk_length);
      if (split_idx >= 0 && split_idx + 1 >= m_config.chunk_length) {
        // 在最佳分割点分割
        int actual_split = split_idx + 1;
        std::vector<int64_t> chunk_tokens(
            generated_tokens.end() - token_counter,
            generated_tokens.end() - token_counter + actual_split);
        chunk_queue.push_back(std::move(chunk_tokens));
        token_counter -= actual_split;
        is_split = true;
      }
    }

    if (!is_split && token_counter >= m_config.chunk_length) {
      std::vector<int64_t> chunk_tokens(
          generated_tokens.end() - token_counter,
          generated_tokens.end());
      chunk_queue.push_back(std::move(chunk_tokens));
      token_counter = 0;
      is_split = true;
    }

    // 如果有多个待解码块，解码第一个
    if (is_split && chunk_queue.size() > 1) {
      auto chunk_to_decode = std::move(chunk_queue.front());
      chunk_queue.pop_front();
      const std::vector<int64_t>* lookahead_ptr = chunk_queue.empty() ? nullptr : &chunk_queue.front();
      auto t_pause = std::chrono::steady_clock::now();
      decode_and_yield(chunk_to_decode, lookahead_ptr, false);
      // 把 decode 耗时从 gpt_gen_start 里扣除
      gpt_gen_start += std::chrono::steady_clock::now() - t_pause;
    }
  }

  {
    auto gpt_gen_end = std::chrono::steady_clock::now();
    double gpt_gen_time_s = std::chrono::duration<double>(gpt_gen_end - gpt_gen_start).count();
    PrintInfo("[StreamingPipeline] GPT generation: {} tokens in {:.3f}s ({:.2f} tokens/s)",
              generated_tokens.size(), gpt_gen_time_s,
              generated_tokens.size() / gpt_gen_time_s);
    if (stats) {
      stats->gpt_tokens += static_cast<int>(generated_tokens.size());
      stats->gpt_time_s += gpt_gen_time_s;
    }
  }

  // 处理剩余的tokens
  if (token_counter > 0) {
    std::vector<int64_t> remaining_tokens(
        generated_tokens.end() - token_counter,
        generated_tokens.end());
    chunk_queue.push_back(std::move(remaining_tokens));
  }

  // 解码所有剩余的分块
  while (!chunk_queue.empty()) {
    auto chunk_to_decode = std::move(chunk_queue.front());
    chunk_queue.pop_front();
    bool is_final = chunk_queue.empty();
    const std::vector<int64_t>* lookahead_ptr = chunk_queue.empty() ? nullptr : &chunk_queue.front();
    decode_and_yield(chunk_to_decode, lookahead_ptr, is_final);
  }

  return last_fade_out;
}

std::vector<float> StreamingPipeline::DecodeChunk(
    const std::vector<int64_t>& chunk_tokens,
    const std::vector<int64_t>& history_tokens,
    const std::vector<int64_t>& lookahead_tokens,
    const SpeakerInfo& speaker_info,
    const std::vector<int64_t>& target_phones,
    Model::Tensor* refer_spec_preconverted,
    Model::Tensor* sv_emb_preconverted,
    Model::Tensor* text_seq_preconverted,
    float noise_scale,
    float speed,
    Model::InferStats* stats) {

  if (chunk_tokens.empty()) {
    return {};
  }

  auto sovits_model = m_edge_pipeline->GetSoVITSModel();
  int sampling_rate = m_edge_pipeline->GetSamplingRate();

  // 截断 history 到 h_len
  size_t h_used = std::min(history_tokens.size(), static_cast<size_t>(m_config.h_len));
  // 截断 lookahead 到 l_len
  size_t l_used = std::min(lookahead_tokens.size(), static_cast<size_t>(m_config.l_len));

  // 截断输入
  if (m_config.max_sovits_tokens > 0) {
    int64_t total = static_cast<int64_t>(h_used + chunk_tokens.size() + l_used);
    if (total > m_config.max_sovits_tokens) {
      int64_t excess = total - m_config.max_sovits_tokens;
      // Reduce history first (least important), then lookahead
      int64_t h_reduce = std::min(static_cast<int64_t>(h_used), excess);
      h_used -= static_cast<size_t>(h_reduce);
      excess -= h_reduce;
      if (excess > 0) {
        l_used -= static_cast<size_t>(std::min(static_cast<int64_t>(l_used), excess));
      }
      PrintDebug("[StreamingPipeline::DecodeChunk] Clamped to max_sovits_tokens={}: h_used={}, chunk={}, l_used={}",
                m_config.max_sovits_tokens, h_used, chunk_tokens.size(), l_used);
    }
  }

  // input_tokens = history[-h_len:] + chunk + lookahead[:l_len]
  std::vector<int64_t> input_tokens;
  input_tokens.reserve(h_used + chunk_tokens.size() + l_used);
  if (h_used > 0) {
    auto h_start = history_tokens.end() - static_cast<ptrdiff_t>(h_used);
    input_tokens.insert(input_tokens.end(), h_start, history_tokens.end());
  }
  input_tokens.insert(input_tokens.end(), chunk_tokens.begin(), chunk_tokens.end());
  if (l_used > 0) {
    input_tokens.insert(input_tokens.end(),
                        lookahead_tokens.begin(),
                        lookahead_tokens.begin() + static_cast<ptrdiff_t>(l_used));
  }

  PrintDebug("[StreamingPipeline::DecodeChunk] input_tokens={}, h_used={}, chunk={}, l_used={}",
            input_tokens.size(), h_used, chunk_tokens.size(), l_used);

  if (target_phones.empty()) {
    PrintWarn("[StreamingPipeline::DecodeChunk] target_phones is empty");
    return {};
  }

  if (!sovits_model || !sovits_model->GetModel()) {
    PrintError("[StreamingPipeline::DecodeChunk] sovits_model is null");
    return {};
  }

  // 准备 pred_semantic 张量 (1, 1, seq_len) — 每次 chunk 不同，不可复用
  auto target_device = sovits_model->GetModel()->GetDevice();
  auto pred_dtype = sovits_model->GetModel()->GetInputDataType("pred_semantic");

  auto pred_semantic = Model::Tensor::Empty(
      {1, 1, static_cast<int64_t>(input_tokens.size())},
      Model::DataType::kInt64,
      Model::DeviceType::kCPU);
  std::memcpy(pred_semantic->Data<int64_t>(), input_tokens.data(),
              input_tokens.size() * sizeof(int64_t));
  auto pred_semantic_final = pred_semantic->To(target_device, pred_dtype);

  // P1-3: refer_spec / sv_emb / text_seq 使用调用方预转换好的张量，无需重复 ->To()
  PrintDebug("[StreamingPipeline::DecodeChunk] Using pre-converted speaker tensors, calling GenerateTensor...");

  auto t_sovits_start = std::chrono::steady_clock::now();
  auto audio_tensor = sovits_model->GenerateTensor(
      pred_semantic_final.get(),
      text_seq_preconverted,
      refer_spec_preconverted,
      sv_emb_preconverted,
      noise_scale,
      speed);
  if (stats) {
    stats->sovits_time_s += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_sovits_start).count();
  }

  PrintDebug("[StreamingPipeline::DecodeChunk] GenerateTensor returned");

  if (!audio_tensor || audio_tensor->ElementCount() == 0) {
    PrintWarn("[StreamingPipeline::DecodeChunk] SoVITS generated empty audio");
    return {};
  }

  // 提取音频数据
  auto audio_cpu = audio_tensor->To(Model::Device(Model::DeviceType::kCPU),
                                    Model::DataType::kFloat32);
  if (!audio_cpu || audio_cpu->ElementCount() == 0) {
    PrintError("[StreamingPipeline::DecodeChunk] audio_cpu is null or empty after To");
    return {};
  }

  size_t audio_size = audio_cpu->ElementCount();
  const float* audio_ptr = audio_cpu->Data<float>();

  if (!audio_ptr) {
    PrintError("[StreamingPipeline::DecodeChunk] audio_ptr is null");
    return {};
  }

  std::vector<float> result(audio_size);
  std::memcpy(result.data(), audio_ptr, audio_size * sizeof(float));

  // 裁剪：只取 chunk 对应的部分，去掉 history 和 lookahead 的音频
  int samples_per_token = static_cast<int>(std::round(
      static_cast<float>(sampling_rate) / 25.0f / speed));
  int h_samples = static_cast<int>(h_used) * samples_per_token;
  int c_samples = static_cast<int>(chunk_tokens.size()) * samples_per_token;

  if (h_samples >= static_cast<int>(audio_size)) {
    PrintWarn("[StreamingPipeline::DecodeChunk] h_samples({}) >= audio_size({}), skipping chunk",
              h_samples, audio_size);
    return {};
  }

  int end = h_samples + c_samples;
  if (end > static_cast<int>(audio_size)) {
    end = static_cast<int>(audio_size);
  }

  result.assign(result.begin() + h_samples, result.begin() + end);

  PrintDebug("[StreamingPipeline::DecodeChunk] audio_size={}, h_samples={}, c_samples={}, result={}",
            audio_size, h_samples, c_samples, result.size());

  return result;
}

std::vector<float> StreamingPipeline::ApplyFade(
    const std::vector<float>& audio,
    const std::vector<float>& fade_in,
    const std::vector<float>& fade_out) {
  if (!m_config.enable_fade) {
    return audio;
  }

  std::vector<float> result = audio;
  int fade_len = m_config.fade_length;

  // 应用淡入
  if (!fade_in.empty() && result.size() > static_cast<size_t>(fade_len)) {
    for (int i = 0; i < fade_len && i < static_cast<int>(fade_in.size()); ++i) {
      float fade_ratio = static_cast<float>(i) / fade_len;
      result[i] = result[i] * fade_ratio + fade_in[i] * (1.0f - fade_ratio);
    }
  }

  // 应用淡出
  if (result.size() > static_cast<size_t>(fade_len)) {
    size_t start = result.size() - fade_len;
    for (int i = 0; i < fade_len; ++i) {
      float fade_ratio = 1.0f - static_cast<float>(i) / fade_len;
      result[start + i] *= fade_ratio;
    }
  }

  return result;
}

std::vector<float> StreamingPipeline::GeneratePause(
    float duration,
    int sampling_rate) {
  int num_samples = static_cast<int>(duration * sampling_rate);
  return std::vector<float>(num_samples, 0.0f);
}

bool StreamingPipeline::LoadMuteMatrix(const std::string& path) {
  PrintInfo("[StreamingPipeline] Loading mute matrix from: {}", path);

  if (!std::filesystem::exists(path)) {
    PrintWarn("[StreamingPipeline] Mute matrix file not found: {}", path);
    return false;
  }

  m_mute_matrix = Model::Tensor::Empty({1025}, Model::DataType::kFloat32, Model::DeviceType::kCPU);
  float* ptr = m_mute_matrix->Data<float>();
  std::fill(ptr, ptr + 1025, 0.0f);

  PrintInfo("[StreamingPipeline] Mute matrix loaded (using default values)");
  return true;
}

int StreamingPipeline::FindBestSplitPoint(const std::vector<int64_t>& tokens, int min_length) const {
  if (!m_mute_matrix || tokens.size() < static_cast<size_t>(min_length + 2)) {
    return -1;
  }

  // 获取 mute_matrix 数据（延长 ToCPU 返回值的生命周期）
  std::unique_ptr<Model::Tensor> mute_cpu_owner;
  const Model::Tensor* mute_cpu = m_mute_matrix.get();
  if (!m_mute_matrix->IsCPU()) {
    mute_cpu_owner = m_mute_matrix->ToCPU();
    mute_cpu = mute_cpu_owner.get();
  }
  const float* mute_scores = mute_cpu->Data<float>();

  // 计算每个位置的分割得分
  std::vector<float> scores(tokens.size());
  for (size_t i = 0; i < tokens.size(); ++i) {
    int64_t token = tokens[i];
    if (token >= 0 && token < 1025) {
      scores[i] = mute_scores[token] - m_config.mute_threshold;
      if (scores[i] < 0) scores[i] = -1.0f;  // 惩罚负分数
    } else {
      scores[i] = -1.0f;
    }
  }

  // 计算累积得分（考虑相邻位置的贡献）
  for (size_t i = 0; i + 1 < tokens.size(); ++i) {
    scores[i] += scores[i + 1];
  }

  // 找到最大得分位置
  float max_score = -1.0f;
  int best_idx = -1;

  for (size_t i = static_cast<size_t>(min_length); i + 1 < tokens.size(); ++i) {
    if (scores[i] >= 0 && scores[i] > max_score) {
      max_score = scores[i];
      best_idx = static_cast<int>(i);
    }
  }

  return best_idx;
}

}  // namespace GPTSoVITS
