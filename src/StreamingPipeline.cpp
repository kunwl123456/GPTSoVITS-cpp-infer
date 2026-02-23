//
// Created by 19254 on 2026/2/8.
//

#include "GPTSoVITS/StreamingPipeline.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>

#include "GPTSoVITS/GPTSoVITSCpp.h"
#include "GPTSoVITS/Text/Sentence.h"
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
    : m_edge_pipeline(edge_pipeline), m_config(config) {
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
    float speed) {
  if (!m_edge_pipeline->HasSpeaker(speaker_name)) {
    PrintError("[StreamingPipeline] Speaker '{}' not found", speaker_name);
    return false;
  }

  PrintInfo("[StreamingPipeline] Starting streaming inference for speaker: {}, text: {}",
            speaker_name, text);

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
        *speaker_info, segment, seg_idx, callback,
        sample_config.temperature, noise_scale, speed, prev_fade_out);

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
    int segment_index,
    AudioChunkCallback callback,
    float temperature,
    float noise_scale,
    float speed,
    const std::vector<float>& prev_fade_out) {
  
  auto gpt_encoder_model = m_edge_pipeline->GetGPTEncoderModel();
  auto gpt_step_model = m_edge_pipeline->GetGPTStepModel();
  auto sovits_model = m_edge_pipeline->GetSoVITSModel();
  auto g2p_pipeline = m_edge_pipeline->GetG2PPipeline();
  int sampling_rate = m_edge_pipeline->GetSamplingRate();

  PrintDebug("[StreamingPipeline::ProcessSegmentStreaming] segment: {}", segment);

  // G2P 处理目标文本
  auto target_bert_res = g2p_pipeline->GetPhoneAndBert(segment, "zh");

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

  if (encoder_output.k_cache && encoder_output.v_cache) {
    PrintDebug("[StreamingPipeline] GPT Encoder cache types: k_cache={}, v_cache={}",
               static_cast<int>(encoder_output.k_cache->Type()),
               static_cast<int>(encoder_output.v_cache->Type()));
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

  // 创建第一个 token tensor (CPU)
  auto current_samples = Model::Tensor::Empty(
      {1, 1}, Model::DataType::kInt64, Model::DeviceType::kCPU);
  current_samples->At<int64_t>(0) = first_token;

  // 准备索引
  auto idx = Model::Tensor::Empty(
      {1}, Model::DataType::kInt64, Model::DeviceType::kCPU);
  idx->At<int64_t>(0) = 0;

  // 从 encoder_output 获取 x_len 和 y_len
  auto x_len = std::move(encoder_output.x_len);
  auto y_len = std::move(encoder_output.y_len);

  // KV Cache
  auto k_cache = std::move(encoder_output.k_cache);
  auto v_cache = std::move(encoder_output.v_cache);

  // 检查 cache 是否需要类型转换
  auto expected_cache_dtype = gpt_step_model->GetModel()->GetInputDataType("k_cache");
  if (k_cache && k_cache->Type() != expected_cache_dtype) {
    PrintDebug("[StreamingPipeline] Converting k_cache from type {} to {}",
               static_cast<int>(k_cache->Type()), static_cast<int>(expected_cache_dtype));
    k_cache = k_cache->To(k_cache->GetDevice(), expected_cache_dtype);
  }
  if (v_cache && v_cache->Type() != expected_cache_dtype) {
    PrintDebug("[StreamingPipeline] Converting v_cache from type {} to {}",
               static_cast<int>(v_cache->Type()), static_cast<int>(expected_cache_dtype));
    v_cache = v_cache->To(v_cache->GetDevice(), expected_cache_dtype);
  }

  // 生成的语义 tokens 列表
  std::vector<int64_t> generated_tokens;
  std::deque<std::vector<int64_t>> chunk_queue;  // 待解码的分块队列
  std::vector<int64_t> history_tokens;           // 已解码的历史tokens

  const int64_t eos_token = 1024;
  if (first_token != eos_token) {
    generated_tokens.push_back(first_token);
  }

  // 边生成边解码
  const int max_steps = 1500;
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

    // 解码
    auto audio_data = DecodeChunk(
        chunk_tokens, history_tokens, lookahead_tokens,
        speaker_info, target_phones, noise_scale, speed);

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

  for (int step = 0; step < max_steps; ++step) {
    auto step_output = gpt_step_model->Step(
        current_samples.get(),
        k_cache.get(),
        v_cache.get(),
        idx.get(),
        x_len.get(),
        y_len.get());

    // 检查输出有效性
    bool output_valid = true;
    if (!step_output.topk_values || step_output.topk_values->ElementCount() == 0) {
      PrintError("[StreamingPipeline] Step {}: topk_values is empty!", step);
      output_valid = false;
    } else if (!step_output.k_cache_new || !step_output.v_cache_new) {
      PrintError("[StreamingPipeline] Step {}: k_cache_new or v_cache_new is null!", step);
      output_valid = false;
    }

    if (!output_valid) {
      consecutive_invalid_count++;
      if (consecutive_invalid_count >= max_consecutive_invalid) {
        PrintError("[StreamingPipeline] GPT Step failed {} times consecutively, terminating at step {}",
                   consecutive_invalid_count, step);
        break;
      }
      // 使用上一个有效 token 继续
      int64_t last_valid_token = current_samples->At<int64_t>(0);
      auto next_token_tensor = Model::Tensor::Empty(
          {1, 1}, Model::DataType::kInt64, Model::DeviceType::kCPU);
      next_token_tensor->At<int64_t>(0) = last_valid_token;
      current_samples = next_token_tensor->Clone();
      generated_tokens.push_back(last_valid_token);
      token_counter++;
      idx->At<int64_t>(0)++;
      continue;
    }

    consecutive_invalid_count = 0;

    // 采样下一个 token
    int64_t next_token = Utils::SampleTopK(
        step_output.topk_values.get(),
        step_output.topk_indices.get(),
        temperature);

    // 验证 token 有效性
    if (next_token < 0 || next_token > 1024) {
      PrintWarn("[StreamingPipeline] Step {}: Invalid token {}, clamping to valid range", step, next_token);
      next_token = std::max<int64_t>(0, std::min<int64_t>(next_token, 1024));
    }

    // 检查 EOS
    if (next_token == eos_token) {
      PrintInfo("[StreamingPipeline] Generated {} tokens before EOS", step + 1);
      break;
    }

    generated_tokens.push_back(next_token);
    token_counter++;

    auto next_token_tensor = Model::Tensor::Empty(
        {1, 1}, Model::DataType::kInt64, Model::DeviceType::kCPU);
    next_token_tensor->At<int64_t>(0) = next_token;
    
    // 先clone用于下一次step输入，再move
    current_samples = next_token_tensor->Clone();

    // 更新 cache 并检查类型
    auto expected_cache_dtype = gpt_step_model->GetModel()->GetInputDataType("k_cache");
    
    if (step_output.k_cache_new) {
      k_cache = std::move(step_output.k_cache_new);
      // 确保类型正确
      if (k_cache && k_cache->Type() != expected_cache_dtype) {
        k_cache = k_cache->To(k_cache->GetDevice(), expected_cache_dtype);
      }
    } else {
      PrintError("[StreamingPipeline] Step {}: k_cache_new is null!", step);
    }
    
    if (step_output.v_cache_new) {
      v_cache = std::move(step_output.v_cache_new);
      // 确保类型正确
      if (v_cache && v_cache->Type() != expected_cache_dtype) {
        v_cache = v_cache->To(v_cache->GetDevice(), expected_cache_dtype);
      }
    } else {
      PrintError("[StreamingPipeline] Step {}: v_cache_new is null!", step);
    }

    // 更新索引
    idx->At<int64_t>(0)++;

    // mute_matrix
    bool is_split = false;
    if (token_counter >= m_config.chunk_length + 2) {
      // 如果启用了 mute_matrix，尝试找到最佳分割点
      if (m_config.enable_mute_matrix && m_mute_matrix) {
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

      // 如果没有使用 mute_matrix 分割，使用固定长度分割
      if (!is_split && token_counter >= m_config.chunk_length) {
        std::vector<int64_t> chunk_tokens(
            generated_tokens.end() - token_counter,
            generated_tokens.end());
        chunk_queue.push_back(std::move(chunk_tokens));
        token_counter = 0;
        is_split = true;
      }
    }

    // 如果有多个待解码块，解码第一个
    if (is_split && chunk_queue.size() > 1) {
      auto chunk_to_decode = std::move(chunk_queue.front());
      chunk_queue.pop_front();
      const std::vector<int64_t>* lookahead_ptr = chunk_queue.empty() ? nullptr : &chunk_queue.front();
      decode_and_yield(chunk_to_decode, lookahead_ptr, false);
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
    float noise_scale,
    float speed) {
  
  if (chunk_tokens.empty()) {
    return {};
  }

  auto sovits_model = m_edge_pipeline->GetSoVITSModel();
  int sampling_rate = m_edge_pipeline->GetSamplingRate();

  // 构建输入tokens：历史 + 当前 + 前瞻
  std::vector<int64_t> input_tokens;
  
  // 添加历史tokens（最后h_len个）
  if (!history_tokens.empty()) {
    size_t h_start = history_tokens.size() > static_cast<size_t>(m_config.h_len) ?
        history_tokens.size() - m_config.h_len : 0;
    input_tokens.insert(input_tokens.end(),
                        history_tokens.begin() + h_start,
                        history_tokens.end());
  }

  // 添加当前分块
  input_tokens.insert(input_tokens.end(), chunk_tokens.begin(), chunk_tokens.end());

  // 添加前瞻tokens（前l_len个）
  if (!lookahead_tokens.empty()) {
    input_tokens.insert(input_tokens.end(), lookahead_tokens.begin(), lookahead_tokens.end());
  }

  PrintDebug("[StreamingPipeline::DecodeChunk] input_tokens size: {}, chunk: {}, history: {}, lookahead: {}",
            input_tokens.size(), chunk_tokens.size(), history_tokens.size(), lookahead_tokens.size());

  // 检查 target_phones 是否有效
  if (target_phones.empty()) {
    PrintWarn("[StreamingPipeline::DecodeChunk] target_phones is empty");
    return {};
  }

  // 准备 pred_semantic 张量 (1, 1, seq_len)
  auto pred_semantic = Model::Tensor::Empty(
      {1, 1, static_cast<int64_t>(input_tokens.size())},
      Model::DataType::kInt64,
      Model::DeviceType::kCPU);
  int64_t* semantic_ptr = pred_semantic->Data<int64_t>();
  std::memcpy(semantic_ptr, input_tokens.data(), input_tokens.size() * sizeof(int64_t));

  // 准备 text_seq 张量 (1, phone_len)
  auto text_seq = Model::Tensor::Empty(
      {1, static_cast<int64_t>(target_phones.size())},
      Model::DataType::kInt64,
      Model::DeviceType::kCPU);
  int64_t* text_ptr = text_seq->Data<int64_t>();
  std::memcpy(text_ptr, target_phones.data(), target_phones.size() * sizeof(int64_t));

  // 准备 refer_spec - 添加空指针检查
  if (!speaker_info.m_refer_spec) {
    PrintError("[StreamingPipeline::DecodeChunk] speaker_info.m_refer_spec is null");
    return {};
  }
  auto refer_spec_input = speaker_info.m_refer_spec->Clone();  // 使用 Clone 确保内存安全
  auto refer_shape = refer_spec_input->Shape();
  if (refer_shape.size() == 2) {
    refer_spec_input->Reshape({1, refer_shape[0], refer_shape[1]});
  }

  // 准备 sv_emb - 添加空指针检查
  if (!speaker_info.m_sv_emb) {
    PrintError("[StreamingPipeline::DecodeChunk] speaker_info.m_sv_emb is null");
    return {};
  }
  auto sv_emb_input = speaker_info.m_sv_emb->Clone();  // 使用 Clone 确保内存安全
  auto sv_shape = sv_emb_input->Shape();
  if (sv_shape.size() == 1) {
    sv_emb_input->Reshape({1, sv_shape[0]});
  }

  // 检查 sovits_model 是否有效
  if (!sovits_model || !sovits_model->GetModel()) {
    PrintError("[StreamingPipeline::DecodeChunk] sovits_model is null");
    return {};
  }

  auto target_device = sovits_model->GetModel()->GetDevice();
  auto pred_dtype = sovits_model->GetModel()->GetInputDataType("pred_semantic");
  auto text_dtype = sovits_model->GetModel()->GetInputDataType("text_seq");
  auto spec_dtype = sovits_model->GetModel()->GetInputDataType("refer_spec");
  auto sv_dtype = sovits_model->GetModel()->GetInputDataType("sv_emb");

  // 类型转换到目标设备
  PrintDebug("[StreamingPipeline::DecodeChunk] Converting tensors to target device...");
  auto pred_semantic_final = pred_semantic->To(target_device, pred_dtype);
  auto text_seq_final = text_seq->To(target_device, text_dtype);
  auto refer_spec_final = refer_spec_input->To(target_device, spec_dtype);
  auto sv_emb_final = sv_emb_input->To(target_device, sv_dtype);
  PrintDebug("[StreamingPipeline::DecodeChunk] Tensor conversion done, calling GenerateTensor...");

  // SoVITS 推理
  auto audio_tensor = sovits_model->GenerateTensor(
      pred_semantic_final.get(),
      text_seq_final.get(),
      refer_spec_final.get(),
      sv_emb_final.get(),
      noise_scale,
      speed);

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

  std::vector<float> audio_vec(audio_size);
  std::memcpy(audio_vec.data(), audio_ptr, audio_size * sizeof(float));

  audio_cpu.reset();

  // 计算应该返回的样本数（仅当前分块的部分）
  int samples_per_token = static_cast<int>((sampling_rate / 25) / speed);  // 语义帧率为 25Hz
  
  // 注意：history_tokens 在构建 input_tokens 时已经被截断到 h_len，
  // 但这里的 history_tokens 是完整的历史，需要重新计算
  size_t effective_history = std::min(history_tokens.size(), static_cast<size_t>(m_config.h_len));
  int h_samples = static_cast<int>(effective_history) * samples_per_token;
  int chunk_samples = static_cast<int>(chunk_tokens.size()) * samples_per_token;

  // 验证计算结果
  if (h_samples < 0 || chunk_samples < 0) {
    PrintError("[StreamingPipeline::DecodeChunk] Invalid sample calculation: h_samples={}, chunk_samples={}", 
               h_samples, chunk_samples);
    return {};
  }

  PrintDebug("[StreamingPipeline::DecodeChunk] audio_size: {}, h_samples: {}, chunk_samples: {}, effective_history: {}",
            audio_size, h_samples, chunk_samples, effective_history);

  std::vector<float> result;
  audio_size = audio_vec.size();
  
  if (static_cast<size_t>(h_samples) >= audio_size) {
    // h_samples 超出音频范围，返回部分音频
    PrintWarn("[StreamingPipeline::DecodeChunk] h_samples ({}) >= audio_size ({}), returning partial audio",
             h_samples, audio_size);
    size_t return_size = std::min(audio_size, static_cast<size_t>(chunk_samples));
    result.assign(audio_vec.begin(), audio_vec.begin() + return_size);
  } else {
    // 提取 h_samples 到 h_samples + chunk_samples 的部分
    size_t available = audio_size - h_samples;
    size_t return_size = std::min(static_cast<size_t>(chunk_samples), available);
    
    if (h_samples + return_size > audio_size) {
      PrintError("[StreamingPipeline::DecodeChunk] Buffer overflow prevented: h_samples({}) + return_size({}) > audio_size({})",
                 h_samples, return_size, audio_size);
      return_size = audio_size > static_cast<size_t>(h_samples) ? audio_size - h_samples : 0;
    }
    
    if (return_size > 0) {
      result.assign(audio_vec.begin() + h_samples, audio_vec.begin() + h_samples + return_size);
    } else {
      PrintWarn("[StreamingPipeline::DecodeChunk] return_size is 0, returning empty");
    }
  }

  PrintDebug("[StreamingPipeline::DecodeChunk] result size: {}", result.size());

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

  // 获取 mute_matrix 数据
  auto mute_cpu = m_mute_matrix->IsCPU() ? m_mute_matrix.get() : m_mute_matrix->ToCPU().get();
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
