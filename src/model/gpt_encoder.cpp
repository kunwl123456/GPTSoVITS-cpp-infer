//
// Created by Huiyicc on 2026/2/17.
//

#include "GPTSoVITS/model/gpt_encoder.h"
#include "GPTSoVITS/plog.h"

#include <algorithm>
#include <stdexcept>

namespace GPTSoVITS::Model {

GPTEncoderOutput GPTEncoderModel::Encode(
    Tensor* phoneme_ids,
    Tensor* prompts,
    Tensor* bert_feature,
    BaseModel::InferenceLease* lease) {

  GPTEncoderOutput output;

  // Ensure inputs are on the correct device and have correct types
  Device model_device = m_model->GetDevice();

  // Prepare phoneme_ids
  std::unique_ptr<Tensor> phoneme_ids_converted;
  Tensor* phoneme_ids_ptr = nullptr;
  if (phoneme_ids->GetDeviceType() != model_device.type ||
      phoneme_ids->Type() != m_model->GetInputDataType("phoneme_ids")) {
    phoneme_ids_converted = phoneme_ids->To(model_device, m_model->GetInputDataType("phoneme_ids"));
    phoneme_ids_ptr = phoneme_ids_converted.get();
  } else {
    phoneme_ids_ptr = phoneme_ids;
  }

  // Prepare prompts (convert to int64 if needed)
  std::unique_ptr<Tensor> prompts_converted;
  Tensor* prompts_ptr = nullptr;
  if (prompts->GetDeviceType() != model_device.type ||
      prompts->Type() != m_model->GetInputDataType("prompts")) {
    prompts_converted = prompts->To(model_device, m_model->GetInputDataType("prompts"));
    prompts_ptr = prompts_converted.get();
  } else {
    prompts_ptr = prompts;
  }

  // Prepare bert_feature
  std::unique_ptr<Tensor> bert_feature_converted;
  Tensor* bert_feature_ptr = nullptr;
  if (bert_feature->GetDeviceType() != model_device.type ||
      bert_feature->Type() != m_model->GetInputDataType("bert_feature")) {
    bert_feature_converted = bert_feature->To(model_device, m_model->GetInputDataType("bert_feature"));
    bert_feature_ptr = bert_feature_converted.get();
  } else {
    bert_feature_ptr = bert_feature;
  }

  std::unique_ptr<Tensor> phoneme_ids_len_tensor;
  const auto& input_names = m_model->GetInputNames();
  bool needs_len = std::find(input_names.begin(), input_names.end(),
                             "phoneme_ids_len") != input_names.end();
  if (needs_len) {
    int64_t seq_len = static_cast<int64_t>(phoneme_ids_ptr->Shape().back());
    phoneme_ids_len_tensor = Tensor::Empty({1}, DataType::kInt64, model_device);
    phoneme_ids_len_tensor->At<int64_t>(0) = seq_len;
  }

  // Prepare inputs
  std::unordered_map<std::string, Tensor*> inputs = {
      {"phoneme_ids", phoneme_ids_ptr},
      {"prompts", prompts_ptr},
      {"bert_feature", bert_feature_ptr}
  };
  if (needs_len) {
    inputs["phoneme_ids_len"] = phoneme_ids_len_tensor.get();
  }

  // Run inference
  std::unordered_map<std::string, std::unique_ptr<Tensor>> outputs;
  m_model->ForwardWithLease(lease, inputs, outputs);

  // Extract outputs
  if (outputs.find("topk_values") != outputs.end()) {
    output.topk_values = std::move(outputs["topk_values"]);
  } else {
    PrintError("GPT Encoder: missing 'topk_values' output");
  }

  if (outputs.find("topk_indices") != outputs.end()) {
    output.topk_indices = std::move(outputs["topk_indices"]);
  } else {
    PrintError("GPT Encoder: missing 'topk_indices' output");
  }

  std::unique_ptr<Tensor> raw_k, raw_v;
  if (outputs.find("k_cache") != outputs.end()) {
    raw_k = std::move(outputs["k_cache"]);
  } else {
    PrintError("GPT Encoder: missing 'k_cache' output");
  }

  if (outputs.find("v_cache") != outputs.end()) {
    raw_v = std::move(outputs["v_cache"]);
  } else {
    PrintError("GPT Encoder: missing 'v_cache' output");
  }

  if (raw_k && raw_v) {
    KVCacheDesc desc;
    const auto& s = raw_k->Shape();
    desc.dtype     = raw_k->Type();
    desc.device    = raw_k->GetDevice();
    desc.raw_shape = s;

    if (s.size() == 5) {
      // TRT layout: [num_layers, batch, num_heads, max_seq_len, head_dim]
      // max_seq_len is pre-allocated (e.g. 2000), use it directly
      desc.num_layers  = s[0];
      m_num_layers     = static_cast<int>(s[0]);
      m_num_heads      = static_cast<int>(s[2]);
      m_head_dim       = static_cast<int>(s[4]);
      desc.max_seq_len = s[3];
    } else {
      // ONNX layout: [num_layers, batch, seq_len, head_dim] (seq_len == actual input len)
      // Use m_max_seq_len from config.json
      desc.num_layers  = s[0];
      m_num_layers     = static_cast<int>(s[0]);
      desc.max_seq_len = static_cast<int64_t>(m_max_seq_len);
    }

    PrintInfo("GPT Encoder KVCache: layers={}, max_seq_len={}, raw_shape=[{}]",
              desc.num_layers, desc.max_seq_len,
              [&]{ std::string r; for (auto v : s) r += std::to_string(v) + ","; return r; }());

    output.kv_cache = KVCacheBuffer::Create(std::move(raw_k), std::move(raw_v), desc);
  }

  if (outputs.find("x_len") != outputs.end()) {
    output.x_len = outputs["x_len"]->ToCPU()->At<int64_t>(0);
  } else {
    PrintError("GPT Encoder: missing 'x_len' output");
  }

  if (outputs.find("y_len") != outputs.end()) {
    output.y_len = outputs["y_len"]->ToCPU()->At<int64_t>(0);
  } else {
    PrintError("GPT Encoder: missing 'y_len' output");
  }

  return output;
}

}  // namespace GPTSoVITS::Model