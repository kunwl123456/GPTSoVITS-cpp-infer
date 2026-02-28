//
// Created by Huiyicc on 2026/2/17.
//

#include "GPTSoVITS/model/gpt_step.h"
#include "GPTSoVITS/plog.h"
#include <fmt/format.h>
#include <fmt/ranges.h>

namespace GPTSoVITS::Model {

GPTStepOutput GPTStepModel::Step(Tensor* samples,
                                 Tensor* k_cache,
                                 Tensor* v_cache,
                                 Tensor* idx,
                                 Tensor* x_len,
                                 Tensor* y_len) {

  GPTStepOutput output;

  // Ensure inputs are on the correct device and have correct types
  Device model_device = m_model->GetDevice();

  // Prepare samples
  std::unique_ptr<Tensor> samples_converted;
  Tensor* samples_ptr = nullptr;
  if (samples->GetDeviceType() != model_device.type ||
      samples->Type() != m_model->GetInputDataType("samples")) {
    samples_converted = samples->To(model_device, m_model->GetInputDataType("samples"));
    samples_ptr = samples_converted.get();
  } else {
    samples_ptr = samples;
  }

  // Prepare k_cache
  std::unique_ptr<Tensor> k_cache_converted;
  Tensor* k_cache_ptr = nullptr;
  if (k_cache->GetDeviceType() != model_device.type ||
      k_cache->Type() != m_model->GetInputDataType("k_cache")) {
    k_cache_converted = k_cache->To(model_device, m_model->GetInputDataType("k_cache"));
    k_cache_ptr = k_cache_converted.get();
  } else {
    k_cache_ptr = k_cache;
  }

  // Prepare v_cache
  std::unique_ptr<Tensor> v_cache_converted;
  Tensor* v_cache_ptr = nullptr;
  if (v_cache->GetDeviceType() != model_device.type ||
      v_cache->Type() != m_model->GetInputDataType("v_cache")) {
    v_cache_converted = v_cache->To(model_device, m_model->GetInputDataType("v_cache"));
    v_cache_ptr = v_cache_converted.get();
  } else {
    v_cache_ptr = v_cache;
  }

  // Prepare idx
  std::unique_ptr<Tensor> idx_converted;
  Tensor* idx_ptr = nullptr;
  if (idx->GetDeviceType() != model_device.type ||
      idx->Type() != m_model->GetInputDataType("idx")) {
    idx_converted = idx->To(model_device, m_model->GetInputDataType("idx"));
    idx_ptr = idx_converted.get();
  } else {
    idx_ptr = idx;
  }

  // Prepare x_len
  std::unique_ptr<Tensor> x_len_converted;
  Tensor* x_len_ptr = nullptr;
  if (x_len->GetDeviceType() != model_device.type ||
      x_len->Type() != m_model->GetInputDataType("x_len")) {
    x_len_converted = x_len->To(model_device, m_model->GetInputDataType("x_len"));
    x_len_ptr = x_len_converted.get();
  } else {
    x_len_ptr = x_len;
  }

  // Prepare y_len
  std::unique_ptr<Tensor> y_len_converted;
  Tensor* y_len_ptr = nullptr;
  if (y_len->GetDeviceType() != model_device.type ||
      y_len->Type() != m_model->GetInputDataType("y_len")) {
    y_len_converted = y_len->To(model_device, m_model->GetInputDataType("y_len"));
    y_len_ptr = y_len_converted.get();
  } else {
    y_len_ptr = y_len;
  }

  // Prepare inputs
  std::unordered_map<std::string, Tensor*> inputs = {
      {"samples", samples_ptr},
      {"k_cache", k_cache_ptr},
      {"v_cache", v_cache_ptr},
      {"idx", idx_ptr},
      {"x_len", x_len_ptr},
      {"y_len", y_len_ptr}
  };

  // Run inference
  std::unordered_map<std::string, std::unique_ptr<Tensor>> outputs;
  m_model->Forward(inputs, outputs);

  // Extract outputs
  if (outputs.find("topk_values") != outputs.end()) {
    output.topk_values = std::move(outputs["topk_values"]);
  } else {
    PrintError("GPT Step: missing 'topk_values' output");
  }

  if (outputs.find("topk_indices") != outputs.end()) {
    output.topk_indices = std::move(outputs["topk_indices"]);
  } else {
    PrintError("GPT Step: missing 'topk_indices' output");
  }

  if (outputs.find("k_cache_new") != outputs.end()) {
    output.k_cache_new = std::move(outputs["k_cache_new"]);
  } else {
    PrintError("GPT Step: missing 'k_cache_new' output");
  }

  if (outputs.find("v_cache_new") != outputs.end()) {
    output.v_cache_new = std::move(outputs["v_cache_new"]);
  } else {
    PrintError("GPT Step: missing 'v_cache_new' output");
  }

  return output;
}

bool GPTStepModel::StepWithIOBinding(Tensor* samples,
                                     Tensor* k_cache_in,
                                     Tensor* v_cache_in,
                                     Tensor* k_cache_out,
                                     Tensor* v_cache_out,
                                     Tensor* idx,
                                     Tensor* x_len,
                                     Tensor* y_len,
                                     Tensor* topk_values_out,
                                     Tensor* topk_indices_out) {

  // 确保所有输入在正确的设备上并有正确的类型
  Device model_device = m_model->GetDevice();

  // 准备输入
  std::unique_ptr<Tensor> samples_converted;
  Tensor* samples_ptr = nullptr;
  if (samples->GetDeviceType() != model_device.type ||
      samples->Type() != m_model->GetInputDataType("samples")) {
    samples_converted = samples->To(model_device, m_model->GetInputDataType("samples"));
    samples_ptr = samples_converted.get();
  } else {
    samples_ptr = samples;
  }

  std::unique_ptr<Tensor> k_cache_converted;
  Tensor* k_cache_ptr = nullptr;
  if (k_cache_in->GetDeviceType() != model_device.type ||
      k_cache_in->Type() != m_model->GetInputDataType("k_cache")) {
    k_cache_converted = k_cache_in->To(model_device, m_model->GetInputDataType("k_cache"));
    k_cache_ptr = k_cache_converted.get();
  } else {
    k_cache_ptr = k_cache_in;
  }

  std::unique_ptr<Tensor> v_cache_converted;
  Tensor* v_cache_ptr = nullptr;
  if (v_cache_in->GetDeviceType() != model_device.type ||
      v_cache_in->Type() != m_model->GetInputDataType("v_cache")) {
    v_cache_converted = v_cache_in->To(model_device, m_model->GetInputDataType("v_cache"));
    v_cache_ptr = v_cache_converted.get();
  } else {
    v_cache_ptr = v_cache_in;
  }

  std::unique_ptr<Tensor> idx_converted;
  Tensor* idx_ptr = nullptr;
  if (idx->GetDeviceType() != model_device.type ||
      idx->Type() != m_model->GetInputDataType("idx")) {
    idx_converted = idx->To(model_device, m_model->GetInputDataType("idx"));
    idx_ptr = idx_converted.get();
  } else {
    idx_ptr = idx;
  }

  std::unique_ptr<Tensor> x_len_converted;
  Tensor* x_len_ptr = nullptr;
  if (x_len->GetDeviceType() != model_device.type ||
      x_len->Type() != m_model->GetInputDataType("x_len")) {
    x_len_converted = x_len->To(model_device, m_model->GetInputDataType("x_len"));
    x_len_ptr = x_len_converted.get();
  } else {
    x_len_ptr = x_len;
  }

  std::unique_ptr<Tensor> y_len_converted;
  Tensor* y_len_ptr = nullptr;
  if (y_len->GetDeviceType() != model_device.type ||
      y_len->Type() != m_model->GetInputDataType("y_len")) {
    y_len_converted = y_len->To(model_device, m_model->GetInputDataType("y_len"));
    y_len_ptr = y_len_converted.get();
  } else {
    y_len_ptr = y_len;
  }

  std::unordered_map<std::string, Tensor*> inputs = {
      {"samples", samples_ptr},
      {"k_cache", k_cache_ptr},
      {"v_cache", v_cache_ptr},
      {"idx", idx_ptr},
      {"x_len", x_len_ptr},
      {"y_len", y_len_ptr}
  };

  std::unordered_map<std::string, Tensor*> outputs;
  if (topk_values_out) {
    outputs["topk_values"] = topk_values_out;
  }
  if (topk_indices_out) {
    outputs["topk_indices"] = topk_indices_out;
  }
  if (k_cache_out) {
    outputs["k_cache_new"] = k_cache_out;
  }
  if (v_cache_out) {
    outputs["v_cache_new"] = v_cache_out;
  }

  // IO Binding 推理
  return m_model->ForwardWithPreallocatedOutput(inputs, outputs);
}

std::unique_ptr<GPTStepContext> GPTStepModel::CreateContext(
    const std::vector<int64_t>& kv_cache_shape,
    int max_steps,
    int top_k) {

  auto ctx = std::make_unique<GPTStepContext>();
  ctx->max_steps = max_steps;

  Device model_device = m_model->GetDevice();
  DataType cache_dtype = m_cache_dtype;

  // 预分配双缓冲 KV cache
  PrintInfo("[GPTStepContext] Allocating double-buffered KV cache: shape={}, dtype={}, device={}",
            fmt::join(kv_cache_shape, "x"),
            static_cast<int>(cache_dtype),
            model_device.type == DeviceType::kCUDA ? "CUDA" : "CPU");

  for (int i = 0; i < 2; ++i) {
    ctx->k_cache[i] = Tensor::Empty(kv_cache_shape, cache_dtype, model_device);
    ctx->v_cache[i] = Tensor::Empty(kv_cache_shape, cache_dtype, model_device);
  }

  // 预分配输出缓冲区
  ctx->topk_values = Tensor::Empty({1, top_k}, DataType::kFloat32, model_device);
  ctx->topk_indices = Tensor::Empty({1, top_k}, DataType::kInt64, model_device);

  // 预分配索引张量 (0 到 max_steps-1)
  PrintInfo("[GPTStepContext] Pre-allocating {} index tensors", max_steps);
  ctx->idx_tensors.reserve(max_steps);
  for (int i = 0; i < max_steps; ++i) {
    auto idx_tensor = Tensor::Empty({1}, DataType::kInt64, model_device);
    int64_t* idx_data = static_cast<int64_t*>(idx_tensor->Data());
    idx_data[0] = static_cast<int64_t>(i);
    ctx->idx_tensors.push_back(std::move(idx_tensor));
  }

  PrintInfo("[GPTStepContext] Context created successfully (zero-alloc ready)");
  return ctx;
}

bool GPTStepModel::StepWithContext(GPTStepContext* ctx,
                                   Tensor* samples,
                                   int step_idx,
                                   Tensor* x_len,
                                   Tensor* y_len) {

  if (!ctx) {
    PrintError("[GPTStepModel] Context is null");
    return false;
  }

  if (step_idx < 0 || step_idx >= ctx->max_steps) {
    PrintError("[GPTStepModel] step_idx {} out of range [0, {})", step_idx, ctx->max_steps);
    return false;
  }

  // 获取当前输入和下一个输出的 cache
  Tensor* k_cache_in = ctx->GetCurrentKCache();
  Tensor* v_cache_in = ctx->GetCurrentVCache();
  Tensor* k_cache_out = ctx->GetNextKCache();
  Tensor* v_cache_out = ctx->GetNextVCache();

  // 使用预分配的索引张量
  Tensor* idx = ctx->idx_tensors[step_idx].get();

  // 调用 StepWithIOBinding (零拷贝)
  bool success = StepWithIOBinding(
      samples,
      k_cache_in,
      v_cache_in,
      k_cache_out,
      v_cache_out,
      idx,
      x_len,
      y_len,
      ctx->topk_values.get(),
      ctx->topk_indices.get()
  );

  if (success) {
    // 自动切换缓冲区 (ping-pong)
    ctx->SwapBuffers();
  }

  return success;
}

}  // namespace GPTSoVITS::Model