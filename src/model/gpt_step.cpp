//
// Created by Huiyicc on 2026/2/17.
//

#include "GPTSoVITS/model/gpt_step.h"
#include "GPTSoVITS/model/gpu_kernels.h"
#include "GPTSoVITS/plog.h"
#include <cstring>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <random>
#ifdef WITH_CUDA
#include <cuda_runtime_api.h>
#endif

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

  // 直接使用输入指针,不做转换检查
  // InferencePipeline负责确保输入已经在正确的设备和类型上
  std::unordered_map<std::string, Tensor*> inputs = {
      {"samples", samples},
      {"k_cache", k_cache_in},
      {"v_cache", v_cache_in},
      {"idx", idx},
      {"x_len", x_len},
      {"y_len", y_len}
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

  // IO Binding
  bool success = m_model->ForwardWithPreallocatedOutput(inputs, outputs);
  if (!success) {
    PrintError("[GPTStepModel] ForwardWithPreallocatedOutput failed");
  }
  return success;
}

std::unique_ptr<GPTStepContext> GPTStepModel::CreateContext(
    const std::vector<int64_t>& kv_cache_shape,
    int max_steps,
    int top_k) {

  auto ctx = std::make_unique<GPTStepContext>();
  ctx->max_steps = max_steps;

  Device model_device = m_model->GetDevice();
  // Use output dtype for cache buffers
  DataType cache_dtype = m_cache_out_dtype;

  // 预分配双缓冲 KV cache
  PrintInfo("[GPTStepContext] Allocating double-buffered KV cache: shape={}, in_dtype={}, out_dtype={}, device={}",
            fmt::join(kv_cache_shape, "x"),
            static_cast<int>(m_cache_dtype),
            static_cast<int>(m_cache_out_dtype),
            model_device.type == DeviceType::kCUDA ? "CUDA" : "CPU");

  for (int i = 0; i < 2; ++i) {
    ctx->k_cache[i] = Tensor::Empty(kv_cache_shape, cache_dtype, model_device);
    ctx->v_cache[i] = Tensor::Empty(kv_cache_shape, cache_dtype, model_device);
  }

  // 如果 input dtype != output dtype，需要在每步推理前做类型转换
  ctx->needs_cache_conversion = (m_cache_dtype != m_cache_out_dtype);
  if (ctx->needs_cache_conversion) {
    PrintInfo("[GPTStepContext] Cache dtype mismatch (in={}, out={}), per-step conversion enabled",
              static_cast<int>(m_cache_dtype), static_cast<int>(m_cache_out_dtype));
  }

  // 预分配输出缓冲区（dtype 从模型输出元数据获取）
  auto topk_val_dtype = m_model->GetOutputDataType("topk_values");
  auto topk_idx_dtype = m_model->GetOutputDataType("topk_indices");
  ctx->topk_values  = Tensor::Empty({1, top_k}, topk_val_dtype, model_device);
  ctx->topk_indices = Tensor::Empty({1, top_k}, topk_idx_dtype, model_device);

  // 预分配索引张量 (0 到 max_steps-1)
  PrintInfo("[GPTStepContext] Pre-allocating {} index tensors", max_steps);
  ctx->idx_tensors.reserve(max_steps);
  for (int i = 0; i < max_steps; ++i) {
    // 先在 CPU 上创建并填充数据
    auto idx_tensor_cpu = Tensor::Empty({1}, DataType::kInt64, Device{DeviceType::kCPU, 0});
    int64_t* idx_data = static_cast<int64_t*>(idx_tensor_cpu->Data());
    idx_data[0] = static_cast<int64_t>(i);

    // 如果目标设备是 CUDA，则传输到 GPU
    if (model_device.type == DeviceType::kCUDA) {
      auto idx_tensor = idx_tensor_cpu->To(model_device,DataType::kInt64);
      ctx->idx_tensors.push_back(std::move(idx_tensor));
    } else {
      ctx->idx_tensors.push_back(std::move(idx_tensor_cpu));
    }
  }

  // Pre-allocate scalar input tensors for the scalar StepWithContext overload
  ctx->current_samples = Tensor::Empty({1, 1}, DataType::kInt64, model_device);
  ctx->x_len_tensor = Tensor::Empty({1}, m_model->GetInputDataType("x_len"), model_device);
  ctx->y_len_tensor = Tensor::Empty({1}, m_model->GetInputDataType("y_len"), model_device);

  PrintInfo("[GPTStepContext] Context created successfully (zero-alloc ready)");
  return ctx;
}

std::unique_ptr<GPTStepContext> GPTStepModel::CreateContext(
    const KVCacheBuffer& init_cache,
    int max_steps,
    int top_k) {

  const auto& desc = init_cache.Desc();

  auto ctx = CreateContext(desc.raw_shape, max_steps, top_k);

  Device target_device = m_model->GetDevice();
  ctx->k_cache[0] = const_cast<KVCacheBuffer&>(init_cache).CurrentK()
                        ->To(target_device, m_cache_out_dtype);
  ctx->v_cache[0] = const_cast<KVCacheBuffer&>(init_cache).CurrentV()
                        ->To(target_device, m_cache_out_dtype);

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

  std::unique_ptr<Tensor> k_conv, v_conv;
  Tensor* k_input = ctx->GetCurrentKCache();
  Tensor* v_input = ctx->GetCurrentVCache();

  if (ctx->needs_cache_conversion) {
    k_conv = k_input->To(k_input->GetDevice(), m_cache_dtype);
    v_conv = v_input->To(v_input->GetDevice(), m_cache_dtype);
    k_input = k_conv.get();
    v_input = v_conv.get();
  }

  std::unordered_map<std::string, Tensor*> inputs = {
      {"samples",  samples},
      {"k_cache",  k_input},
      {"v_cache",  v_input},
      {"idx",      ctx->idx_tensors[step_idx].get()},
      {"x_len",    x_len},
      {"y_len",    y_len}
  };
  std::unordered_map<std::string, Tensor*> outputs = {
      {"topk_values",  ctx->topk_values.get()},
      {"topk_indices", ctx->topk_indices.get()},
      {"k_cache_new",  ctx->GetNextKCache()},
      {"v_cache_new",  ctx->GetNextVCache()}
  };

  bool success = m_model->ForwardWithPreallocatedOutput(inputs, outputs);
  if (success) ctx->SwapBuffers();
  return success;
}

bool GPTStepModel::StepWithContext(GPTStepContext* ctx,
                                   int64_t current_token,
                                   int step_idx,
                                   int64_t x_len,
                                   int64_t y_len) {
  if (!ctx) {
    PrintError("[GPTStepModel] Context is null (scalar overload)");
    return false;
  }

  // Update current_samples tensor with the new token value
  if (ctx->current_samples->IsCPU()) {
    ctx->current_samples->At<int64_t>(0) = current_token;
  }
#ifdef WITH_CUDA
  else {
    cudaMemcpy(ctx->current_samples->Data(), &current_token,
               sizeof(int64_t), cudaMemcpyHostToDevice);
  }
#endif

  // Update x_len / y_len tensors (constant across steps, but cheap to set)
  if (ctx->x_len_tensor->IsCPU()) {
    // dtype may be int32 or int64 — write via int64 then let the tensor handle it
    if (ctx->x_len_tensor->Type() == DataType::kInt32) {
      ctx->x_len_tensor->At<int32_t>(0) = static_cast<int32_t>(x_len);
      ctx->y_len_tensor->At<int32_t>(0) = static_cast<int32_t>(y_len);
    } else {
      ctx->x_len_tensor->At<int64_t>(0) = x_len;
      ctx->y_len_tensor->At<int64_t>(0) = y_len;
    }
  }
#ifdef WITH_CUDA
  else {
    if (ctx->x_len_tensor->Type() == DataType::kInt32) {
      int32_t xl = static_cast<int32_t>(x_len);
      int32_t yl = static_cast<int32_t>(y_len);
      cudaMemcpy(ctx->x_len_tensor->Data(), &xl, sizeof(int32_t), cudaMemcpyHostToDevice);
      cudaMemcpy(ctx->y_len_tensor->Data(), &yl, sizeof(int32_t), cudaMemcpyHostToDevice);
    } else {
      cudaMemcpy(ctx->x_len_tensor->Data(), &x_len, sizeof(int64_t), cudaMemcpyHostToDevice);
      cudaMemcpy(ctx->y_len_tensor->Data(), &y_len, sizeof(int64_t), cudaMemcpyHostToDevice);
    }
  }
#endif

  return StepWithContext(ctx, ctx->current_samples.get(), step_idx,
                         ctx->x_len_tensor.get(), ctx->y_len_tensor.get());
}

bool GPTStepModel::StepWithGPUSampling(GPTStepContext* ctx,
                                       int64_t current_token,
                                       int step_idx,
                                       int64_t x_len,
                                       int64_t y_len,
                                       float temperature) {
#ifdef WITH_CUDA
  if (!ctx || !ctx->enable_gpu_sampling) {
    PrintError("[GPTStepModel] GPU sampling not enabled or ctx is null");
    return false;
  }

  if (step_idx < 0 || step_idx >= ctx->max_steps) {
    PrintError("[GPTStepModel] step_idx {} out of range [0, {})", step_idx, ctx->max_steps);
    return false;
  }

  // Update current_samples (GPU tensor)
  cudaMemcpyAsync(ctx->current_samples->Data(), &current_token,
                  sizeof(int64_t), cudaMemcpyHostToDevice, ctx->cuda_stream);

  // Update x_len / y_len tensors (only once)
  if (!ctx->x_y_len_initialized) {
    if (ctx->x_len_tensor->Type() == DataType::kInt32) {
      int32_t xl = static_cast<int32_t>(x_len);
      int32_t yl = static_cast<int32_t>(y_len);
      cudaMemcpyAsync(ctx->x_len_tensor->Data(), &xl, sizeof(int32_t), 
                      cudaMemcpyHostToDevice, ctx->cuda_stream);
      cudaMemcpyAsync(ctx->y_len_tensor->Data(), &yl, sizeof(int32_t), 
                      cudaMemcpyHostToDevice, ctx->cuda_stream);
    } else {
      cudaMemcpyAsync(ctx->x_len_tensor->Data(), &x_len, sizeof(int64_t), 
                      cudaMemcpyHostToDevice, ctx->cuda_stream);
      cudaMemcpyAsync(ctx->y_len_tensor->Data(), &y_len, sizeof(int64_t), 
                      cudaMemcpyHostToDevice, ctx->cuda_stream);
    }
    ctx->x_y_len_initialized = true;
  }

  // Prepare cache input (with type conversion if needed)
  std::unique_ptr<Tensor> k_conv, v_conv;
  Tensor* k_input = ctx->GetCurrentKCache();
  Tensor* v_input = ctx->GetCurrentVCache();

  if (ctx->needs_cache_conversion) {
    k_conv = k_input->To(k_input->GetDevice(), m_cache_dtype);
    v_conv = v_input->To(v_input->GetDevice(), m_cache_dtype);
    k_input = k_conv.get();
    v_input = v_conv.get();
  }

  // Prepare inputs/outputs for inference
  std::unordered_map<std::string, Tensor*> inputs = {
      {"samples",  ctx->current_samples.get()},
      {"k_cache",  k_input},
      {"v_cache",  v_input},
      {"idx",      ctx->idx_tensors[step_idx].get()},
      {"x_len",    ctx->x_len_tensor.get()},
      {"y_len",    ctx->y_len_tensor.get()}
  };
  std::unordered_map<std::string, Tensor*> outputs = {
      {"topk_values",  ctx->topk_values.get()},
      {"topk_indices", ctx->topk_indices.get()},
      {"k_cache_new",  ctx->GetNextKCache()},
      {"v_cache_new",  ctx->GetNextVCache()}
  };

  // Run inference
  bool success = m_model->ForwardWithPreallocatedOutput(inputs, outputs);
  if (!success) {
    PrintError("[GPTStepModel] Forward failed");
    return false;
  }

  // GPU 采样 (在同一个 stream 上执行)
  auto topk_val_dtype = ctx->topk_values->Type();
  cudaError_t err;
  
  if (topk_val_dtype == DataType::kFloat16) {
    err = GPU::LaunchSampleTopKFP16Kernel(
        ctx->topk_values->Data(),
        ctx->topk_indices->Data<int64_t>(),
        ctx->top_k,
        temperature,
        ctx->out_token_gpu->Data<int64_t>(),
        static_cast<uint64_t*>(ctx->rng_state->Data()),
        ctx->cuda_stream);
  } else {
    err = GPU::LaunchSampleTopKKernel(
        ctx->topk_values->Data(),
        ctx->topk_indices->Data<int64_t>(),
        ctx->top_k,
        temperature,
        ctx->out_token_gpu->Data<int64_t>(),
        static_cast<uint64_t*>(ctx->rng_state->Data()),
        ctx->cuda_stream);
  }

  if (err != cudaSuccess) {
    PrintError("[GPTStepModel] GPU sampling kernel failed: {}", cudaGetErrorString(err));
    return false;
  }

  ctx->SwapBuffers();
  return true;
#else
  PrintError("[GPTStepModel] GPU sampling requires CUDA support");
  return false;
#endif
}

int64_t GPTStepModel::GetSampledTokenGPU(GPTStepContext* ctx) {
#ifdef WITH_CUDA
  if (!ctx || !ctx->enable_gpu_sampling || !ctx->out_token_gpu) {
    return -1;
  }

  int64_t token = 0;
  cudaMemcpyAsync(&token, ctx->out_token_gpu->Data<int64_t>(), 
                  sizeof(int64_t), cudaMemcpyDeviceToHost, ctx->cuda_stream);
  cudaStreamSynchronize(ctx->cuda_stream);
  return token;
#else
  return -1;
#endif
}

bool GPTStepModel::EnableGPUSampling(GPTStepContext* ctx, uint64_t seed) {
#ifdef WITH_CUDA
  if (!ctx) {
    PrintError("[GPTStepModel] Context is null");
    return false;
  }

  Device model_device = m_model->GetDevice();
  if (model_device.type != DeviceType::kCUDA) {
    PrintWarn("[GPTStepModel] GPU sampling only available on CUDA device");
    return false;
  }

  if (model_device.stream) {
    ctx->cuda_stream = static_cast<cudaStream_t>(model_device.stream);
    ctx->owns_stream = false;
    PrintDebug("[GPTStepModel] Using model's CUDA stream");
  } else if (!ctx->cuda_stream) {
    cudaError_t err = cudaStreamCreate(&ctx->cuda_stream);
    if (err != cudaSuccess) {
      PrintError("[GPTStepModel] Failed to create CUDA stream: {}", cudaGetErrorString(err));
      return false;
    }
    ctx->owns_stream = true;
    PrintDebug("[GPTStepModel] Created new CUDA stream");
  }

  ctx->rng_state = Tensor::Empty({1}, DataType::kUInt64, model_device);
  uint64_t init_seed = (seed != 0) ? seed : static_cast<uint64_t>(std::random_device{}());
  cudaMemcpyAsync(ctx->rng_state->Data(), &init_seed, sizeof(uint64_t), 
                  cudaMemcpyHostToDevice, ctx->cuda_stream);

  ctx->out_token_gpu = Tensor::Empty({1}, DataType::kInt64, model_device);

  ctx->enable_gpu_sampling = true;
  ctx->top_k = static_cast<int>(ctx->topk_values->Shape().back());
  
  PrintInfo("[GPTStepModel] GPU sampling enabled (seed={}, top_k={}, own_stream={})", 
            init_seed, ctx->top_k, ctx->owns_stream);
  return true;
#else
  PrintWarn("[GPTStepModel] GPU sampling requires CUDA support");
  return false;
#endif
}

}  // namespace GPTSoVITS::Model