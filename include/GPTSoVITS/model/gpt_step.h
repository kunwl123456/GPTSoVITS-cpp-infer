//
// Created by Huiyicc on 2026/2/17.
//

#ifndef GSV_CPP_GPT_STEP_H
#define GSV_CPP_GPT_STEP_H

#include <memory>
#include <vector>

#include "GPTSoVITS/model/base.h"
#include "GPTSoVITS/model/tensor.h"
#include "GPTSoVITS/model/kv_cache.h"
#include "GPTSoVITS/Utils/exception.h"
#include "GPTSoVITS/model/backend/backend_config.h"

namespace GPTSoVITS::Model {

/**
 * @brief GPT Step output structure
 */
struct GPTStepOutput {
  std::unique_ptr<Tensor> topk_values;      // [1, top_k]
  std::unique_ptr<Tensor> topk_indices;     // [1, top_k]
  std::unique_ptr<Tensor> k_cache_new;      // [num_layers, batch, num_heads, seq_len, head_dim]
  std::unique_ptr<Tensor> v_cache_new;      // [num_layers, batch, num_heads, seq_len, head_dim]
};

/**
 * @brief KV Cache 双缓冲上下文
 */
struct GPTStepContext {
  // 双缓冲 KV cache (ping-pong)
  std::unique_ptr<Tensor> k_cache[2];  // [num_layers, batch, max_seq_len, head_dim]
  std::unique_ptr<Tensor> v_cache[2];  // [num_layers, batch, max_seq_len, head_dim]

  // 输出缓冲区 (复用)
  std::unique_ptr<Tensor> topk_values;   // [1, top_k]
  std::unique_ptr<Tensor> topk_indices;  // [1, top_k]

  // 预分配的索引张量 (避免每次创建)
  std::vector<std::unique_ptr<Tensor>> idx_tensors;  // [max_steps] 个 [1] 张量

  // 预分配的标量输入张量
  std::unique_ptr<Tensor> current_samples;  // [1, 1] on model device
  std::unique_ptr<Tensor> x_len_tensor;     // [1] on model device, model dtype
  std::unique_ptr<Tensor> y_len_tensor;     // [1] on model device, model dtype

  // 当前使用的缓冲区索引 (0 或 1)
  int current_buffer = 0;

  // 最大生成步数
  int max_steps = 1000;

  // 是否需要 cache 类型转换 (input dtype != output dtype)
  bool needs_cache_conversion = false;

  /**
   * @brief 获取当前输入 cache
   */
  Tensor* GetCurrentKCache() { return k_cache[current_buffer].get(); }
  Tensor* GetCurrentVCache() { return v_cache[current_buffer].get(); }

  /**
   * @brief 获取下一个输出 cache
   */
  Tensor* GetNextKCache() { return k_cache[1 - current_buffer].get(); }
  Tensor* GetNextVCache() { return v_cache[1 - current_buffer].get(); }

  /**
   * @brief 切换缓冲区 (ping-pong)
   */
  void SwapBuffers() { current_buffer = 1 - current_buffer; }

  /**
   * @brief 重置到初始状态
   */
  void Reset() { current_buffer = 0; }
};

/**
 * @brief GPT Step Model
 *
 * GPT Step performs single-step autoregressive generation.
 * This implementation supports zero-copy IO binding to minimize memory transfers
 * between CPU and GPU, which is critical for performance during multi-step generation.
 */
class GPTStepModel {
protected:
  std::unique_ptr<BaseModel> m_model;
  int m_top_k = 5;
  bool m_supports_iobinding = false;
  DataType m_cache_dtype = DataType::kFloat32;      // model INPUT k_cache dtype
  DataType m_cache_out_dtype = DataType::kFloat32;  // model OUTPUT k_cache_new dtype

public:
  GPTStepModel() = default;
  virtual ~GPTStepModel() = default;

  /**
   * @brief Initialize model with backend
   * @tparam MODEL_BACKEND Backend type (e.g., ONNXBackend)
   * @param model_path Path to gpt_step.onnx
   * @param device Device to run on
   * @param work_thread_num Number of threads
   */
  template <typename MODEL_BACKEND>
  void Init(const std::string& model_path,
            const Device& device = DeviceType::kCPU,
            int work_thread_num = 1) {
    m_model = std::make_unique<MODEL_BACKEND>();
    if (!m_model->Load(model_path, device, work_thread_num)) {
      THROW_ERRORN("Failed to load GPTStepModel from: {}", model_path);
    }

    DetectCacheDTypes();
    m_supports_iobinding = true;
  }

  template <typename MODEL_BACKEND>
  void Init(const std::string& model_path, const BackendConfig& config) {
    m_model = std::make_unique<MODEL_BACKEND>();
    if (!m_model->Load(model_path, config)) {
      THROW_ERRORN("Failed to load GPTStepModel from: {}", model_path);
    }

    DetectCacheDTypes();
    m_supports_iobinding = true;
  }

  /**
   * @brief Run single GPT step inference (standard forward)
   *
   * @param samples Previous token sample [1, 1]
   * @param k_cache Key cache [num_layers, batch, num_heads, seq_len, head_dim]
   * @param v_cache Value cache [num_layers, batch, num_heads, seq_len, head_dim]
   * @param idx Current step index [1]
   * @param x_len Prompt length [1]
   * @param y_len Target length [1]
   * @return GPTStepOutput containing next topk, indices, and updated cache
   */
  GPTStepOutput Step(Tensor* samples,
                     Tensor* k_cache,
                     Tensor* v_cache,
                     Tensor* idx,
                     Tensor* x_len,
                     Tensor* y_len);

  /**
   * @brief Run GPT step with IO binding for zero-copy (if supported)
   *
   * This method uses pre-allocated tensors to avoid memory allocations
   * during generation loops. It's more efficient for multi-step generation.
   *
   * @param samples Previous token sample [1, 1]
   * @param k_cache_in Input key cache (will be consumed)
   * @param v_cache_in Input value cache (will be consumed)
   * @param k_cache_out Output key cache (will be written to)
   * @param v_cache_out Output value cache (will be written to)
   * @param idx Current step index [1]
   * @param x_len Prompt length [1]
   * @param y_len Target length [1]
   * @param topk_values_out Output buffer for topk values [1, top_k]
   * @param topk_indices_out Output buffer for topk indices [1, top_k]
   * @return true if successful
   */
  bool StepWithIOBinding(Tensor* samples,
                         Tensor* k_cache_in,
                         Tensor* v_cache_in,
                         Tensor* k_cache_out,
                         Tensor* v_cache_out,
                         Tensor* idx,
                         Tensor* x_len,
                         Tensor* y_len,
                         Tensor* topk_values_out,
                         Tensor* topk_indices_out);

  /**
   * @brief 创建双缓冲上下文
   *
   * 预分配所有必要的缓冲区，避免生成循环中的内存分配。
   *
   * @param kv_cache_shape KV cache 的形状 [num_layers, batch, max_seq_len, head_dim]
   * @param max_steps 最大生成步数 (默认 1000)
   * @param top_k Top-K 采样参数 (默认 5)
   * @return 预分配的上下文对象
   */
  std::unique_ptr<GPTStepContext> CreateContext(
      const std::vector<int64_t>& kv_cache_shape,
      int max_steps = 1000,
      int top_k = 5);

  /**
   * @brief Create context from a KVCacheBuffer.
   *
   * Derives shape from init_cache.Desc() and copies initial k/v data into
   * the context, so the pipeline does not need to call CopyFrom manually.
   *
   * @param init_cache  Encoder output KV cache (read-only)
   * @param max_steps   Maximum generation steps
   * @param top_k       Top-K sampling parameter
   */
  std::unique_ptr<GPTStepContext> CreateContext(
      const KVCacheBuffer& init_cache,
      int max_steps = 1000,
      int top_k = 5);

  /**
   * @brief 使用上下文进行推理
   *
   * @param ctx 预分配的上下文
   * @param samples 当前 token [1, 1]
   * @param step_idx 当前步数 (0-based)
   * @param x_len Prompt 长度 [1]
   * @param y_len Target 长度 [1]
   * @return true 如果成功
   *
   * @note 调用后需要手动调用 ctx->SwapBuffers() 切换缓冲区
   */
  bool StepWithContext(GPTStepContext* ctx,
                       Tensor* samples,
                       int step_idx,
                       Tensor* x_len,
                       Tensor* y_len);

  /**
   * @brief Scalar overload — no Tensor* plumbing in the pipeline.
   *
   * Updates the pre-allocated current_samples / x_len / y_len tensors inside
   * ctx (including CUDA memcpy when needed) and delegates to the Tensor* overload.
   *
   * @param ctx           Pre-allocated context from CreateContext
   * @param current_token Current token value (scalar)
   * @param step_idx      Current step index (0-based)
   * @param x_len         Prompt length (scalar, constant across steps)
   * @param y_len         Target text length (scalar, constant across steps)
   */
  bool StepWithContext(GPTStepContext* ctx,
                       int64_t current_token,
                       int step_idx,
                       int64_t x_len,
                       int64_t y_len);

  /**
   * @brief Check if IO binding is supported
   */
  [[nodiscard]] bool SupportsIOBinding() const { return m_supports_iobinding; }

  /**
   * @brief Get cache data type
   */
  [[nodiscard]] DataType GetCacheDType() const { return m_cache_dtype; }

  /**
   * @brief Get cache output data type
   */
  [[nodiscard]] DataType GetCacheOutDType() const { return m_cache_out_dtype; }

  /**
   * @brief Get the underlying model
   */
  [[nodiscard]] const BaseModel* GetModel() const { return m_model.get(); }

private:
  void DetectCacheDTypes() {
    for (const auto& name : m_model->GetInputNames()) {
      if (name == "k_cache" || name == "v_cache") {
        m_cache_dtype = m_model->GetInputDataType(name);
        break;
      }
    }
    // Default output dtype to input dtype; override if model says otherwise
    m_cache_out_dtype = m_cache_dtype;
    for (const auto& name : m_model->GetOutputNames()) {
      if (name == "k_cache_new" || name == "v_cache_new") {
        m_cache_out_dtype = m_model->GetOutputDataType(name);
        break;
      }
    }
  }
};

}  // namespace GPTSoVITS::Model

#endif  // GSV_CPP_GPT_STEP_H