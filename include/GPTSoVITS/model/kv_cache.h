//
// Created by Huiyicc on 2026/2/28.
//

#ifndef GSV_CPP_KV_CACHE_H
#define GSV_CPP_KV_CACHE_H

#include <memory>
#include <vector>

#include "GPTSoVITS/model/tensor.h"

namespace GPTSoVITS::Model {

/**
 * @brief Semantic descriptor for a KV cache buffer.
 *
 * Decouples pipeline logic from raw tensor shapes so that ONNX (dynamic
 * seq_len) and TRT (pre-allocated fixed seq_len) backends can be handled
 * uniformly.
 */
struct KVCacheDesc {
  int64_t num_layers  = 0;
  int64_t max_seq_len = 0;   // semantic capacity; pipeline uses this, NOT raw shape
  DataType dtype  = DataType::kFloat32;
  Device   device;
  std::vector<int64_t> raw_shape;  // original tensor shape from backend (private to model layer)
};

/**
 * @brief Double-buffered KV cache with semantic metadata.
 *
 * Wraps the encoder output k/v tensors and pre-allocates a second buffer
 * for ping-pong during autoregressive generation.  The pipeline only ever
 * calls MaxSeqLen() — it never inspects raw tensor shapes.
 */
class KVCacheBuffer {
public:
  /**
   * @brief Create from encoder output tensors.
   *
   * k and v are moved into buffer slot 0.  A second slot is allocated with
   * the same shape so that GPTStepModel can use this directly for IO binding.
   *
   * @param k   Key cache from encoder (ownership transferred)
   * @param v   Value cache from encoder (ownership transferred)
   * @param desc Semantic descriptor (max_seq_len must be set by caller)
   */
  static std::unique_ptr<KVCacheBuffer> Create(
      std::unique_ptr<Tensor> k,
      std::unique_ptr<Tensor> v,
      const KVCacheDesc& desc);

  [[nodiscard]] int64_t MaxSeqLen() const { return desc_.max_seq_len; }
  [[nodiscard]] int64_t NumLayers() const { return desc_.num_layers; }
  [[nodiscard]] const KVCacheDesc& Desc() const { return desc_; }

  Tensor* CurrentK() { return k_[cur_].get(); }
  Tensor* CurrentV() { return v_[cur_].get(); }
  Tensor* NextK()    { return k_[1 - cur_].get(); }
  Tensor* NextV()    { return v_[1 - cur_].get(); }
  void Swap()  { cur_ = 1 - cur_; }
  void Reset() { cur_ = 0; }

private:
  KVCacheBuffer() = default;

  KVCacheDesc desc_;
  std::unique_ptr<Tensor> k_[2];
  std::unique_ptr<Tensor> v_[2];
  int cur_ = 0;
};

}  // namespace GPTSoVITS::Model

#endif  // GSV_CPP_KV_CACHE_H
