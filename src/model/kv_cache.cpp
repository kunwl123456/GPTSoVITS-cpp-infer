//
// Created by Huiyicc on 2026/2/28.
//

#include "GPTSoVITS/model/kv_cache.h"

namespace GPTSoVITS::Model {

std::unique_ptr<KVCacheBuffer> KVCacheBuffer::Create(
    std::unique_ptr<Tensor> k,
    std::unique_ptr<Tensor> v,
    const KVCacheDesc& desc) {

  auto buf = std::unique_ptr<KVCacheBuffer>(new KVCacheBuffer());
  buf->desc_ = desc;

  // initial encoder output
  buf->k_[0] = std::move(k);
  buf->v_[0] = std::move(v);

  // pre-allocated second buffer for ping-pong (same shape, same device/dtype)
  buf->k_[1] = Tensor::Empty(buf->k_[0]->Shape(), desc.dtype, desc.device);
  buf->v_[1] = Tensor::Empty(buf->v_[0]->Shape(), desc.dtype, desc.device);

  buf->cur_ = 0;
  return buf;
}

}  // namespace GPTSoVITS::Model
