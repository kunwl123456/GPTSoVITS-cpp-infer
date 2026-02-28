//
// Created by Huiyicc on 2026/2/28.
//

#ifndef GSV_CPP_MODEL_IO_H
#define GSV_CPP_MODEL_IO_H

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "GPTSoVITS/model/base.h"
#include "GPTSoVITS/model/tensor.h"

namespace GPTSoVITS::Model {

/**
 * @brief Helper that eliminates repetitive device/dtype conversion boilerplate.
 *
 * Usage:
 * @code
 *   ModelInputs inputs;
 *   inputs.Set("samples", samples_ptr)
 *         .Set("k_cache", k_cache_ptr)
 *         .Set("x_len",   x_len_ptr);
 *   inputs.PrepareFor(*m_model);
 *   m_model->Forward(inputs.Raw(), outputs);
 * @endcode
 */
class ModelInputs {
public:
  /**
   * @brief Register a named input tensor (raw pointer, not owned).
   * @return *this for chaining
   */
  ModelInputs& Set(const std::string& name, Tensor* t);

  /**
   * @brief Convert all registered tensors to the device/dtype expected by model.
   *
   * Tensors that already match are passed through without allocation.
   * Converted tensors are owned by this object until the next call or destruction.
   */
  void PrepareFor(const BaseModel& model);

  /**
   * @brief Return the prepared input map (valid after PrepareFor).
   */
  [[nodiscard]] const std::unordered_map<std::string, Tensor*>& Raw() const { return prepared_; }

private:
  std::unordered_map<std::string, Tensor*> raw_;
  std::unordered_map<std::string, Tensor*> prepared_;
  std::vector<std::unique_ptr<Tensor>> owned_;  // converted temporaries
};

}  // namespace GPTSoVITS::Model

#endif  // GSV_CPP_MODEL_IO_H
