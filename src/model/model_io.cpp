//
// Created by Huiyicc on 2026/2/28.
//

#include "GPTSoVITS/model/model_io.h"

namespace GPTSoVITS::Model {

ModelInputs& ModelInputs::Set(const std::string& name, Tensor* t) {
  raw_[name] = t;
  return *this;
}

void ModelInputs::PrepareFor(const BaseModel& model) {
  owned_.clear();
  prepared_.clear();

  Device model_device = model.GetDevice();

  for (auto& [name, t] : raw_) {
    if (!t) {
      prepared_[name] = nullptr;
      continue;
    }
    DataType expected_dtype = model.GetInputDataType(name);
    bool needs_convert = (t->GetDeviceType() != model_device.type ||
                          t->Type() != expected_dtype);
    if (needs_convert) {
      auto converted = t->To(model_device, expected_dtype);
      prepared_[name] = converted.get();
      owned_.push_back(std::move(converted));
    } else {
      prepared_[name] = t;
    }
  }
}

}  // namespace GPTSoVITS::Model
