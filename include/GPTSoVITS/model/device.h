//
// Created by Huiyicc on 2026/1/10.
//

#ifndef GPT_SOVITS_CPP_DEVICE_H
#define GPT_SOVITS_CPP_DEVICE_H

#include <cstdint>

namespace GPTSoVITS::Model {

enum class DeviceType {
  kCPU = 0,
  kCUDA = 1,
  // kDirectML = 2,
  // kCoreML = 3
};

struct Device {
  DeviceType type;
  int device_id;
  void* stream;

  Device(DeviceType t = DeviceType::kCPU, int id = 0, void* s = nullptr)
      : type(t), device_id(id), stream(s) {}

  bool operator==(const Device& other) const {
    return type == other.type && device_id == other.device_id;
  }
  bool operator!=(const Device& other) const {
    return !operator==(other);
  }

  bool StrictEquals(const Device& other) const {
    return type == other.type && device_id == other.device_id &&
           stream == other.stream;
  }
};

}  // namespace GPTSoVITS::Model

#endif  // GPT_SOVITS_CPP_DEVICE_H