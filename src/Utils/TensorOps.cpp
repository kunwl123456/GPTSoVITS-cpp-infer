//
// Created by 19254 on 2026/2/22.
//
// 张量操作工具实现
//

#include "GPTSoVITS/Utils/TensorOps.h"

#include <algorithm>
#include <cstring>

#include "GPTSoVITS/plog.h"

#ifdef WITH_CUDA
#include <cuda_runtime.h>
#endif

namespace GPTSoVITS {
namespace Utils {

// ============================================================================
// 张量拼接操作
// ============================================================================

std::unique_ptr<Model::Tensor> ConcatTensors(
    const Model::Tensor* a,
    const Model::Tensor* b,
    int axis) {
  if (!a || !b) {
    PrintError("[ConcatTensors] Null tensor input");
    return nullptr;
  }
  return ConcatTensors({a, b}, axis);
}

std::unique_ptr<Model::Tensor> ConcatTensors(
    const std::vector<const Model::Tensor*>& tensors,
    int axis) {
  if (tensors.empty()) {
    PrintError("[ConcatTensors] Empty tensor list");
    return nullptr;
  }
  
  if (tensors.size() == 1) {
    return tensors[0]->Clone();
  }
  
  // 验证所有张量的类型、设备和维度一致性
  Model::DataType dtype = tensors[0]->Type();
  Model::Device device = tensors[0]->GetDevice();
  auto base_shape = tensors[0]->Shape();
  
  if (axis < 0) axis += static_cast<int>(base_shape.size());
  if (axis < 0 || axis >= static_cast<int>(base_shape.size())) {
    PrintError("[ConcatTensors] Invalid axis: {}", axis);
    return nullptr;
  }
  
  // 计算输出形状
  std::vector<int64_t> out_shape = base_shape;
  int64_t total_dim = 0;
  
  for (auto t : tensors) {
    if (t->Type() != dtype) {
      PrintError("[ConcatTensors] Data type mismatch");
      return nullptr;
    }
    if (t->GetDevice() != device) {
      PrintError("[ConcatTensors] Device mismatch");
      return nullptr;
    }
    auto s = t->Shape();
    if (s.size() != base_shape.size()) {
      PrintError("[ConcatTensors] Dimension mismatch");
      return nullptr;
    }
    for (size_t i = 0; i < s.size(); ++i) {
      if (static_cast<int>(i) != axis && s[i] != base_shape[i]) {
        PrintError("[ConcatTensors] Shape mismatch on non-concat axis");
        return nullptr;
      }
    }
    total_dim += s[axis];
  }
  out_shape[axis] = total_dim;
  
  // 创建输出张量
  auto out_tensor = Model::Tensor::Empty(out_shape, dtype, device);
  
  // 执行拼接
  size_t element_size = Model::Tensor::ElementSize(dtype);
  uint8_t* dst_ptr = out_tensor->Data<uint8_t>();
  
  if (axis == 0) {
    // 沿axis=0拼接，直接拷贝
    for (auto t : tensors) {
      size_t bytes = t->ByteSize();
      if (device.type == Model::DeviceType::kCPU) {
        std::memcpy(dst_ptr, t->Data(), bytes);
      } else {
#ifdef WITH_CUDA
        cudaMemcpy(dst_ptr, t->Data(), bytes, cudaMemcpyDeviceToDevice);
#endif
      }
      dst_ptr += bytes;
    }
  } else if (base_shape.size() == 2 && axis == 1) {
    // 沿axis=1拼接二维张量
    int64_t row_count = base_shape[0];
    for (int64_t r = 0; r < row_count; ++r) {
      for (auto t : tensors) {
        size_t row_bytes = t->Shape()[1] * element_size;
        void* src = static_cast<uint8_t*>(t->Data()) + r * row_bytes;
        if (device.type == Model::DeviceType::kCPU) {
          std::memcpy(dst_ptr, src, row_bytes);
        } else {
#ifdef WITH_CUDA
          cudaMemcpy(dst_ptr, src, row_bytes, cudaMemcpyDeviceToDevice);
#endif
        }
        dst_ptr += row_bytes;
      }
    }
  } else {
    std::vector<Model::Tensor*> tensor_ptrs;
    for (auto t : tensors) {
      // 创建临时unique_ptr但释放所有权
      auto tmp = const_cast<Model::Tensor*>(t);
      tensor_ptrs.push_back(tmp);
    }
    return Model::Tensor::Concat(tensor_ptrs, axis);
  }
  
  return out_tensor;
}

// ============================================================================
// 张量类型转换
// ============================================================================

std::unique_ptr<Model::Tensor> SafeCast(
    const Model::Tensor* tensor,
    Model::DataType target_dtype,
    const Model::Device* target_device) {
  
  if (!tensor) {
    PrintError("[SafeCast] Null tensor input");
    return nullptr;
  }
  
  // 源和目标一致，返回Clone
  bool same_dtype = (tensor->Type() == target_dtype);
  bool same_device = (!target_device || tensor->GetDevice() == *target_device);
  
  if (same_dtype && same_device) {
    return tensor->Clone();
  }
  
  // 记录潜在精度损失
  if (tensor->Type() == Model::DataType::kFloat32 && 
      target_dtype == Model::DataType::kFloat16) {
    PrintDebug("[SafeCast] FP32 -> FP16 conversion, potential precision loss");
  }
  
  // 执行转换
  if (same_device) {
    return tensor->ToType(target_dtype);
  } else if (same_dtype) {
    return tensor->ToDevice(*target_device);
  } else {
    // 综合转换：选择最优路径
    // 非标准转换在CPU上进行更稳定
    auto temp_cpu = tensor->ToCPU();
    auto temp_typed = temp_cpu->ToType(target_dtype);
    if (target_device && target_device->type != Model::DeviceType::kCPU) {
      return temp_typed->ToDevice(*target_device);
    }
    return temp_typed;
  }
}

std::unique_ptr<Model::Tensor> TransformTensor(
    const Model::Tensor* tensor,
    Model::Device target_device,
    Model::DataType target_dtype) {
  
  return SafeCast(tensor, target_dtype, &target_device);
}

// ============================================================================
// 张量创建辅助函数
// ============================================================================

std::unique_ptr<Model::Tensor> CreateTensor(
    const void* data,
    const std::vector<int64_t>& shape,
    Model::DataType dtype,
    Model::Device device) {
  
  auto tensor = Model::Tensor::Empty(shape, dtype, device);
  size_t bytes = tensor->ByteSize();
  
  if (device.type == Model::DeviceType::kCPU) {
    std::memcpy(tensor->Data(), data, bytes);
  } else {
#ifdef WITH_CUDA
    cudaMemcpy(tensor->Data(), data, bytes, cudaMemcpyHostToDevice);
#endif
  }
  
  return tensor;
}

std::unique_ptr<Model::Tensor> Zeros(
    const std::vector<int64_t>& shape,
    Model::DataType dtype,
    Model::Device device) {
  
  auto tensor = Model::Tensor::Empty(shape, dtype, device);
  size_t bytes = tensor->ByteSize();
  
  if (device.type == Model::DeviceType::kCPU) {
    std::memset(tensor->Data(), 0, bytes);
  } else {
#ifdef WITH_CUDA
    cudaMemset(tensor->Data(), 0, bytes);
#endif
  }
  
  return tensor;
}

// ============================================================================
// 张量视图操作 (零拷贝)
// ============================================================================

std::unique_ptr<Model::Tensor> Unsqueeze(
    const Model::Tensor* tensor,
    int axis) {
  
  if (!tensor) return nullptr;
  
  auto shape = tensor->Shape();
  if (axis < 0) axis += static_cast<int>(shape.size()) + 1;
  if (axis < 0 || axis > static_cast<int>(shape.size())) {
    PrintError("[Unsqueeze] Invalid axis: {}", axis);
    return nullptr;
  }
  
  std::vector<int64_t> new_shape;
  new_shape.reserve(shape.size() + 1);
  new_shape.insert(new_shape.end(), shape.begin(), shape.begin() + axis);
  new_shape.push_back(1);
  new_shape.insert(new_shape.end(), shape.begin() + axis, shape.end());
  
  return tensor->View(new_shape);
}

std::unique_ptr<Model::Tensor> Reshape(
    const Model::Tensor* tensor,
    const std::vector<int64_t>& new_shape) {
  
  if (!tensor) return nullptr;
  return tensor->View(new_shape);
}

// ============================================================================
// 张量切片操作
// ============================================================================

std::unique_ptr<Model::Tensor> TakeLast(
    const Model::Tensor* tensor,
    int64_t n,
    int axis) {
  
  if (!tensor) return nullptr;
  
  auto shape = tensor->Shape();
  if (axis < 0) axis += static_cast<int>(shape.size());
  if (axis < 0 || axis >= static_cast<int>(shape.size())) {
    PrintError("[TakeLast] Invalid axis: {}", axis);
    return nullptr;
  }
  
  n = std::min(n, shape[axis]);
  if (n <= 0) return nullptr;
  
  int64_t start = shape[axis] - n;
  return tensor->Slice(start, shape[axis], axis);
}

std::unique_ptr<Model::Tensor> TakeFirst(
    const Model::Tensor* tensor,
    int64_t n,
    int axis) {
  
  if (!tensor) return nullptr;
  
  auto shape = tensor->Shape();
  if (axis < 0) axis += static_cast<int>(shape.size());
  if (axis < 0 || axis >= static_cast<int>(shape.size())) {
    PrintError("[TakeFirst] Invalid axis: {}", axis);
    return nullptr;
  }
  
  n = std::min(n, shape[axis]);
  if (n <= 0) return nullptr;
  
  return tensor->Slice(0, n, axis);
}

}  // namespace Utils
}  // namespace GPTSoVITS
