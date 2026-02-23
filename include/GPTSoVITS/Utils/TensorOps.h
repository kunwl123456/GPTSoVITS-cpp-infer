//
// Created by 19254 on 2026/2/22.
//
// 张量操作
//

#ifndef GPT_SOVITS_CPP_UTILS_TENSOR_OPS_H
#define GPT_SOVITS_CPP_UTILS_TENSOR_OPS_H

#include <memory>
#include <vector>

#include "GPTSoVITS/model/tensor.h"

namespace GPTSoVITS {
namespace Utils {

// ============================================================================
// 张量拼接操作
// ============================================================================

/**
 * @brief 沿指定轴拼接两个张量
 * 
 * @param a 第一个张量
 * @param b 第二个张量
 * @param axis 拼接轴 (默认0)
 * @return 拼接后的新张量
 */
std::unique_ptr<Model::Tensor> ConcatTensors(
    const Model::Tensor* a,
    const Model::Tensor* b,
    int axis = 0);

/**
 * @brief 沿指定轴拼接多个张量
 * 
 * @param tensors 张量指针数组
 * @param axis 拼接轴
 * @return 拼接后的新张量
 */
std::unique_ptr<Model::Tensor> ConcatTensors(
    const std::vector<const Model::Tensor*>& tensors,
    int axis = 0);

// ============================================================================
// 张量类型转换
// ============================================================================

/**
 * @brief 安全的张量类型转换
 *
 * - 源类型 == 目标类型：返回Clone
 * - 设备转换 + 类型转换：选择最优路径
 * - FP16 -> FP32：安全转换，无精度损失
 * - FP32 -> FP16：可能损失精度，记录警告
 * 
 * @param tensor 输入张量
 * @param target_dtype 目标数据类型
 * @param target_device 目标设备 (可选，默认保持原设备)
 * @return 转换后的新张量
 */
std::unique_ptr<Model::Tensor> SafeCast(
    const Model::Tensor* tensor,
    Model::DataType target_dtype,
    const Model::Device* target_device = nullptr);

/**
 * @brief 将张量转换到目标设备和类型
 * 
 * 自动检测并优化转换路径：
 * - 如果源和目标一致，返回Clone
 * - 否则选择最少拷贝次数的路径
 * 
 * @param tensor 输入张量
 * @param target_device 目标设备
 * @param target_dtype 目标数据类型
 * @return 转换后的新张量
 */
std::unique_ptr<Model::Tensor> TransformTensor(
    const Model::Tensor* tensor,
    Model::Device target_device,
    Model::DataType target_dtype);

// ============================================================================
// 张量创建辅助函数
// ============================================================================

/**
 * @brief 从数据创建张量
 * 
 * @param data 数据指针
 * @param shape 张量形状
 * @param dtype 数据类型
 * @param device 目标设备
 * @return 新张量
 */
std::unique_ptr<Model::Tensor> CreateTensor(
    const void* data,
    const std::vector<int64_t>& shape,
    Model::DataType dtype,
    Model::Device device);

/**
 * @brief 从向量创建一维张量
 * 
 * @tparam T 数据类型
 * @param data 数据向量
 * @param dtype 张量数据类型
 * @param device 目标设备
 * @return 新张量
 */
template<typename T>
std::unique_ptr<Model::Tensor> CreateTensorFromVector(
    const std::vector<T>& data,
    Model::DataType dtype,
    Model::Device device) {
  return CreateTensor(data.data(), {static_cast<int64_t>(data.size())}, dtype, device);
}

/**
 * @brief 创建空张量
 * 
 * @param shape 张量形状
 * @param dtype 数据类型
 * @param device 目标设备
 * @return 填充零的张量
 */
std::unique_ptr<Model::Tensor> Zeros(
    const std::vector<int64_t>& shape,
    Model::DataType dtype,
    Model::Device device);

// ============================================================================
// 张量视图操作
// ============================================================================

/**
 * @brief 扩展张量维度
 * 
 * 在指定位置插入大小为1的维度
 * 零拷贝视图
 * 
 * @param tensor 输入张量
 * @param axis 插入位置
 * @return 视图张量 (共享底层数据)
 */
std::unique_ptr<Model::Tensor> Unsqueeze(
    const Model::Tensor* tensor,
    int axis);

/**
 * @brief 重塑张量形状
 * 
 * @param tensor 输入张量
 * @param new_shape 新形状 (元素总数必须相同)
 * @return 视图张量 (共享底层数据)
 */
std::unique_ptr<Model::Tensor> Reshape(
    const Model::Tensor* tensor,
    const std::vector<int64_t>& new_shape);

// ============================================================================
// 张量切片操作
// ============================================================================

/**
 * @brief 取张量的最后n个元素 (沿指定轴)
 * 
 * @param tensor 输入张量
 * @param n 元素数量
 * @param axis 切片轴 (默认0)
 * @return 切片后的张量
 */
std::unique_ptr<Model::Tensor> TakeLast(
    const Model::Tensor* tensor,
    int64_t n,
    int axis = 0);

/**
 * @brief 取张量的前n个元素 (沿指定轴)
 * 
 * @param tensor 输入张量
 * @param n 元素数量
 * @param axis 切片轴 (默认0)
 * @return 切片后的张量
 */
std::unique_ptr<Model::Tensor> TakeFirst(
    const Model::Tensor* tensor,
    int64_t n,
    int axis = 0);

}  // namespace Utils
}  // namespace GPTSoVITS

#endif  // GPT_SOVITS_CPP_UTILS_TENSOR_OPS_H
