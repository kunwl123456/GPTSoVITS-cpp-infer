//
// Created by 19254 on 2026/2/22.
//
// 精度上下文管理
//

#ifndef GPT_SOVITS_CPP_UTILS_PRECISION_H
#define GPT_SOVITS_CPP_UTILS_PRECISION_H

#include <cmath>
#include <string>
#include <unordered_map>

#include "GPTSoVITS/model/tensor.h"

namespace GPTSoVITS {

// 前向声明
namespace Model {
class BaseModel;
}

namespace Utils {

/**
 * @brief 精度模式枚举
 */
enum class PrecisionMode {
  kFP32,      ///< 全FP32精度
  kFP16,      ///< 全FP16精度
  kMixed,     ///< 混合精度 (计算FP16，关键路径FP32)
  kAuto,      ///< 自动检测
  // kINT8Dyn,   ///< INT8动态量化 (预留)
  // kINT8Static,///< INT8静态量化 (预留)
};

/**
 * @brief 精度上下文 - 管理整个推理流程中的精度策略
 * 
 * 参考 Python 实现的精度检测逻辑：
 * - 从 GPT Encoder 输入检测模型精度
 * - 从 GPT Step 输入检测 Cache 精度
 */
class PrecisionContext {
public:
  PrecisionContext();
  ~PrecisionContext() = default;
  
  // ========================================================================
  // 精度配置
  // ========================================================================
  
  /**
   * @brief 设置精度模式
   */
  void SetMode(PrecisionMode mode);
  
  /**
   * @brief 获取当前精度模式
   */
  PrecisionMode GetMode() const { return mode_; }
  
  /**
   * @brief 获取计算精度 (用于矩阵运算)
   */
  Model::DataType GetComputeDType() const { return compute_dtype_; }
  
  /**
   * @brief 获取Cache精度 (用于KV Cache)
   */
  Model::DataType GetCacheDType() const { return cache_dtype_; }
  
  /**
   * @brief 获取输出精度 (用于最终输出)
   */
  Model::DataType GetOutputDType() const { return output_dtype_; }
  
  /**
   * @brief 获取模型版本 (v2, v2ProPlus等)
   */
  const std::string& GetVersion() const { return version_; }
  
  /**
   * @brief 设置模型版本
   */
  void SetVersion(const std::string& version) { version_ = version; }
  
  // ========================================================================
  // 精度自动检测
  // ========================================================================
  
  /**
   * @brief 从模型输入检测精度
   * 
   * @param model 模型基类接口
   */
  void DetectFromModel(Model::BaseModel* model);
  
  /**
   * @brief 从输入名称和类型映射检测精度
   * 
   * @param input_types 输入名称 -> ONNX类型字符串 的映射
   */
  void DetectFromInputTypes(const std::unordered_map<std::string, std::string>& input_types);
  
  /**
   * @brief 从Cache输入类型检测Cache精度
   * 
   * @param cache_type Cache的ONNX类型字符串
   */
  void DetectCacheDType(const std::string& cache_type);
  
  // ========================================================================
  // 精度转换工具
  // ========================================================================
  
  /**
   * @brief 检查类型是否兼容
   * 
   * @param source 源类型
   * @param target 目标类型
   * @return true 如果可以安全转换
   */
  bool IsCompatible(Model::DataType source, Model::DataType target) const;
  
  /**
   * @brief 检查是否需要转换
   */
  bool NeedConversion(Model::DataType source, Model::DataType target) const {
    return source != target;
  }
  
  /**
   * @brief 获取描述字符串
   */
  std::string ToString() const;
  
  // ========================================================================
  // 静态工具函数
  // ========================================================================
  
  /**
   * @brief 从ONNX类型字符串解析数据类型
   * 
   * @param onnx_type ONNX类型字符串 (如 "tensor(float16)")
   * @return 对应的 DataType
   */
  static Model::DataType ParseOnnxType(const std::string& onnx_type);
  
  /**
   * @brief 将 DataType 转换为字符串
   */
  static std::string DataTypeToString(Model::DataType dtype);
  
  /**
   * @brief 检查是否是Pro版本模型
   */
  bool IsProVersion() const {
    return version_.find("Pro") != std::string::npos;
  }
  
private:
  PrecisionMode mode_ = PrecisionMode::kAuto;
  Model::DataType compute_dtype_ = Model::DataType::kFloat32;
  Model::DataType cache_dtype_ = Model::DataType::kFloat32;
  Model::DataType output_dtype_ = Model::DataType::kFloat32;
  std::string version_ = "v2";
  bool auto_detected_ = false;
};

// ============================================================================
// 全局精度转换函数
// ============================================================================

/**
 * @brief 安全的数值类型转换 (标量版本)
 * 
 * 处理边界情况：
 * - 溢出时返回类型最大/最小值
 * - NaN/Inf传播
 * 
 * @tparam To 目标类型
 * @tparam From 源类型
 * @param value 输入值
 * @return 转换后的值
 */
template<typename To, typename From>
To SafeNumericCast(From value);

// 特化版本声明
template<> float SafeNumericCast<float, int64_t>(int64_t value);
template<> int64_t SafeNumericCast<int64_t, float>(float value);
template<> float SafeNumericCast<float, double>(double value);
template<> int SafeNumericCast<int, float>(float value);

/**
 * @brief 检查浮点数值是否有效
 * 
 * @param value 浮点数值
 * @return true 如果不是 NaN 或 Inf
 */
inline bool IsValidFloat(float value) {
  return std::isfinite(value);
}

/**
 * @brief 检查浮点数值是否有效
 */
inline bool IsValidFloat(double value) {
  return std::isfinite(value);
}

/**
 * @brief 安全地clamp浮点值到有效范围
 * 
 * @param value 输入值
 * @param min_val 最小值
 * @param max_val 最大值
 * @return clamp后的值
 */
inline float SafeClamp(float value, float min_val, float max_val) {
  if (!std::isfinite(value)) return 0.0f;
  return std::max(min_val, std::min(max_val, value));
}

/**
 * @brief 安全地clamp浮点值到有效范围
 */
inline double SafeClamp(double value, double min_val, double max_val) {
  if (!std::isfinite(value)) return 0.0;
  return std::max(min_val, std::min(max_val, value));
}

}  // namespace Utils
}  // namespace GPTSoVITS

#endif  // GPT_SOVITS_CPP_UTILS_PRECISION_H
