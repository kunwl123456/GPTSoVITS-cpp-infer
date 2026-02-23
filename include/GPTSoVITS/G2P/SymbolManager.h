//
// Created by Huiyicc on 2026/2/23.
//

#ifndef GPT_SOVITS_CPP_SYMBOL_MANAGER_H
#define GPT_SOVITS_CPP_SYMBOL_MANAGER_H

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace GPTSoVITS::G2P {

/**
 * @brief 符号映射管理器
 *
 * 管理音素符号到ID的映射，支持从JSON配置动态加载，
 * 不同版本可以共享相同或不同的符号表。
 */
class SymbolManager {
public:
  using SymbolMap = std::unordered_map<std::string, int>;

  /**
   * @brief 获取全局单例实例
   */
  static SymbolManager& Instance();

  /**
   * @brief 从JSON字符串加载符号映射
   * @param json_content JSON字符串，格式为 {"symbol": id, ...}
   * @param version 版本标识，用于区分不同模型的符号表
   * @return 是否加载成功
   */
  bool LoadFromJson(const std::string& json_content, const std::string& version = "default");

  /**
   * @brief 从JSON文件加载符号映射
   * @param json_path JSON文件路径
   * @param version 版本标识
   * @return 是否加载成功
   */
  bool LoadFromFile(const std::string& json_path, const std::string& version = "default");

  /**
   * @brief 从config.json的symbol_to_id字段加载
   * @param config_path config.json文件路径
   * @param version 版本标识
   * @return 是否加载成功
   */
  bool LoadFromConfig(const std::string& config_path, const std::string& version = "default");

  /**
   * @brief 获取指定版本的符号映射
   * @param version 版本标识
   * @return 符号映射指针，不存在返回nullptr
   */
  const SymbolMap* GetSymbols(const std::string& version = "default") const;

  /**
   * @brief 查找符号对应的ID
   * @param symbol 符号字符串
   * @param version 版本标识
   * @return ID值，未找到返回-1
   */
  int FindSymbol(const std::string& symbol, const std::string& version = "default") const;

  /**
   * @brief 检查符号是否存在
   */
  bool HasSymbol(const std::string& symbol, const std::string& version = "default") const;

  /**
   * @brief 获取符号数量
   */
  size_t GetSymbolCount(const std::string& version = "default") const;

  /**
   * @brief 获取当前激活的版本
   */
  const std::string& GetActiveVersion() const { return m_active_version; }

  /**
   * @brief 设置当前激活的版本
   */
  void SetActiveVersion(const std::string& version);

  /**
   * @brief 使用激活版本查找符号
   */
  int FindSymbolActive(const std::string& symbol) const;

  /**
   * @brief 使用激活版本检查符号
   */
  bool HasSymbolActive(const std::string& symbol) const;

  /**
   * @brief 获取激活版本的符号映射
   */
  const SymbolMap* GetSymbolsActive() const;

  /**
   * @brief 初始化内置默认符号表（兼容旧版本）
   */
  void InitDefaultSymbols();

  /**
   * @brief 检查指定版本是否已加载
   */
  bool HasVersion(const std::string& version) const;

  /**
   * @brief 清除指定版本的符号表
   */
  void ClearVersion(const std::string& version);

  /**
   * @brief 清除所有符号表
   */
  void ClearAll();

private:
  SymbolManager();
  ~SymbolManager() = default;
  SymbolManager(const SymbolManager&) = delete;
  SymbolManager& operator=(const SymbolManager&) = delete;

  // 版本 -> 符号映射
  std::unordered_map<std::string, SymbolMap> m_version_symbols;
  // 当前激活的版本
  std::string m_active_version = "default";
  // 线程安全锁
  mutable std::mutex m_mutex;
};

}  // namespace GPTSoVITS::G2P

#endif  // GPT_SOVITS_CPP_SYMBOL_MANAGER_H
