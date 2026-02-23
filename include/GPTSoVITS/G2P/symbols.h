//
// Created by Huiyicc on 24-11-10.
//

#ifndef GPT_SOVITS_CPP_SYMBOLS_H
#define GPT_SOVITS_CPP_SYMBOLS_H

#include <set>
#include <vector>
#include <string>
#include <unordered_map>


namespace GPTSoVITS::G2P {

extern const std::unordered_map<std::string_view, int> g_Symbols;

/**
 * @brief 初始化符号表从config.json
 * @param config_path config.json文件路径
 * @param version 版本标识（可选，默认使用config中的version字段）
 * @return 是否成功
 *
 * 此函数会调用SymbolManager加载符号表，并设置当前激活版本
 */
bool InitSymbolsFromConfig(const std::string& config_path,
                            const std::string& version = "");

/**
 * @brief 设置当前使用的符号表版本
 * @param version 版本标识
 */
void SetActiveSymbolVersion(const std::string& version);

/**
 * @brief 获取当前激活的符号表版本
 */
std::string GetActiveSymbolVersion();

/**
 * @brief 查找符号ID（使用当前激活版本）
 * @param symbol 符号字符串
 * @return ID值，未找到返回-1
 */
int FindSymbolId(const std::string& symbol);

/**
 * @brief 检查符号是否存在（使用当前激活版本）
 */
bool HasSymbol(const std::string& symbol);

}  // namespace GPTSoVITS::G2P

#endif //GPT_SOVITS_CPP_SYMBOLS_H