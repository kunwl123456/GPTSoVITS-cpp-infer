//
// Created by 19254 on 2026/2/8.
//
// 创建说话人
//
// 演示使用 InferencePipeline 创建说话人并导出数据包
//

#include <iostream>
#include <string>

#include "GPTSoVITS/GPTSoVITS.h"
#include "GPTSoVITS/InferencePipeline.h"

#ifdef _HOST_WINDOWS_
#define MODEL_PATH R"(F:\Engcode\AIAssistant\GPT-SoVITS-Devel\GPT-SoVITS_minimal_inference\onnx_export\firefly_v2_proplus_fp16)"
#else
#define MODEL_PATH R"(/Users/huiyi/code/python/GPT-SoVITS_minimal_inference/onnx_export/firefly_v2_proplus_fp16)"
#endif

int main(int argc, char* argv[]) {
#ifdef _WIN32
  std::system("chcp 65001");
#endif

  try {
    std::cout << "========================================" << std::endl;
    std::cout << "  创建说话人" << std::endl;
    std::cout << "========================================" << std::endl;

    // 解析命令行参数
    std::string speaker_name = "firefly";
    std::string ref_audio_path =FS_PATH(R"(./看，这尊雕像就是匹诺康尼大名鼎鼎的卡通人物钟表小子.wav)").string() ;
    std::string ref_text = "看，这尊雕像就是匹诺康尼大名鼎鼎的卡通人物钟表小子";
    std::string ref_lang = "zh";
    std::string output_path = "firefly.gsppkg";

    if (argc >= 2) speaker_name = argv[1];
    if (argc >= 3) ref_audio_path = argv[2];
    if (argc >= 4) ref_text = argv[3];
    if (argc >= 5) ref_lang = argv[4];
    if (argc >= 6) output_path = argv[5];

    std::cout << "\n配置信息:" << std::endl;
    std::cout << "  说话人名称: " << speaker_name << std::endl;
    std::cout << "  参考音频: " << ref_audio_path << std::endl;
    std::cout << "  参考文本: " << ref_text << std::endl;
    std::cout << "  参考语言: " << ref_lang << std::endl;
    std::cout << "  输出路径: " << output_path << std::endl;

    // 创建 Pipeline 配置
    GPTSoVITS::PipelineConfig config = GPTSoVITS::PipelineConfig::FullCUDA(MODEL_PATH);
    config.resources_path = "./res";  // 资源文件路径
    config.verbose = true;

    // 创建 Pipeline
    std::cout << "\n[1/3] 初始化 Pipeline..." << std::endl;
    GPTSoVITS::InferencePipeline pipeline(config);

    std::cout << pipeline.GetModelInfo() << std::endl;

    // 创建说话人
    std::cout << "\n[2/3] 创建说话人..." << std::endl;
    auto* speaker = pipeline.CreateSpeaker(
        speaker_name, ref_lang, ref_audio_path, ref_text);

    if (!speaker) {
      std::cerr << "错误：创建说话人失败" << std::endl;
      return 1;
    }

    std::cout << "  说话人创建成功: " << speaker->Name() << std::endl;
    std::cout << "  语言: " << speaker->Lang() << std::endl;

    // 导出说话人数据包
    std::cout << "\n[3/3] 导出说话人数据包..." << std::endl;
    if (!pipeline.ExportSpeaker(speaker_name, output_path)) {
      std::cerr << "错误：无法导出说话人数据包" << std::endl;
      return 1;
    }

    std::cout << "\n✓ 说话人数据包已导出: " << output_path << std::endl;

  } catch (const std::exception& e) {
    std::cerr << "\n✗ 错误: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
