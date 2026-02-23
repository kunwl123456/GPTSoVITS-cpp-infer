//
// Created by 19254 on 2026/2/8.
//
// 推理端
//
// 演示使用 InferencePipeline 加载说话人数据包并推理
//

#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>

#include "GPTSoVITS/AudioTools.h"
#include "GPTSoVITS/InferencePipeline.h"

#ifdef _HOST_WINDOWS_
#define MODEL_PATH R"(F:\Engcode\AIAssistant\GPT-SoVITS-Devel\GPT-SoVITS_minimal_inference\onnx_export\firefly_v2_proplus_fp16)"
#else
#define MODEL_PATH R"(/Users/huiyi/code/python/GPT-SoVITS_minimal_inference/onnx_export/firefly_v2_proplus_fp16)"
#endif

#ifdef _HOST_WINDOWS_
auto device = GPTSoVITS::Model::Device(GPTSoVITS::Model::DeviceType::kCUDA, 0);
#else
auto device = GPTSoVITS::Model::Device(GPTSoVITS::Model::DeviceType::kCPU, 0);
#endif

int main(int argc, char* argv[]) {
#ifdef _WIN32
  std::system("chcp 65001");
#endif

  try {
    std::cout << "========================================" << std::endl;
    std::cout << "  推理端" << std::endl;
    std::cout << "========================================" << std::endl;

    // 解析命令行参数
    std::string speaker_package = "firefly.gsppkg";
    std::string text = {
      "皆さん、我在インターネット上看到someone把几国language混在一起speak。我看到之后be like：それは我じゃないか！私もtry一tryです。\n"
      "虽然是混乱している句子ですけど、中文日本語プラスEnglish、挑戦スタート！\n"
      "我study日本語的时候，もし有汉字，我会很happy。\n"
      "Bueause中国人として、when I see汉字，すぐに那个汉字がわかります。\n"
      "But 我hate外来語、什么マクドナルド、スターバックス、グーグル、ディズニーランド、根本记不住カタカナhow to写、太難しい。\n"
      "2021年6月25日,今天32°C。以上です，byebye！"
    };
    std::string text_lang = "zh";
    std::string speaker_name = "firefly";
    std::string output_path = "edge_output.wav";

    if (argc >= 2) speaker_package = argv[1];
    if (argc >= 3) text = argv[2];
    if (argc >= 4) text_lang = argv[3];
    if (argc >= 5) speaker_name = argv[4];
    if (argc >= 6) output_path = argv[5];

    std::cout << "\n配置信息:" << std::endl;
    std::cout << "  模型路径: " << MODEL_PATH << std::endl;
    std::cout << "  说话人数据包: " << speaker_package << std::endl;
    std::cout << "  文本: " << text << std::endl;
    std::cout << "  语言: " << text_lang << std::endl;

    // 创建 Pipeline 配置（边缘模式）
    GPTSoVITS::PipelineConfig config = GPTSoVITS::PipelineConfig::Edge(
        MODEL_PATH,
        device.type,
        device.device_id);
    config.resources_path = "./res";
    config.verbose = true;

    // 创建 Pipeline
    std::cout << "\n[1/3] 初始化 Pipeline..." << std::endl;
    GPTSoVITS::InferencePipeline pipeline(config);

    std::cout << pipeline.GetModelInfo() << std::endl;

    // 导入说话人数据包
    std::cout << "\n[2/3] 导入说话人数据包..." << std::endl;
    if (!pipeline.ImportSpeaker(speaker_package, speaker_name)) {
      std::cerr << "错误：无法导入说话人数据包" << std::endl;
      return 1;
    }

    std::cout << "  已导入说话人: " << speaker_name << std::endl;
    std::cout << "  可用说话人: ";
    for (const auto& name : pipeline.ListSpeakers()) {
      std::cout << name << " ";
    }
    std::cout << std::endl;

    // 执行推理
    std::cout << "\n[3/3] 执行推理..." << std::endl;
    auto start_time = std::chrono::steady_clock::now();

    auto audio = pipeline.Infer(speaker_name, text, text_lang);

    auto end_time = std::chrono::steady_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    if (audio) {
      // 保存音频
      audio->SaveToFile(output_path);

      // 计算统计信息
      int sampling_rate = 32000;
      int num_samples = audio->ReadSamples().size();
      double audio_duration_s = static_cast<double>(num_samples) / sampling_rate;

      std::cout << "\n推理统计:" << std::endl;
      std::cout << "  总耗时: " << std::fixed << std::setprecision(2)
                << elapsed_ms << " ms" << std::endl;
      std::cout << "  音频时长: " << std::fixed << std::setprecision(3)
                << audio_duration_s << " s" << std::endl;
      std::cout << "  样本数: " << num_samples << std::endl;

      if (audio_duration_s > 0) {
        double rtf = elapsed_ms / 1000.0 / audio_duration_s;
        std::cout << "  实时率 (RTF): " << std::fixed << std::setprecision(4)
                  << rtf << std::endl;
      }

      std::cout << "\n✓ 音频已保存到: " << output_path << std::endl;

    } else {
      std::cerr << "\n✗ 推理失败！" << std::endl;
      return 1;
    }

  } catch (const std::exception& e) {
    std::cerr << "\n✗ 错误: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
