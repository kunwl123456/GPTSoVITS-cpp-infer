//
// Created by 19254 on 2026/2/8.
//
// 多说话人场景
//
// 演示多个说话人共享模型权重的场景
//

#include <chrono>
#include <iostream>
#include <string>

#include "GPTSoVITS/AudioTools.h"
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
    std::cout << "  多说话人场景" << std::endl;
    std::cout << "========================================" << std::endl;

    // 创建 Pipeline 配置（边缘模式）
    GPTSoVITS::PipelineConfig config = GPTSoVITS::PipelineConfig::Edge(
        MODEL_PATH,
        GPTSoVITS::Model::DeviceType::kCUDA,
        0);
    config.resources_path = "./res";
    config.verbose = true;

    // 创建 Pipeline（模型只加载一次）
    std::cout << "\n[1/2] 初始化 Pipeline（共享模型权重）..." << std::endl;
    GPTSoVITS::InferencePipeline pipeline(config);

    std::cout << pipeline.GetModelInfo() << std::endl;

    // 导入多个说话人数据包（共享模型权重，只加载特征数据）
    std::cout << "\n[2/2] 导入多个说话人..." << std::endl;

    // 多个说话人
    std::vector<std::pair<std::string,std::string>> speaker_packages = {
        {"firefly","firefly.gsppkg"},
        {"firefly-normal","firefly-normal.gsppkg"},
    };

    for (auto& [speaker_name,speaker_path]:speaker_packages) {
      if (pipeline.ImportSpeaker(speaker_path, speaker_name)) {
        std::cout << "  已导入说话人: " << speaker_name << std::endl;
      } else {
        std::cout << "  提示: 请先创建说话人数据包" << std::endl;
      }
    }


    // 列出所有说话人
    std::cout << "\n已加载的说话人: ";
    for (const auto& name : pipeline.ListSpeakers()) {
      std::cout << name << " ";
    }
    std::cout << std::endl;

    // 演示多说话人推理
    std::cout << "\n========================================" << std::endl;
    std::cout << "  多说话人推理示例" << std::endl;
    std::cout << "========================================" << std::endl;

    std::string text = "你好，我是测试文本。";

    for (const auto& name : pipeline.ListSpeakers()) {
      std::cout << "\n推理说话人 [" << name << "]: " << text << std::endl;

      auto start = std::chrono::steady_clock::now();
      auto audio = pipeline.Infer(name, text);
      auto end = std::chrono::steady_clock::now();

      if (audio) {
        double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
        std::string output_file = name + "_output.wav";
        audio->SaveToFile(output_file);

        std::cout << "  耗时: " << elapsed_ms << " ms" << std::endl;
        std::cout << "  保存到: " << output_file << std::endl;
      } else {
        std::cout << "  推理失败" << std::endl;
      }
    }

    std::cout << "  模型权重只加载一次，多个说话人共享" << std::endl;
    std::cout << "  说话人特征数据按需加载到设备" << std::endl;
    std::cout << "  调用 ReleaseDeviceCache() 释放设备缓存" << std::endl;

    // 设备缓存管理
    // auto& speaker = pipeline.GetSpeakerManager().GetSpeaker("speaker_name");
    // if (speaker) {
    //   speaker->ReleaseAllDeviceCaches();  // 释放设备缓存，保留 CPU 数据
    // }

  } catch (const std::exception& e) {
    std::cerr << "\n✗ 错误: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
