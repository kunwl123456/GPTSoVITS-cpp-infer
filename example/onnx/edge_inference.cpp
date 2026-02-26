//
// ONNX Edge Inference Example
// 边缘推理：导入说话人数据包，执行推理
//

#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>

#include "GPTSoVITS/AudioTools.h"
#include "GPTSoVITS/GPTSoVITS.h"
#include "GPTSoVITS/InferencePipeline.h"
#include "GPTSoVITS/model/sample_config.h"

#ifdef _WIN32
#define HOST_WINDOWS
#endif

#ifdef HOST_WINDOWS
auto device = GPTSoVITS::Model::Device(GPTSoVITS::Model::DeviceType::kCUDA, 0);
#else
auto device = GPTSoVITS::Model::Device(GPTSoVITS::Model::DeviceType::kCPU, 0);
#endif

#ifdef HOST_WINDOWS
#define MODEL_PATH \
  R"(F:\Engcode\AIAssistant\GPT-SoVITS-Devel\GPT-SoVITS_minimal_inference\onnx_export\firefly_v2_proplus_fp16)"
#else
#define MODEL_PATH \
  R"(/Users/huiyi/code/python/GPT-SoVITS_minimal_inference/onnx_export/firefly_v2_proplus_fp16)"
#endif

static void PrintUsage(const char* prog) {
  std::cout << "用法: " << prog << " [speaker_package] [text] [lang] [speaker_name] [output]\n";
}

int main(int argc, char* argv[]) {
#ifdef _WIN32
  std::system("chcp 65001");
#endif

  if (argc >= 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
    PrintUsage(argv[0]);
    return 0;
  }

  std::string speaker_package = argc >= 2 ? argv[1] : "firefly.gsppkg";
  std::string text            = argc >= 3 ? argv[2] : (
    "皆さん、我在インターネット上看到someone把几国language混在一起speak。\n"
    "虽然是混乱している句子ですけど、中文日本語プラスEnglish、挑戦スタート！\n"
    "我study日本語的时候，もし有汉字，我会很happy。\n"
    "2021年6月25日,今天32°C。以上です，byebye！"
  );
  std::string text_lang    = argc >= 4 ? argv[3] : "zh";
  std::string speaker_name = argc >= 5 ? argv[4] : "firefly";
  std::string output_path  = argc >= 6 ? argv[5] : "edge_output.wav";

  try {
    std::cout << "========================================\n";
    std::cout << "  ONNX Edge Inference\n";
    std::cout << "========================================\n";

    // ==================== 预热阶段（不计入基准）====================
    std::cout << "\n[Warmup] 初始化 Pipeline...\n";

    GPTSoVITS::PipelineConfig config = GPTSoVITS::PipelineConfig::Edge(
        MODEL_PATH, device.type, device.device_id);
    config.resources_path = "./res";
    config.verbose        = false;

    GPTSoVITS::InferencePipeline pipeline(config);

    std::cout << "[Warmup] 导入说话人数据包...\n";
    if (!pipeline.ImportSpeaker(speaker_package, speaker_name)) {
      std::cerr << "错误：无法导入说话人数据包: " << speaker_package << "\n";
      return 1;
    }

    // 首次推理预热
    std::cout << "[Warmup] 执行预热推理...\n";
    pipeline.Infer(speaker_name, "预热", text_lang);
    std::cout << "[Warmup] 完成，开始基准推理\n\n";

    // ==================== 基准推理（计时从此开始）====================
    std::cout << "配置:\n";
    std::cout << "  模型路径:     " << MODEL_PATH << "\n";
    std::cout << "  说话人数据包: " << speaker_package << "\n";
    std::cout << "  说话人:       " << speaker_name << "\n";
    std::cout << "  文本:         " << text.substr(0, 40) << "...\n";
    std::cout << "  语言:         " << text_lang << "\n";
    std::cout << "  设备:         "
              << (device.type == GPTSoVITS::Model::DeviceType::kCUDA ? "CUDA" : "CPU") << "\n\n";

    auto t_start = std::chrono::steady_clock::now();
    GPTSoVITS::Model::InferStats stats;
    double first_segment_latency_ms = 0.0;
    auto audio   = pipeline.Infer(speaker_name, text, text_lang, {}, 0.5f, 1.0f, &stats,
                                  [&]() {
                                    first_segment_latency_ms =
                                        std::chrono::duration<double, std::milli>(
                                            std::chrono::steady_clock::now() - t_start).count();
                                  });
    auto t_end   = std::chrono::steady_clock::now();

    if (!audio) {
      std::cerr << "推理失败\n";
      return 1;
    }

    audio->SaveToFile(output_path);

    double elapsed_ms     = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    int    num_samples    = static_cast<int>(audio->ReadSamples().size());
    int    sampling_rate  = 32000;
    double audio_duration = static_cast<double>(num_samples) / sampling_rate;
    double rtf            = (elapsed_ms / 1000.0) / audio_duration;

    std::cout << "--- Inference Performance Summary ---\n";
    std::cout << "GPT Semantic Gen:       " << std::fixed << std::setprecision(3)
              << stats.gpt_time_s << "s"
              << " (" << std::fixed << std::setprecision(2) << stats.TokensPerSec() << " tokens/s,"
              << " " << stats.gpt_tokens << " tokens)\n";
    std::cout << "SoVITS Audio Decode:    " << std::fixed << std::setprecision(3)
              << stats.sovits_time_s << "s\n";
    if (first_segment_latency_ms > 0.0)
      std::cout << "First Segment Latency:  " << std::fixed << std::setprecision(2)
                << first_segment_latency_ms << " ms\n";
    std::cout << "Total Inference Time:   " << std::fixed << std::setprecision(3)
              << elapsed_ms / 1000.0 << "s\n";
    std::cout << "Total Audio Duration:   " << std::fixed << std::setprecision(3)
              << audio_duration << "s\n";
    std::cout << "Real Time Factor (RTF): " << std::fixed << std::setprecision(4) << rtf << "\n";
    if (rtf < 1.0)
      std::cout << "Performance:            Real-time (RTF < 1.0)\n";
    else
      std::cout << "Performance:            Non-real-time (RTF >= 1.0)\n";
    std::cout << "-------------------------------------\n";
    std::cout << "\n音频已保存到: " << output_path << "\n";

  } catch (const std::exception& e) {
    std::cerr << "错误: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
