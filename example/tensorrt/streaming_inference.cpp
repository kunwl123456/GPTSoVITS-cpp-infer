//
// ONNX Streaming Inference Example
// 流式推理：实时音频生成，分块输出
//

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "GPTSoVITS/AudioTools.h"
#include "GPTSoVITS/EdgePipeline.h"
#include "GPTSoVITS/GPTSoVITS.h"
#include "GPTSoVITS/InferencePipeline.h"
#include "GPTSoVITS/StreamingPipeline.h"
#include "GPTSoVITS/model/backend/onnx_backend.h"
#include "GPTSoVITS/model/backend/tensorrt_backend.h"
#include "GPTSoVITS/model/sample_config.h"

#ifdef _WIN32
#define HOST_WINDOWS
#endif

#ifdef HOST_WINDOWS
auto device = GPTSoVITS::Model::Device(GPTSoVITS::Model::DeviceType::kCUDA, 0);
#else
auto device = GPTSoVITS::Model::Device(GPTSoVITS::Model::DeviceType::kCUDA, 0);
#endif

#ifdef HOST_WINDOWS
#define MODEL_PATH \
  R"(/home/autogame/3rd/GPT-SoVITS_minimal_inference/onnx_export/v2proplus_base_fp16)"
#else
#define MODEL_PATH \
  R"(/home/autogame/3rd/GPT-SoVITS_minimal_inference/onnx_export/v2proplus_base_fp16)"
#endif

static std::string readFile(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) throw std::runtime_error("无法打开文件: " + path);
  std::stringstream buf;
  buf << file.rdbuf();
  return buf.str();
}

static void PrintUsage(const char* prog) {
  std::cout << "用法: " << prog << " [选项]\n"
            << "  --speaker-package <path>  说话人数据包 (默认: firefly.gsppkg)\n"
            << "  --text <text>             目标文本\n"
            << "  --lang <lang>             语言 (默认: zh)\n"
            << "  --speaker-name <name>     说话人名称 (默认: firefly)\n"
            << "  --output <path>           输出路径 (默认: streaming_output.wav)\n"
            << "  --temperature <float>     采样温度 (默认: 1.0)\n"
            << "  --top-k <int>             Top-K (默认: 40)\n"
            << "  --top-p <float>           Top-P (默认: 0.6)\n"
            << "  --noise-scale <float>     噪声缩放 (默认: 0.35)\n"
            << "  --speed <float>           语速 (默认: 1.0)\n"
            << "  --chunk-length <int>      分块长度 token 数 (默认: 24)\n"
            << "  --pause-length <float>    段落停顿秒数 (默认: 0.3)\n";
}

struct Stats {
  int    total_chunks            = 0;
  int    total_samples           = 0;
  double first_packet_latency_ms = 0.0;
  double total_audio_duration_s  = 0.0;
  bool   first_packet_received   = false;
  std::chrono::steady_clock::time_point infer_start;
};

int main(int argc, char* argv[]) {
#ifdef _WIN32
  std::system("chcp 65001");
#endif

  std::string speaker_package = "firefly.gsppkg";
  std::string text = (
    "皆さん、我在インターネット上看到someone把几国language混在一起speak。\n"
    "虽然是混乱している句子ですけど、中文日本語プラスEnglish、挑戦スタート！\n"
    "我study日本語的时候，もし有汉字，我会很happy。\n"
    "2021年6月25日,今天32°C。以上です，byebye！"
  );
  std::string text_lang    = "zh";
  std::string speaker_name = "firefly";
  std::string output_path  = "streaming_output_trt.wav";
  float temperature  = 1.0f;
  int   top_k        = 40;
  float top_p        = 0.6f;
  float noise_scale  = 0.35f;
  float speed        = 1.0f;
  int   chunk_length = 24;
  float pause_length = 0.3f;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if      (arg == "--help"            || arg == "-h") { PrintUsage(argv[0]); return 0; }
    else if (arg == "--speaker-package" && i+1 < argc)  speaker_package = argv[++i];
    else if (arg == "--text"            && i+1 < argc)  text            = argv[++i];
    else if (arg == "--lang"            && i+1 < argc)  text_lang       = argv[++i];
    else if (arg == "--speaker-name"    && i+1 < argc)  speaker_name    = argv[++i];
    else if (arg == "--output"          && i+1 < argc)  output_path     = argv[++i];
    else if (arg == "--temperature"     && i+1 < argc)  temperature     = std::stof(argv[++i]);
    else if (arg == "--top-k"           && i+1 < argc)  top_k           = std::stoi(argv[++i]);
    else if (arg == "--top-p"           && i+1 < argc)  top_p           = std::stof(argv[++i]);
    else if (arg == "--noise-scale"     && i+1 < argc)  noise_scale     = std::stof(argv[++i]);
    else if (arg == "--speed"           && i+1 < argc)  speed           = std::stof(argv[++i]);
    else if (arg == "--chunk-length"    && i+1 < argc)  chunk_length    = std::stoi(argv[++i]);
    else if (arg == "--pause-length"    && i+1 < argc)  pause_length    = std::stof(argv[++i]);
  }

  try {
    std::filesystem::path modelPath = FS_PATH(MODEL_PATH);

    std::cout << "========================================\n";
    std::cout << "  TRT Streaming Inference\n";
    std::cout << "========================================\n";

    // ==================== 预热阶段（不计入基准）====================
    std::cout << "\n[Warmup] 加载模型...\n";

    auto g2p = std::make_shared<GPTSoVITS::G2P::G2PPipline>();
    {
      auto bert = std::make_unique<GPTSoVITS::Model::CNBertModel>();
      bert->Init<GPTSoVITS::Model::TensorRTBackend>(
          (modelPath / "bert.onnx").string(),
          (GPTSoVITS::GetGlobalResourcesPath() / "bert_tokenizer.json").string(),
          device);
      g2p->RegisterLangProcess("zh", std::make_unique<GPTSoVITS::G2P::G2PZH>(), std::move(bert), true);
      g2p->RegisterLangProcess("en", std::make_unique<GPTSoVITS::G2P::G2PEN>(), nullptr, true);
      g2p->RegisterLangProcess("ja", std::make_unique<GPTSoVITS::G2P::G2PJA>(), nullptr, true);
      g2p->SetDefaultLang("zh");
    }

    auto enc_model = std::make_shared<GPTSoVITS::Model::GPTEncoderModel>();
    enc_model->Init<GPTSoVITS::Model::TensorRTBackend>(
        (modelPath / "gpt_encoder.engine").string(), device);

    auto step_model = std::make_shared<GPTSoVITS::Model::GPTStepModel>();
    step_model->Init<GPTSoVITS::Model::TensorRTBackend>(
        (modelPath / "gpt_step.engine").string(), device);

    auto sovits_model = std::make_shared<GPTSoVITS::Model::SoVITSModel>();
    sovits_model->Init<GPTSoVITS::Model::TensorRTBackend>(
        (modelPath / "sovits.engine").string(), device);

    auto edge_pipeline = std::make_shared<GPTSoVITS::EdgePipeline>(
        readFile((modelPath / "config.json").string()),
        MODEL_PATH, g2p, enc_model, step_model, sovits_model);

    if (!edge_pipeline->ImportSpeaker(speaker_package, speaker_name)) {
      std::cerr << "错误：无法导入说话人数据包: " << speaker_package << "\n";
      return 1;
    }
    std::cout << "[Warmup] 完成，开始基准推理\n\n";

    GPTSoVITS::StreamingConfig streaming_config;
    streaming_config.chunk_length = chunk_length;
    streaming_config.pause_length = pause_length;
    streaming_config.fade_length  = 1280;
    streaming_config.h_len        = 512;
    streaming_config.l_len        = 16;
    streaming_config.enable_fade  = true;

    GPTSoVITS::Model::SampleConfig sample_config;
    sample_config.temperature = temperature;
    sample_config.top_k       = top_k;
    sample_config.top_p       = top_p;
    if (!sample_config.Validate()) {
      std::cerr << "错误：无效的采样配置\n";
      return 1;
    }

    auto streaming_pipeline = std::make_shared<GPTSoVITS::StreamingPipeline>(
        edge_pipeline, streaming_config);

    std::cout << "配置:\n";
    std::cout << "  说话人:   " << speaker_name << "\n";
    std::cout << "  文本:     " << text.substr(0, 40) << "...\n";
    std::cout << "  语言:     " << text_lang << "\n";
    std::cout << "  设备:     "
              << (device.type == GPTSoVITS::Model::DeviceType::kCUDA ? "CUDA" : "CPU") << "\n";
    std::cout << "  采样参数: temperature=" << temperature
              << ", top_k=" << top_k << ", top_p=" << top_p << "\n";
    std::cout << "  流参数:   chunk_length=" << chunk_length
              << ", pause_length=" << pause_length << "s\n\n";

    Stats stats;
    GPTSoVITS::Model::InferStats infer_stats;
    std::vector<float> accumulated_audio;
    int sampling_rate = edge_pipeline->GetSamplingRate();

    std::cout << "开始流推理...\n";
    std::cout << "----------------------------------------\n";

    stats.infer_start = std::chrono::steady_clock::now();

    bool success = streaming_pipeline->InferSpeakerStreaming(
        speaker_name, text, text_lang,
        [&](const GPTSoVITS::AudioChunk& chunk) {
          if (!stats.first_packet_received) {
            auto now = std::chrono::steady_clock::now();
            stats.first_packet_latency_ms =
                std::chrono::duration<double, std::milli>(now - stats.infer_start).count();
            stats.first_packet_received = true;
            std::cout << ">>> 首包延迟: " << std::fixed << std::setprecision(2)
                      << stats.first_packet_latency_ms << " ms\n";
          }
          stats.total_chunks++;
          stats.total_samples += static_cast<int>(chunk.audio_data.size());
          stats.total_audio_duration_s += chunk.duration;
          accumulated_audio.insert(accumulated_audio.end(),
                                   chunk.audio_data.begin(), chunk.audio_data.end());
          std::cout << "分块 " << chunk.segment_index << "-" << chunk.chunk_index
                    << " | 时长: " << std::fixed << std::setprecision(3) << chunk.duration
                    << "s | 样本数: " << chunk.audio_data.size()
                    << " | 累计: " << std::fixed << std::setprecision(2)
                    << stats.total_audio_duration_s << "s\n";
        },
        sample_config, noise_scale, speed, &infer_stats);

    auto t_end = std::chrono::steady_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(
        t_end - stats.infer_start).count();

    std::cout << "----------------------------------------\n";

    if (!success) {
      std::cerr << "流推理失败\n";
      return 1;
    }

    // 保存音频
    auto audio_tools = GPTSoVITS::AudioTools::FromByte(accumulated_audio, sampling_rate);
    if (audio_tools) {
      audio_tools->SaveToFile(output_path);
    }

    double rtf = total_ms / 1000.0 / stats.total_audio_duration_s;

    std::cout << "\n--- Streaming Inference Performance Summary ---\n";
    std::cout << "Total Chunks:           " << stats.total_chunks << "\n";
    std::cout << "GPT Semantic Gen:       " << std::fixed << std::setprecision(3)
              << infer_stats.gpt_time_s << "s"
              << " (" << std::fixed << std::setprecision(2) << infer_stats.TokensPerSec() << " tokens/s,"
              << " " << infer_stats.gpt_tokens << " tokens)\n";
    std::cout << "SoVITS Audio Decode:    " << std::fixed << std::setprecision(3)
              << infer_stats.sovits_time_s << "s\n";
    if (stats.first_packet_received)
      std::cout << "First Packet Latency:   " << std::fixed << std::setprecision(2)
                << stats.first_packet_latency_ms << " ms\n";
    std::cout << "Total Inference Time:   " << std::fixed << std::setprecision(3)
              << total_ms / 1000.0 << "s\n";
    std::cout << "Total Audio Duration:   " << std::fixed << std::setprecision(3)
              << stats.total_audio_duration_s << "s\n";
    std::cout << "Real Time Factor (RTF): " << std::fixed << std::setprecision(4) << rtf << "\n";
    std::cout << "Performance:            " << (rtf < 1.0 ? "Real-time (RTF < 1.0)" : "Non-real-time (RTF >= 1.0)") << "\n";
    std::cout << "-----------------------------------------------\n";
    std::cout << "\n音频已保存到: " << output_path << "\n";

  } catch (const std::exception& e) {
    std::cerr << "错误: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
