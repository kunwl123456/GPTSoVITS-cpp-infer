//
// Created by 19254 on 2026/2/22.
//
// 流推理示例 - 实时音频生成

#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "GPTSoVITS/EdgePipeline.h"
#include "GPTSoVITS/GPTSoVITS.h"
#include "GPTSoVITS/StreamingPipeline.h"
#include "GPTSoVITS/model/backend/onnx_backend.h"
#include "GPTSoVITS/model/sample_config.h"
#include "cpptrace/from_current.hpp"

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

std::string readFile(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) throw std::runtime_error("无法打开文件: " + path);

  std::stringstream buffer;
  buffer << file.rdbuf();
  return buffer.str();
}

// 音频分块统计
struct StreamingStats {
  int total_chunks = 0;
  int total_samples = 0;
  double first_packet_latency_ms = 0.0;
  double total_time_ms = 0.0;
  double total_audio_duration_s = 0.0;
  std::chrono::steady_clock::time_point start_time;
  std::chrono::steady_clock::time_point first_packet_time;
  bool first_packet_received = false;
};

// 音频分块处理器
class AudioChunkProcessor {
public:
  AudioChunkProcessor(int sampling_rate)
      : sampling_rate_(sampling_rate), accumulated_audio_() {}

  void ProcessChunk(const GPTSoVITS::AudioChunk& chunk, StreamingStats& stats) {
    if (!stats.first_packet_received) {
      stats.first_packet_time = std::chrono::steady_clock::now();
      stats.first_packet_latency_ms =
          std::chrono::duration<double, std::milli>(stats.first_packet_time -
                                                    stats.start_time)
              .count();
      stats.first_packet_received = true;

      std::cout << "\n>>> 首包延迟: " << std::fixed << std::setprecision(2)
                << stats.first_packet_latency_ms << " ms" << std::endl;
    }

    stats.total_chunks++;
    stats.total_samples += chunk.audio_data.size();
    stats.total_audio_duration_s += chunk.duration;

    // 累积音频数据
    accumulated_audio_.insert(accumulated_audio_.end(),
                              chunk.audio_data.begin(), chunk.audio_data.end());

    // 打印分块信息
    std::cout << "分块 " << chunk.segment_index << "-" << chunk.chunk_index
              << " | 时长: " << std::fixed << std::setprecision(3)
              << chunk.duration << "s | 样本数: " << chunk.audio_data.size()
              << " | 累计音频: " << std::fixed << std::setprecision(2)
              << stats.total_audio_duration_s << "s" << std::endl;

    // 如果是最后一个分块，保存到文件
    if (chunk.is_last) {
      SaveToFile("streaming_output.wav");
    }
  }

  void SaveToFile(const std::string& filename) {
    if (accumulated_audio_.empty()) {
      std::cerr << "警告：没有音频数据可保存" << std::endl;
      return;
    }

    // 使用 AudioTools 保存 WAV 文件
    auto audio_tools =
        GPTSoVITS::AudioTools::FromByte(accumulated_audio_, sampling_rate_);
    if (audio_tools) {
      audio_tools->SaveToFile(filename);
      std::cout << "\n>>> 音频已保存到: " << filename << std::endl;
    } else {
      std::cerr << "错误：无法创建音频文件" << std::endl;
    }
  }

private:
  int sampling_rate_;
  std::vector<float> accumulated_audio_;
};

void PrintUsage(const char* program_name) {
  std::cout << "用法: " << program_name << " [选项]" << std::endl;
  std::cout << "选项:" << std::endl;
  std::cout
      << "  --speaker-package <path>  说话人数据包路径 (默认: firefly.gsppkg)"
      << std::endl;
  std::cout << "  --text <text>             要生成的文本" << std::endl;
  std::cout << "  --lang <lang>             文本语言 (默认: zh)" << std::endl;
  std::cout << "  --speaker-name <name>     说话人名称 (默认: firefly)"
            << std::endl;
  std::cout
      << "  --output <path>           输出音频路径 (默认: streaming_output.wav)"
      << std::endl;
  std::cout << "  --temperature <float>     采样温度 (默认: 1.0)" << std::endl;
  std::cout << "  --top-k <int>             Top-K 采样参数 (默认: 15)"
            << std::endl;
  std::cout << "  --top-p <float>           Top-P 采样参数 (默认: 0.6)"
            << std::endl;
  std::cout << "  --noise-scale <float>     噪声缩放 (默认: 0.35，与Python一致)" << std::endl;
  std::cout << "  --speed <float>           语速 (默认: 1.0)" << std::endl;
  std::cout << "  --chunk-length <int>      分块长度，token数 (默认: 24)"
            << std::endl;
  std::cout << "  --pause-length <float>    段落间停顿，秒 (默认: 0.3)"
            << std::endl;
  std::cout << "  --help                    显示帮助信息" << std::endl;
}

int main(int argc, char* argv[]) {
#ifdef _WIN32
  std::system("chcp 65001");
#endif

  try {
    std::cout << "========================================" << std::endl;
    std::cout << "  GPT-SoVITS-CPP 流推理示例" << std::endl;
    std::cout << "  支持分布式推理" << std::endl;
    std::cout << "========================================" << std::endl;

    // 默认参数
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
    std::string output_path = "streaming_output.wav";
    float temperature = 1.0f;
    int top_k = 40;
    float top_p = 0.6f;
    float noise_scale = 0.5f;
    float speed = 1.0f;
    int chunk_length = 24;
    float pause_length = 0.3f;

    for (int i = 1; i < argc; ++i) {
      std::string arg = argv[i];
      if (arg == "--help" || arg == "-h") {
        PrintUsage(argv[0]);
        return 0;
      } else if (arg == "--speaker-package" && i + 1 < argc) {
        speaker_package = argv[++i];
      } else if (arg == "--text" && i + 1 < argc) {
        text = argv[++i];
      } else if (arg == "--lang" && i + 1 < argc) {
        text_lang = argv[++i];
      } else if (arg == "--speaker-name" && i + 1 < argc) {
        speaker_name = argv[++i];
      } else if (arg == "--output" && i + 1 < argc) {
        output_path = argv[++i];
      } else if (arg == "--temperature" && i + 1 < argc) {
        temperature = std::stof(argv[++i]);
      } else if (arg == "--top-k" && i + 1 < argc) {
        top_k = std::stoi(argv[++i]);
      } else if (arg == "--top-p" && i + 1 < argc) {
        top_p = std::stof(argv[++i]);
      } else if (arg == "--noise-scale" && i + 1 < argc) {
        noise_scale = std::stof(argv[++i]);
      } else if (arg == "--speed" && i + 1 < argc) {
        speed = std::stof(argv[++i]);
      } else if (arg == "--chunk-length" && i + 1 < argc) {
        chunk_length = std::stoi(argv[++i]);
      } else if (arg == "--pause-length" && i + 1 < argc) {
        pause_length = std::stof(argv[++i]);
      }
    }

    std::filesystem::path modelPath = FS_PATH(MODEL_PATH);

    std::cout << "\n配置信息:" << std::endl;
    std::cout << "  模型路径: " << MODEL_PATH << std::endl;
    std::cout << "  说话人数据包: " << speaker_package << std::endl;
    std::cout << "  文本: " << text << std::endl;
    std::cout << "  语言: " << text_lang << std::endl;
    std::cout << "  设备: "
              << (device.type == GPTSoVITS::Model::DeviceType::kCUDA ? "CUDA"
                                                                     : "CPU")
              << std::endl;
    std::cout << "  采样参数: temperature=" << temperature
              << ", top_k=" << top_k << ", top_p=" << top_p << std::endl;
    std::cout << "  流推理参数: chunk_length=" << chunk_length
              << ", pause_length=" << pause_length << "s" << std::endl;

    if (!std::filesystem::exists(modelPath)) {
      std::cerr << "错误：模型路径不存在: " << MODEL_PATH << std::endl;
      return 1;
    }

    if (!std::filesystem::exists(speaker_package)) {
      std::cerr << "错误：说话人数据包不存在: " << speaker_package << std::endl;
      return 1;
    }

    std::cout << "\n[1/4] 加载 G2P Pipeline..." << std::endl;
    auto g2p_pipeline = std::make_shared<GPTSoVITS::G2P::G2PPipline>();

    // 中文 BERT 模型
    auto bert_model = std::make_unique<GPTSoVITS::Model::CNBertModel>();
    auto bert_path = modelPath / "bert.onnx";
    auto tokenizer_path =
        GPTSoVITS::GetGlobalResourcesPath() / "bert_tokenizer.json";

    if (!std::filesystem::exists(bert_path)) {
      std::cerr << "错误：BERT 模型不存在: " << bert_path << std::endl;
      return 1;
    }

    bert_model->Init<GPTSoVITS::Model::ONNXBackend>(
        bert_path.string(), tokenizer_path.string(), device);

    g2p_pipeline->RegisterLangProcess("zh",
                                      std::make_unique<GPTSoVITS::G2P::G2PZH>(),
                                      std::move(bert_model), true);
    g2p_pipeline->RegisterLangProcess(
        "en", std::make_unique<GPTSoVITS::G2P::G2PEN>(), nullptr, true);
    g2p_pipeline->RegisterLangProcess(
        "ja", std::make_unique<GPTSoVITS::G2P::G2PJA>(), nullptr, true);
    g2p_pipeline->SetDefaultLang("zh");

    std::cout << "  G2P Pipeline 加载完成" << std::endl;

    std::cout << "\n[2/4] 加载边缘推理模型..." << std::endl;

    // GPT Encoder
    auto gpt_encoder_path = modelPath / "gpt_encoder.onnx";
    if (!std::filesystem::exists(gpt_encoder_path)) {
      std::cerr << "错误：GPT Encoder 模型不存在: " << gpt_encoder_path
                << std::endl;
      return 1;
    }
    auto gpt_encoder_model =
        std::make_shared<GPTSoVITS::Model::GPTEncoderModel>();
    gpt_encoder_model->Init<GPTSoVITS::Model::ONNXBackend>(
        gpt_encoder_path.string(), device);
    std::cout << "  - GPT Encoder 加载完成" << std::endl;

    // GPT Step
    auto gpt_step_path = modelPath / "gpt_step.onnx";
    if (!std::filesystem::exists(gpt_step_path)) {
      std::cerr << "错误：GPT Step 模型不存在: " << gpt_step_path << std::endl;
      return 1;
    }
    auto gpt_step_model = std::make_shared<GPTSoVITS::Model::GPTStepModel>();
    gpt_step_model->Init<GPTSoVITS::Model::ONNXBackend>(gpt_step_path.string(),
                                                        device);
    std::cout << "  - GPT Step 加载完成" << std::endl;

    // SoVITS 模型
    auto sovits_path = modelPath / "sovits.onnx";
    if (!std::filesystem::exists(sovits_path)) {
      std::cerr << "错误：SoVITS 模型不存在: " << sovits_path << std::endl;
      return 1;
    }
    auto sovits_model = std::make_shared<GPTSoVITS::Model::SoVITSModel>();
    sovits_model->Init<GPTSoVITS::Model::ONNXBackend>(sovits_path.string(),
                                                      device);
    std::cout << "  - SoVITS 加载完成" << std::endl;


    auto config_path = modelPath / "config.json";
    if (!std::filesystem::exists(config_path)) {
      std::cerr << "错误：配置文件不存在: " << config_path << std::endl;
      return 1;
    }
    auto config_content = readFile(config_path.string());

    // 创建边缘 Pipeline
    auto edge_pipeline = std::make_shared<GPTSoVITS::EdgePipeline>(
        config_content, MODEL_PATH, g2p_pipeline, gpt_encoder_model,
        gpt_step_model, sovits_model);

    std::cout << "\n模型信息:" << std::endl;
    std::cout << edge_pipeline->GetModelInfo() << std::endl;

    // ==================== 步骤 3: 导入说话人数据包 ====================
    std::cout << "[3/4] 导入说话人数据包..." << std::endl;
    if (!edge_pipeline->ImportSpeaker(speaker_package, speaker_name)) {
      std::cerr << "错误：无法导入说话人数据包: " << speaker_package
                << std::endl;
      return 1;
    }

    std::cout << "  已导入说话人: " << speaker_name << std::endl;
    std::cout << "  可用说话人: ";
    for (const auto& name : edge_pipeline->ListSpeakers()) {
      std::cout << name << " ";
    }
    std::cout << std::endl;

    // ==================== 步骤 4: 配置并执行流推理 ====================
    std::cout << "\n[4/4] 配置并执行流推理..." << std::endl;

    // 配置流推理参数
    GPTSoVITS::StreamingConfig streaming_config;
    streaming_config.chunk_length = chunk_length;  // 分块长度（token数）
    streaming_config.pause_length = pause_length;  // 段落间停顿（秒）
    streaming_config.fade_length = 1280;            // 淡入淡出长度（采样点）
    streaming_config.h_len = 512;                  // 历史token长度
    streaming_config.l_len = 16;                   // 前瞻token长度
    streaming_config.enable_fade = true;           // 启用淡入淡出

    // 创建采样配置
    GPTSoVITS::Model::SampleConfig sample_config;
    sample_config.temperature = temperature;
    sample_config.top_k = top_k;
    sample_config.top_p = top_p;

    // 验证采样配置
    if (!sample_config.Validate()) {
      std::cerr << "错误：无效的采样配置" << std::endl;
      return 1;
    }

    // 创建流推理 Pipeline
    auto streaming_pipeline = std::make_shared<GPTSoVITS::StreamingPipeline>(
        edge_pipeline, streaming_config);

    // 创建音频处理器和统计
    AudioChunkProcessor audio_processor(edge_pipeline->GetSamplingRate());
    StreamingStats stats;
    stats.start_time = std::chrono::steady_clock::now();

    // 开始流推理
    std::cout << "\n开始流推理..." << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    bool success = streaming_pipeline->InferSpeakerStreaming(
        speaker_name, text, text_lang,
        [&audio_processor, &stats](const GPTSoVITS::AudioChunk& chunk) {
          audio_processor.ProcessChunk(chunk, stats);
        },
        sample_config,  // 采样配置
        noise_scale,    // 噪声缩放
        speed           // 语速
    );

    auto end_time = std::chrono::steady_clock::now();
    stats.total_time_ms =
        std::chrono::duration<double, std::milli>(end_time - stats.start_time)
            .count();

    std::cout << "----------------------------------------" << std::endl;

    if (success) {
      // 打印统计信息
      std::cout << "\n流推理统计:" << std::endl;
      std::cout << "  总分块数: " << stats.total_chunks << std::endl;
      std::cout << "  总样本数: " << stats.total_samples << std::endl;

      if (stats.first_packet_received) {
        std::cout << "  首包延迟: " << std::fixed << std::setprecision(2)
                  << stats.first_packet_latency_ms << " ms" << std::endl;
      }

      std::cout << "  总耗时: " << std::fixed << std::setprecision(2)
                << stats.total_time_ms << " ms" << std::endl;
      std::cout << "  音频时长: " << std::fixed << std::setprecision(3)
                << stats.total_audio_duration_s << " s" << std::endl;

      if (stats.total_audio_duration_s > 0) {
        double rtf =
            stats.total_time_ms / 1000.0 / stats.total_audio_duration_s;
        std::cout << "  实时率 (RTF): " << std::fixed << std::setprecision(4)
                  << rtf << std::endl;
        if (rtf < 1.0) {
          std::cout << "  性能: 实时 (RTF < 1.0)" << std::endl;
        } else {
          std::cout << "  性能: 非实时 (RTF >= 1.0)" << std::endl;
        }
      }

      std::cout << "\n流推理完成！" << std::endl;
    } else {
      std::cerr << "\n流推理失败！" << std::endl;
      return 1;
    }

    // } CPPTRACE_CATCH(const std::exception& e) {
  } catch (const std::exception& e) {
    std::cerr << "\n错误: " << e.what() << std::endl;
    // cpptrace::from_current_exception().print();
    return 1;
  }

  return 0;
}