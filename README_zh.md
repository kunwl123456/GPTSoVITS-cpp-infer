<div align="center">

# ⚡ GPT-SoVITS C++ SDK

**生产级 | 零拷贝 | 多语言绑定**

[![License](https://img.shields.io/badge/license-apache-blue.svg)](LICENSE)
[![C++](https://img.shields.io/badge/C++-17-blue.svg)](https://isocpp.org/)
[![GPU](https://img.shields.io/badge/CUDA-12.6+-orange.svg)](https://developer.nvidia.com/cuda-zone)
[![ONNX](https://img.shields.io/badge/ONNX-Optimized-brightgreen.svg)](https://onnxruntime.ai/)
[![TensorRT](https://img.shields.io/badge/TensorRT-Planned-76B900.svg)](https://developer.nvidia.com/tensorrt)

[简体中文](./README_zh.md) | [English](./README.md)

**"Python 是原型，C++ 才是武器。"**

---

本仓库是 [GPT-SoVITS Minimal Inference](https://github.com/GPT-SoVITS-Devel/GPT-SoVITS_minimal_inference) 的生产推理引擎。
Python 仓库负责模型导出与技术预览，后续模型生产代码将全面转移至本 C++ 绑定。

</div>

---

## 🌟 核心愿景

Python 项目已经证明了方向。现在我们不再对性能客气了。

这个 C++ SDK 把 Python pipeline 里的每一项优化——KV-Cache 预分配、IOBinding 零拷贝、Lookahead 流式——全部保留，然后干掉了最后一个瓶颈：**Python 解释器本身**。

目标：**更快 🏎️**、**可嵌入 🔩**、**绑定一切 🌍**、**零运行时税 💀**。

---

## 🚀 性能对比

*测试环境: I7 12700 | RTX 2080TI (22G) | CUDA 12.9 | FP16 精度*

*测试文本: 中日英混合多语言，输出音频约 19 秒*

```
皆さん、我在インターネット上看到someone把几国language混在一起speak。
虽然是混乱している句子ですけど、中文日本語プラスEnglish、挑戦スタート！
我study日本語的时候，もし有汉字，我会很happy。
2021年6月25日,今天32°C。以上です，byebye！
```

| 指标                          | Python ONNX  | **C++ 分布式**   | Python ONNX 流式 | **C++ 流式**      |
|:------------------------------|:-------------|:-----------------|:----------------|:------------------|
| **推理速度 (↑)**              | 172.4 tok/s  | **215.1 tok/s**  | 167.5 tok/s     | **198.6 tok/s**   |
| **RTF (↓)**                   | 0.3325       | **0.2398**       | 0.3100          | 0.4894            |
| **首包延迟 (↓)**              | 2.683 s      | **1.210 s**      | **1.000 s**         | 1.250 s       |
| **显存占用**                  | 3.9 G        | 3.6 G            | 4.5 G           | 4.0 G             |

> C++ 分布式 vs Python ONNX：**吞吐 +24.8%**，**RTF -27.9%**，**首包延迟 -54.9%**。

---

## 🏗️ 架构设计

本 SDK 采用**分布式推理模型**——说话人创建与推理完全解耦：

```
[云端 / 离线]                        [边缘 / 生产]
  参考音频                              .gsppkg 文件
     ↓                                      ↓
CreateSpeaker()              →      ImportSpeaker()
ExportSpeaker(.gsppkg)                      ↓
                                    Infer() / InferStreaming()
                                            ↓
                                         音频输出
```

### Pipeline 模式

| 模式 | 说明                           |
|:-----|:-----------------------------|
| **Edge Pipeline** | 纯推理，从 `.gsppkg` 加载说话人，显存占用最小 |
| **Streaming Pipeline** | 分块实时生成，带 CrossFade 去伪影       |
| **Full Pipeline** | 说话人创建 + 推理，适合大部分场景           |

### 模型栈

```
CNBertModel      → 音素 + BERT 特征提取
GPTEncoderModel  → 上下文编码（一次性）
GPTStepModel     → 自回归解码（O(1)/步，KV-Cache）
SoVITSModel      → 神经声码器 → PCM 音频
```

---

## 🏁 快速开始

### 环境要求

- CMake 3.20+
- ONNX Runtime 1.16+（GPU 推理需 CUDA 构建版本）
- CUDA 12.6+（可选，GPU 加速）

### 编译

```bash
# CPU 构建
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release

# CUDA 构建（推荐）
cmake -B build -S . \
  -DENABLE_STATIC_RUNTIME=1 \
  -DONNXRUNTIME_PATH=/path/to/onnxruntime \
  -DENABLE_CUDA=1 \
  -DCUDNN_PATH=/path/to/cudnn \
  -DCUDA_TOOLKIT_ROOT_DIR=/path/to/cuda
cmake --build build --config Release
```

### 第一步 — 导出模型（Python 侧）

```bash
# 在 GPT-SoVITS_minimal_inference 仓库中执行
python export_onnx.py \
    --gpt_path "pretrained_models/GPT_weights_v2ProPlus/your_model.ckpt" \
    --sovits_path "pretrained_models/SoVITS_weights_v2ProPlus/your_model.pth" \
    --cnhubert_base_path pretrained_models/chinese-hubert-base \
    --bert_path pretrained_models/chinese-roberta-wwm-ext-large \
    --output_dir "onnx_export/my_model_fp16" \
    --max_len 1000 \
    --validate \
    --validation_device cuda
```

### 第二步 — 创建说话人数据包

```bash
./build/gpt_sovits_cpp_cloud_create_onnx \
    --ref-audio ref.wav \
    --ref-text "参考文本" \
    --lang zh \
    --speaker-name my_speaker \
    --output my_speaker.gsppkg
```

### 第三步 — 推理

```bash
# 分布式（边缘）推理
./build/gpt_sovits_cpp_edge_inference_onnx \
    --speaker-package my_speaker.gsppkg \
    --text "要合成的文本" \
    --output output.wav

# 流式推理
./build/gpt_sovits_cpp_streaming_onnx \
    --speaker-package my_speaker.gsppkg \
    --text "要合成的文本" \
    --chunk-length 24 \
    --output output.wav
```

---

## 🔧 C++ API 使用示例

### 创建并导出说话人

```cpp
#include "GPTSoVITS/InferencePipeline.h"

// Full 模式：加载全部模型，包含说话人创建所需模型
GPTSoVITS::PipelineConfig config = GPTSoVITS::PipelineConfig::Full(
    "/path/to/model", GPTSoVITS::Model::DeviceType::kCUDA, 0);
config.resources_path = "./res";

GPTSoVITS::InferencePipeline pipeline(config);

// 从参考音频创建说话人
pipeline.CreateSpeaker("my_speaker", "zh", "ref.wav", "参考文本");

// 导出为可移植数据包
pipeline.ExportSpeaker("my_speaker", "my_speaker.gsppkg");
```

### 导入说话人

```cpp
// Edge 模式：纯推理，不加载创建模型，显存占用最小
GPTSoVITS::PipelineConfig config = GPTSoVITS::PipelineConfig::Edge(
    "/path/to/model", GPTSoVITS::Model::DeviceType::kCUDA, 0);
config.resources_path = "./res";

GPTSoVITS::InferencePipeline pipeline(config);
pipeline.ImportSpeaker("my_speaker.gsppkg", "my_speaker");  // 第二个参数：重命名（可选）
```

### 分布式推理

```cpp
GPTSoVITS::Model::SampleConfig sample_config;
sample_config.temperature = 1.0f;
sample_config.top_k       = 40;
sample_config.top_p       = 0.6f;

GPTSoVITS::Model::InferStats stats;
double first_latency_ms = 0.0;
auto t_start = std::chrono::steady_clock::now();

auto audio = pipeline.Infer(
    "my_speaker", "要合成的文本", "zh",
    sample_config, /*noise_scale=*/0.35f, /*speed=*/1.0f,
    &stats,
    [&]() {  // 第一个句段完成后触发
        first_latency_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t_start).count();
    });

audio->SaveToFile("output.wav");
// stats.TokensPerSec(), stats.gpt_time_s, stats.sovits_time_s
```

### 流式推理

```cpp
#include "GPTSoVITS/EdgePipeline.h"
#include "GPTSoVITS/StreamingPipeline.h"

// 构建 EdgePipeline（模型可共享，支持多个 StreamingPipeline 复用）
auto edge = std::make_shared<GPTSoVITS::EdgePipeline>(config_json, model_path,
                                                       g2p, enc, step, sovits);
edge->ImportSpeaker("my_speaker.gsppkg", "my_speaker");

GPTSoVITS::StreamingConfig stream_cfg;
stream_cfg.chunk_length = 24;    // 每块 token 数
stream_cfg.pause_length = 0.3f;  // 句段间停顿（秒）
stream_cfg.h_len        = 512;   // 历史 token 数（用于 CrossFade）
stream_cfg.l_len        = 16;    // 前瞻 token 数（用于 CrossFade）
stream_cfg.enable_fade  = true;

auto streaming = std::make_shared<GPTSoVITS::StreamingPipeline>(edge, stream_cfg);

GPTSoVITS::Model::InferStats stats;
streaming->InferSpeakerStreaming(
    "my_speaker", "要合成的文本", "zh",
    [](const GPTSoVITS::AudioChunk& chunk) {
        // chunk.audio_data  — PCM float32 采样数据
        // chunk.duration    — 时长（秒）
        // chunk.is_first / chunk.is_last
        // 送入音频设备或累积写文件
    },
    /*sample_config=*/{}, /*noise_scale=*/0.35f, /*speed=*/1.0f,
    &stats);
```

---

## 💎 核心优化

### 1. IOBinding 零拷贝（KV-Cache）
ONNX Runtime IOBinding 让 KV-Cache 张量在整个自回归过程中常驻显存。每个 token 生成步骤无 PCIe 传输，无 `cudaMemcpy`。

### 2. Ping-Pong Cache 缓冲区
预分配 `k_cache` / `v_cache` 输出缓冲区，每步仅交换指针——热循环内零分配。

### 3. 流式推理去伪影
Lookahead + History Window 机制，在 Chunk 边界进行线性加权融合（CrossFade）。无论 chunk 多小，彻底消除咔哒声。

---

## 🗺️ 路线图

- [x] **ONNX Runtime** 后端（CPU + CUDA）
- [x] **分布式推理**（说话人数据包工作流）
- [x] **流式推理**（带 CrossFade）
- [x] **多语言 G2P**（中文 / 英文 / 日文）
- [x] **InferStats** — tokens/s、RTF、首包延迟
- [ ] **TensorRT** 后端
- [ ] **INT8** 量化
- [ ] **多语言绑定**：
    - [ ] C API
    - [ ] Python 绑定
    - [ ] Rust 绑定
    - [ ] Go 绑定
    - [ ] Android 封装
    - (... 更多绑定)

---

## 🤝 致谢

基于 [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) 与 [GPT-SoVITS_minimal_inference](https://github.com/GPT-SoVITS-Devel/GPT-SoVITS_minimal_inference) 的工程化工作构建。

**如果本项目对你有帮助，点个 ⭐ 吧，免费的，而且真的很重要。🤗**
