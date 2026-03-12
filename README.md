<div align="center">

# ⚡ GPT-SoVITS C++ SDK

**Production-Grade | Zero-Copy | Multi-Language Binding**

[![License](https://img.shields.io/badge/license-apache-blue.svg)](LICENSE)
[![C++](https://img.shields.io/badge/C++-17-blue.svg)](https://isocpp.org/)
[![GPU](https://img.shields.io/badge/CUDA-12.6+-orange.svg)](https://developer.nvidia.com/cuda-zone)
[![ONNX](https://img.shields.io/badge/ONNX-Optimized-brightgreen.svg)](https://onnxruntime.ai/)
[![TensorRT](https://img.shields.io/badge/TensorRT-Planned-76B900.svg)](https://developer.nvidia.com/tensorrt)

[简体中文](./README_zh.md) | [English](./README.md)

**"Python was the prototype. C++ is the weapon."**

---

The production inference engine for [GPT-SoVITS Minimal Inference](https://github.com/GPT-SoVITS-Devel/GPT-SoVITS_minimal_inference).
The Python repo handles model export and tech preview. This repo is where the real work happens.

</div>

---

## 🌟 Core Vision

The Python project proved the concept. Now we're done being polite about performance.

This C++ SDK takes every optimization from the Python pipeline — KV-Cache pre-allocation, IOBinding zero-copy, lookahead streaming — and removes the last remaining bottleneck: **the Python interpreter itself**.

Goals: **Faster 🏎️**, **Embeddable 🔩**, **Bindable to Everything 🌍**, **No Runtime Tax 💀**.

---

## 🚀 Performance Benchmarks

*Environment: I7 12700 | RTX 2080TI (22G) | CUDA 12.9 | FP16 Precision*

*Test text: multilingual ZH/JA/EN mixed, ~19s audio output*

**ONNX:**

| Metric                       | Python ONNX | **C++ ONNX**    | Python ONNX Stream | **C++ ONNX Stream** |
|:-----------------------------|:------------|:----------------|:-------------------|:--------------------|
| **Inference Speed (↑)**      | 172.4 tok/s | **215.1 tok/s** | 167.5 tok/s        | **222.73 tok/s**    |
| **RTF (↓)**                  | 0.3325      | **0.2398**      | 0.3100             | 0.4894              |
| **First Packet Latency (↓)** | 2.683 s     | **1.210 s**     | **1.000 s**        | 1.250 s             |
| **VRAM Usage (↓)**           | 3.9 G       | **3.6 G**       | 4.5 G              | **4.0** G           |

**TRT:**

| Metric                       | Python TRT  | **C++ TRT**      | **C++ TRT Stream** |
|:-----------------------------|:------------|:-----------------|:-------------------|
| **Inference Speed (↑)**      | 291.6 tok/s | **357.66 tok/s** | **355.65 tok/s**   |
| **RTF (↓)**                  | 0.2096      | **0.1020**       | **0.1205**         |
| **First Packet Latency (↓)** | 2.683 s     | 0.5 s            | **0.46 S**         |
| **VRAM Usage (↓)**           | 3.4 G       | 2.8 G            | **2.3 G**          | 

---

## 🏗️ Architecture

This SDK implements a **distributed inference model** — speaker creation and inference are decoupled:

```
[Cloud / Offline]                    [Edge / Production]
  Reference Audio                      .gsppkg file
       ↓                                    ↓
  CreateSpeaker()              →    ImportSpeaker()
  ExportSpeaker(.gsppkg)                    ↓
                                      Infer() / InferStreaming()
                                            ↓
                                       Audio Output
```

### Pipeline Modes

| Mode | Description |
|:-----|:------------|
| **Edge Pipeline** | Inference only. Loads speaker from `.gsppkg`. Minimal VRAM. |
| **Streaming Pipeline** | Chunk-based real-time generation with crossfade. |
| **Full Pipeline** | Speaker creation + inference. For most scenarios. |

### Model Stack

```
CNBertModel      → Phoneme + BERT features
GPTEncoderModel  → Context encoding (one-shot)
GPTStepModel     → Autoregressive decoding (O(1) per step, KV-cache)
SoVITSModel      → Neural vocoder → PCM audio
```

---

## 🏁 Quick Start

### Prerequisites

- CMake 3.20+
- ONNX Runtime 1.16+ (CUDA build for GPU)
- CUDA 12.6+ (optional, for GPU)

### Build

```bash
# CPU build (ONNX only)
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release

# CUDA build with ONNX
cmake -B build -S . \
  -DENABLE_CUDA=1 \
  -DONNXRUNTIME_PATH=/path/to/onnxruntime \
  -DCUDA_TOOLKIT_ROOT_DIR=/path/to/cuda \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release

# TensorRT build
cmake -B build -S . \
  -DUSE_TENSORRT=1 \
  -DENABLE_CUDA=1 \
  -DTENSORRT_PATH=/path/to/tensorrt \
  -DCUDA_TOOLKIT_ROOT_DIR=/path/to/cuda \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

### Step 1 — Export Model (Python side)

**For ONNX:**

```bash
# In GPT-SoVITS_minimal_inference repo
python export_onnx.py \
    --gpt_path "pretrained_models/GPT_weights_v2ProPlus/your_model.ckpt" \
    --sovits_path "pretrained_models/SoVITS_weights_v2ProPlus/your_model.pth" \
    --cnhubert_base_path pretrained_models/chinese-hubert-base \
    --bert_path pretrained_models/chinese-roberta-wwm-ext-large \
    --output_dir "onnx_export/my_onnx_model" \
    --max_len 1000 \
    --validate \
    --validation_device cuda

# optimize
python onnx_to_fp16.py \
    --input_dir "onnx_export/my_onnx_model"
    --output_dir "onnx_export/my_onnx_model_fp16"
```

**For TensorRT:**

```bash
python onnx2trt.py \
    --input_dir onnx_export/my_onnx_model \
    --output_dir onnx_export/my_trt_model \
    --shape_profile fitted \
    --precision auto
```

**Shape Profile Notes:**
- C++ SDK uses **fitted** profile by default (optimized for 3-8s audio, ~10s max per segment)
- Suitable for ≤24GB VRAM with sentence-based inference
- For longer audio or larger VRAM, modify `GetProfileDefs()` in `src/model/backend/tensorrt_backend.cpp`

### Step 2 — Create Speaker Package

**ONNX:**

```bash
./build/example/gpt_sovits_cpp_cloud_create_onnx \
    my_speaker \
    ref.wav \
    "参考文本" \
    zh \
    my_speaker.gsppkg
```

**TensorRT:**

```bash
./build/example/gpt_sovits_cpp_cloud_create_trt \
    my_speaker \
    ref.wav \
    "参考文本" \
    zh \
    my_speaker.gsppkg
```

### Step 3 — Run Inference

**ONNX Edge Inference:**

```bash
./build/example/gpt_sovits_cpp_edge_inference_onnx \
    my_speaker.gsppkg \
    "要合成的文本" \
    zh \
    my_speaker \
    output.wav
```

**ONNX Streaming Inference:**

```bash
./build/example/gpt_sovits_cpp_streaming_onnx \
    --speaker-package my_speaker.gsppkg \
    --text "要合成的文本" \
    --lang zh \
    --chunk-length 24 \
    --output output.wav
```

**TensorRT Edge Inference:**

```bash
./build/example/gpt_sovits_cpp_edge_inference_trt \
    my_speaker.gsppkg \
    "要合成的文本" \
    zh \
    my_speaker \
    output_trt.wav
```

**TensorRT Streaming Inference:**

```bash
./build/example/gpt_sovits_cpp_streaming_trt \
    --speaker-package my_speaker.gsppkg \
    --text "要合成的文本" \
    --lang zh \
    --chunk-length 24 \
    --output output_trt.wav
```

---

## 🔧 C++ API Usage

### Create & Export Speaker

**ONNX Backend:**

```cpp
#include "GPTSoVITS/InferencePipeline.h"

// Full mode: loads all models including speaker creation models
GPTSoVITS::PipelineConfig config = GPTSoVITS::PipelineConfig::Full(
    "/path/to/model", GPTSoVITS::Model::DeviceType::kCUDA, 0);
config.resources_path = "./res";
config.backend = GPTSoVITS::Model::BackendType::kONNX;

GPTSoVITS::InferencePipeline pipeline(config);

// Create speaker from reference audio
pipeline.CreateSpeaker("my_speaker", "zh", "ref.wav", "参考文本");

// Export to portable package
pipeline.ExportSpeaker("my_speaker", "my_speaker.gsppkg");
```

**TensorRT Backend:**

```cpp
#include "GPTSoVITS/InferencePipeline.h"

GPTSoVITS::PipelineConfig config = GPTSoVITS::PipelineConfig::Full(
    "/path/to/model", GPTSoVITS::Model::DeviceType::kCUDA, 0);
config.resources_path = "./res";
config.backend = GPTSoVITS::Model::BackendType::kTensorRT;
config.engine_cache_dir = "./trt_cache";  // TensorRT engine cache

GPTSoVITS::InferencePipeline pipeline(config);

// TensorRT engines will be built automatically on first run
pipeline.CreateSpeaker("my_speaker", "zh", "ref.wav", "参考文本");
pipeline.ExportSpeaker("my_speaker", "my_speaker.gsppkg");
```

### Import Speaker

```cpp
// Edge mode: inference only, no creation models loaded
GPTSoVITS::PipelineConfig config = GPTSoVITS::PipelineConfig::Edge(
    "/path/to/model", GPTSoVITS::Model::DeviceType::kCUDA, 0);
config.resources_path = "./res";

GPTSoVITS::InferencePipeline pipeline(config);
pipeline.ImportSpeaker("my_speaker.gsppkg", "my_speaker");  // second arg: rename (optional)
```

### Edge Inference

```cpp
GPTSoVITS::Model::SampleConfig sample_config;
sample_config.temperature = 1.0f;
sample_config.top_k       = 40;
sample_config.top_p       = 0.6f;

GPTSoVITS::Model::InferStats stats;
double first_latency_ms = 0.0;

auto audio = pipeline.Infer(
    "my_speaker", "要合成的文本", "zh",
    sample_config, /*noise_scale=*/0.35f, /*speed=*/1.0f,
    &stats,
    [&]() {  // fires after first segment is ready
        first_latency_ms = /* elapsed since infer start */;
    });

audio->SaveToFile("output.wav");
// stats.TokensPerSec(), stats.gpt_time_s, stats.sovits_time_s
```

### Streaming Inference

**ONNX Backend:**

```cpp
#include "GPTSoVITS/EdgePipeline.h"
#include "GPTSoVITS/StreamingPipeline.h"

// Build EdgePipeline (shared models, can serve multiple StreamingPipelines)
auto g2p = std::make_shared<GPTSoVITS::G2P::G2PPipline>();
auto bert = std::make_unique<GPTSoVITS::Model::CNBertModel>();
bert->Init<GPTSoVITS::Model::ONNXBackend>(model_path + "/bert.onnx",
                                          "./res/bert_tokenizer.json", device);
g2p->RegisterLangProcess("zh", std::make_unique<GPTSoVITS::G2P::G2PZH>(),
                         std::move(bert), true);

auto enc = std::make_shared<GPTSoVITS::Model::GPTEncoderModel>();
enc->Init<GPTSoVITS::Model::ONNXBackend>(model_path + "/gpt_encoder.onnx", device);

auto step = std::make_shared<GPTSoVITS::Model::GPTStepModel>();
step->Init<GPTSoVITS::Model::ONNXBackend>(model_path + "/gpt_step.onnx", device);

auto sovits = std::make_shared<GPTSoVITS::Model::SoVITSModel>();
sovits->Init<GPTSoVITS::Model::ONNXBackend>(model_path + "/sovits.onnx", device);

auto edge = std::make_shared<GPTSoVITS::EdgePipeline>(config_json, model_path,
                                                       g2p, enc, step, sovits);
edge->ImportSpeaker("my_speaker.gsppkg", "my_speaker");

GPTSoVITS::StreamingConfig stream_cfg;
stream_cfg.chunk_length = 24;    // tokens per chunk
stream_cfg.pause_length = 0.3f;  // silence between sentences (s)
stream_cfg.h_len        = 512;   // history tokens for crossfade
stream_cfg.l_len        = 16;    // lookahead tokens for crossfade
stream_cfg.enable_fade  = true;

auto streaming = std::make_shared<GPTSoVITS::StreamingPipeline>(edge, stream_cfg);

GPTSoVITS::Model::InferStats stats;
streaming->InferSpeakerStreaming(
    "my_speaker", "要合成的文本", "zh",
    [](const GPTSoVITS::AudioChunk& chunk) {
        // chunk.audio_data  — PCM float32 samples
        // chunk.duration    — seconds
        // chunk.is_first / chunk.is_last
        // feed to audio device or accumulate
    },
    /*sample_config=*/{}, /*noise_scale=*/0.35f, /*speed=*/1.0f,
    &stats);
```

**TensorRT Backend:**

```cpp
#include "GPTSoVITS/EdgePipeline.h"
#include "GPTSoVITS/StreamingPipeline.h"

// Build EdgePipeline with TensorRT backend
auto g2p = std::make_shared<GPTSoVITS::G2P::G2PPipline>();
auto bert = std::make_unique<GPTSoVITS::Model::CNBertModel>();
bert->Init<GPTSoVITS::Model::TensorRTBackend>(model_path + "/bert.onnx",
                                              "./res/bert_tokenizer.json", device);
g2p->RegisterLangProcess("zh", std::make_unique<GPTSoVITS::G2P::G2PZH>(),
                         std::move(bert), true);

auto enc = std::make_shared<GPTSoVITS::Model::GPTEncoderModel>();
enc->Init<GPTSoVITS::Model::TensorRTBackend>(model_path + "/gpt_encoder.engine", device);

auto step = std::make_shared<GPTSoVITS::Model::GPTStepModel>();
step->Init<GPTSoVITS::Model::TensorRTBackend>(model_path + "/gpt_step.engine", device);

auto sovits = std::make_shared<GPTSoVITS::Model::SoVITSModel>();
sovits->Init<GPTSoVITS::Model::TensorRTBackend>(model_path + "/sovits.engine", device);

auto edge = std::make_shared<GPTSoVITS::EdgePipeline>(config_json, model_path,
                                                       g2p, enc, step, sovits);
edge->ImportSpeaker("my_speaker.gsppkg", "my_speaker");

// Same streaming configuration as ONNX
GPTSoVITS::StreamingConfig stream_cfg;
stream_cfg.chunk_length = 24;
stream_cfg.pause_length = 0.3f;
stream_cfg.h_len        = 512;
stream_cfg.l_len        = 16;
stream_cfg.enable_fade  = true;

auto streaming = std::make_shared<GPTSoVITS::StreamingPipeline>(edge, stream_cfg);

// Streaming callback remains the same
streaming->InferSpeakerStreaming(
    "my_speaker", "要合成的文本", "zh",
    [](const GPTSoVITS::AudioChunk& chunk) {
        // Process audio chunks in real-time
    },
    /*sample_config=*/{}, /*noise_scale=*/0.35f, /*speed=*/1.0f);
```

---

## 💎 Key Optimizations

### 1. IOBinding Zero-Copy (KV-Cache)
ONNX Runtime IOBinding keeps KV-cache tensors resident in VRAM across every autoregressive step. No PCIe round-trips, no `cudaMemcpy` per token.

### 2. Ping-Pong Cache Buffers
Pre-allocated `k_cache` / `v_cache` output buffers. Each step swaps pointers — zero allocation in the hot loop.

### 3. Artifact-Free Streaming
Lookahead + history window with linear crossfade at chunk boundaries. No clicks, no pops, even at aggressive chunk sizes.

---

## 🗺️ Roadmap

- [x] **ONNX Runtime** backend (CPU + CUDA)
- [x] **Distributed inference** (speaker package workflow)
- [x] **Streaming inference** with crossfade
- [x] **Multi-language G2P** (ZH / EN / JA)
- [x] **InferStats** — tokens/s, RTF, first-packet latency
- [x] **TensorRT** backend
- [ ] **INT8** quantization
- [ ] **Language Bindings**:
    - [ ] C API
    - [ ] Python binding
    - [ ] Rust binding
    - [ ] Go binding
    - [ ] Android / iOS wrapper
    - (... more bindings)

---

## 🤝 Acknowledgments

Built on top of [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) and the engineering work in [GPT-SoVITS_minimal_inference](https://github.com/GPT-SoVITS-Devel/GPT-SoVITS_minimal_inference).

**If this project helps you, drop a ⭐. It's free and it means a lot. 🤗**
