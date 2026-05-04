# CosyVoice.cpp

语言： [English](README.md)

> 非官方说明：本仓库**与 CosyVoice 官方团队无隶属关系**，也未获得官方背书或维护。本项目是社区开发者发起和维护的 C++/GGML 移植实现。

> **当前状态提示：** 当前 CPU、CUDA、Metal 和 SYCL 后端均可正常运行。Vulkan 后端目前无法正常工作。用于生产前请先阅读[后端测试情况](#后端测试情况)。

本项目将原始 CosyVoice 项目发布的 Python 推理流程迁移到 C++/GGML，目前主要支持 **CosyVoice3**。

本仓库仅提供独立社区实现，不包含任何官方支持承诺。

本项目提供：
- 核心 C/C++ 推理库（`cosyvoice`）
- 命令行合成工具（`cosyvoice-cli`）
- OpenAI Speech 兼容 API 服务（`cosyvoice-server`）
- GGUF 量化工具（`quantize`）

## 目录
- [文档](#文档)
- [AI 使用说明](#ai-使用说明)
- [快速开始](#快速开始)
- [推理流程](#推理流程)
- [构建](#构建)
- [依赖解析方式](#依赖解析方式)
- [CMake 选项](#cmake-选项)
- [常见构建矩阵](#常见构建矩阵)
- [GGML 后端/构建选项](#ggml-后端构建选项)
- [使用自定义依赖](#使用自定义依赖)
- [模型转 GGUF](#模型转-gguf)
- [工具使用说明](#工具使用说明)
- [后端测试情况](#后端测试情况)
- [故障排查](#故障排查)
- [第三方许可说明](#第三方许可说明)
- [许可证说明](#许可证说明)
- [欢迎贡献](#欢迎贡献)

## 文档
- API 索引：[docs/API_zh.md](docs/API_zh.md)
- 工具说明：[docs/TOOLS_zh.md](docs/TOOLS_zh.md)

## AI 使用说明
- 项目代码主要由作者手工实现。
- 文档内容部分由 AI 协助撰写与整理。
- 文档可能存在错误或与实现不同步的情况；如有疑问请以源码与头文件为准，也欢迎提交 Issue/PR 纠正。

## 第三方许可说明
- 已打包依赖的许可证信息见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
- FFT 实现（`src/fft.cpp`）参考/改造自 KissFFT（BSD-3-Clause），并加入了项目内 SIMD 优化；详见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
- tokenizer 实现基于 llama.cpp（MIT）改造。

## 许可证说明
- **本仓库代码**：MIT（见 `LICENSE`）。
- **上游参考**：原始 CosyVoice 项目代码与模型为 Apache-2.0。
- **实现说明**：本仓库是基于模型架构与推理行为的独立 C++/GGML 重实现，并非官方 fork 或官方发布。
- **GGUF 模型产物**：发布的模型文件继续保持 Apache-2.0。
  - ModelScope：<https://modelscope.cn/models/Lourdle/Fun-CosyVoice3-0.5B-2512-GGUF>
  - Hugging Face：<https://huggingface.co/Lourdle/Fun-CosyVoice3-0.5B-2512-GGUF>
- **模型许可证文件**：[MODEL_LICENSE.md](MODEL_LICENSE.md)

## 快速开始

### 预编译发布版 (Releases)

本仓库提供的 Releases 不包含 GGML 后端库。要使用它们：
1. 从本仓库的 Releases 页面下载 `cosyvoice-cli` 或 `cosyvoice-server`。
2. 下载与您的硬件（如 CUDA、CPU 等）和操作系统匹配的 `llama.cpp` release。
3. 解压 `llama.cpp` release，并将 `cosyvoice` 的可执行文件放入其中，即包含 GGML 后端共享库（如 `ggml.dll`、`ggml-cuda.dll` 等）的目录。
4. 在该目录下运行可执行文件。它们默认会自动加载程序所在目录的 GGML 后端。

### Python 环境准备（用于模型转换）
1. 先用本仓库的 `convert_model_to_gguf.py` 将上游 CosyVoice 模型权重转换为 GGUF。
2. 配置并编译本项目。
3. （可选）用 `quantize` 对 GGUF 模型量化。
4. 使用 `cosyvoice-cli` 进行文件合成，或使用 `cosyvoice-server` 提供 OpenAI Speech 兼容 HTTP API。

## 推理流程
本项目支持两条等价推理路径：

1. **前端 + TTS 一体流程（首次推荐）**
   - 输入：参考音频（以及按模式要求的参考文本）+ 目标文本
   - 流程：前端提取 `prompt_speech` -> 使用 `prompt_speech` 与目标文本执行 TTS
   - 模式说明：
     - `zero-shot` 模式需要 `--prompt-text`
     - `instruct` / `cross-lingual` 模式下 `--prompt-text` 会被忽略

2. **复用 `prompt_speech`（批量合成推荐）**
   - 可将 `prompt_speech` 保存为 `.gguf`（例如通过 `--prompt-speech-output` 或 API `cosyvoice_prompt_speech_save_to_file`）
   - 后续直接传入 `--prompt-speech <file>` 即可合成，**不需要再次运行前端 ONNX**

简化理解：先把参考条件（音色/说话人信息）编码成 `prompt_speech`，再将 `prompt_speech` 与目标文本结合生成语音。

## 构建

### 环境要求
- CMake >= 3.24
- 支持 C++20 的 C/C++ 编译器
- Git（当本地缺少 GGML 源码时用于自动拉取）
- 目前 CPU 路径中的部分数据处理要求 x86 CPU 支持 AVX2
- 对 CPU 侧数学运算较重的路径（如 `log`、三角函数），当前仅 MSVC 构建可启用 SIMD 加速；其他工具链目前回退为标量实现

后端/运行时依赖会随构建选项变化（CUDA/Vulkan/CPU、ONNX Runtime、ICU 等）。

### 1）配置
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
```

### 2）编译
```bash
cmake --build build --config Release
```

构建产物默认输出到：
- `build/bin`（可执行文件与运行时 DLL）
- `build/lib`（库文件）

## 依赖解析方式
顶层 CMake 按以下顺序解析依赖：

- **PCRE2**
  - 从 `vendor/pcre2` 构建静态库（`pcre2-8`、`pcre2-16`）。
- **GGML**
  - 使用 `GGML_SOURCE_DIR`（默认 `vendor/ggml`）。
  - 若目录不存在，会自动克隆 `https://github.com/ggml-org/ggml.git`。
- **ICU**（用于文本规范化，除非通过 `COSYVOICE_NO_ICU` 关闭）
  - 解析顺序：`ICU_PREBUILT_DIR` -> `find_package(ICU)` -> Windows 自动下载 -> Linux/macOS 使用系统 ICU。
- **ONNX Runtime**（用于前端，除非通过 `COSYVOICE_NO_FRONTEND` 关闭）
  - 解析顺序：`ORT_PREBUILT_DIR` -> `find_package(onnxruntime)` -> 自动下载。

常用缓存变量：
- `GGML_SOURCE_DIR`
- `ICU_PREBUILT_DIR`
- `ORT_PREBUILT_DIR`

默认值：
- `GGML_SOURCE_DIR=vendor/ggml`
- `ICU_PREBUILT_DIR=<build_dir>/_deps/icu`
- `ORT_PREBUILT_DIR=<build_dir>/_deps/onnxruntime`

说明：
- 如果 `GGML_SOURCE_DIR` 下没有 GGML 源码，CMake 会尝试自动克隆 GGML。
- 如果 ICU/ONNX Runtime 未被 `find_package` 找到，CMake 会在配置的预编译目录中使用或下载对应二进制。
- Windows 下预编译依赖 DLL 会复制到可执行文件旁。

## 音频后端与 FFmpeg

本项目的音频辅助 API 支持两种后端：

- `MINIAUDIO`（默认）：提供 WAV I/O 与基本 PCM 帮助函数。
- `FFMPEG`（可选）：在链接的 FFmpeg 运行时提供所需编码器时，启用更多编码/解码格式。

通过 CMake 配置音频后端：将 `COSYVOICE_AUDIO_BACKEND` 设为 `MINIAUDIO` 或 `FFMPEG`。
默认值为 `MINIAUDIO`。

示例：
```bash
cmake -S . -B build -DCOSYVOICE_AUDIO_BACKEND=MINIAUDIO
cmake -S . -B build -DCOSYVOICE_AUDIO_BACKEND=FFMPEG
cmake -S . -B build -DCOSYVOICE_AUDIO_BACKEND=FFMPEG -DFFMPEG_PREBUILT_DIR=/path/to/ffmpeg
```

如果启用 FFmpeg 支持，公开音频 API 的函数名保持不变。可使用 `cosyvoice_audio_supported_encoding_formats()` 查询当前链接的 FFmpeg 运行时真正支持哪些格式。

FFmpeg 使用要点：

- 在 Windows 上，构建脚本默认在未提供 `FFMPEG_PREBUILT_DIR` 时下载 BtbN 的预编译 FFmpeg。
- 在 Linux/macOS 上，若系统提供 FFmpeg（apt/homebrew），项目会优先使用系统库；否则可通过 `FFMPEG_PREBUILT_DIR` 指定预编译位置。
- API 层支持 `wav`、`mp3`、`aac`、`flac`、`m4a`、`opus`，但具体可用格式取决于当前链接的 FFmpeg 构建。库会在运行时探测可用编码器，并通过 API / CLI / server 帮助文本暴露支持集合。
- `m4a` 是这里提供的非标准便捷扩展。OpenAI Speech 标准并没有定义 `m4a`，只在你的客户端/服务端理解这个扩展时使用。
- 如果客户端请求了运行时不可用的格式，服务/CLI 会建议回退到 `wav` 或 `pcm`。
- 在 Windows 上，构建脚本会把找到的 FFmpeg 运行时 DLL 复制到可执行文件目录。若你使用自定义预编译 FFmpeg，请确认其 `bin` / `lib` 目录结构符合 `cmake/Dependencies.cmake` 的预期。

许可证提醒：

- 本仓库代码采用 MIT 许可。FFmpeg 预编译包可能是 LGPL 或 GPL，取决于编译选项。使用包含 GPL 编码器的 FFmpeg 构建并重新分发时，可能会对你的发行物带来 GPL 约束。详见 [FFmpeg-NOTICE.md](FFmpeg-NOTICE.md)。

## CMake 选项
项目级选项：
- `BUILD_SHARED_LIBS=ON/OFF`（默认：`ON`）
- `COSYVOICE_NO_AUDIO=ON/OFF`（默认：`OFF`）
- `COSYVOICE_NO_FRONTEND=ON/OFF`（默认：`OFF`）
- `COSYVOICE_NO_ICU=ON/OFF`（默认：`OFF`）
- `COSYVOICE_AUDIO_BACKEND=MINIAUDIO/FFMPEG`（默认：`MINIAUDIO`）

依赖路径选项：
- `GGML_SOURCE_DIR=<path>`
- `ICU_PREBUILT_DIR=<path>`
- `ORT_PREBUILT_DIR=<path>`
- `FFMPEG_PREBUILT_DIR=<path>`

GGML 后端相关选项可直接透传（例如 `GGML_CUDA`、`GGML_VULKAN` 等）。

## 常见构建矩阵
| 场景 | 推荐 CMake 参数 |
|---|---|
| CUDA 后端 | `-DGGML_CUDA=ON` |
| Vulkan 后端 | `-DGGML_VULKAN=ON` |
| 仅 CPU | 通常不需要额外后端参数 |
| 仅核心能力（无 frontend / ICU） | `-DCOSYVOICE_NO_FRONTEND=ON -DCOSYVOICE_NO_ICU=ON` |
| 关闭音频辅助 API | `-DCOSYVOICE_NO_AUDIO=ON` |

## GGML 后端/构建选项
本项目通过 CMake 集成 GGML，可在根工程直接传入 GGML 后端开关。

常见示例（后端具体配置建议参考 `llama.cpp` / GGML [文档](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)）：
```bash
# CUDA 示例
cmake -S . -B build-cuda -DGGML_CUDA=ON
```

项目选项：
- `COSYVOICE_NO_AUDIO=ON/OFF`（关闭/启用音频辅助 API）
- `COSYVOICE_NO_FRONTEND=ON/OFF`（关闭/启用 ONNX 前端）
- `COSYVOICE_NO_ICU=ON/OFF`（关闭/启用 ICU 文本规范化）
- `BUILD_SHARED_LIBS=ON/OFF`

常见组合示例：
```bash
# 仅核心功能构建（关闭 ONNX 前端与 ICU 文本规范化）
cmake -S . -B build-core -DCOSYVOICE_NO_FRONTEND=ON -DCOSYVOICE_NO_ICU=ON

# 无音频辅助 API 构建（CLI 走 WAV 输出回退路径）
cmake -S . -B build-noaudio -DCOSYVOICE_NO_AUDIO=ON
```

## 使用自定义依赖
可以通过缓存变量指定自定义依赖路径：

```bash
cmake -S . -B build \
  -DGGML_SOURCE_DIR=/path/to/ggml \
  -DICU_PREBUILT_DIR=/path/to/icu \
  -DORT_PREBUILT_DIR=/path/to/onnxruntime
```

也可以直接使用构建目录下的默认预编译依赖位置：
- `<build_dir>/_deps/icu`
- `<build_dir>/_deps/onnxruntime`

只要按期望目录结构把文件放进去，CMake 会自动识别（不需要额外 `-D`）。

期望的关键目录/文件：
- ICU：`include/unicode/utypes.h`（以及 `lib*` / `bin*` 下的库和 DLL）
- ONNX Runtime：`include/onnxruntime_c_api.h`（以及 `lib` 下的运行库文件）

说明：
- 如果 `GGML_SOURCE_DIR` 下没有 GGML 源码，CMake 会尝试自动克隆 GGML。
- 如果 ICU/ONNX Runtime 未被 `find_package` 找到，CMake 会在配置的预编译目录中使用或下载对应二进制。
- Windows 下会在构建后将所需 DLL 复制到可执行文件目录，便于本地直接运行。

## 模型转 GGUF
可使用本仓库的转换脚本 `convert_model_to_gguf.py`，将上游 CosyVoice 模型权重转换为 `cosyvoice.cpp` 可用的 GGUF。

先安装 Python 依赖：
```bash
pip install -r requirements.txt
```

最小用法：
```bash
python convert_model_to_gguf.py \
  --yaml_config /path/to/cosyvoice.yaml \
  --ftype f16 \
  --gguf_model /path/to/CosyVoice3-2512_F16.gguf
```

完整参数示例：
```bash
python convert_model_to_gguf.py \
  --yaml_config /path/to/cosyvoice.yaml \
  --llm_model /path/to/llm.pt \
  --blank_llm /path/to/CosyVoice-BlankEN \
  --flow_model /path/to/flow.pt \
  --hift_model /path/to/hift.pt \
  --gguf_model /path/to/CosyVoice3-2512_Q8_0.gguf \
  --ftype q8_0 \
  --tag 2512
```

`--ftype` 可选值：
- `default`, `f32`, `f16`, `q8_0`, `q5_0`, `q5_1`, `q4_0`, `q4_1`

未显式传入时的默认路径规则：
- `--llm_model` -> `<yaml_dir>/llm.pt`
- `--blank_llm` -> `<yaml_dir>/CosyVoice-BlankEN`
- `--flow_model` -> `<yaml_dir>/flow.pt`
- `--hift_model` -> `<yaml_dir>/hift.pt`

转换后建议：
1. 先确认生成的 `.gguf` 文件可用。
2. （可选）再使用本仓库 `quantize` 工具量化。

## 工具使用说明
本仓库包含 3 个面向使用者的工具：
- `cosyvoice-cli`：本地文件式 TTS 合成（支持复用 prompt_speech，以及前端 + TTS 一体流程）。
- `cosyvoice-server`：OpenAI Speech 兼容 HTTP API 服务，适合服务化接入。
- `quantize`：GGUF 量化工具，用于将模型转换为更小/更快的量化格式。

完整命令、参数和示例见：
- [docs/TOOLS_zh.md](docs/TOOLS_zh.md)

## 后端测试情况
当前各后端测试结果如下：

| 后端 | 状态 | 备注 |
|---|---:|---|
| CPU | 可运行 | 感谢 @[jasagiri](https://github.com/jasagiri) 帮助定位问题。已在 Windows、Linux 和 Mac 上测试。 |
| CUDA | 可运行 | 已在 Ada Lovelace GPU (Windows & Linux) 上测试。 |
| Metal | 可运行 | 感谢 @[jasagiri](https://github.com/jasagiri) 的支持与代码贡献。 |
| SYCL | 可运行 | 已在 Windows 11 x64 上的 Intel Raptor Lake 集成显卡上验证。 |
| Vulkan | 不能运行 | 目前无法正常运行。 |
| 其它 | 未测试 | |

## 故障排查
- CMake 找不到 GGML：设置 `-DGGML_SOURCE_DIR=...`，或使用默认 `vendor/ggml` 并确保本机可用 Git（用于自动克隆）。
- ICU/ONNX Runtime 检测失败：可安装系统包（适用平台），或将预编译文件放到 `<build_dir>/_deps/icu` 与 `<build_dir>/_deps/onnxruntime`。
- Windows 运行时缺库：检查 `build/bin` 下是否存在构建后复制的依赖 DLL。
- 后端相关情况见[后端测试情况](#后端测试情况)。

## 欢迎贡献
欢迎提交 Issue 和 Pull Request，尤其是：
- 后端稳定性修复
- 跨平台正确性改进
- 性能与内存优化
- 文档与工具改进

如果根因在 GGML，请优先向上游 GGML 提交修复补丁。
