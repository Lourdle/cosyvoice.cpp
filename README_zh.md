# CosyVoice.cpp

语言： [English](README.md)

> 非官方说明：本仓库**与 CosyVoice 官方团队无隶属关系**，也未获得官方背书或维护。本项目是社区开发者发起和维护的 C++/GGML 移植实现。

> **当前状态提示：** 当前测试中 CUDA 稳定性问题已解决，Windows/Linux 下 CUDA 均可正常生成音频；CPU 和 Vulkan 仍无法正常运行。用于生产前请先阅读[已知问题](#已知问题)。

本项目将原始 CosyVoice 项目发布的 Python 推理流程迁移到 C++/GGML，目前主要支持 **CosyVoice3**。

本仓库仅提供独立社区实现，不包含任何官方支持承诺。

本项目提供：
- 核心 C/C++ 推理库（`cosyvoice`）
- 命令行合成工具（`cosyvoice-cli`）
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
- [量化工具（`tools/quantize`）](#量化工具toolsquantize)
- [CLI 工具（`tools/cli`）](#cli-工具toolscli)
- [CLI 快速索引](#cli-快速索引)
- [已知问题](#已知问题)
- [故障排查](#故障排查)
- [第三方许可说明](#第三方许可说明)
- [许可证说明](#许可证说明)
- [欢迎贡献](#欢迎贡献)

## 文档
- API 索引：[docs/API_zh.md](docs/API_zh.md)

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
1. 先用本仓库的 `convert_model_to_gguf.py` 将上游 CosyVoice 模型权重转换为 GGUF。
2. 配置并编译本项目。
3. （可选）用 `quantize` 对 GGUF 模型量化。
4. 使用 `cosyvoice-cli` 合成语音。

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
顶层 CMake 对依赖的处理逻辑如下：

- **PCRE2**
  - 从 `vendor/pcre2` 构建静态库（`pcre2-8`、`pcre2-16`）。
- **GGML**
  - 使用 `GGML_SOURCE_DIR`（默认 `vendor/ggml`）。
  - 若目录不存在，会自动克隆 `https://github.com/ggml-org/ggml.git`。
- **ICU**（用于文本规范化，除非关闭）
  - 由 `COSYVOICE_NO_ICU` 控制。
  - 如果 `ICU_PREBUILT_DIR` 可用，会优先直接使用该目录。
  - 否则尝试 `find_package(ICU)`。
  - Windows 下若仍找不到，会自动下载预编译 ICU。
  - Linux/macOS 下若仍找不到，需要安装系统 ICU。
- **ONNX Runtime**（用于前端，除非关闭）
  - 由 `COSYVOICE_NO_FRONTEND` 控制。
  - 如果 `ORT_PREBUILT_DIR` 可用，会优先直接使用该目录。
  - 否则尝试 `find_package(onnxruntime)`。
  - 若仍找不到，会自动下载预编译 ONNX Runtime。

常用依赖路径缓存变量：
- `GGML_SOURCE_DIR`
- `ICU_PREBUILT_DIR`
- `ORT_PREBUILT_DIR`

默认值：
- `GGML_SOURCE_DIR=vendor/ggml`
- `ICU_PREBUILT_DIR=<build_dir>/_deps/icu`
- `ORT_PREBUILT_DIR=<build_dir>/_deps/onnxruntime`

依赖优先级（实际解析顺序）：
- **GGML**：`GGML_SOURCE_DIR` -> 若缺失则自动克隆 GGML 仓库。
- **ICU**：若 `ICU_PREBUILT_DIR` 可用则优先使用该目录 -> `find_package(ICU)` ->（Windows）自动下载预编译 ICU ->（Linux/macOS）需手动安装系统 ICU。
- **ONNX Runtime**：若 `ORT_PREBUILT_DIR` 可用则优先使用该目录 -> `find_package(onnxruntime)` -> 自动下载预编译 ONNX Runtime。

平台说明：
- **Windows**：构建后会将预编译依赖 DLL 复制到可执行文件目录。
- **Linux/macOS**：预编译共享库按安装规则放到库安装目录。

## CMake 选项
项目级选项：
- `BUILD_SHARED_LIBS=ON/OFF`（默认：`ON`）
- `COSYVOICE_NO_AUDIO=ON/OFF`（默认：`OFF`）
- `COSYVOICE_NO_FRONTEND=ON/OFF`（默认：`OFF`）
- `COSYVOICE_NO_ICU=ON/OFF`（默认：`OFF`）

依赖路径选项：
- `GGML_SOURCE_DIR=<path>`
- `ICU_PREBUILT_DIR=<path>`
- `ORT_PREBUILT_DIR=<path>`

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

## 量化工具（`tools/quantize`）
可执行文件名：`quantize`

基本用法：
```bash
quantize -f input.gguf -o output-q4k.gguf -t Q4_K
```

查看帮助：
```bash
quantize --help
```

支持的量化类型：
- `F16`, `Q8_0`, `Q5_0`, `Q5_1`, `Q4_0`, `Q4_1`
- `Q6_K`, `Q5_K`, `Q4_K`, `Q3_K`, `Q2_K`
- `COPY`

支持通过重复 `-c/--custom-string` 写入自定义字符串元数据。

## CLI 工具（`tools/cli`）
可执行文件名：`cosyvoice-cli`

查看帮助：
```bash
cosyvoice-cli --help
```

### A）使用已有 prompt_speech 做 TTS
```bash
cosyvoice-cli \
  --model model.gguf \
  --prompt-speech prompt_speech.gguf \
  --text "Hello from CosyVoice" \
  --output out.wav
```

### B）前端 + TTS 一体流程（运行时提取 prompt_speech）
```bash
cosyvoice-cli \
  --model model.gguf \
  --speech-tokenizer speech_tokenizer.onnx \
  --campplus campplus.onnx \
  --prompt-audio ref.wav \
  --prompt-text "参考语音文本" \
  --text "target text" \
  --output out.wav
```

### C）仅前端模式（只生成 prompt_speech）
```bash
cosyvoice-cli \
  --frontend-only \
  --speech-tokenizer speech_tokenizer.onnx \
  --campplus campplus.onnx \
  --prompt-audio ref.wav \
  --prompt-text "参考语音文本" \
  --prompt-speech-output prompt_speech.gguf
```

### 参数说明

核心参数：
- `--help, -h`：显示帮助并退出。
- `--model, -m <file>`：TTS 使用的 CosyVoice 模型文件（`.gguf`）。
- `--text, -t <text>`：待合成文本。
- `--output, -o <file>`：输出音频文件路径。
  - 常规构建：输出格式由文件扩展名决定。
  - `COSYVOICE_NO_AUDIO=ON`：输出始终为 WAV。
- `--speed, -s <value>`：语速倍率，默认 `1.0`，必须大于 `0`。
- `--seed <value>`：采样与内部噪声生成的随机种子，必须是无符号 32 位整数；默认随机。
- `--max-llm-len <value>`：LLM 最大输入 token 数（`n_max_seq`），默认 `2048`，必须为正整数。
- `--threads, -j <value>`：模型推理使用的 CPU 线程数，必须是无符号 32 位整数；默认 `0`（使用当前硬件并发数）。
- `--llm-kv-cache-type <f32|f16|q8_0|q5_1|q5_0|q4_1|q4_0>`：LLM KV cache 类型，默认 `q8_0`。
- `--mode <zero-shot|instruct|cross-lingual>`：TTS 模式。默认按 `--instruction` 自动判定。
- `--instruction, -i <text>`：instruct 模式指令文本。

采样覆盖参数：
- `--temperature <value>`：采样温度，必须 `> 0`。
- `--top-k <value>`：top-k 采样大小，必须 `>= 0`。
- `--top-p <value>`：top-p 采样阈值，必须在 `[0, 1]`。
- `--win-size <value>`：重复惩罚窗口大小，必须 `> 0`。
- `--tau-r <value>`：重复惩罚系数，必须 `>= 0`。
- `--min-token-text-ratio <value>`：最小 token/text 比率，必须 `>= 0`。
- `--max-token-text-ratio <value>`：最大 token/text 比率，必须 `>= 0`，且在与 `--min-token-text-ratio` 同时设置时不能更小。

日志参数：
- `--verbose, -v`：显示完整运行信息（上下文、高级采样、内存、完整耗时）。
- `--quiet, -q`：抑制非错误日志输出。
- `--quiet` 与 `--verbose` 不能同时使用。

前端参数（仅在启用前端构建时可用，即 `COSYVOICE_NO_FRONTEND=OFF`）：
- `--frontend-only`：仅运行前端，保存 `prompt_speech` 后退出。
- `--speech-tokenizer <file>`：前端 speech tokenizer ONNX 文件。
- `--campplus <file>`：前端 campplus ONNX 文件。
- `--prompt-audio <file>`：前端参考音频。
  - 若使用 `COSYVOICE_NO_AUDIO=ON` 构建，则改用：
    - `--prompt-audio-16k <16k_pcm_file>`：16 kHz float PCM 文件。
    - `--prompt-audio-24k <24k_pcm_file>`：24 kHz float PCM 文件。
- `--prompt-text <text>`：参考音频对应文本。
- `--prompt-speech-output <file>`：将生成的 `prompt_speech` 保存到文件。

提示源相关参数：
- `--prompt-speech <file>`：直接使用已保存的 `prompt_speech` 文件。
- 提示源二选一：
  - 使用 `--prompt-speech`，或
  - 提供前端输入（`--speech-tokenizer`、`--campplus`、音频输入，以及按模式要求是否提供 `--prompt-text`）。
- 同时传入 `--prompt-speech` 和前端输入会报错。
- 典型复用流程：先用 `--frontend-only` 生成并保存 `prompt_speech.gguf`，后续合成时直接用 `--prompt-speech`。

文本规范化：
- `--disable-text-normalization`：关闭分词前 ICU 文本规范化。
- 该选项仅在启用 ICU 时可用（`COSYVOICE_NO_ICU=OFF`）。

运行日志：
- 模型加载前会先打印基础请求信息（模型路径、模式、提示源、输出路径、语速、解析后的 CPU 线程数、seed 来源）。
- 模型加载阶段会显示转圈动画（`| / - \\`）。
- 默认输出为简洁分区样式。
- `--verbose` 显示完整运行细节（含上下文/内存明细与完整阶段耗时）。
- `--quiet` 不显示运行信息和耗时（错误信息仍会输出）。
- 采样参数会在同一行显示来源，例如：`temperature : 1.0000 (model default)`。
- `--verbose` 下内存显示的是生成后值，并附带相对生成前快照的增量（delta）。

必填与可选：
- 常规 TTS 必填：`--model`、`--text`、`--output`，以及一个提示源（`--prompt-speech` 或前端输入）。
- `--frontend-only` 必填：`--speech-tokenizer`、`--campplus`、音频输入、`--prompt-speech-output`。
- 可选参数默认值来自 CLI 或模型元数据：
  - CLI 默认：`--speed=1.0`、`--max-llm-len=2048`、`--threads=0`（硬件并发数）、`--llm-kv-cache-type=q8_0`、`--mode=auto`。
  - 采样默认（`temperature/top_k/top_p/win_size/tau_r/token-text ratio`）来自模型配置，未传参数时不变。

## CLI 快速索引

### 不同场景的必填参数

| 场景 | 必填 |
|---|---|
| 使用已有 prompt_speech 做 TTS | `--model`、`--prompt-speech`、`--text`、`--output` |
| 前端 + TTS 一体流程 | `--model`、`--speech-tokenizer`、`--campplus`、`--prompt-audio`（或 `--prompt-audio-16k` + `--prompt-audio-24k`）、`--text`、`--output` |
| 仅前端 | `--frontend-only`、`--speech-tokenizer`、`--campplus`、音频输入、`--prompt-speech-output` |

### 默认值速查

| 选项 | 默认值 | 来源 |
|---|---|---|
| `--mode` | `auto` | CLI |
| `--speed` | `1.0` | CLI |
| `--max-llm-len` | `2048` | CLI |
| `--threads` | `0`（硬件并发数） | CLI |
| `--llm-kv-cache-type` | `q8_0` | CLI |
| `--seed` | 随机 | 运行时 |
| `temperature/top_k/top_p/win_size/tau_r/min/max_token_text_ratio` | 模型元数据 | 模型 |

### 常用命令模板

```bash
# 复用 prompt_speech
cosyvoice-cli --model model.gguf --prompt-speech prompt_speech.gguf --text "hello" --output out.wav

# 前端 + TTS 一条命令
cosyvoice-cli --model model.gguf --speech-tokenizer speech_tokenizer.onnx --campplus campplus.onnx --prompt-audio ref.wav --prompt-text "参考文本" --text "目标文本" --output out.wav

# 仅前端（预先生成 prompt_speech）
cosyvoice-cli --frontend-only --speech-tokenizer speech_tokenizer.onnx --campplus campplus.onnx --prompt-audio ref.wav --prompt-text "参考文本" --prompt-speech-output prompt_speech.gguf
```

### 参数校验与模式行为

- `--frontend-only` 必需：`--speech-tokenizer`、`--campplus`、音频输入、`--prompt-speech-output`。
- 常规 TTS 必需：`--model`、`--text`、`--output`，以及一个提示源。
- 未提供 `--prompt-speech` 时：
  - 必须提供前端输入；
  - `zero-shot` 模式下必须提供 `--prompt-text`；
  - `instruct`/`cross-lingual` 模式下 `--prompt-text` 会被忽略。
- `--mode` 行为：
  - `auto`：有 `--instruction` 则用 `instruct`，否则用 `zero-shot`。
  - `instruct` 且缺少 `--instruction`：给出警告并回退到 `zero-shot`。
  - `zero-shot` 且提供 `--instruction`：给出警告并忽略 `--instruction`。
  - 传入未知 mode：给出警告并自动判定模式。
- 若未启用前端（`COSYVOICE_NO_FRONTEND=ON`），则必须使用 `--prompt-speech`。
- 在 `--frontend-only` 模式下，`--seed` 会被接受但忽略（会打印警告）。
- 在 `--frontend-only` 模式下，`--threads` 会被接受但忽略（会打印警告）。
- 在 `--frontend-only` 模式下，采样覆盖参数会被接受但忽略（会打印警告）。

## 已知问题
当前生成稳定性与后端关系较大。

已测试现象：
- **CUDA 后端（Windows + Linux）：**
  - 当前测试中 CUDA 稳定性问题已解决。
  - Windows CUDA 与 Linux CUDA 均可正常运行。
- **CPU / Vulkan：**
  - 当前测试中仍无法正常运行（例如噪音/输出异常）。

补充说明：
- 目前仅在 Ada Lovelace 架构显卡上做过验证。
- 其他后端尚未测试。

## 故障排查
- CMake 找不到 GGML：设置 `-DGGML_SOURCE_DIR=...`，或使用默认 `vendor/ggml` 并确保本机可用 Git（用于自动克隆）。
- ICU/ONNX Runtime 检测失败：可安装系统包（适用平台），或将预编译文件放到 `<build_dir>/_deps/icu` 与 `<build_dir>/_deps/onnxruntime`。
- Windows 运行时缺库：检查 `build/bin` 下是否存在构建后复制的依赖 DLL。
- CPU/Vulkan 输出异常或有噪音：当前属于已知问题，建议优先使用 CUDA 后端。

## 欢迎贡献
欢迎提交 Issue 和 Pull Request，尤其是：
- 后端稳定性修复
- 跨平台正确性改进
- 性能与内存优化
- 文档与工具改进

如果根因在 GGML，请优先向上游 GGML 提交修复补丁。
