# 从源码构建

## 环境要求
- CMake >= 3.28
- 支持 C++20 的 C/C++ 编译器
- Git（当本地缺少 GGML 源码时用于自动拉取）
- 不要求特定的 x86 指令集：CPU DSP 路径通过运行时分派自动选择 CPU 支持的最快层级，最低可回退到纯 x86-64 标量基线（见 [SIMD_zh.md](SIMD_zh.md)）
- SIMD 数学路径（如 `log`、三角函数）在所有受支持的工具链上均按层级编译（MSVC `/arch:...`、GCC/Clang `-m...`）；非 x86 平台可选 SIMDe 模拟 SSE4.2+FMA3 类，否则回退为标量实现

后端/运行时依赖会随构建选项变化（CUDA/Vulkan/CPU、ONNX Runtime、ICU 等）。

## 基本构建

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

构建产物默认输出到：
- `build/bin`（可执行文件与运行时 DLL）
- `build/lib`（库文件）

## Server 构建须知（非 Windows 平台）

在 Linux/macOS 上编译 `cosyvoice-server` 需在以下两方面额外留意：

**C23 `#embed` 嵌入 WebUI 资源**
WebUI 资源嵌入通过 C23 `#embed` 指令将 HTML/CSS/JS 打包到可执行文件中，因此需要 **C 编译器**支持 C23——即 GCC 15+ 或 Clang 19+。Windows 通过原生 RC 工具嵌入资源，无需特殊编译器。

示例——指定支持 C23 的 C 编译器：
```bash
# Ubuntu/Debian — 用 clang-20 作为 C 编译器
sudo apt install clang-20
cmake -B build -DCMAKE_C_COMPILER=clang-20 -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

**C++20 模块**
服务端使用了 C++20 模块接口（封装 nlohmann/json 和 cpp-httplib）。**建议**使用 **Ninja** 生成器（1.11+）配合较新的 C++ 编译器（GCC 14+ / Clang 16+ / MSVC 14.34+）以获得完整的模块扫描支持：
- 在 cmake 配置时添加 `-G Ninja`。
- Windows：Visual Studio 生成器完整支持模块扫描——无需额外参数。
- 不支持的生成器或旧版编译器：CMake 会自动检测并**回退到预编译头（PCH）**。

Linux/macOS 推荐配置：
```bash
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

## CMake 选项

### 项目选项

| 选项 | 取值 | 默认值 | 说明 |
|---|---|---|---|
| `BUILD_SHARED_LIBS` | ON / OFF | ON | 构建为共享库 |
| `COSYVOICE_NO_AUDIO` | ON / OFF | OFF | 关闭音频辅助 API |
| `COSYVOICE_NO_FRONTEND` | ON / OFF | OFF | 关闭 ONNX 前端 |
| `COSYVOICE_NO_ICU` | ON / OFF | OFF | 关闭 ICU 文本规范化 |
| `COSYVOICE_AUDIO_BACKEND` | MINIAUDIO / FFMPEG | MINIAUDIO | 音频编解码后端 |
| `COSYVOICE_CLI_NO_PLAYBACK` | ON / OFF | 未设置（跟随 `COSYVOICE_NO_AUDIO`） | 关闭 CLI 播放功能 |
| `COSYVOICE_SERVER_NO_WEBUI` | ON / OFF | OFF | 关闭嵌入式 WebUI，仅以 API 模式运行；非 Windows 平台可移除 C23 `#embed` 依赖 |
| `COSYVOICE_SERVER_DEFAULT_MODE` | WEBUI / API | WEBUI | 命令行未指定 `--api` 或 `--webui` 时的默认模式。设为 `API` 可用于纯后端部署 |

### 后端选项

GGML 后端选项可直接从根工程透传。常见示例：

```bash
# CUDA 后端
cmake -B build -DGGML_CUDA=ON

# Vulkan 后端
cmake -B build -DGGML_VULKAN=ON
```

完整后端选项列表及推荐配置请参考 [GGML 文档](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)。

### Metal 后端（`GGML_METAL`）

- `GGML_METAL` 在 Apple Silicon 上**默认为 ON**（见 ggml 自身 CMakeLists），也可用 `-DGGML_METAL=ON/OFF` 强制指定。
- GGML 本身对 Metal 无需任何特殊处理：所有平台统一浅克隆（`--depth=1`）最新 master，且不应用任何补丁。此前需要打补丁的 Metal PAD beg-padding 支持已进入上游 [ggml-org/llama.cpp#29561](https://github.com/ggml-org/llama.cpp/pull/29561)（ggml 提交 `46fc5b3b`），该提交还一并加入了 circular 填充与 permute 源的支持。
- 若已有的 `vendor/ggml` 早于该提交，CMake 会发出警告（不中断），因为它的 Metal PAD 内核会导致推理中止——删掉 `vendor/ggml` 并重新运行 CMake 即可拉取新版本。

```bash
# 强制开启/关闭 Metal（Apple Silicon 默认开启）
cmake -B build -DGGML_METAL=ON
```

### 依赖路径选项

| 选项 | 说明 |
|---|---|
| `GGML_SOURCE_DIR=<path>` | GGML 源码路径（默认 `vendor/ggml`），目录不存在时自动克隆 |
| `ICU_PREBUILT_DIR=<path>` | ICU 预编译二进制路径（默认 `<build_dir>/_deps/icu`） |
| `ORT_PREBUILT_DIR=<path>` | ONNX Runtime 预编译二进制路径（默认 `<build_dir>/_deps/onnxruntime`） |
| `FFMPEG_PREBUILT_DIR=<path>` | FFmpeg 预编译二进制路径 |
| `SIMDE_INCLUDE_DIR=<path>` | ARM64/aarch64（含 Android 交叉编译）的 SIMDe 头文件目录（可选，未提供时自动降级为纯标量）——见 [SIMDe（SIMD Everywhere）](#simdesimd-everywhere) |

### 链接时优化（LTO）

默认启用（`CMAKE_INTERPROCEDURAL_OPTIMIZATION=TRUE`）。通过折叠完全相同的 SIMD tier 函数体和内联跨 TU 小函数，减少约 3% 二进制体积。已在 MSVC、GCC、Clang 上验证安全——各编译单元的 `/arch` 标记在 LTO codegen 中保持隔离。

```bash
# 需要更快迭代时可禁用
cmake -B build -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF
```

### 常见构建矩阵

| 场景 | 推荐 CMake 参数 |
|---|---|
| CUDA 后端 | `-DGGML_CUDA=ON` |
| Vulkan 后端 | `-DGGML_VULKAN=ON` |
| 仅 CPU | 通常不需要额外后端参数 |
| 仅核心能力（无 frontend / ICU） | `-DCOSYVOICE_NO_FRONTEND=ON -DCOSYVOICE_NO_ICU=ON` |
| 关闭音频辅助 API | `-DCOSYVOICE_NO_AUDIO=ON` |
| 关闭 CLI 播放功能 | `-DCOSYVOICE_CLI_NO_PLAYBACK=ON` |

常见组合示例：

```bash
# 仅核心功能构建（关闭 ONNX 前端与 ICU 文本规范化）
cmake -B build-core -DCMAKE_BUILD_TYPE=Release -DCOSYVOICE_NO_FRONTEND=ON -DCOSYVOICE_NO_ICU=ON

# 无音频辅助 API 构建（CLI 走 WAV 输出回退路径）
cmake -B build-noaudio -DCMAKE_BUILD_TYPE=Release -DCOSYVOICE_NO_AUDIO=ON

# 关闭 CLI 播放功能（音频辅助 API 仍可用）
cmake -B build-noplay -DCMAKE_BUILD_TYPE=Release -DCOSYVOICE_CLI_NO_PLAYBACK=ON
```

## 依赖解析方式

顶层 CMake 按以下顺序解析依赖：

- **PCRE2**：从 `vendor/pcre2` 构建静态库（`pcre2-8`、`pcre2-16`）。
- **GGML**：使用 `GGML_SOURCE_DIR`（默认 `vendor/ggml`）。若目录不存在，会自动克隆 `https://github.com/ggml-org/ggml.git`。
- **ICU**（用于文本规范化，除非通过 `COSYVOICE_NO_ICU` 关闭）：解析顺序：`ICU_PREBUILT_DIR` → `find_package(ICU)` → Windows 自动下载 → Linux/macOS 使用系统 ICU。
- **ONNX Runtime**（用于前端，除非通过 `COSYVOICE_NO_FRONTEND` 关闭）：解析顺序：`ORT_PREBUILT_DIR` → `find_package(onnxruntime)` → 自动下载。

Windows 下预编译依赖 DLL 会复制到可执行文件旁。

## 使用自定义依赖

可以通过缓存变量指定自定义依赖路径：

```bash
cmake -B build \
  -DGGML_SOURCE_DIR=/path/to/ggml \
  -DICU_PREBUILT_DIR=/path/to/icu \
  -DORT_PREBUILT_DIR=/path/to/onnxruntime \
  -DSIMDE_INCLUDE_DIR=/path/to/simde
```

也可以直接使用构建目录下的默认预编译依赖位置：
- `<build_dir>/_deps/icu`
- `<build_dir>/_deps/onnxruntime`

只要按期望目录结构把文件放进去，CMake 会自动识别（不需要额外 `-D`）。

期望的关键目录/文件：
- ICU：`include/unicode/utypes.h`（以及 `lib*` / `bin*` 下的库和 DLL）
- ONNX Runtime：`include/onnxruntime_c_api.h`（以及 `lib` 下的运行库文件）

## SIMDe（SIMD Everywhere）

[SIMD_zh.md](SIMD_zh.md) 描述的各层级代码按 x86 内联函数编写。[SIMDe](https://github.com/simd-everywhere/simde) 是一个纯头文件库，可将这些 x86 内联函数翻译到其他指令集——在 ARM64/aarch64 上，模拟的 SSE4.2+FMA3 类代码无需修改即可编译为 NEON。

CMake 的处理方式（`CMakeLists.txt`）：

- **x86_64**：完全不需要 SIMDe——每个层级都在自己的指令集上原生执行。
- **ARM64/aarch64（含 Android 交叉编译）**：SIMDe 为**可选**。CMake 会在 `/opt/homebrew/include`、`/usr/local/include`、`/usr/include`、`vendor/simde/`（或 `-DSIMDE_INCLUDE_DIR=<path>`）中查找 `simde/x86/sse4.2.h`：找到时经 NEON 模拟 SSE4.2+FMA3 类（此时标量层甚至不会被编译——非 x86 的分派总是直接选择模拟类）；找不到时自动降级为纯标量内核。

获取 SIMDe：

```bash
# macOS
brew install simde

# Debian/Ubuntu（部分发行版可能没有该包）
apt install libsimde-dev

# 通用方式——克隆并用 SIMDE_INCLUDE_DIR 指向
git clone --depth=1 https://github.com/simd-everywhere/simde.git
cmake -B build -DSIMDE_INCLUDE_DIR=/path/to/simde
```

Android 相关细节见 [docs/build-android_zh.md](build-android_zh.md)。

## 音频后端与 FFmpeg

本项目的音频辅助 API 支持两种后端：

- `MINIAUDIO`（默认）：提供 WAV I/O 与基本 PCM 帮助函数。
- `FFMPEG`（可选）：在链接的 FFmpeg 运行时提供所需编码器时，启用更多编码/解码格式。

通过 CMake 配置音频后端：将 `COSYVOICE_AUDIO_BACKEND` 设为 `MINIAUDIO` 或 `FFMPEG`。默认值为 `MINIAUDIO`。

示例：
```bash
cmake -B build -DCOSYVOICE_AUDIO_BACKEND=MINIAUDIO
cmake -B build -DCOSYVOICE_AUDIO_BACKEND=FFMPEG
cmake -B build -DCOSYVOICE_AUDIO_BACKEND=FFMPEG -DFFMPEG_PREBUILT_DIR=/path/to/ffmpeg
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

- 本仓库代码采用 MIT 许可。FFmpeg 预编译包可能是 LGPL 或 GPL，取决于编译选项。使用包含 GPL 编码器的 FFmpeg 构建并重新分发时，可能会对你的发行物带来 GPL 约束。详见 [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md#ffmpeg-optional-audio-backend)。

## 故障排查

- CMake 找不到 GGML：设置 `-DGGML_SOURCE_DIR=...`，或使用默认 `vendor/ggml` 并确保本机可用 Git（用于自动克隆）。
- ICU/ONNX Runtime 检测失败：可安装系统包（适用平台），或将预编译文件放到 `<build_dir>/_deps/icu` 与 `<build_dir>/_deps/onnxruntime`。
- Windows 运行时缺库：检查 `build/bin` 下是否存在构建后复制的依赖 DLL。
- 后端相关情况见 [README_zh.md — 后端测试情况](../README_zh.md#后端测试情况)。
