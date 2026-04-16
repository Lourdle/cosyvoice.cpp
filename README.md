# CosyVoice.cpp

Language: [中文](README_zh.md)

> Unofficial project notice: this repository is **not** affiliated with, endorsed by, or maintained by the official CosyVoice team. It is a community-maintained C++/GGML port created by an independent developer.

> **Current status notice:** CUDA stability issues are resolved in current tests, and both Windows and Linux CUDA runs can generate normal audio. CPU and Vulkan backends are still not running normally. Please review [Known Issues](#known-issues) before production use.

C++/GGML port of the Python CosyVoice inference pipeline released by the original CosyVoice project, currently focused on **CosyVoice3**.

This repository ships independent engineering work and does not contain official support commitments.

This project provides:
- A core C/C++ inference library (`cosyvoice`)
- A CLI synthesis tool (`cosyvoice-cli`)
- A GGUF quantization tool (`quantize`)

## Contents
- [Documentation](#documentation)
- [AI Usage Disclosure](#ai-usage-disclosure)
- [Quick Start](#quick-start)
- [Inference Pipeline](#inference-pipeline)
- [Build](#build)
- [Dependency Resolution](#dependency-resolution)
- [CMake Options](#cmake-options)
- [Build Matrix (Typical)](#build-matrix-typical)
- [GGML Backend/Build Options](#ggml-backendbuild-options)
- [Using Custom Dependencies](#using-custom-dependencies)
- [Model Conversion to GGUF](#model-conversion-to-gguf)
- [Quantization Tool (`tools/quantize`)](#quantization-tool-toolsquantize)
- [CLI Tool (`tools/cli`)](#cli-tool-toolscli)
- [CLI Quick Reference](#cli-quick-reference)
- [Known Issues](#known-issues)
- [Troubleshooting](#troubleshooting)
- [Third-Party Notices](#third-party-notices)
- [Licensing](#licensing)
- [Contributing](#contributing)

## Documentation
- API index: [docs/API.md](docs/API.md)

## AI Usage Disclosure
- Most project code is written by the author.
- Parts of the documentation are drafted and edited with AI assistance.
- Documentation may contain mistakes or lag behind implementation details; when in doubt, treat source code and header files as the ground truth, and feel free to open an issue/PR.

## Third-Party Notices
- See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for bundled dependency license details.
- Tokenizer implementation is adapted from llama.cpp (MIT).

## Licensing
- **Repository code**: MIT (`LICENSE`).
- **Upstream reference**: the original CosyVoice project code and models are under Apache-2.0.
- **Implementation note**: this repository is an independent C++/GGML re-implementation based on model architecture and inference behavior, and is not an official fork or release.
- **GGUF model artifacts**: published model files remain under Apache-2.0.
  - ModelScope: <https://modelscope.cn/models/Lourdle/Fun-CosyVoice3-0.5B-2512-GGUF>
  - Hugging Face: <https://huggingface.co/Lourdle/Fun-CosyVoice3-0.5B-2512-GGUF>
- **Model license file**: [MODEL_LICENSE.md](MODEL_LICENSE.md)

## Quick Start
1. Convert upstream CosyVoice model weights to GGUF (via this repository's `convert_model_to_gguf.py`).
2. Configure and build this project.
3. (Optional) Quantize the GGUF model with `quantize`.
4. Run `cosyvoice-cli` for synthesis.

## Inference Pipeline
This project supports two equivalent inference paths:

1. **End-to-end frontend + TTS (recommended for first run)**
   - Input: reference audio (and reference text when required by mode) + target text
   - Flow: frontend extracts `prompt_speech` -> TTS runs with `prompt_speech` and target text
   - Mode notes:
     - `zero-shot` requires `--prompt-text`
     - `instruct` and `cross-lingual` ignore `--prompt-text`

2. **Reuse saved `prompt_speech` (recommended for batch/repeated synthesis)**
   - Save `prompt_speech` as `.gguf` first (for example via `--prompt-speech-output` or API `cosyvoice_prompt_speech_save_to_file`)
   - Later runs can pass `--prompt-speech <file>` directly, so frontend ONNX does **not** need to run again

In short: encode reference conditions (voice/speaker traits) into `prompt_speech` first, then combine `prompt_speech` with target text to generate audio.

## Build

### Requirements
- CMake >= 3.24
- C/C++ toolchain with C++20 support
- Git (used to fetch GGML automatically when missing)
- x86 CPU with AVX2 support is currently required for parts of the CPU data path
- For CPU math-heavy paths (for example `log` and trigonometric functions), SIMD acceleration is currently enabled only in MSVC builds; other toolchains currently fall back to scalar implementations

Backend/runtime requirements depend on your build options (CUDA/Vulkan/CPU, ONNX Runtime, ICU, etc.).

### 1) Configure
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
```

### 2) Build
```bash
cmake --build build --config Release
```

Build outputs are placed in:
- `build/bin` (executables/runtime DLLs)
- `build/lib` (libraries)

## Dependency Resolution
The top-level CMake project handles dependencies as follows:

- **PCRE2**
  - Built from `vendor/pcre2` as static libraries (`pcre2-8`, `pcre2-16`).
- **GGML**
  - Uses `GGML_SOURCE_DIR` (default: `vendor/ggml`).
  - If missing, CMake clones `https://github.com/ggml-org/ggml.git` automatically.
- **ICU** (used by text normalization unless disabled)
  - Controlled by `COSYVOICE_NO_ICU`.
  - If `ICU_PREBUILT_DIR` is available, uses it directly.
  - Otherwise tries `find_package(ICU)`.
  - On Windows, if still not found, prebuilt ICU is downloaded automatically.
  - On Linux/macOS, install system ICU if still not found.
- **ONNX Runtime** (used by frontend unless disabled)
  - Controlled by `COSYVOICE_NO_FRONTEND`.
  - If `ORT_PREBUILT_DIR` is available, uses it directly.
  - Otherwise tries `find_package(onnxruntime)`.
  - If still not found, prebuilt ONNX Runtime is downloaded automatically.

Useful dependency path cache variables:
- `GGML_SOURCE_DIR`
- `ICU_PREBUILT_DIR`
- `ORT_PREBUILT_DIR`

Default values:
- `GGML_SOURCE_DIR=vendor/ggml`
- `ICU_PREBUILT_DIR=<build_dir>/_deps/icu`
- `ORT_PREBUILT_DIR=<build_dir>/_deps/onnxruntime`

Dependency priority (effective order):
- **GGML**: `GGML_SOURCE_DIR` -> auto clone GGML repository if missing.
- **ICU**: use `ICU_PREBUILT_DIR` if available -> `find_package(ICU)` -> (Windows) auto download prebuilt ICU -> (Linux/macOS) install system ICU manually.
- **ONNX Runtime**: use `ORT_PREBUILT_DIR` if available -> `find_package(onnxruntime)` -> auto download prebuilt ONNX Runtime.

Platform notes:
- **Windows**: prebuilt dependency DLLs are copied next to executables after build.
- **Linux/macOS**: prebuilt shared libraries are installed under library install directories.

## CMake Options
Project-level options:
- `BUILD_SHARED_LIBS=ON/OFF` (default: `ON`)
- `COSYVOICE_NO_AUDIO=ON/OFF` (default: `OFF`)
- `COSYVOICE_NO_FRONTEND=ON/OFF` (default: `OFF`)
- `COSYVOICE_NO_ICU=ON/OFF` (default: `OFF`)

Dependency path options:
- `GGML_SOURCE_DIR=<path>`
- `ICU_PREBUILT_DIR=<path>`
- `ORT_PREBUILT_DIR=<path>`

GGML backend options are passed through from GGML CMake (for example `GGML_CUDA`, `GGML_VULKAN`, etc.).

## Build Matrix (Typical)
| Scenario | Recommended CMake flags |
|---|---|
| CUDA backend | `-DGGML_CUDA=ON` |
| Vulkan backend | `-DGGML_VULKAN=ON` |
| CPU-only | no backend flag required |
| Core-only (no frontend / ICU) | `-DCOSYVOICE_NO_FRONTEND=ON -DCOSYVOICE_NO_ICU=ON` |
| No-audio helper API | `-DCOSYVOICE_NO_AUDIO=ON` |

## GGML Backend/Build Options
This project vendors/uses GGML through CMake, so GGML backend switches can be passed from this root build.

Typical examples (refer to `llama.cpp` / GGML [docs](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md) for backend-specific options and recommended settings):
```bash
# CUDA example
cmake -S . -B build-cuda -DGGML_CUDA=ON
```

Project options:
- `COSYVOICE_NO_AUDIO=ON/OFF` (disable/enable audio helper APIs)
- `COSYVOICE_NO_FRONTEND=ON/OFF` (disable/enable ONNX frontend)
- `COSYVOICE_NO_ICU=ON/OFF` (disable/enable ICU text normalization)
- `BUILD_SHARED_LIBS=ON/OFF`

Practical combinations:
```bash
# Core-only build (no ONNX frontend, no ICU text norm)
cmake -S . -B build-core -DCMAKE_BUILD_TYPE=Release -DCOSYVOICE_NO_FRONTEND=ON -DCOSYVOICE_NO_ICU=ON

# No-audio build (CLI output forced to WAV fallback path)
cmake -S . -B build-noaudio -DCMAKE_BUILD_TYPE=Release -DCOSYVOICE_NO_AUDIO=ON
```

## Using Custom Dependencies
You can point CMake to custom dependency locations with cache variables:

```bash
cmake -S . -B build \
  -DGGML_SOURCE_DIR=/path/to/ggml \
  -DICU_PREBUILT_DIR=/path/to/icu \
  -DORT_PREBUILT_DIR=/path/to/onnxruntime
```

You can also use the default prebuilt locations under your build directory:
- `<build_dir>/_deps/icu`
- `<build_dir>/_deps/onnxruntime`

If you place files there with the expected layout, CMake will pick them up automatically (without extra `-D` flags).

Expected markers/layout:
- ICU: `include/unicode/utypes.h` (and platform libs/dlls under `lib*` / `bin*`)
- ONNX Runtime: `include/onnxruntime_c_api.h` and runtime library files under `lib`

Notes:
- If `GGML_SOURCE_DIR` does not contain GGML sources, CMake will try to clone GGML.
- If ICU/ONNX Runtime are not found by `find_package`, CMake will use/download prebuilt binaries into the configured prebuilt directories.
- On Windows, prebuilt DLLs are copied next to built executables for local running.

## Model Conversion to GGUF
Use this repository's conversion script (`convert_model_to_gguf.py`) to convert upstream CosyVoice model weights to GGUF for `cosyvoice.cpp`.

Install Python dependencies first:
```bash
pip install -r requirements.txt
```

Minimal usage:
```bash
python convert_model_to_gguf.py \
  --yaml_config /path/to/cosyvoice.yaml \
  --ftype f16 \
  --gguf_model /path/to/CosyVoice3-2512_F16.gguf
```

Full example:
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

`--ftype` options:
- `default`, `f32`, `f16`, `q8_0`, `q5_0`, `q5_1`, `q4_0`, `q4_1`

Default path behavior (when not explicitly provided):
- `--llm_model` -> `<yaml_dir>/llm.pt`
- `--blank_llm` -> `<yaml_dir>/CosyVoice-BlankEN`
- `--flow_model` -> `<yaml_dir>/flow.pt`
- `--hift_model` -> `<yaml_dir>/hift.pt`

After conversion:
1. Verify the generated `.gguf` file.
2. (Optional) Quantize it with this repository's `quantize` tool.

## Quantization Tool (`tools/quantize`)
Executable name: `quantize`

Basic usage:
```bash
quantize -f input.gguf -o output-q4k.gguf -t Q4_K
```

Show help:
```bash
quantize --help
```

Supported quantization types:
- `F16`, `Q8_0`, `Q5_0`, `Q5_1`, `Q4_0`, `Q4_1`
- `Q6_K`, `Q5_K`, `Q4_K`, `Q3_K`, `Q2_K`
- `COPY`

Custom metadata strings are supported with repeated `-c/--custom-string`.

## CLI Tool (`tools/cli`)
Executable name: `cosyvoice-cli`

Show help:
```bash
cosyvoice-cli --help
```

### A) TTS from an existing prompt_speech file
```bash
cosyvoice-cli \
  --model model.gguf \
  --prompt-speech prompt_speech.gguf \
  --text "Hello from CosyVoice" \
  --output out.wav
```

### B) End-to-end frontend + TTS (extract prompt_speech on the fly)
```bash
cosyvoice-cli \
  --model model.gguf \
  --speech-tokenizer speech_tokenizer.onnx \
  --campplus campplus.onnx \
  --prompt-audio ref.wav \
  --prompt-text "reference transcript" \
  --text "target text" \
  --output out.wav
```

### C) Frontend-only mode (generate prompt_speech and exit)
```bash
cosyvoice-cli \
  --frontend-only \
  --speech-tokenizer speech_tokenizer.onnx \
  --campplus campplus.onnx \
  --prompt-audio ref.wav \
  --prompt-text "reference transcript" \
  --prompt-speech-output prompt_speech.gguf
```

### Option Reference

Core options:
- `--help, -h`: Show help message and exit.
- `--model, -m <file>`: CosyVoice model file (`.gguf`) used for TTS.
- `--text, -t <text>`: Text to synthesize.
- `--output, -o <file>`: Output audio path.
  - Normal build: format is inferred from file extension.
  - `COSYVOICE_NO_AUDIO=ON`: output is always WAV.
- `--speed, -s <value>`: Speech speed multiplier. Default: `1.0`. Must be `> 0`.
- `--seed <value>`: Random seed for sampling and internal noise generation. Must be an unsigned 32-bit integer. Default: random.
- `--max-llm-len <value>`: Maximum input token count for LLM (`n_max_seq`). Default: `2048`. Must be a positive integer.
- `--threads, -j <value>`: CPU thread count for model inference. Must be an unsigned 32-bit integer. Default: `0` (use current hardware concurrency).
- `--llm-kv-cache-type <f32|f16|q8_0|q5_1|q5_0|q4_1|q4_0>`: LLM KV cache type. Default: `q8_0`.
- `--mode <zero-shot|instruct|cross-lingual>`: TTS mode. Default: auto-detect from `--instruction`.
- `--instruction, -i <text>`: Instruction text for instruct mode.

Sampling override options:
- `--temperature <value>`: Sampling temperature, must be `> 0`.
- `--top-k <value>`: Top-k sampling size, must be `>= 0`.
- `--top-p <value>`: Top-p sampling threshold, must be in `[0, 1]`.
- `--win-size <value>`: Repetition window size, must be `> 0`.
- `--tau-r <value>`: Repetition penalty coefficient, must be `>= 0`.
- `--min-token-text-ratio <value>`: Minimum generated token/text ratio, must be `>= 0`.
- `--max-token-text-ratio <value>`: Maximum generated token/text ratio, must be `>= 0` and not less than `--min-token-text-ratio` when both are set.

Log options:
- `--verbose, -v`: Show detailed runtime sections (context, advanced sampling, memory, full timings).
- `--quiet, -q`: Suppress non-error logs.
- `--quiet` and `--verbose` cannot be used together.

Frontend options (available when frontend is compiled, i.e. `COSYVOICE_NO_FRONTEND=OFF`):
- `--frontend-only`: Run frontend only, save `prompt_speech`, then exit.
- `--speech-tokenizer <file>`: Frontend speech tokenizer ONNX file.
- `--campplus <file>`: Frontend campplus ONNX file.
- `--prompt-audio <file>`: Reference audio file for frontend.
  - If built with `COSYVOICE_NO_AUDIO=ON`, use:
    - `--prompt-audio-16k <16k_pcm_file>`: 16 kHz float PCM file.
    - `--prompt-audio-24k <24k_pcm_file>`: 24 kHz float PCM file.
- `--prompt-text <text>`: Transcript of the reference audio.
- `--prompt-speech-output <file>`: Save generated `prompt_speech` to file.

Prompt source options:
- `--prompt-speech <file>`: Use a saved `prompt_speech` file.
- Choose exactly one source:
  - a saved `--prompt-speech`, or
  - frontend inputs (`--speech-tokenizer`, `--campplus`, audio input, optional/required `--prompt-text` depending on mode).
- Using `--prompt-speech` and frontend inputs together is rejected.
- Typical reuse workflow: generate and save `prompt_speech.gguf` with `--frontend-only`, then run future synthesis with `--prompt-speech` directly.

Text normalization:
- `--disable-text-normalization`: Disable ICU text normalization before tokenization.
- This option exists only when ICU is enabled (`COSYVOICE_NO_ICU=OFF`).

Runtime logs:
- Basic request info (model path, mode, prompt source, output, speed, resolved CPU thread count, seed source) is shown before model loading.
- During model loading, a spinner (`| / - \\`) is shown in the console.
- Default output is concise and formatted with sections.
- `--verbose` shows full runtime details, including context/memory breakdown and full timing stages.
- `--quiet` suppresses runtime info and timings (errors still print).
- Sampling lines include source directly inline, e.g. `temperature : 1.0000 (model default)`.
- Memory in `--verbose` shows post-generation values with deltas against pre-generation snapshot.

Required vs optional:
- Required for normal TTS: `--model`, `--text`, `--output`, and one prompt source (`--prompt-speech` OR frontend inputs).
- Required for `--frontend-only`: `--speech-tokenizer`, `--campplus`, audio input, `--prompt-speech-output`.
- Optional parameters use defaults from CLI or model metadata:
  - CLI defaults: `--speed=1.0`, `--max-llm-len=2048`, `--threads=0` (hardware concurrency), `--llm-kv-cache-type=q8_0`, `--mode=auto`.
  - Sampling defaults (`temperature`, `top_k`, `top_p`, `win_size`, `tau_r`, token/text ratios) come from model config unless overridden by CLI.

## CLI Quick Reference

### Required arguments by scenario

| Scenario | Required |
|---|---|
| TTS with existing prompt_speech | `--model`, `--prompt-speech`, `--text`, `--output` |
| End-to-end frontend + TTS | `--model`, `--speech-tokenizer`, `--campplus`, `--prompt-audio` (or `--prompt-audio-16k` + `--prompt-audio-24k`), `--text`, `--output` |
| Frontend-only | `--frontend-only`, `--speech-tokenizer`, `--campplus`, audio input, `--prompt-speech-output` |

### Defaults at a glance

| Option | Default | Source |
|---|---|---|
| `--mode` | `auto` | CLI |
| `--speed` | `1.0` | CLI |
| `--max-llm-len` | `2048` | CLI |
| `--threads` | `0` (hardware concurrency) | CLI |
| `--llm-kv-cache-type` | `q8_0` | CLI |
| `--seed` | random | runtime |
| `temperature`, `top_k`, `top_p`, `win_size`, `tau_r`, `min/max_token_text_ratio` | model metadata | model |

### Typical command templates

```bash
# Reuse prompt_speech
cosyvoice-cli --model model.gguf --prompt-speech prompt_speech.gguf --text "hello" --output out.wav

# Frontend + TTS in one command
cosyvoice-cli --model model.gguf --speech-tokenizer speech_tokenizer.onnx --campplus campplus.onnx --prompt-audio ref.wav --prompt-text "ref text" --text "target" --output out.wav

# Frontend-only (prepare prompt_speech for later reuse)
cosyvoice-cli --frontend-only --speech-tokenizer speech_tokenizer.onnx --campplus campplus.onnx --prompt-audio ref.wav --prompt-text "ref text" --prompt-speech-output prompt_speech.gguf
```

### Validation and Mode Behavior

- `--frontend-only` requires: `--speech-tokenizer`, `--campplus`, audio input, and `--prompt-speech-output`.
- Normal TTS requires: `--model`, `--text`, `--output`, and one prompt source.
- If `--prompt-speech` is not provided:
  - frontend inputs are required;
  - `--prompt-text` is required in `zero-shot` mode;
  - `--prompt-text` is ignored in `instruct` and `cross-lingual` modes.
- `--mode` behavior:
  - `auto`: resolves to `instruct` when `--instruction` is provided, otherwise `zero-shot`.
  - `instruct` without `--instruction`: warning, then fallback to `zero-shot`.
  - `zero-shot` with `--instruction`: warning, and `--instruction` is ignored.
  - unrecognized mode value: warning, then auto-detect.
- If frontend is not available (`COSYVOICE_NO_FRONTEND=ON`), `--prompt-speech` is mandatory.
- In `--frontend-only` mode, `--seed` is accepted but ignored (warning will be printed).
- In `--frontend-only` mode, `--threads` is accepted but ignored (warning will be printed).
- In `--frontend-only` mode, sampling override options are accepted but ignored (warning will be printed).

## Known Issues
Current generation stability is backend-dependent.

Tested observations:
- **CUDA backend (Windows + Linux):**
  - CUDA stability issues are resolved in current tests.
  - Windows CUDA and Linux CUDA runs are both normal.
- **CPU / Vulkan backends:**
  - Still cannot run normally in current tests (for example noisy/incorrect output).

Additional note:
- Tests were performed on Ada Lovelace GPUs only.
- Other backends are not tested yet.

## Troubleshooting
- CMake cannot find GGML: set `-DGGML_SOURCE_DIR=...` or keep default `vendor/ggml` and ensure Git is available for auto-clone.
- ICU/ONNX Runtime detection issues: either install system packages (where applicable) or place prebuilt files into `<build_dir>/_deps/icu` and `<build_dir>/_deps/onnxruntime`.
- Executable starts but misses runtime libraries on Windows: ensure post-build copied DLLs exist next to binaries in `build/bin`.
- CPU/Vulkan output is noisy or incorrect: this is currently a known issue; use CUDA backend when possible.

## Contributing
Contributions are welcome.

Please feel free to open issues or submit pull requests for:
- Backend stability fixes
- Cross-platform correctness improvements
- Performance and memory optimizations
- Documentation/tooling improvements

If the root cause is in GGML, please submit fixes/patches upstream to GGML.
