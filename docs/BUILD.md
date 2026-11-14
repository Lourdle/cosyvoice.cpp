# Building from Source

## Requirements
- CMake >= 3.28
- C/C++ toolchain with C++20 support
- Git (used to fetch GGML automatically when missing)
- No specific x86 ISA is required: CPU DSP paths use runtime dispatch and automatically fall back to the best tier the CPU supports (down to the plain x86-64 scalar baseline) — see [SIMD.md](SIMD.md)
- SIMD math paths (for example `log` and trigonometric functions) are compiled per tier on all supported toolchains (MSVC `/arch:...`, GCC/Clang `-m...`); non-x86 targets optionally emulate the SSE4.2+FMA3 class via SIMDe and otherwise fall back to scalar

Backend/runtime requirements depend on your build options (CUDA/Vulkan/CPU, ONNX Runtime, ICU, etc.).

## Basic Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

Build outputs are placed in:
- `build/bin` (executables/runtime DLLs)
- `build/lib` (libraries)

## Server Build Considerations (non-Windows)

On Linux/macOS, `cosyvoice-server` requires extra attention in two areas:

**C23 `#embed` for WebUI resources**
The WebUI resource embedding uses the C23 `#embed` directive to bundle HTML/CSS/JS into the executable, which needs a **C compiler** that supports C23 — GCC 15+ or Clang 19+. Windows embeds resources via its native RC tool, so no special C compiler is needed.

Example — specifying a C23-capable C compiler:
```bash
# Ubuntu/Debian — use clang-20 as the C compiler
sudo apt install clang-20
cmake -B build -DCMAKE_C_COMPILER=clang-20 -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

**C++20 modules**
The server uses C++20 module interfaces (for nlohmann-json and cpp-httplib). A **Ninja** generator (1.11+) with a modern C++ compiler (GCC 14+ / Clang 16+ / MSVC 14.34+) is **recommended** for full module scanning:
- Add `-G Ninja` to your cmake configure command.
- Windows: the Visual Studio generator fully supports module scanning — no extra flags.
- Unsupported generators or older compilers: CMake detects the gap and automatically **falls back to precompiled headers (PCH)**.

Recommended full configure for Linux/macOS:
```bash
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

## CMake Options

### Project Options

| Option | Values | Default | Description |
|---|---|---|---|
| `BUILD_SHARED_LIBS` | ON / OFF | ON | Build cosyvoice as a shared library |
| `COSYVOICE_NO_AUDIO` | ON / OFF | OFF | Disable audio helper APIs |
| `COSYVOICE_NO_FRONTEND` | ON / OFF | OFF | Disable ONNX frontend |
| `COSYVOICE_NO_ICU` | ON / OFF | OFF | Disable ICU text normalization |
| `COSYVOICE_AUDIO_BACKEND` | MINIAUDIO / FFMPEG | MINIAUDIO | Audio encoding/decoding backend |
| `COSYVOICE_CLI_NO_PLAYBACK` | ON / OFF | unset (follows `COSYVOICE_NO_AUDIO`) | Disable CLI playback |
| `COSYVOICE_SERVER_NO_WEBUI` | ON / OFF | OFF | Disable embedded WebUI; server starts in API-only mode. Removes C23 `#embed` dependency on non-Windows. |
| `COSYVOICE_SERVER_DEFAULT_MODE` | WEBUI / API | WEBUI | Default server mode when neither `--api` nor `--webui` is given. Set to `API` for headless deployments. |

### Backend Options

GGML backend options are passed through from GGML CMake. Typical examples:

```bash
# CUDA backend
cmake -B build -DGGML_CUDA=ON

# Vulkan backend
cmake -B build -DGGML_VULKAN=ON
```

Refer to the [GGML documentation](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md) for the full list of backend-specific options and recommended settings.

### Metal backend (`GGML_METAL`)

- `GGML_METAL` defaults to **ON on Apple Silicon** (see ggml's own CMakeLists) and can be forced with `-DGGML_METAL=ON/OFF`.
- GGML itself needs no special handling for Metal: it is shallow-cloned at latest master (`--depth=1`) on every platform and no patch is applied. The Metal PAD beg-padding support that used to be patched in is upstream since [ggml-org/llama.cpp#29561](https://github.com/ggml-org/llama.cpp/pull/29561) (ggml commit `46fc5b3b`), which also added circular padding and support for permuted sources.
- If an existing `vendor/ggml` predates that commit, CMake warns (without failing) because its Metal PAD kernel would abort inference — delete `vendor/ggml` and re-run CMake to pick up a fresh one.

```bash
# Force Metal on/off (default: ON on Apple Silicon)
cmake -B build -DGGML_METAL=ON
```

### Dependency Path Options

| Option | Description |
|---|---|
| `GGML_SOURCE_DIR=<path>` | Path to GGML sources (default: `vendor/ggml`). If missing, CMake auto-clones. |
| `ICU_PREBUILT_DIR=<path>` | Path to ICU prebuilt binaries (default: `<build_dir>/_deps/icu`) |
| `ORT_PREBUILT_DIR=<path>` | Path to ONNX Runtime prebuilt binaries (default: `<build_dir>/_deps/onnxruntime`) |
| `FFMPEG_PREBUILT_DIR=<path>` | Path to FFmpeg prebuilt binaries |
| `SIMDE_INCLUDE_DIR=<path>` | SIMDe headers for ARM64/aarch64 (including Android cross-compilation) — see [SIMDe (SIMD Everywhere)](#simde-simd-everywhere) |

### Link-time optimization (LTO)

Enabled by default (`CMAKE_INTERPROCEDURAL_OPTIMIZATION=TRUE`). Reduces binary size ~3% by deduplicating identical SIMD tier bodies and inlining small cross-TU helpers. Verified safe on MSVC, GCC, and Clang — per-object `/arch` flags are preserved through LTO codegen.

```bash
# Disable if faster iteration is needed
cmake -B build -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF
```

### Build Matrix

| Scenario | Recommended CMake flags |
|---|---|
| CUDA backend | `-DGGML_CUDA=ON` |
| Vulkan backend | `-DGGML_VULKAN=ON` |
| CPU-only | no backend flag required |
| Core-only (no frontend / ICU) | `-DCOSYVOICE_NO_FRONTEND=ON -DCOSYVOICE_NO_ICU=ON` |
| No-audio helper API | `-DCOSYVOICE_NO_AUDIO=ON` |
| Disable CLI playback | `-DCOSYVOICE_CLI_NO_PLAYBACK=ON` |

Practical build examples:

```bash
# Core-only build (no ONNX frontend, no ICU text norm)
cmake -B build-core -DCMAKE_BUILD_TYPE=Release -DCOSYVOICE_NO_FRONTEND=ON -DCOSYVOICE_NO_ICU=ON

# No-audio build (CLI output forced to WAV fallback path)
cmake -B build-noaudio -DCMAKE_BUILD_TYPE=Release -DCOSYVOICE_NO_AUDIO=ON

# Disable CLI playback (audio helper APIs still available)
cmake -B build-noplay -DCMAKE_BUILD_TYPE=Release -DCOSYVOICE_CLI_NO_PLAYBACK=ON
```

## Dependency Resolution

The top-level CMake project resolves dependencies in this order:

- **PCRE2**: Built from `vendor/pcre2` as static libraries (`pcre2-8`, `pcre2-16`).
- **GGML**: Uses `GGML_SOURCE_DIR` (default: `vendor/ggml`). If missing, CMake clones `https://github.com/ggml-org/ggml.git` automatically.
- **ICU** (used by text normalization unless disabled with `COSYVOICE_NO_ICU`): Resolution order: `ICU_PREBUILT_DIR` → `find_package(ICU)` → Windows auto-download → system ICU on Linux/macOS.
- **ONNX Runtime** (used by the frontend unless disabled with `COSYVOICE_NO_FRONTEND`): Resolution order: `ORT_PREBUILT_DIR` → `find_package(onnxruntime)` → auto-download.

On Windows, prebuilt dependency DLLs are copied next to built executables.

## Using Custom Dependencies

You can point CMake to custom dependency locations with cache variables:

```bash
cmake -B build \
  -DGGML_SOURCE_DIR=/path/to/ggml \
  -DICU_PREBUILT_DIR=/path/to/icu \
  -DORT_PREBUILT_DIR=/path/to/onnxruntime \
  -DSIMDE_INCLUDE_DIR=/path/to/simde
```

You can also use the default prebuilt locations under your build directory:
- `<build_dir>/_deps/icu`
- `<build_dir>/_deps/onnxruntime`

If you place files there with the expected layout, CMake will pick them up automatically (without extra `-D` flags).

Expected markers/layout:
- ICU: `include/unicode/utypes.h` (and platform libs/dlls under `lib*` / `bin*`)
- ONNX Runtime: `include/onnxruntime_c_api.h` and runtime library files under `lib`

## SIMDe (SIMD Everywhere)

The SIMD tiers described in [SIMD.md](SIMD.md) are written against x86 intrinsics. [SIMDe](https://github.com/simd-everywhere/simde) is a header-only library that maps these x86 intrinsics to other ISAs — on ARM64/aarch64 the emulated SSE4.2+FMA3 class compiles to NEON without modification.

How CMake handles it (`CMakeLists.txt`):

- **x86_64**: SIMDe is never needed — every tier runs natively on its own ISA.
- **ARM64/aarch64 (incl. Android cross-compilation)**: SIMDe is **optional**. If CMake finds `simde/x86/sse4.2.h` (in `/opt/homebrew/include`, `/usr/local/include`, `/usr/include`, `vendor/simde/` or via `-DSIMDE_INCLUDE_DIR=<path>`) the SSE4.2+FMA3 class is emulated via NEON; the scalar tier is not even built then, because dispatch on non-x86 always selects the emulated class. Without SIMDe the build automatically falls back to scalar-only kernels.

Getting SIMDe:

```bash
# macOS
brew install simde

# Debian/Ubuntu (package may not exist on all distros)
apt install libsimde-dev

# Anywhere — clone and point SIMDE_INCLUDE_DIR at it
git clone --depth=1 https://github.com/simd-everywhere/simde.git
cmake -B build -DSIMDE_INCLUDE_DIR=/path/to/simde
```

For Android specifics, see [docs/build-android.md](build-android.md).

## Audio Backend & FFmpeg

This project supports two audio backends for encoding/decoding helper APIs:

- `MINIAUDIO` (default): provides WAV I/O and basic PCM helpers.
- `FFMPEG` (optional): enables encoding/decoding for additional formats when the linked FFmpeg runtime provides the required encoders.

Control the audio backend via CMake: set `COSYVOICE_AUDIO_BACKEND` to `MINIAUDIO` or `FFMPEG`. Default: `MINIAUDIO`.

Examples:
```bash
cmake -B build -DCOSYVOICE_AUDIO_BACKEND=MINIAUDIO
cmake -B build -DCOSYVOICE_AUDIO_BACKEND=FFMPEG
cmake -B build -DCOSYVOICE_AUDIO_BACKEND=FFMPEG -DFFMPEG_PREBUILT_DIR=/path/to/ffmpeg
```

If you build with FFmpeg support, the public audio API keeps the same function names. Use `cosyvoice_audio_supported_encoding_formats()` to query the actual formats available in the linked FFmpeg runtime.

FFmpeg usage notes:

- On Windows the build scripts download prebuilt FFmpeg (BtbN builds) by default when `FFMPEG_PREBUILT_DIR` is not provided.
- On Linux/macOS the system-provided FFmpeg (homebrew/apt) will be used when available; otherwise point `FFMPEG_PREBUILT_DIR` to your prebuilt location.
- The API surface includes `wav`, `mp3`, `aac`, `flac`, `m4a`, and `opus`, but the usable subset depends on the linked FFmpeg build. The library probes available encoders at runtime and exposes the supported set via the API and CLI/server help messages.
- `m4a` is a non-standard convenience extension here. OpenAI Speech does not define it; use `response_format` only if your client/server understands this project-specific extension.
- If a requested format is not supported at runtime, the server/CLI will instruct you to use `wav` or `pcm` instead.
- On Windows, the build will copy the FFmpeg runtime DLLs it found into the executable directory. If you use a custom prebuilt FFmpeg, make sure the `bin` and `lib` layout matches the expectations in `cmake/Dependencies.cmake`.

License reminder:

- The repository code is MIT. FFmpeg prebuilt binaries may be LGPL or GPL depending on build options. Using a GPL-enabled FFmpeg build may impose GPL obligations on your redistributed binaries. See [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md#ffmpeg-optional-audio-backend).

## Troubleshooting

- CMake cannot find GGML: set `-DGGML_SOURCE_DIR=...` or keep default `vendor/ggml` and ensure Git is available for auto-clone.
- ICU/ONNX Runtime detection issues: either install system packages (where applicable) or place prebuilt files into `<build_dir>/_deps/icu` and `<build_dir>/_deps/onnxruntime`.
- Executable starts but misses runtime libraries on Windows: ensure post-build copied DLLs exist next to binaries in `build/bin`.
- Backend-specific issues are summarized in [README.md — Backend Test Status](../README.md#backend-test-status).
