# Third-Party Notices

This project includes bundled third-party components under `vendor/` and also
contains code adapted/referenced from external open-source projects.

## GGML

- Upstream: https://github.com/ggml-org/ggml
- Local path: `vendor/ggml` (auto-cloned by the build if absent)
- Usage in this project: **core dependency** — the tensor compute library that
  backs the LLM, Flow (DiT), and HiFT inference and all accelerator backends
  (CUDA / Metal / Vulkan / SYCL). GGML is the foundation split out of the
  llama.cpp project.
- Local modifications: none — the tree is used as unmodified upstream code.
- Upstream copyright: Copyright (C) 2020-2026 Georgi Gerganov and GGML contributors
- License: MIT
- License text location: `LICENSE` at the root of the upstream repository (it ships
  with the upstream clone the build fetches). `vendor/ggml/` is git-ignored and may be a
  shallow local copy without the LICENSE file — if you redistribute, restore the
  full upstream tree (delete `vendor/ggml` and let CMake re-clone) and ship the
  MIT text with your binary.

## llama.cpp (tokenizer implementation reference)

- Upstream: https://github.com/ggml-org/llama.cpp
- Usage in this project: tokenizer implementation is adapted/modified from llama.cpp
- Upstream copyright: Copyright (c) 2023-2026 The llama.cpp developers
- License: MIT
- Note: the runtime compute library (GGML) shipped alongside llama.cpp is
  listed in its own section above.

## ONNX Runtime

- Upstream: https://github.com/microsoft/onnxruntime
- Usage in this project: frontend speech pipeline (SpeechTokenizer +
  CampPlus speaker embedding) via the C API; prebuilt binaries are fetched
  into `<build>/_deps/onnxruntime` at configure time or supplied via
  `ORT_PREBUILT_DIR`.
- License: MIT
- License text location: `LICENSE` in the released ONNX Runtime package
  (also at https://github.com/microsoft/onnxruntime/blob/main/LICENSE)

## ICU (International Components for Unicode)

- Upstream: https://github.com/unicode-org/icu
- Usage in this project: optional text normalization frontend; prebuilt
  binaries are fetched into `<build>/_deps/icu` or supplied via
  `ICU_PREBUILT_DIR`. Disable with `-DCOSYVOICE_NO_ICU=ON`.
- License: Unicode License v3 (permissive, MIT-style)
- License text location: https://www.unicode.org/license.txt

## SIMDe (SIMD Everywhere)

- Upstream: https://github.com/simd-everywhere/simde
- Usage in this project: optional header-only dependency that emulates the
  SSE4.2+FMA3 SIMD tier on non-x86 ISAs (ARM64/NEON); auto-detected or
  supplied via `SIMDE_INCLUDE_DIR`. x86 builds never use it.
- License: MIT
- License text location: https://github.com/simd-everywhere/simde/blob/master/COPYING

## miniaudio

- Upstream: https://github.com/mackron/miniaudio
- Local path: `vendor/miniaudio/miniaudio.h`
- License: dual-licensed by upstream as Public Domain (The Unlicense) or MIT-0
- License text location: embedded at the end of `vendor/miniaudio/miniaudio.h`

## PCRE2

- Upstream: https://github.com/PCRE2Project/pcre2
- Local path: `vendor/pcre2`
- License: BSD-3-Clause WITH PCRE2-exception
- License text location: `vendor/pcre2/LICENCE.md`

### Note on PCRE2 JIT / SLJIT

PCRE2 documents that `deps/sljit` carries its own license when present.
If your local PCRE2 source includes that directory, ensure the corresponding
license file is retained and distributed.

## KissFFT (FFT reference/adaptation)

- Upstream: https://github.com/mborgerding/kissfft
- Local usage: `src/fft.cpp`, `src/fft.h`
- Usage in this project: FFT core logic references/adapts the KissFFT mixed-radix
  design; SIMD optimization/integration is implemented in this project.
- Upstream copyright: Copyright (c) 2003-2010 Mark Borgerding
- License: BSD-3-Clause
- License text location: https://github.com/mborgerding/kissfft/blob/master/COPYING

## cpp-httplib

- Upstream: https://github.com/yhirose/cpp-httplib
- Local path: `vendor/cpp-httplib`
- License: MIT
- License text location: header comment in `vendor/cpp-httplib/httplib.h`

## nlohmann/json

- Upstream: https://github.com/nlohmann/json
- Local path: `vendor/nlohmann`
- License: MIT
- License text location: header comment in `vendor/nlohmann/json.hpp`

## sse_mathfun / AVX_mathfun (portable SIMD math reference)

- Upstream: http://gruntthepeon.free.fr/ssemath/ (Julien Pommier)
- Local usage: `src/simd-math.h`
- Usage in this project: the polynomial approximations for log/log10/sin/cos
  follow the Cephes-derived algorithms popularized by sse_mathfun.h /
  AVX_mathfun.h; re-implemented in this project on top of portable SSE/AVX
  intrinsics (also usable via SIMDe on ARM64).
- License: zlib
- License text location: https://zlib.net/zlib_license.html

## FFmpeg (optional audio backend)

- Upstream: https://ffmpeg.org
- Usage in this project: optional audio backend (`COSYVOICE_AUDIO_BACKEND=FFMPEG`) for
  multi-format audio encoding/decoding. FFmpeg is not part of this repository; it is
  linked dynamically, and the build installs its shared libraries next to the binaries.
- How it is obtained: on Windows, when no usable FFmpeg is found in
  `FFMPEG_PREBUILT_DIR`, the build downloads a BtbN prebuilt package from
  https://github.com/BtbN/FFmpeg-Builds/releases (win64 or winarm64,
  `ffmpeg-n8.1-latest-*-lgpl-shared-8.1.zip`). On other platforms FFmpeg is located via
  `FFMPEG_PREBUILT_DIR` or the system library search path.
- License: the default BtbN package is LGPL-2.1-or-later and contains no GPL-only
  components. A GPL-enabled FFmpeg is only involved if you build or supply one yourself
  (see below).

### Redistributing with the default LGPL build

The repository code stays MIT-licensed. When you redistribute a package that includes
the default LGPL FFmpeg shared libraries, you must:

1. Ship this notices file and the LGPL license text that comes with FFmpeg
2. Keep FFmpeg replaceable by a modified version of your own — dynamic linking already
   achieves this — and make the source of any FFmpeg modifications you make available
3. Preserve the copyright notices that ship with FFmpeg

### Redistributing with a GPL-enabled FFmpeg (optional)

If you build or ship a GPL-enabled FFmpeg, you must:

1. License the combined work under GPL-2.0-or-later or a compatible license
2. Provide the complete corresponding source of the combined work to all recipients
3. State clearly that a GPL-enabled FFmpeg is included

Complying with FFmpeg's licenses is the responsibility of the user/distributor, not of
this project's maintainers.

### Further reading

- https://ffmpeg.org/legal.html
- https://github.com/FFmpeg/FFmpeg/blob/master/LICENSE.md
