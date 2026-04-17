# Third-Party Notices

This project includes bundled third-party components under `vendor/` and also
contains code adapted/referenced from external open-source projects.

## miniaudio

- Upstream: https://github.com/mackron/miniaudio
- Local path: `vendor/miniaudio/miniaudio.h`
- License: dual-licensed by upstream as Public Domain (The Unlicense) or MIT No Attribution (MIT-0 style text)
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
- Usage in this project: FFT core logic references/adapts the KissFFT mixed-radix design; SIMD optimization/integration is implemented in this project.
- Upstream copyright: Copyright (c) 2003-2010 Mark Borgerding
- License: BSD-3-Clause
- License text location: https://github.com/mborgerding/kissfft/blob/master/COPYING

## llama.cpp (tokenizer implementation reference)

- Upstream: https://github.com/ggml-org/llama.cpp
- Usage in this project: tokenizer implementation is adapted/modified from llama.cpp
- Upstream license: MIT
