# Environment Variables

The runtime library (`cosyvoice`) and its tools read a small set of environment variables. All of them are optional and read on a best-effort basis; invalid values are ignored with a warning where applicable.

| Variable | Scope | Purpose |
|---|---|---|
| `COSYVOICE_SIMD_LEVEL` | Library load | Cap the CPU DSP SIMD dispatch tier (x86-64 only) |
| `COSYVOICE_MAX_WEIGHT_BUFFER_SIZE` | Model load | Override the largest per-buffer weight allocation, in bytes |
| `COSYVOICE_HIFT_PHASE_LOG` | Streaming TTS | Log HiFT streaming frame accounting |
| `COSYVOICE_DUMP_MEL` | Streaming TTS | Append new mel frames to a file for continuity debugging |

## `COSYVOICE_SIMD_LEVEL`

Read **once at library load** by every consumer of the library (also affecting `cosyvoice-cli`, `cosyvoice-server`, and `quantize`). Caps the runtime CPU DSP dispatch tier on x86-64. See [SIMD.md — Runtime control API](SIMD.md#runtime-control-api) for the full semantics.

Accepted values (case-insensitive):

- `auto` — no cap; use the fastest tier the CPU supports (default).
- `scalar` (alias `none`) — disable SIMD at runtime.
- `sse42` (alias `sse4.2`) — cap at the SSE4.2-class tier.
- `avx` — cap at the AVX-class tier.
- `avx2` — cap at the AVX2-class tier.
- `avx10-256` (aliases `avx10_256`, `avx10.1-256`, `avx10_1_256`) — hide the 512-bit tiers; AVX10-256 stays available.
- `avx512` (alias `avx-512`) — allow everything (same as `auto`).

Invalid values are silently ignored. Changing the variable after the library is loaded has no effect. Precedence: an explicit `cosyvoice_set_simd_level()` call (e.g. `--simd-level` on the CLI/server) > the environment variable > `AUTO`.

Example:

```bash
COSYVOICE_SIMD_LEVEL=avx2 cosyvoice-cli --model model.gguf --prompt-speech prompt_speech.gguf --text "hello" --output out.wav
```

Only applicable on x86-64. Non-x86 builds (including SIMDe-on-ARM64) fix the tier at compile time and ignore this variable.

## `COSYVOICE_MAX_WEIGHT_BUFFER_SIZE`

Set at model load. Overrides the largest weight buffer the loader asks a backend to allocate, in bytes. It exists to exercise the multi-buffer weights path and to work around backends whose drivers impose an allocation limit. The value is parsed as a decimal byte count; `0` removes the cap entirely.

- Setting a value smaller than the largest tensor causes a startup error explaining that the buffer limit is too small.
- Setting a value that is *very small* (below 64 MiB) emits a warning that many device buffers will be created, which can slow loading and fragment device memory.
- An invalid (non-numeric) value is ignored with a warning.

This complements the limit a backend reports through `ggml_backend_get_max_size`; on backends that report no limit (such as CPU and CUDA, which return `SIZE_MAX`) it is the only way to force the multi-buffer allocation path.

Example:

```bash
# Force 256 MiB weight buffers
COSYVOICE_MAX_WEIGHT_BUFFER_SIZE=268435456 cosyvoice-cli --model model.gguf --prompt-speech prompt_speech.gguf --text "hello" --output out.wav
```

## `COSYVOICE_HIFT_PHASE_LOG`

Setting the variable to any value (even an empty one) enables HiFT streaming frame accounting — the code only checks that the variable is *set* (`getenv` returns non-null), so to disable it the variable must be unset. While streaming, each chunk logs a line with `prompt_frames`, `input_frames`, `new_frames`, `cache_frames`, `emitted_frames`, and `emitted_total` to both the GGML log callback and stderr. Intended for debugging HiFT cache/streaming seams.

Example:

```bash
COSYVOICE_HIFT_PHASE_LOG=1 cosyvoice-cli --model model.gguf --prompt-speech prompt_speech.gguf --text "hello" --output out.wav --stream
```

## `COSYVOICE_DUMP_MEL`

Set to a file path. During streaming TTS, each chunk appends its newly-generated mel frames to that file in a simple binary format so the chunked mel sequence can be inspected for continuity across streaming joins:

```
[uint32 n_new_frames][uint32 mel_dimension][float32 * (n_new_frames * mel_dimension)]
```

The file is opened in append mode, so it accumulates across calls and runs. The variable must be **unset** to disable it — an empty string is still "set" (`getenv` returns non-null), in which case the code tries to open the empty path and silently does nothing.

Example:

```bash
COSYVOICE_DUMP_MEL=/tmp/mel.bin cosyvoice-cli --model model.gguf --prompt-speech prompt_speech.gguf --text "hello" --output out.wav --stream
```

This is a debugging aid; the format is not a stable API and may change without notice.
