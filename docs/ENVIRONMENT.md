# Environment Variables

The runtime library (`cosyvoice`) and its tools read a small set of environment variables. All of them are optional and read on a best-effort basis; invalid values are ignored with a warning where applicable.

| Variable | Scope | Purpose |
|---|---|---|
| `COSYVOICE_SIMD_LEVEL` | Library load | Cap the CPU DSP SIMD dispatch tier (x86-64 only) |
| `COSYVOICE_MAX_WEIGHT_BUFFER_SIZE` | Model load | Override the largest per-buffer weight allocation, in bytes |
| `COSYVOICE_NO_MMAP_ZEROCOPY` | Model load | Disable using weights zero-copy from the file mapping (CPU backends) |
| `COSYVOICE_LOG_VERBOSE` | Logging | Print INFO/DEBUG log output in Release builds |
| `COSYVOICE_HIFT_PHASE_LOG` | Streaming TTS | Log HiFT streaming frame accounting |
| `COSYVOICE_DUMP_MEL` | Streaming TTS | Append new mel frames to a file for continuity debugging |
| `COSYVOICE_STAGE_TIMING` | TTS inference | Log per-stage inference timings |
| `COSYVOICE_SCHED_RESERVE` | Model load | Pre-reserve the Flow buffers for a worst-case mel length |
| `COSYVOICE_MEMORY_POOL_BYTES` | Runtime init | Preallocate a process-wide pool for internal C++ containers, in bytes (0/unset disables) |

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

## `COSYVOICE_NO_MMAP_ZEROCOPY`

Read at model load. When a model is loaded from a file, CPU backends use the file mapping in place: weight tensors point into the mapped GGUF instead of being copied into a separate CPU buffer. This saves roughly one model-sized copy of resident memory and speeds up loading. Set this variable to any value other than `0` to force the previous copy behavior.

- Only applies when the active device is the CPU backend and it supports `buffer_from_host_ptr`; GPU loads always copy the weights to the device.
- While the context is alive the mapped file stays open. On Windows the file is opened with `FILE_SHARE_DELETE`, so it can still be renamed or deleted; the deletion takes effect when the context is released, and the original mapping stays readable until then.

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

## `COSYVOICE_LOG_VERBOSE`

Controls whether INFO/DEBUG messages from the library are printed. Debug builds (`_DEBUG`) enable it by default; Release builds keep it off unless the variable is set (any value, even an empty one) or one of the diagnostics below (`COSYVOICE_STAGE_TIMING`, `COSYVOICE_HIFT_PHASE_LOG`) is set. Warning and error messages are always printed.

All output goes through the active GGML log callback; the default callback prints these levels to stdout.

## `COSYVOICE_HIFT_PHASE_LOG`

Setting the variable to any value (even an empty one) enables HiFT streaming frame accounting — the code only checks that the variable is *set* (`getenv` returns non-null), so to disable it the variable must be unset. While streaming, each chunk logs a line with `prompt_frames`, `input_frames`, `new_frames`, `cache_frames`, `emitted_frames`, and `emitted_total` through the GGML log callback at INFO. Setting this variable also enables verbose output (see `COSYVOICE_LOG_VERBOSE`). Intended for debugging HiFT cache/streaming seams.

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

## `COSYVOICE_STAGE_TIMING`

Setting the variable to any value (even an empty one) enables per-stage inference timing. The library times the pipeline and prints each line through the GGML log callback at INFO; setting this variable also enables verbose output, so the lines are visible in Release builds.

- `stage timing: text: ...` — once per high-level TTS call: emoji stripping, text normalization and tokenization.
- `stage timing: ...` — once per synthesis round (one LLM job plus one vocoder pass): LLM prefill/decode/epilogue, flow encode, DiT diffusion, mel assembly and HiFT, including per-token/per-step averages and the number of graphs that had to be rebuilt. In streaming mode the line is prefixed with the chunk index; the first chunk's time is the time to first audio. `buffers=`/`t2w=` are the current scheduler compute buffer and Flow intermediate buffer sizes, so growth across rounds is visible.
- `stage timing: stream: ...` — summary of a streaming request: chunk count, total wall time and per-stage sums.
- `stage timing: sched re-reserved ...` — the graph scheduler had to reserve its compute buffers again, which synchronizes the backends; the line includes the buffer size before the re-reservation and the round line counts these events as `sched_reallocs`.
- `stage timing: token2wav re-allocated (dit context|mel): X -> Y MiB` — the intermediate Flow buffer was freed and allocated again at the new size while its contents grew; this only happens without `COSYVOICE_SCHED_RESERVE` or beyond the reserved length. The round line counts these events as `t2w_reallocs`.

While the variable is unset the instrumentation only costs one branch per stage boundary and never reads the clock.

Example:

```bash
COSYVOICE_STAGE_TIMING=1 cosyvoice-cli --model model.gguf --prompt-speech prompt_speech.gguf --text "hello" --output out.wav --stream
```

## `COSYVOICE_SCHED_RESERVE`

Pre-reserves the Flow (DiT) buffers at model load for a worst-case mel length: the scheduler compute buffers, the intermediate buffer that carries the DiT step state and the accumulated mel between steps, and the host-side mel cache. The backend scheduler re-plans its allocation whenever the graph layout changes and only ever grows its buffers, so on the default (non-KV) Flow path a streaming request reallocates them chunk after chunk as the accumulated mel grows. Reserving up front moves those allocations to load time and keeps them out of the inference path. Only the Flow path is covered; the LLM and HiFT graphs keep the previous behavior.

The value is a mel frame count:

- unset, negative or not a number — disabled (an invalid value logs a warning).
- `0` — use the length derived from the context parameters: `n_max_seq * token_mel_ratio` (the longest mel the model can emit).
- `N > 0` — reserve for `N` mel frames, clamped to the derived maximum.

The reserved length is a fixed cost in compute memory: the scheduler buffers are pre-grown to that worst case even for short requests (roughly 0.7 MiB per frame with the reference model). Size it to the longest audio you expect; longer requests still work and simply grow the buffers as before. The reservation is skipped when a DiT KV cache is configured (`dit_kv_fixed_slots`/`dit_kv_offloadable_slots`), because that path already keeps the per-chunk graph size bounded.

No reservation is performed when the variable is unset, which is the default.

Example:

```bash
COSYVOICE_SCHED_RESERVE=2048 cosyvoice-cli --model model.gguf --prompt-speech prompt_speech.gguf --text "hello" --output out.wav --stream
```

## `COSYVOICE_MEMORY_POOL_BYTES`

Preallocates a process-wide memory pool for the library's internal C++ containers (tokenizer, text chunking, prompt/TTS state, and other `src/` runtime structures). The pool is backed by a `std::pmr::synchronized_pool_resource` with an arena reserved at runtime initialization. Requests up to 32 KiB are served from the pool; larger or over-aligned requests go directly to the system allocator. When the arena is exhausted the pool continues from the system heap, so the value is not a hard memory cap.

The value is a size in bytes:

- unset or `0` —?disabled (the default).
- `N > 0` —?enable the pool with an `N`-byte initial arena; the size is rounded up to the arena alignment (64 KiB).
- negative or not a number —?disabled with a warning.

The pool is process-global and can be initialized only once. The environment variable is applied when the library is loaded, before any library call, so it must be set before the process starts. `cosyvoice_init_runtime()` can be used instead of the variable; because the variable is applied at load, it takes precedence over a later explicit call, which is rejected as already initialized. Enabling the pool after pooled containers have already allocated is rejected with a warning and leaves the system allocator active.

Pool setup and teardown are process-lifetime: the arena is not returned to the OS and is reused for the life of the process. The pool does not cover ggml/backend tensors, ICU-internal allocations, or `tools/`/`tests/` code. The startup banner and any warnings are printed once when the library is loaded, before the verbose log filter is installed, so they are visible even without `COSYVOICE_LOG_VERBOSE`.

Example:

```bash
COSYVOICE_MEMORY_POOL_BYTES=33554432 COSYVOICE_LOG_VERBOSE=1 cosyvoice-cli --model model.gguf --prompt-speech prompt_speech.gguf --text "hello" --output out.wav --stream
```
