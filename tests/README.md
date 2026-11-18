# Tests

Model-backed tests that exercise the runtime directly, without the HTTP server.
They load a CosyVoice 3 GGUF model (and, for TTS, a `prompt_speech.gguf`) from
disk and print pass/fail results to stdout. They are manual tools: they need a
real model, a working backend, and enough VRAM/RAM, so they are opt-in at
configure time and are not run by a plain `ctest`.

## Build

```sh
cmake -S . -B build -DCOSYVOICE_BUILD_TESTS=ON
cmake --build build --config Release --target test-llm-graph-reuse test-tts-stream test-kv-offload test-mmap-zerocopy test-mmap-delete
```

The executables land in `build/bin/<config>/` next to the shared library they
load (`cosyvoice`, `ggml`, and the backend DLLs).

## Requirements

- A CosyVoice 3 model in GGUF format (e.g. `CosyVoice3-2512_F32.gguf` or a
  quantized variant such as `CosyVoice3-2512_Q4_K_M.gguf`).
- A `prompt_speech.gguf` produced by the frontend export / `--prompt-speech-output`
  flow (only for `test-tts-stream`).
- A backend that supports the model. `CUDA0` for NVIDIA GPUs, `CPU`, or `MTL0`
  on Apple Silicon. Use the name printed by the backend registry at startup if
  unsure.

## `test-llm-graph-reuse`

Regression for embedding-index lifetime in reused LLM graphs. It compares
repeated single-token decoding (which reuses the Flash Attention graph) against
fresh full-prefix evaluations. It needs a GGUF with non-F32 speech embeddings,
such as Q4_K_M, but no reference recording, ONNX assets, or network access.

```sh
build/bin/test-llm-graph-reuse /path/to/CosyVoice3-2512_Q4_K_M.gguf MTL0
build/bin/test-llm-graph-reuse /path/to/CosyVoice3-2512_Q4_K_M.gguf CPU
```

The test fails if a reused graph changes the next-token distribution by more
than the allowed backend rounding tolerance. In particular, it catches the
unprotected `emb_pos` tensor being overwritten after its first use.

## `test-tts-stream`

Streaming TTS regression and listening tool. It synthesizes the same text twice
with the same seed and prompt — once through the streaming callback path and
once in a single non-streaming pass — then reports how the streamed audio lines
up with the full render, and saves both to wav so they can be compared by ear.

The tool exists mainly to check streaming seams (chunk boundaries) without
starting the HTTP server: it prints the per-chunk sizes, the sample step across
every seam, and the RMS/max deviation from the non-streaming reference.

### Usage

All arguments are positional. Provide `text` (the 5th argument or later) and
omit trailing arguments to use their defaults.

```sh
test-tts-stream [model.gguf] [backend] [chunk_tokens] [out_prefix] text \
                [instruction] [prompt_speech.gguf] \
                [speed] [seed] [diffusion_steps] [threads] [reset|continue] \
                [no-save]
```

| Argument | Default | Meaning |
|---|---|---|
| `model.gguf` | `CosyVoice3-2512_F32.gguf` | Model to load. |
| `backend` | `CUDA0` | Backend name (`CUDA0`, `CPU`, `MTL0`, ...). |
| `chunk_tokens` | `28` | Streaming LLM chunk size. |
| `out_prefix` | `cfv` | Writes `<prefix>_stream.wav` and `<prefix>_ns.wav`. |
| `text` | — (required) | Text to synthesize. |
| `instruction` | — | Instruction text; enables instruct mode. Omit or pass `-` for zero-shot. |
| `prompt_speech.gguf` | `prompt_speech.gguf` | Prompt speech file. |
| `speed` | `1.0` | Playback speed; `1.0` disables time stretching. |
| `seed` | `42` | Sampler seed (strict, reproducible). |
| `diffusion_steps` | `4` | Flow diffusion steps; `0` uses the GGUF value. |
| `threads` | `4` | CPU threads for the backend. |
| `reset\|continue` | `reset` | Sampler RNG policy across sessions. |
| `no-save` | off | Skip writing the wav files. |

Because text and instruction are positional, quote values that contain spaces.
A literal `-` for the instruction selects zero-shot mode, which is handy when
later optional arguments are also set.

### Output

```
stream: <N> samples (<s> s), <K> chunks in <ms>
  chunk 0: <n0> samples
  ...
non-stream: <N> samples (<s> s) in <ms>
max step overall: <step> @ sample <i> (<ms>)
seam 0: step=<step> rms64=<rms>
diff vs non-stream (n=<N>): rms=<rms> max=<max> @ sample <i> (<ms>)
saved <prefix>_stream.wav / <prefix>_ns.wav
```

- `chunk k` — samples emitted by each streaming callback.
- `seam k` — absolute sample-to-sample step right at the boundary after chunk
  `k`. A large step with a quiet `rms64` in the neighbourhood is a click.
- `max step overall` — the largest sample step anywhere in the streamed audio,
  including natural speech transients.
- `diff vs non-stream` — RMS and max deviation from the single-pass render.
  Streaming and non-streaming audio differ by design (the shared flow decoder
  attends over chunks while streaming), so this is a regression baseline rather
  than a pass/fail threshold.

### Examples

```sh
# Instruct mode, defaults, chunk 28.
build/bin/test-tts-stream CosyVoice3-2512_F32.gguf CUDA0 28 out \
    "今天天气不错，我们出去走走吧。" "清晰地說中文。"

# Zero-shot, smaller chunks, custom seed, keep the wavs.
build/bin/test-tts-stream CosyVoice3-2512_F32.gguf CUDA0 14 out \
    "今天天气不错，我们出去走走吧。" - "" 1.0 7

# CPU run with a quantized model and no wav output.
build/bin/test-tts-stream CosyVoice3-2512_Q4_K_M.gguf CPU 28 out \
    "Hello world." - "" 1.0 42 0 8 no-save
```

## `test-kv-offload`

Regression for the KV cache offload paths. It runs in two phases:

1. Dedicated buffer policy with two offloadable DiT KV slots — checks that the
   offloaded DiT cache is reported by `cosyvoice_get_memory_usage()` and is
   released by `cosyvoice_empty_buffer_cache()`.
2. Balanced buffer policy (LLM KV cache offload slot) — clears the buffer cache,
   twice, and synthesizes again. This catches the freed-context use and the
   double free on destruction that used to happen after
   `cosyvoice_empty_buffer_cache()`.

```sh
build/bin/test-kv-offload /path/to/CosyVoice3-2512_Q4_K_M.gguf CPU prompt_speech.gguf "你好。"
```

The model is loaded twice (once per phase), so the DiT KV cache length and the
LLM sequence length are kept small; it prints `PASS` on success.

## `test-mmap-zerocopy`

Regression for using the model file mapping in place on CPU backends. It loads
the model twice - once with the default zero-copy path and once with
`COSYVOICE_NO_MMAP_ZEROCOPY=1` forcing the copy path - and requires
bit-identical audio for the same text and seed.

```sh
build/bin/test-mmap-zerocopy /path/to/CosyVoice3-2512_Q4_K_M.gguf CPU prompt_speech.gguf "你好。"
```

It also prints the `parameters` memory usage of both loads; the zero-copy load
reports at least the size of the model file because the mapping backs the
weights. Only CPU backends use the zero-copy path, so pass `CPU` as the backend.

## `test-mmap-delete`

Checks that deleting the model file while a zero-copy context is alive is safe.
It copies the given model to a work file, loads the copy on a CPU backend,
synthesizes once, deletes the work file, and synthesizes again. The second pass
must succeed and produce bit-identical audio, and the work file name must be
gone once the context is released. On Windows this relies on the mapping being
opened with `FILE_SHARE_DELETE`.

```sh
build/bin/test-mmap-delete model-Q4_0.gguf work.gguf CPU prompt_speech.gguf "你好。"
```

The work file is copied from the source model when it does not exist yet, so it
can be pointed at any writable location with room for one model copy. If the work
file already exists it is used as is, which allows hard-linking it to the model on
space-constrained drives.
