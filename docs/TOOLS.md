# Tooling Guide

Language: [中文](TOOLS_zh.md)

Back to root: [README.md](../README.md)

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

## Server Tool (`tools/server`)
Executable name: `cosyvoice-server`

Provides an OpenAI Speech-compatible API (no WebUI).

Endpoints:
- `GET /healthz`
- `GET /v1/models`
- `POST /v1/audio/speech`

Authentication behavior:
- If `--api-key` is provided: requests must include `Authorization: Bearer <api_key>`.
- If `--api-key` is omitted: authentication is disabled.

Core startup options:
- `--model <file.gguf>`: required model file.
- `--backend-path <dir>`: GGML backend directory. If omitted, the server will load the GGML backend from the executable's directory.
- `--served-model-name <name>`: exposed model id used by API requests. If omitted, the server will use the model architecture from `cosyvoice_get_architecture()` when available, and fall back to a name derived from the model filename otherwise.
- `--voice-prompt <voice=prompt_speech.gguf>`: map one `voice` name to one `prompt_speech` file (repeatable).
- `--host <host>`, `--port <port>`: bind address. Defaults: `127.0.0.1:8080`.

Runtime tuning options:
- `--inference-buffer-policy <shared|balanced|dedicated>`: inference buffer policy. Default: `balanced`.
- `--max-llm-len <value>`: maximum LLM sequence length. Default: `2048`.
- `--threads, -j <value>`: CPU thread count. Default: `0` (hardware concurrency).
- `--llm-kv-cache-type <f32|f16|q8_0|q5_1|q5_0|q4_1|q4_0>`: default `q8_0`.
- `--seed <value>`: default per-request seed (used when request does not provide extension field `seed`).

Sampling default overrides (server-level):
- `--temperature <value>` (`> 0`)
- `--top-k <value>` (`>= 0`)
- `--top-p <value>` (`[0, 1]`)
- `--win-size <value>` (`> 0`)
- `--tau-r <value>` (`>= 0`)
- `--min-token-text-ratio <value>` (`>= 0`)
- `--max-token-text-ratio <value>` (`>= 0`, and not less than `min_token_text_ratio`)

Single-voice shortcut:
```bash
cosyvoice-server \
  --model model.gguf \
  --voice alloy \
  --prompt-speech prompt_speech.gguf
```

Multi-voice startup:
```bash
cosyvoice-server \
  --model model.gguf \
  --served-model-name cosyvoice-3 \
  --voice-prompt alloy=prompt_alloy.gguf \
  --voice-prompt nova=prompt_nova.gguf \
  --api-key sk-local-demo \
  --host 0.0.0.0 \
  --port 8080
```

Speech request behavior (`POST /v1/audio/speech`):
- Required fields: `model`, `input`, `voice`.
- Optional standard fields: `instructions`, `speed`, `response_format`.
- Optional extension fields: `seed`, `temperature`, `top_k`, `top_p`, `win_size`, `tau_r`, `min_token_text_ratio`, `max_token_text_ratio`.
- Seed precedence: request `seed` extension > server `--seed` > random seed per request.
- Mode selection is automatic: non-empty `instructions` -> instruct mode; otherwise zero-shot mode.
- Supported `response_format`: `wav`, `pcm`, plus FFmpeg-backed formats `mp3`, `aac`, `flac`, `m4a`, and `opus` when the linked FFmpeg runtime actually provides the required encoders. The server prints the runtime-supported subset in its help text.
- `m4a` is a project-specific extension, not part of the OpenAI Speech standard.
- `wav` is supported through the audio encoder in normal builds, and through an internal PCM16 WAV fallback when audio helpers are disabled.
- `pcm` returns raw 16-bit little-endian mono PCM payload.
- If a requested format is not available at runtime the server will return an error explaining the unsupported format; clients should fall back to `wav` or `pcm`.

OpenAI Python client example:
```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="sk-local-demo")

audio = client.audio.speech.create(
    model="cosyvoice-3",
    voice="alloy",
    input="Hello from cosyvoice-server",
    instructions="Speak with a warm and calm style.",
    response_format="mp3",
)

with open("out.mp3", "wb") as f:
    f.write(audio.read())
```

Non-standard extension fields (OpenAI SDK `extra_body`):
- Standard OpenAI Speech requests usually use: `model`, `input`, `voice`, optional `instructions/speed/response_format`.
- `cosyvoice-server` additionally supports request-level sampling extensions:
  - `seed`, `temperature`, `top_k`, `top_p`, `win_size`, `tau_r`, `min_token_text_ratio`, `max_token_text_ratio`
- With the OpenAI Python SDK, pass these extensions via `extra_body`:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="sk-local-demo")

audio = client.audio.speech.create(
    model="cosyvoice-3",
    voice="alloy",
    input="Hello from cosyvoice-server",
    response_format="wav",
    extra_body={
        "seed": 123,
        "temperature": 0.8,
        "top_k": 32,
        "top_p": 0.9,
        "win_size": 10,
        "tau_r": 0.2,
        "min_token_text_ratio": 1.0,
        "max_token_text_ratio": 5.0,
    },
)

with open("out.wav", "wb") as f:
    f.write(audio.read())
```

Validation rules for these extensions:
- `seed`: uint32 (`[0, 4294967295]`)
- `temperature`: `> 0`
- `top_k`: `>= 0`
- `top_p`: `[0, 1]`
- `win_size`: `> 0`
- `tau_r`: `>= 0`
- `min_token_text_ratio`: `>= 0`
- `max_token_text_ratio`: `>= 0` and `>= min_token_text_ratio` (when both are set)

If validation fails, the server returns OpenAI-style `400` with an error object.

Script helper:
- `tools/server/synthesize_via_api.py` uses OpenAI SDK directly and maps extension CLI flags to `extra_body` automatically.

`synthesize_via_api.py` usage:
- This script is meant for public use (quick smoke tests and local integration checks), not just internal testing.
- Dependency:

```bash
pip install openai
```

- Standard request example:

```bash
python tools/server/synthesize_via_api.py \
  --base-url http://127.0.0.1:8080 \
  --model cosyvoice-3 \
  --voice alloy \
  --text "Hello from API" \
  --response-format wav \
  --output out.wav
```

- Batch generation (single process, sequential requests):

```bash
python tools/server/synthesize_via_api.py \
  --base-url http://127.0.0.1:8080 \
  --model cosyvoice-3 \
  --voice alloy \
  --text "Batch request sample" \
  --response-format wav \
  --output samples/out.wav \
  --requests 5
```

When `--requests > 1`, outputs are saved as `out_001.wav`, `out_002.wav`, ...

Concurrent stress test helper:
- `tools/server/batch_tts_stress_test.py` sends multiple concurrent requests and keeps all generated audio files.

```bash
python tools/server/batch_tts_stress_test.py \
  --base-url http://127.0.0.1:8080 \
  --model cosyvoice-3 \
  --workers 4 \
  --repeat 8 \
  --out-dir build/bin/Release/server_batch
```

This script prints per-request output file paths so you can listen and compare results.

- With non-standard extension fields:

```bash
python tools/server/synthesize_via_api.py \
  --base-url http://127.0.0.1:8080 \
  --model cosyvoice-3 \
  --voice alloy \
  --text "Hello with sampling overrides" \
  --response-format wav \
  --seed 123 \
  --temperature 0.8 \
  --top-k 32 \
  --top-p 0.9 \
  --win-size 10 \
  --tau-r 0.2 \
  --min-token-text-ratio 1.0 \
  --max-token-text-ratio 5.0
```

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
- `--backend-path <dir>`: GGML backend directory. If omitted, the CLI will load the GGML backend from the executable's directory.
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
