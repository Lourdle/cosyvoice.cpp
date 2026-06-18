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

### Options

| Option | Description |
|--------|-------------|
| `-h, --help` | Show help message |
| `-f, --file <arg>` | Input GGUF file path |
| `-o, --output-file <arg>` | Output GGUF file path |
| `-t, --type <arg>` | Default quantization type |
| `-c, --custom-string <arg> <arg>` | Custom metadata key-value pair (repeatable) |
| `-M, --tensor-map <arg>` | JSON file mapping tensor name regex to quantization type |

### Tensor Map (Selective Quantization)

Use `-M/--tensor-map` to apply different quantization types to different tensors.
Unlisted tensors use the `--type` default.

JSON format (PCRE2 regex patterns as keys):
```json
{
    "blk\\.\\d+\\.attn_(q|k|v|o)\\.weight": "Q4_K",
    "token_embd\\.weight": "Q8_0",
    "output\\.weight": "COPY"
}
```

#### Pattern Matching Priority

A tensor may match multiple patterns. The winning pattern is determined by:

1. **Literal (exact) patterns** always beat regex patterns.
   - `"tensor0": "Q5_K"` (literal) wins over `"tensor.*": "Q4_K"` for tensor `tensor0`.
2. **Among regex patterns, the longest textual pattern wins** (more specific).
   - `"layers\\..*\\.mlp\\.down_proj\\.weight": "Q5_K"` beats `"layers.*": "Q4_K"` for `layers.0.mlp.down_proj.weight`.
3. Ties are broken by JSON insertion order.

Patterns that never match any tensor produce a warning.

#### Pre-built Profiles

Ready-to-use tensor maps are available under `tools/quantize/profiles/`, organized by model:

```
tools/quantize/profiles/
└── cosyvoice3-2512/
    ├── Q4_K_S.json
    └── Q4_K_M.json
```

Example:
```bash
quantize -f model.gguf -o model-q4_k_-_s.gguf -t Q4_K -M tools/quantize/profiles/cosyvoice3-2512/Q4_K_S.json
```

Custom metadata strings are supported with repeated `-c/--custom-string`.

## Server Tool (`tools/server`)
Executable name: `cosyvoice-server`

The server runs in **two modes**:

| Mode | Selector | Description |
|---|---|---|
| **WebUI** (default) | no `--api` flag | Serves a full-featured web interface for model/speaker management, TTS generation, and speaker extraction. |
| **API** | `--api` flag | Headless OpenAI Speech-compatible API server. |

### Quick Start

#### WebUI mode (default)
```bash
# No arguments — WebUI opens with a browser-based model loader
cosyvoice-server

# With pre-loaded model and optional voices
cosyvoice-server \
  --model model.gguf \
  --voice-prompt alloy=prompt_alloy.gguf \
  --host 0.0.0.0 \
  --port 8080
```

Open http://127.0.0.1:8080 in your browser. When no `--model` is given, the WebUI provides a model loader on first use.

#### API mode
```bash
cosyvoice-server \
  --model model.gguf \
  --voice-prompt alloy=prompt_alloy.gguf \
  --served-model-name cosyvoice-3 \
  --api-key sk-local-demo \
  --host 0.0.0.0 \
  --port 8080 \
  --api
```

### WebUI Features

The WebUI provides a modern single-page application with the following capabilities:

#### Model Management
- **Load a model** at runtime: enter a `.gguf` file path, select backend and threads, then click "Load Model".
- **Unload a model**: releases the model from memory without restarting the server.
- **Parameter configuration**: KV cache types, buffer policy, max LLM length — configurable before loading.
- **Dynamic backend selection**: queries available GGML backends at startup (Auto / CPU / CUDA / Vulkan / Metal).

#### Speaker (Voice) Management
- **Register from `.gguf` prompt speech**: provide a name and path to a pre-extracted `prompt_speech.gguf` file.
- **Extract from audio file**: upload an audio file (via file selection or drag-and-drop), enter the prompt text, and extract — requires ONNX frontend models.
- **Record and extract**: record audio directly from the browser microphone, trim the waveform, and extract a voice embedding.
- **Audio trimmer**: visual waveform with start/end markers for fine-grained selection before extraction.
- **Save speaker**: export a registered speaker's prompt speech as a `.gguf` file.
- **Delete speaker**: remove a registered speaker from the server.
- **persistent state**: speaker names and settings are recovered from `localStorage` across page reloads.

#### TTS Generation
- **Text input** with configurable voice, mode (zero-shot / instruct / cross-lingual), and output format.
- **Advanced sampling controls**: temperature, top-k, top-p, repetition window, tau-r, seed (with "lock for session").
- **Instruct mode**: enter instructions to control speaking style (requires an instruct-capable model).
- **Audio playback**: generated audio plays directly in the browser with a live waveform visualizer.
- **Download**: save generated audio as a file.
- **History panel**: recent generations are cached up to 20 entries; click to replay or re-download.
- **Responsive output formats**: `wav`, `mp3`, `flac`, `opus`, `aac`, `m4a`, `pcm` (subset depends on FFmpeg runtime).
- **Theme toggle**: light / dark mode, persisted in `localStorage`.

#### Drag & Drop
- Audio files can be dragged onto the Extract tab or the speaker registration area.

### Core Options

| Option | Description |
|--------|-------------|
| `--help, -h` | Show help message and exit. |
| `--model, -m <file>` | CosyVoice model file (`.gguf`). |
| `--backend-path <dir>` | GGML backend directory. If omitted, the server will load the GGML backend from the executable's directory. |
| `--backend <name>` | GGML backend name (e.g. `cpu`, `cuda0`, `vulkan`, `metal`). Default: `auto` (best available). Mutually exclusive with `--cpu`/`--cuda`. |
| `--cpu` | Use CPU backend (equivalent to `--backend cpu`). Mutually exclusive with `--cuda`/`--backend`. |
| `--cuda` | Use CUDA backend (equivalent to `--backend cuda0`). Mutually exclusive with `--cpu`/`--backend`. |
| `--served-model-name <name>` | Exposed model id used by API/WebUI requests. If omitted, the server will use the model architecture from `cosyvoice_get_architecture()` when available, and fall back to a name derived from the model filename otherwise. |
| `--host <host>` | Listen host. Default: `127.0.0.1`. |
| `--port <port>` | Listen port. Default: `8080`. |
| `--api-key <key>` | Require `Authorization: Bearer <key>` on API and WebUI routes when provided. |
| `--concurrency <value>` | Concurrent request slots. Default: `1`. (API mode only) |

### Mode Selection

| Option | Description |
|--------|-------------|
| `--api` | Enable OpenAI-compatible API mode. Without this flag the server starts in WebUI mode. |

### Voice Mapping

| Option | Description |
|--------|-------------|
| `--voice-prompt <voice=prompt_speech.gguf>` | Map one `voice` name to one `prompt_speech` file (repeatable). |
| `--voice <voice>` | Voice name for single mapping mode. Default: `alloy`. |
| `--prompt-speech <file>` | Prompt speech file for single mapping mode (combines with `--voice` as shorthand for `--voice-prompt alloy=file`). |

In **API mode**, at least one voice mapping is required. In **WebUI mode**, voice mappings are optional — speakers can be registered at runtime.

### WebUI Frontend Options

These options point to ONNX models for speaker extraction (from audio files or microphone recording):

| Option | Description |
|--------|-------------|
| `--frontend-model <dir>` | Frontend ONNX model directory. |
| `--speech-tokenizer <file>` | Speech tokenizer ONNX model file. |
| `--campplus <file>` | Campplus ONNX model file. |

If provided at startup, the WebUI pre-populates the frontend configuration fields. The frontend can also be loaded at runtime through the WebUI.

### Runtime Tuning Options

| Option | Description |
|--------|-------------|
| `--max-llm-len <value>` | Maximum LLM sequence length. Default: `2048`. |
| `--threads, -j <value>` | CPU thread count. Default: `0` (hardware concurrency). |
| `--inference-buffer-policy <shared\|balanced\|dedicated>` | Inference buffer policy. Default: `balanced`. |
| `--llm-kv-cache-type <f32\|f16\|q8_0\|q5_1\|q5_0\|q4_1\|q4_0\|k=<type>,v=<type>[,fallback=<type>]>` | KV cache type. Single type (e.g. `q8_0`) uses the same format for K and V. Default: `k=q8_0,v=q4_0,fallback=q8_0`. |
| `--seed <value>` | Default random seed for the built-in sampler (when request does not provide a seed). |

### Sampling Default Overrides (Server-level)

| Option | Description |
|--------|-------------|
| `--temperature <value>` | Sampling temperature (`> 0`). |
| `--top-k <value>` | Sampling top-k (`>= 0`). |
| `--top-p <value>` | Sampling top-p in `[0, 1]`. |
| `--win-size <value>` | Repetition window size (`> 0`). |
| `--tau-r <value>` | Repetition penalty coefficient (`>= 0`). |
| `--min-token-text-ratio <value>` | Minimum token/text ratio (`>= 0`). |
| `--max-token-text-ratio <value>` | Maximum token/text ratio (`>= 0`, and not less than `--min-token-text-ratio`). |

### Text Normalization & Splitting

| Option | Description |
|--------|-------------|
| `--disable-text-normalization` | Disable ICU text normalization (only when `COSYVOICE_NO_ICU=OFF`). |
| `--disable-text-splitting` | Disable fragment splitting in TTS context. |
| `--disable-fast-split` | Disable fast token-based splitting in TTS context. |

### Output Postprocess

| Option | Description |
|--------|-------------|
| `--disable-fade-in` | Disable the default 20 ms fade-in on output audio. |

### Log Options

| Option | Description |
|--------|-------------|
| `--verbose, -v` | Verbose logs (advanced sampling, memory, full timings). |
| `--quiet, -q` | Suppress non-error logs. |

---

### WebUI HTTP API Reference

The WebUI exposes the following REST endpoints, consumed by the frontend JavaScript:

#### `GET /`

Serves the WebUI HTML page with server-side injected configuration (available backends, frontend model paths, etc.).

#### `GET /ping`

Simple liveness check. Returns `pong`.

#### `GET /backends`

Returns a JSON array of available GGML backends:
```json
[
  {"name": "auto", "description": "Choose the best available backend automatically"},
  {"name": "CPU", "description": "...", "has_device_id": false},
  {"name": "CUDA0", "description": "...", "has_device_id": true}
]
```

#### `GET /formats`

Returns a JSON array of supported audio output format names:
```json
["wav", "pcm", "mp3", "flac", "opus"]
```
The set depends on the linked FFmpeg runtime.

#### `GET /status`

Returns server status as JSON:
```json
{
  "status": "ok",
  "model_loaded": true,
  "model": "cosyvoice-3",
  "sample_rate": 24000,
  "max_llm_len": 2048,
  "k_cache_type": "Q8_0",
  "v_cache_type": "Q4_0",
  "buffer_policy": "balanced",
  "model_arch": "cosyvoice3-2512",
  "frontend_available": false,
  "speakers": ["alloy", "nova"]
}
```

#### `GET /speaker`

Returns a JSON array of registered speaker names:
```json
["alloy", "nova"]
```

#### `POST /speaker` — Register a speaker

Two content types are supported:

**JSON** (register from `.gguf` prompt speech):
```json
{"type": "gguf", "name": "alloy", "path": "/data/prompts/alloy.gguf"}
```

**Multipart** (extract from audio):
```
POST /speaker?type=extract
Content-Type: multipart/form-data
Fields: name, text, audio (file)
```

Returns `200` on success, `409` if name exists, `503` if no model loaded.

#### `DELETE /speaker/:name`

Unregisters the named speaker. Returns `200` on success, `404` if not found.

#### `POST /speaker/save`

Exports a speaker's prompt speech to a `.gguf` file on the server filesystem.

JSON body:
```json
{"name": "alloy", "path": "/data/exports/alloy_export.gguf"}
```

#### `POST /tts`

Generates speech. JSON body:

```json
{
  "text": "Hello from CosyVoice",
  "voice": "alloy",
  "mode": "zero-shot",
  "response_format": "wav",
  "speed": 1.0,
  "seed": 12345,
  "temperature": 0.8,
  "top_k": 32,
  "top_p": 0.9,
  "win_size": 10,
  "tau_r": 0.2,
  "min_token_text_ratio": 1.0,
  "max_token_text_ratio": 5.0,
  "instruction": "Speak with a warm and calm style.",
  "fadein": true
}
```

Returns raw audio bytes with the appropriate `Content-Type` header (`audio/wav`, `audio/mpeg`, etc.).

#### `GET /model/defaults`

Returns default model parameters as JSON:
```json
{
  "max_llm_len": 2048,
  "temperature": 1.0,
  "top_k": 50,
  "top_p": 0.9,
  ...
}
```

#### `POST /model/load`

Loads a model into the server at runtime. JSON body:
```json
{
  "model_path": "/data/models/cosyvoice-3.gguf",
  "backend": "auto",
  "n_threads": 8,
  "max_llm_len": 2048,
  "buffer_policy": "balanced",
  "k_cache_type": "q8_0",
  "v_cache_type": "q4_0"
}
```

#### `POST /model/unload`

Unloads the current model. JSON body not required.

#### `GET /frontend/model`

Returns the current frontend ONNX model configuration as JSON.

#### `POST /frontend/model/load`

Loads frontend ONNX models at runtime. JSON body:
```json
{
  "speech_tokenizer": "/data/models/speech_tokenizer.onnx",
  "campplus": "/data/models/campplus.onnx"
}
```

#### `POST /frontend/model/unload`

Unloads frontend ONNX models.

---

### API Mode Endpoints (OpenAI-compatible)

- `GET /healthz`
- `GET /v1/models`
- `POST /v1/audio/speech`

Authentication behavior:
- If `--api-key` is provided: requests must include `Authorization: Bearer <api_key>`.
- If `--api-key` is omitted: authentication is disabled.

#### Speech request behavior (`POST /v1/audio/speech`):
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

### Request/Response Format Handling

Both modes (WebUI and API) share the same audio encoding logic:

| Format | Content-Type | Backend |
|--------|-------------|---------|
| `wav` | `audio/wav` | Audio encoder or internal PCM16 WAV fallback |
| `pcm` | `audio/L16` | Raw PCM16 little-endian bytes |
| `mp3` | `audio/mpeg` | FFmpeg encoder (runtime-dependent) |
| `flac` | `audio/flac` | FFmpeg encoder (runtime-dependent) |
| `opus` | `audio/opus` | FFmpeg encoder (runtime-dependent) |
| `aac` | `audio/aac` | FFmpeg encoder (runtime-dependent) |
| `m4a` | `audio/mp4` | FFmpeg encoder (runtime-dependent) |

### Helper Scripts

#### `tools/server/synthesize_via_api.py`

Uses OpenAI SDK directly and maps extension CLI flags to `extra_body` automatically.

- Dependency: `pip install openai`
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

#### `tools/server/batch_tts_stress_test.py`

Concurrent stress test helper that sends multiple concurrent requests and keeps all generated audio files.

```bash
python tools/server/batch_tts_stress_test.py \
  --base-url http://127.0.0.1:8080 \
  --model cosyvoice-3 \
  --workers 4 \
  --repeat 8 \
  --out-dir build/bin/Release/server_batch
```

This script prints per-request output file paths so you can listen and compare results.

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

### D) Interactive mode (REPL)
```bash
cosyvoice-cli \
  --model model.gguf \
  --prompt-speech prompt_speech.gguf \
  --interactive
```

When neither `--text` nor `--output` is provided, interactive mode is enabled automatically.
Type text at the `> ` prompt to synthesize, or use slash commands:

- `/play [code]`: Play cached audio (requires audio support; disabled with `COSYVOICE_CLI_NO_PLAYBACK=ON`). Press `Ctrl+C` to stop playback.
- `/save <file> [code]`: Save cached audio to file.
- `/list`: List cached audio.
- `/query [code]`: Show audio details.
- `/delete [code]`: Delete cached audio.
- `/clear`: Clear cached audio.
- `/seed [value]`: Show or set next seed.
- `/seed-policy <fixed|random>`: Show or set seed policy.
- `/help`: Show command list.
- `/exit`: Exit interactive mode. `Ctrl+C` also exits.

The `[code]` parameter defaults to the last synthesized audio when omitted.

### Option Reference

Core options:
- `--help, -h`: Show help message and exit.
- `--interactive`: Run in interactive REPL mode.
- `--model, -m <file>`: CosyVoice model file (`.gguf`) used for TTS.
- `--backend-path <dir>`: GGML backend directory. If omitted, the CLI will load the GGML backend from the executable's directory.
- `--backend <name>`: GGML backend name (e.g. `cpu`, `cuda0`, `vulkan`, `metal`). Default: `auto` (best available). Mutually exclusive with `--cpu`/`--cuda`.
- `--cpu`: Use CPU backend (equivalent to `--backend cpu`). Mutually exclusive with `--cuda`/`--backend`.
- `--cuda`: Use CUDA backend (equivalent to `--backend cuda0`). Mutually exclusive with `--cpu`/`--backend`.
- `--text, -t <text>`: Text to synthesize.
- `--output, -o <file>`: Output audio path.
  - Normal build: format is inferred from file extension.
  - `COSYVOICE_NO_AUDIO=ON`: output is always WAV.
- `--speed, -s <value>`: Speech speed multiplier. Default: `1.0`. Must be `> 0`.
- `--seed <value>`: Random seed for sampling and internal noise generation. Must be an unsigned 32-bit integer. Default: random.
- `--max-llm-len <value>`: Maximum input token count for LLM (`n_max_seq`). Default: `2048`. Must be a positive integer.
- `--threads, -j <value>`: CPU thread count for model inference. Must be an unsigned 32-bit integer. Default: `0` (use current hardware concurrency).
- `--llm-kv-cache-type <f32|f16|q8_0|q5_1|q5_0|q4_1|q4_0|k=<type>,v=<type>[,fallback=<type>]>`: LLM KV cache type. Single type (e.g. `q8_0`) uses the same format for K and V. Use separate K/V types (e.g. `k=q8_0,v=q4_0`) for different formats. Default: `k=q8_0,v=q4_0,fallback=q8_0`.
- `--inference-buffer-policy <shared|balanced|dedicated>`: Inference buffer policy. Only effective in interactive mode; non-interactive mode always uses `shared` for minimal memory footprint and fastest single-shot synthesis. Default: `balanced`.
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
- `--disable-text-splitting`: Disable fragment splitting before synthesis.
- `--disable-fast-split`: Disable fast token-based fragment synthesis.
- `--disable-fade-in`: Disable the default 20 ms output fade-in.

Runtime logs:
- Basic request info (model path, mode, prompt source, output, speed, resolved CPU thread count, seed source) is shown before model loading.
- During model loading, a spinner (`| / - \\`) is shown in the console.
- Backend info now includes UMA (unified memory architecture) detection result and effective buffer policy.
- When UMA is detected and the requested buffer policy is `balanced`, the library automatically switches to `dedicated` for better throughput; a log message is printed in this case.
- Default output is concise and formatted with sections.
- `--verbose` shows full runtime details, including context/memory breakdown and full timing stages.
- `--quiet` suppresses runtime info and timings (errors still print).
- Sampling lines include source directly inline, e.g. `temperature : 1.0000 (model default)`.
- Memory in `--verbose` shows post-generation values with deltas against pre-generation snapshot.

Required vs optional:
- Required for normal TTS: `--model`, `--text`, `--output`, and one prompt source (`--prompt-speech` OR frontend inputs).
- Required for `--frontend-only`: `--speech-tokenizer`, `--campplus`, audio input, `--prompt-speech-output`.
- Optional parameters use defaults from CLI or model metadata:
  - CLI defaults: `--speed=1.0`, `--max-llm-len=2048`, `--threads=0` (hardware concurrency), `--llm-kv-cache-type=k=q8_0,v=q4_0,fallback=q8_0`, `--mode=auto`.
  - Sampling defaults (`temperature`, `top_k`, `top_p`, `win_size`, `tau_r`, token/text ratios) come from model config unless overridden by CLI.

## CLI Quick Reference

### Required arguments by scenario

| Scenario | Required |
|---|---|
| TTS with existing prompt_speech | `--model`, `--prompt-speech`, `--text`, `--output` |
| End-to-end frontend + TTS | `--model`, `--speech-tokenizer`, `--campplus`, `--prompt-audio` (or `--prompt-audio-16k` + `--prompt-audio-24k`), `--text`, `--output` |
| Frontend-only | `--frontend-only`, `--speech-tokenizer`, `--campplus`, audio input, `--prompt-speech-output` |
| Interactive mode | `--model`, one prompt source, no `--text`/`--output` required |

### Defaults at a glance

| Option | Default | Source |
|---|---|---|
| `--mode` | `auto` | CLI |
| `--speed` | `1.0` | CLI |
| `--max-llm-len` | `2048` | CLI |
| `--threads` | `0` (hardware concurrency) | CLI |
| `--llm-kv-cache-type` | `k=q8_0,v=q4_0,fallback=q8_0` | CLI |
| `--inference-buffer-policy` | `balanced` (interactive) / `shared` (non-interactive) | CLI |
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

# Interactive mode
cosyvoice-cli --model model.gguf --prompt-speech prompt_speech.gguf --interactive
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
