# 工具使用说明

语言： [English](TOOLS.md)

返回仓库主页：[README_zh.md](../README_zh.md)

## 量化工具（`tools/quantize`）
可执行文件名：`quantize`

基本用法：
```bash
quantize -f input.gguf -o output-q4k.gguf -t Q4_K
```

查看帮助：
```bash
quantize --help
```

支持的量化类型：
- `F16`, `Q8_0`, `Q5_0`, `Q5_1`, `Q4_0`, `Q4_1`
- `Q6_K`, `Q5_K`, `Q4_K`, `Q3_K`, `Q2_K`
- `COPY`

支持通过重复 `-c/--custom-string` 写入自定义字符串元数据。

## Server 工具（`tools/server`）
可执行文件名：`cosyvoice-server`

提供 OpenAI Speech 兼容 API（不包含 WebUI）。

接口：
- `GET /healthz`
- `GET /v1/models`
- `POST /v1/audio/speech`

鉴权行为：
- 传入 `--api-key`：请求必须包含 `Authorization: Bearer <api_key>`。
- 不传 `--api-key`：不做鉴权。

核心启动参数：
- `--model <file.gguf>`：必填模型文件。
- `--backend-path <dir>`：GGML backend 所在目录。如果不指定此选项，将默认加载程序所在目录的 GGML 后端。
- `--served-model-name <name>`：API 对外模型名（请求中的 `model` 必须匹配）。如果省略，服务器会优先使用 `cosyvoice_get_architecture()` 返回的模型架构名；如果没有可用架构名，则回退为由模型文件名推导的名称。
- `--voice-prompt <voice=prompt_speech.gguf>`：将一个 `voice` 名映射到一个 `prompt_speech` 文件（可重复传入）。
- `--host <host>`、`--port <port>`：监听地址，默认 `127.0.0.1:8080`。

运行时调优参数：
- `--inference-buffer-policy <shared|balanced|dedicated>`：推理缓冲区策略，默认 `balanced`。
- `--max-llm-len <value>`：LLM 最大序列长度，默认 `2048`。
- `--threads, -j <value>`：CPU 线程数，默认 `0`（硬件并发）。
- `--llm-kv-cache-type <f32|f16|q8_0|q5_1|q5_0|q4_1|q4_0>`：默认 `q8_0`。
- `--seed <value>`：请求级默认 seed（当请求扩展字段未传 `seed` 时使用）。

TTS 后处理参数：
- `--disable-fade-in`：关闭默认的 20 ms 输出淡入。

采样默认值覆盖（服务级）：
- `--temperature <value>`（`> 0`）
- `--top-k <value>`（`>= 0`）
- `--top-p <value>`（`[0, 1]`）
- `--win-size <value>`（`> 0`）
- `--tau-r <value>`（`>= 0`）
- `--min-token-text-ratio <value>`（`>= 0`）
- `--max-token-text-ratio <value>`（`>= 0` 且不小于 `min_token_text_ratio`）

单音色快捷启动：
```bash
cosyvoice-server \
  --model model.gguf \
  --voice alloy \
  --prompt-speech prompt_speech.gguf
```

多音色启动示例：
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

语音请求行为（`POST /v1/audio/speech`）：
- 必填字段：`model`、`input`、`voice`。
- 标准可选字段：`instructions`、`speed`、`response_format`。
- 扩展可选字段：`seed`、`temperature`、`top_k`、`top_p`、`win_size`、`tau_r`、`min_token_text_ratio`、`max_token_text_ratio`。
- seed 优先级：请求扩展字段 `seed` > 服务端 `--seed` > 每次请求随机 seed。
- 模式自动判定：`instructions` 非空时走 instruct；否则走 zero-shot。
- 支持 `response_format`：`wav`、`pcm`，以及 FFmpeg 后端提供的 `mp3`、`aac`、`flac`、`m4a`、`opus`（前提是当前链接的 FFmpeg 运行时确实提供对应编码器）。服务启动时会在帮助文本中打印运行时实际支持的子集。
- `m4a` 是本项目提供的便捷扩展，不属于 OpenAI Speech 标准。
- `wav` 在正常构建中通过音频编码器生成；当关闭音频辅助 API 时，会回退到内部 PCM16 WAV 实现。
- `pcm` 返回原始 16 位小端单声道 PCM 负载。
- 如果客户端请求了运行时不可用的格式，服务会返回说明该格式不受支持的错误；客户端应回退到 `wav` 或 `pcm`。

OpenAI Python 客户端示例：
```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="sk-local-demo")

audio = client.audio.speech.create(
    model="cosyvoice-3",
    voice="alloy",
    input="你好，这里是 cosyvoice-server",
    instructions="语气温和、平稳。",
    response_format="mp3",
)

with open("out.mp3", "wb") as f:
    f.write(audio.read())
```

非标准扩展字段（OpenAI SDK `extra_body`）
- 标准 OpenAI Speech 请求通常使用：`model`、`input`、`voice`，以及可选 `instructions/speed/response_format`。
- `cosyvoice-server` 额外支持“请求级采样扩展字段”：
  - `seed`、`temperature`、`top_k`、`top_p`、`win_size`、`tau_r`、`min_token_text_ratio`、`max_token_text_ratio`
- 在 OpenAI Python SDK 中，通过 `extra_body` 传递这些扩展字段：

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="sk-local-demo")

audio = client.audio.speech.create(
    model="cosyvoice-3",
    voice="alloy",
    input="你好，这里是 cosyvoice-server",
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

这些扩展字段的校验规则：
- `seed`：uint32（`[0, 4294967295]`）
- `temperature`：`> 0`
- `top_k`：`>= 0`
- `top_p`：`[0, 1]`
- `win_size`：`> 0`
- `tau_r`：`>= 0`
- `min_token_text_ratio`：`>= 0`
- `max_token_text_ratio`：`>= 0`，且在同时设置时必须 `>= min_token_text_ratio`

校验失败时，服务端会返回 OpenAI 风格 `400` 错误对象。

脚本辅助：
- `tools/server/synthesize_via_api.py` 已改为直接使用 OpenAI SDK，并会把扩展参数自动放入 `extra_body`。

`synthesize_via_api.py` 使用说明：
- 该脚本建议对外发布，适合作为快速联调和本地冒烟测试工具，不仅是内部临时脚本。
- 依赖安装：

```bash
pip install openai
```

- 标准请求示例：

```bash
python tools/server/synthesize_via_api.py \
  --base-url http://127.0.0.1:8080 \
  --model cosyvoice-3 \
  --voice alloy \
  --text "你好，这里是 API" \
  --response-format wav \
  --output out.wav
```

- 批量生成（单进程顺序请求）：

```bash
python tools/server/synthesize_via_api.py \
  --base-url http://127.0.0.1:8080 \
  --model cosyvoice-3 \
  --voice alloy \
  --text "批量请求示例" \
  --response-format wav \
  --output samples/out.wav \
  --requests 5
```

当 `--requests > 1` 时，会自动生成 `out_001.wav`、`out_002.wav` ...

并发压测脚本：
- `tools/server/batch_tts_stress_test.py` 会并发发送多次请求，并保留全部输出音频文件。

```bash
python tools/server/batch_tts_stress_test.py \
  --base-url http://127.0.0.1:8080 \
  --model cosyvoice-3 \
  --workers 4 \
  --repeat 8 \
  --out-dir build/bin/Release/server_batch
```

脚本会输出每个请求对应的音频文件路径，便于你逐个试听和对比。

- 带非标准扩展字段示例：

```bash
python tools/server/synthesize_via_api.py \
  --base-url http://127.0.0.1:8080 \
  --model cosyvoice-3 \
  --voice alloy \
  --text "带采样扩展参数的请求" \
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

## CLI 工具（`tools/cli`）
可执行文件名：`cosyvoice-cli`

查看帮助：
```bash
cosyvoice-cli --help
```

### A）使用已有 prompt_speech 做 TTS
```bash
cosyvoice-cli \
  --model model.gguf \
  --prompt-speech prompt_speech.gguf \
  --text "Hello from CosyVoice" \
  --output out.wav
```

### B）前端 + TTS 一体流程（运行时提取 prompt_speech）
```bash
cosyvoice-cli \
  --model model.gguf \
  --speech-tokenizer speech_tokenizer.onnx \
  --campplus campplus.onnx \
  --prompt-audio ref.wav \
  --prompt-text "参考语音文本" \
  --text "target text" \
  --output out.wav
```

### C）仅前端模式（只生成 prompt_speech）
```bash
cosyvoice-cli \
  --frontend-only \
  --speech-tokenizer speech_tokenizer.onnx \
  --campplus campplus.onnx \
  --prompt-audio ref.wav \
  --prompt-text "参考语音文本" \
  --prompt-speech-output prompt_speech.gguf
```

### D）交互模式（REPL）
```bash
cosyvoice-cli \
  --model model.gguf \
  --prompt-speech prompt_speech.gguf \
  --interactive
```

当同时未提供 `--text` 和 `--output` 时，会自动进入交互模式。
在 `> ` 提示符下输入文本即可合成，或使用斜杠命令：

- `/play [code]`：播放缓存音频（需要音频支持；`COSYVOICE_CLI_NO_PLAYBACK=ON` 时不可用）。
- `/save <file> [code]`：保存缓存音频到文件。
- `/list`：列出缓存音频。
- `/query [code]`：查看音频详情。
- `/delete [code]`：删除缓存音频。
- `/clear`：清空缓存。
- `/seed [value]`：显示或设置下一次 seed。
- `/seed-policy <fixed|random>`：显示或设置 seed 策略。
- `/help`：显示命令列表。
- `/exit`：退出交互模式。

当省略 `[code]` 时，默认使用最近一次合成的音频。

### 参数说明

核心参数：
- `--help, -h`：显示帮助并退出。
- `--interactive`：进入交互式 REPL 模式。
- `--model, -m <file>`：TTS 使用的 CosyVoice 模型文件（`.gguf`）。
- `--backend-path <dir>`：GGML backend 所在目录。如果不指定此选项，将默认加载程序所在目录的 GGML 后端。
- `--text, -t <text>`：待合成文本。
- `--output, -o <file>`：输出音频文件路径。
  - 常规构建：输出格式由文件扩展名决定。
  - `COSYVOICE_NO_AUDIO=ON`：输出始终为 WAV。
- `--speed, -s <value>`：语速倍率，默认 `1.0`，必须大于 `0`。
- `--seed <value>`：采样与内部噪声生成的随机种子，必须是无符号 32 位整数；默认随机。
- `--max-llm-len <value>`：LLM 最大输入 token 数（`n_max_seq`），默认 `2048`，必须为正整数。
- `--threads, -j <value>`：模型推理使用的 CPU 线程数，必须是无符号 32 位整数；默认 `0`（使用当前硬件并发数）。
- `--llm-kv-cache-type <f32|f16|q8_0|q5_1|q5_0|q4_1|q4_0>`：LLM KV cache 类型，默认 `q8_0`。
- `--mode <zero-shot|instruct|cross-lingual>`：TTS 模式。默认按 `--instruction` 自动判定。
- `--instruction, -i <text>`：instruct 模式指令文本。

采样覆盖参数：
- `--temperature <value>`：采样温度，必须 `> 0`。
- `--top-k <value>`：top-k 采样大小，必须 `>= 0`。
- `--top-p <value>`：top-p 采样阈值，必须在 `[0, 1]`。
- `--win-size <value>`：重复惩罚窗口大小，必须 `> 0`。
- `--tau-r <value>`：重复惩罚系数，必须 `>= 0`。
- `--min-token-text-ratio <value>`：最小 token/text 比率，必须 `>= 0`。
- `--max-token-text-ratio <value>`：最大 token/text 比率，必须 `>= 0`，且在与 `--min-token-text-ratio` 同时设置时不能更小。

日志参数：
- `--verbose, -v`：显示完整运行信息（上下文、高级采样、内存、完整耗时）。
- `--quiet, -q`：抑制非错误日志输出。
- `--quiet` 与 `--verbose` 不能同时使用。

前端参数（仅在启用前端构建时可用，即 `COSYVOICE_NO_FRONTEND=OFF`）：
- `--frontend-only`：仅运行前端，保存 `prompt_speech` 后退出。
- `--speech-tokenizer <file>`：前端 speech tokenizer ONNX 文件。
- `--campplus <file>`：前端 campplus ONNX 文件。
- `--prompt-audio <file>`：前端参考音频。
  - 若使用 `COSYVOICE_NO_AUDIO=ON` 构建，则改用：
    - `--prompt-audio-16k <16k_pcm_file>`：16 kHz float PCM 文件。
    - `--prompt-audio-24k <24k_pcm_file>`：24 kHz float PCM 文件。
- `--prompt-text <text>`：参考音频对应文本。
- `--prompt-speech-output <file>`：将生成的 `prompt_speech` 保存到文件。

提示源相关参数：
- `--prompt-speech <file>`：直接使用已保存的 `prompt_speech` 文件。
- 提示源二选一：
  - 使用 `--prompt-speech`，或
  - 提供前端输入（`--speech-tokenizer`、`--campplus`、音频输入，以及按模式要求是否提供 `--prompt-text`）。
- 同时传入 `--prompt-speech` 和前端输入会报错。
- 典型复用流程：先用 `--frontend-only` 生成并保存 `prompt_speech.gguf`，后续合成时直接用 `--prompt-speech`。

文本规范化：
- `--disable-text-normalization`：关闭分词前 ICU 文本规范化。
- 该选项仅在启用 ICU 时可用（`COSYVOICE_NO_ICU=OFF`）。
- `--disable-text-splitting`：关闭合成前的文本分片。
- `--disable-fast-split`：关闭基于 token 的快速分片合成。
- `--disable-fade-in`：关闭默认的 20 ms 输出淡入。

运行日志：
- 模型加载前会先打印基础请求信息（模型路径、模式、提示源、输出路径、语速、解析后的 CPU 线程数、seed 来源）。
- 模型加载阶段会显示转圈动画（`| / - \\`）。
- 默认输出为简洁分区样式。
- `--verbose` 显示完整运行细节（含上下文/内存明细与完整阶段耗时）。
- `--quiet` 不显示运行信息和耗时（错误信息仍会输出）。
- 采样参数会在同一行显示来源，例如：`temperature : 1.0000 (model default)`。
- `--verbose` 下内存显示的是生成后值，并附带相对生成前快照的增量（delta）。

必填与可选：
- 常规 TTS 必填：`--model`、`--text`、`--output`，以及一个提示源（`--prompt-speech` 或前端输入）。
- `--frontend-only` 必填：`--speech-tokenizer`、`--campplus`、音频输入、`--prompt-speech-output`。
- 可选参数默认值来自 CLI 或模型元数据：
  - CLI 默认：`--speed=1.0`、`--max-llm-len=2048`、`--threads=0`（硬件并发数）、`--llm-kv-cache-type=q8_0`、`--mode=auto`。
  - 采样默认（`temperature/top_k/top_p/win_size/tau_r/token-text ratio`）来自模型配置，未传参数时不变。

## CLI 快速索引

### 不同场景的必填参数

| 场景 | 必填 |
|---|---|
| 使用已有 prompt_speech 做 TTS | `--model`、`--prompt-speech`、`--text`、`--output` |
| 前端 + TTS 一体流程 | `--model`、`--speech-tokenizer`、`--campplus`、`--prompt-audio`（或 `--prompt-audio-16k` + `--prompt-audio-24k`）、`--text`、`--output` |
| 仅前端 | `--frontend-only`、`--speech-tokenizer`、`--campplus`、音频输入、`--prompt-speech-output` |
| 交互模式 | `--model`、一个提示源，无需 `--text`/`--output` |

### 默认值速查

| 选项 | 默认值 | 来源 |
|---|---|---|
| `--mode` | `auto` | CLI |
| `--speed` | `1.0` | CLI |
| `--max-llm-len` | `2048` | CLI |
| `--threads` | `0`（硬件并发数） | CLI |
| `--llm-kv-cache-type` | `q8_0` | CLI |
| `--seed` | 随机 | 运行时 |
| `temperature/top_k/top_p/win_size/tau_r/min/max_token_text_ratio` | 模型元数据 | 模型 |

### 常用命令模板

```bash
# 复用 prompt_speech
cosyvoice-cli --model model.gguf --prompt-speech prompt_speech.gguf --text "hello" --output out.wav

# 前端 + TTS 一条命令
cosyvoice-cli --model model.gguf --speech-tokenizer speech_tokenizer.onnx --campplus campplus.onnx --prompt-audio ref.wav --prompt-text "参考文本" --text "目标文本" --output out.wav

# 仅前端（预先生成 prompt_speech）
cosyvoice-cli --frontend-only --speech-tokenizer speech_tokenizer.onnx --campplus campplus.onnx --prompt-audio ref.wav --prompt-text "参考文本" --prompt-speech-output prompt_speech.gguf

# 交互模式
cosyvoice-cli --model model.gguf --prompt-speech prompt_speech.gguf --interactive
```

### 参数校验与模式行为

- `--frontend-only` 必需：`--speech-tokenizer`、`--campplus`、音频输入、`--prompt-speech-output`。
- 常规 TTS 必需：`--model`、`--text`、`--output`，以及一个提示源。
- 未提供 `--prompt-speech` 时：
  - 必须提供前端输入；
  - `zero-shot` 模式下必须提供 `--prompt-text`；
  - `instruct`/`cross-lingual` 模式下 `--prompt-text` 会被忽略。
- `--mode` 行为：
  - `auto`：有 `--instruction` 则用 `instruct`，否则用 `zero-shot`。
  - `instruct` 且缺少 `--instruction`：给出警告并回退到 `zero-shot`。
  - `zero-shot` 且提供 `--instruction`：给出警告并忽略 `--instruction`。
  - 传入未知 mode：给出警告并自动判定模式。
- 若未启用前端（`COSYVOICE_NO_FRONTEND=ON`），则必须使用 `--prompt-speech`。
- 在 `--frontend-only` 模式下，`--seed` 会被接受但忽略（会打印警告）。
- 在 `--frontend-only` 模式下，`--threads` 会被接受但忽略（会打印警告）。
- 在 `--frontend-only` 模式下，采样覆盖参数会被接受但忽略（会打印警告）。
