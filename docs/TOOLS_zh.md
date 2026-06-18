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

### 参数列表

| 参数 | 说明 |
|------|------|
| `-h, --help` | 显示帮助 |
| `-f, --file <arg>` | 输入 GGUF 文件路径 |
| `-o, --output-file <arg>` | 输出 GGUF 文件路径 |
| `-t, --type <arg>` | 默认量化类型 |
| `-c, --custom-string <arg> <arg>` | 自定义字符串元数据（可重复） |
| `-M, --tensor-map <arg>` | Tensor 名称正则到量化类型的 JSON 映射文件 |

### Tensor Map（选择性量化）

使用 `-M/--tensor-map` 可以为不同 tensor 指定不同的量化类型。
不在映射中的 tensor 会使用 `--type` 指定的默认类型。

JSON 格式（key 为 PCRE2 正则表达式）：
```json
{
    "blk\\.\\d+\\.attn_(q|k|v|o)\\.weight": "Q4_K",
    "token_embd\\.weight": "Q8_0",
    "output\\.weight": "COPY"
}
```

#### 匹配优先级

一个 tensor 名可能匹配多个 pattern，按以下规则确定胜出者：

1. **Literal（精确）pattern 始终优先于正则 pattern**。
   - `"tensor0": "Q5_K"`（literal）对 tensor `tensor0` 胜过 `"tensor.*": "Q4_K"`。
2. **正则之间，选 pattern 字符串最长的那条**（越具体越长）。
   - `"layers\\..*\\.mlp\\.down_proj\\.weight": "Q5_K"` 对 `layers.0.mlp.down_proj.weight` 胜过 `"layers.*": "Q4_K"`。
3. 平手按 JSON 中的顺序。

从未匹配过任何 tensor 的 pattern 会在运行结束后给出警告。

#### 预制 Profile

预制 tensor map 放在 `tools/quantize/profiles/`，按模型分组：

```
tools/quantize/profiles/
└── cosyvoice3-2512/
    ├── Q4_K_S.json
    └── Q4_K_M.json
```

示例：
```bash
quantize -f model.gguf -o model-q4_k_s.gguf -t Q4_K -M tools/quantize/profiles/cosyvoice3-2512/Q4_K_S.json
```

支持通过重复 `-c/--custom-string` 写入自定义字符串元数据。

## Server 工具（`tools/server`）
可执行文件名：`cosyvoice-server`

服务端有**两种运行模式**：

| 模式 | 选择方式 | 说明 |
|------|----------|------|
| **WebUI**（默认） | 不加 `--api` | 提供完整的 Web 界面，支持模型/音色管理、TTS 生成、音色提取等功能。 |
| **API** | 加 `--api` | 无界面 OpenAI Speech 兼容 API 服务。 |

### 快速开始

#### WebUI 模式（默认）
```bash
# 不带参数直接启动——WebUI 会在网页端提供模型加载功能
cosyvoice-server

# 预加载模型和音色
cosyvoice-server \
  --model model.gguf \
  --voice-prompt alloy=prompt_alloy.gguf \
  --host 0.0.0.0 \
  --port 8080
```

在浏览器中打开 http://127.0.0.1:8080。如果未指定 `--model`，WebUI 会在首页提供模型加载面板。

#### API 模式
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

### WebUI 功能特性

WebUI 是一个现代化的单页应用，提供以下功能：

#### 模型管理
- **运行时加载模型**：输入 `.gguf` 文件路径，选择后端和线程数，点击「Load Model」加载。
- **卸载模型**：释放模型内存，无需重启服务端。
- **参数配置**：KV cache 类型、buffer 策略、最大 LLM 长度——可在加载前配置。
- **动态后端选择**：启动时自动探测可用的 GGML 后端（Auto / CPU / CUDA / Vulkan / Metal）。

#### 音色管理
- **从 `.gguf` prompt speech 注册**：输入名称和预提取的 `prompt_speech.gguf` 文件路径。
- **从音频文件提取**：上传音频文件（文件选择或拖拽），输入参考文本，提取音色——需要 ONNX 前端模型。
- **麦克风录音提取**：在浏览器中直接录音，可视化波形裁剪后提取音色嵌入。
- **音频裁剪器**：可视化波形，支持起点/终点标记进行精确裁剪。
- **保存音色**：将已注册音色的 prompt speech 导出为 `.gguf` 文件。
- **删除音色**：从服务端移除已注册的音色。
- **持久化状态**：音色名称和参数通过 `localStorage` 在页面刷新间保持。

#### TTS 生成
- **文本输入**：可选择音色、模式（zero-shot / instruct / cross-lingual）和输出格式。
- **高级采样控制**：temperature、top-k、top-p、重复惩罚窗口、tau-r、随机种子（支持锁定）。
- **Instruct 模式**：输入指令控制说话风格（需要模型支持 instruct）。
- **音频播放**：生成的音频直接在浏览器中播放，配有波形可视化。
- **下载**：保存生成的音频文件。
- **历史记录**：最近最多 20 条生成记录，可点击回放或重新下载。
- **多格式输出**：`wav`、`mp3`、`flac`、`opus`、`aac`、`m4a`、`pcm`（可用格式取决于 FFmpeg 运行时）。
- **主题切换**：亮色/暗色模式，通过 `localStorage` 持久化。

#### 拖拽支持
- 音频文件可直接拖入提取标签页或音色注册区域。

### 核心启动参数

| 参数 | 说明 |
|------|------|
| `--help, -h` | 显示帮助并退出。 |
| `--model, -m <file>` | CosyVoice 模型文件（`.gguf`）。 |
| `--backend-path <dir>` | GGML backend 所在目录。如果不指定，默认加载程序所在目录的 GGML 后端。 |
| `--backend <name>` | GGML 后端名称（如 `cpu`、`cuda0`、`vulkan`、`metal`）。默认 `auto`（自动选择最佳后端）。与 `--cpu`/`--cuda` 互斥。 |
| `--cpu` | 使用 CPU 后端（等价于 `--backend cpu`）。与 `--cuda`/`--backend` 互斥。 |
| `--cuda` | 使用 CUDA 后端（等价于 `--backend cuda0`）。与 `--cpu`/`--backend` 互斥。 |
| `--served-model-name <name>` | API/WebUI 对外模型名。如果省略，优先使用 `cosyvoice_get_architecture()` 返回的模型架构名；否则从文件名推导。 |
| `--host <host>` | 监听地址，默认 `127.0.0.1`。 |
| `--port <port>` | 监听端口，默认 `8080`。 |
| `--api-key <key>` | 传入时要求请求包含 `Authorization: Bearer <key>`。 |

### 模式选择

| 参数 | 说明 |
|------|------|
| `--api` | 启用 OpenAI 兼容 API 模式。不加此标志默认为 WebUI 模式。 |

### 音色映射

| 参数 | 说明 |
|------|------|
| `--voice-prompt <voice=prompt_speech.gguf>` | 将一个 `voice` 名映射到一个 `prompt_speech` 文件（可重复传入）。 |
| `--voice <voice>` | 单音色模式的音色名，默认 `alloy`。 |
| `--prompt-speech <file>` | 单音色模式的 prompt speech 文件（与 `--voice` 配合使用，等价于 `--voice-prompt alloy=file`）。 |

**API 模式**下至少需要一个音色映射。**WebUI 模式**下音色映射可选——可以在运行时通过界面注册。

### WebUI 前端模型参数

这些参数指向 ONNX 模型文件，用于音色提取（从音频文件或麦克风录音）：

| 参数 | 说明 |
|------|------|
| `--frontend-model <dir>` | 前端 ONNX 模型目录。 |
| `--speech-tokenizer <file>` | Speech tokenizer ONNX 模型文件。 |
| `--campplus <file>` | Campplus ONNX 模型文件。 |

如果在启动时提供，WebUI 会自动填入前端配置字段。前端也可以在运行时通过 WebUI 界面加载。

### 运行时调优参数

| 参数 | 说明 |
|------|------|
| `--max-llm-len <value>` | LLM 最大序列长度，默认 `2048`。 |
| `--threads, -j <value>` | CPU 线程数，默认 `0`（硬件并发数）。 |
| `--concurrency, -c <value>` | 并发请求槽数，默认 `1`（仅 API 模式）。 |
| `--inference-buffer-policy <shared\|balanced\|dedicated>` | 推理缓冲区策略，默认 `balanced`。 |
| `--llm-kv-cache-type <f32\|f16\|q8_0\|q5_1\|q5_0\|q4_1\|q4_0\|k=<type>,v=<type>[,fallback=<type>]>` | KV cache 类型。单一类型（如 `q8_0`）为 K 和 V 使用相同格式。默认 `k=q8_0,v=q4_0,fallback=q8_0`。 |
| `--seed <value>` | 默认随机种子（当请求未传 seed 时使用）。 |

### 采样默认值覆盖（服务级）

| 参数 | 说明 |
|------|------|
| `--temperature <value>` | 采样温度（`> 0`）。 |
| `--top-k <value>` | 采样 top-k（`>= 0`）。 |
| `--top-p <value>` | 采样 top-p（`[0, 1]`）。 |
| `--win-size <value>` | 重复惩罚窗口大小（`> 0`）。 |
| `--tau-r <value>` | 重复惩罚系数（`>= 0`）。 |
| `--min-token-text-ratio <value>` | 最小 token/text 比率（`>= 0`）。 |
| `--max-token-text-ratio <value>` | 最大 token/text 比率（`>= 0`，且不小于 `--min-token-text-ratio`）。 |

### 文本规范化与分片

| 参数 | 说明 |
|------|------|
| `--disable-text-normalization` | 关闭 ICU 文本规范化（仅 `COSYVOICE_NO_ICU=OFF` 时可用）。 |
| `--disable-text-splitting` | 关闭 TTS 中的文本分片。 |
| `--disable-fast-split` | 关闭基于 token 的快速分片。 |

### 输出后处理

| 参数 | 说明 |
|------|------|
| `--disable-fade-in` | 关闭默认的 20 ms 输出淡入。 |

### 日志参数

| 参数 | 说明 |
|------|------|
| `--verbose, -v` | 详细日志（高级采样、内存、完整耗时）。 |
| `--quiet, -q` | 抑制非错误日志。 |

---

### WebUI HTTP API 参考

WebUI 暴露以下 REST 接口，由前端 JavaScript 调用：

#### `GET /`

返回 WebUI HTML 页面，包含服务端注入的配置（可用后端、前端模型路径等）。

#### `GET /ping`

简单存活检测。返回 `pong`。

#### `GET /backends`

返回可用 GGML 后端的 JSON 数组：
```json
[
  {“name”: “auto”, “description”: “自动选择最佳后端”},
  {“name”: “CPU”, “description”: “…”, “has_device_id”: false},
  {“name”: “CUDA0”, “description”: “…”, “has_device_id”: true}
]
```

#### `GET /formats`

返回支持的音频输出格式名称 JSON 数组：
```json
[“wav”, “pcm”, “mp3”, “flac”, “opus”]
```
具体可用格式取决于链接的 FFmpeg 运行时。

#### `GET /status`

返回服务端状态 JSON：
```json
{
  “status”: “ok”,
  “model_loaded”: true,
  “model”: “cosyvoice-3”,
  “sample_rate”: 24000,
  “max_llm_len”: 2048,
  “k_cache_type”: “Q8_0”,
  “v_cache_type”: “Q4_0”,
  “buffer_policy”: “balanced”,
  “model_arch”: “cosyvoice3-2512”,
  “frontend_available”: false,
  “speakers”: [“alloy”, “nova”]
}
```

#### `GET /speaker`

返回已注册音色名称的 JSON 数组：
```json
[“alloy”, “nova”]
```

#### `POST /speaker` — 注册音色

支持两种 Content-Type：

**JSON**（从 `.gguf` prompt speech 注册）：
```json
{“type”: “gguf”, “name”: “alloy”, “path”: “/data/prompts/alloy.gguf”}
```

**Multipart**（从音频提取）：
```
POST /speaker?type=extract
Content-Type: multipart/form-data
字段: name, text, audio (文件)
```

成功返回 `200`，名称已存在返回 `409`，未加载模型返回 `503`。

#### `DELETE /speaker/:name`

注销指定音色。成功返回 `200`，音色不存在返回 `404`。

#### `POST /speaker/save`

将音色的 prompt speech 导出为 `.gguf` 文件到服务端磁盘。

JSON 请求体：
```json
{“name”: “alloy”, “path”: “/data/exports/alloy_export.gguf”}
```

#### `POST /tts`

生成语音。JSON 请求体：

```json
{
  “text”: “你好，这里是 CosyVoice”,
  “voice”: “alloy”,
  “mode”: “zero-shot”,
  “response_format”: “wav”,
  “speed”: 1.0,
  “seed”: 12345,
  “temperature”: 0.8,
  “top_k”: 32,
  “top_p”: 0.9,
  “win_size”: 10,
  “tau_r”: 0.2,
  “min_token_text_ratio”: 1.0,
  “max_token_text_ratio”: 5.0,
  “instruction”: “语气温和、平稳。”,
  “fadein”: true
}
```

返回原始音频字节，带对应 `Content-Type` 头（`audio/wav`、`audio/mpeg` 等）。

#### `GET /model/defaults`

返回模型默认参数 JSON：
```json
{
  “max_llm_len”: 2048,
  “temperature”: 1.0,
  “top_k”: 50,
  “top_p”: 0.9,
  …
}
```

#### `POST /model/load`

运行时加载模型。JSON 请求体：
```json
{
  “model_path”: “/data/models/cosyvoice-3.gguf”,
  “backend”: “auto”,
  “n_threads”: 8,
  “max_llm_len”: 2048,
  “buffer_policy”: “balanced”,
  “k_cache_type”: “q8_0”,
  “v_cache_type”: “q4_0”
}
```

#### `POST /model/unload`

卸载当前模型（无需 JSON 请求体）。

#### `GET /frontend/model`

返回当前前端 ONNX 模型配置 JSON。

#### `POST /frontend/model/load`

运行时加载前端 ONNX 模型。JSON 请求体：
```json
{
  “speech_tokenizer”: “/data/models/speech_tokenizer.onnx”,
  “campplus”: “/data/models/campplus.onnx”
}
```

#### `POST /frontend/model/unload`

卸载前端 ONNX 模型。

---

### API 模式端点（OpenAI 兼容）

- `GET /healthz`
- `GET /v1/models`
- `POST /v1/audio/speech`

鉴权行为：
- 传入 `--api-key`：请求必须包含 `Authorization: Bearer <api_key>`。
- 不传 `--api-key`：不做鉴权。

#### 语音请求行为（`POST /v1/audio/speech`）：
- 必填字段：`model`、`input`、`voice`。
- 标准可选字段：`instructions`、`speed`、`response_format`。
- 扩展可选字段：`seed`、`temperature`、`top_k`、`top_p`、`win_size`、`tau_r`、`min_token_text_ratio`、`max_token_text_ratio`。
- seed 优先级：请求扩展字段 `seed` > 服务端 `--seed` > 随机 seed。
- 模式自动判定：`instructions` 非空时走 instruct；否则走 zero-shot。
- 支持 `response_format`：`wav`、`pcm`，以及 FFmpeg 后端提供的 `mp3`、`aac`、`flac`、`m4a`、`opus`。
- `m4a` 是本项目的便捷扩展，不属于 OpenAI Speech 标准。
- `wav` 通过音频编码器生成；关闭音频辅助 API 时回退到内部 PCM16 WAV 实现。
- `pcm` 返回原始 16 位小端单声道 PCM 负载。
- 格式不可用时返回错误。

OpenAI Python 客户端示例：
```python
from openai import OpenAI

client = OpenAI(base_url=”http://127.0.0.1:8080/v1”, api_key=”sk-local-demo”)

audio = client.audio.speech.create(
    model=”cosyvoice-3”,
    voice=”alloy”,
    input=”你好，这里是 cosyvoice-server”,
    instructions=”语气温和、平稳。”,
    response_format=”mp3”,
)

with open(“out.mp3”, “wb”) as f:
    f.write(audio.read())
```

非标准扩展字段（OpenAI SDK `extra_body`）：
```python
from openai import OpenAI

client = OpenAI(base_url=”http://127.0.0.1:8080/v1”, api_key=”sk-local-demo”)

audio = client.audio.speech.create(
    model=”cosyvoice-3”,
    voice=”alloy”,
    input=”你好，这里是 cosyvoice-server”,
    response_format=”wav”,
    extra_body={
        “seed”: 123,
        “temperature”: 0.8,
        “top_k”: 32,
        “top_p”: 0.9,
        “win_size”: 10,
        “tau_r”: 0.2,
        “min_token_text_ratio”: 1.0,
        “max_token_text_ratio”: 5.0,
    },
)

with open(“out.wav”, “wb”) as f:
    f.write(audio.read())
```

校验规则：
- `seed`：uint32（`[0, 4294967295]`）
- `temperature`：`> 0`
- `top_k`：`>= 0`
- `top_p`：`[0, 1]`
- `win_size`：`> 0`
- `tau_r`：`>= 0`
- `min_token_text_ratio`：`>= 0`
- `max_token_text_ratio`：`>= 0` 且同时设置时 `>= min_token_text_ratio`

校验失败时返回 OpenAI 风格 `400` 错误对象。

### 请求/响应格式处理

两种模式（WebUI 和 API）共享相同的音频编码逻辑：

| 格式 | Content-Type | 后端 |
|------|-------------|------|
| `wav` | `audio/wav` | 音频编码器或内部 PCM16 WAV 回退 |
| `pcm` | `audio/L16` | 原始 PCM16 小端字节 |
| `mp3` | `audio/mpeg` | FFmpeg 编码器（运行时决定） |
| `flac` | `audio/flac` | FFmpeg 编码器（运行时决定） |
| `opus` | `audio/opus` | FFmpeg 编码器（运行时决定） |
| `aac` | `audio/aac` | FFmpeg 编码器（运行时决定） |
| `m4a` | `audio/mp4` | FFmpeg 编码器（运行时决定） |

### 辅助脚本

#### `tools/server/synthesize_via_api.py`

直接使用 OpenAI SDK，自动将扩展参数放入 `extra_body`。

依赖安装：`pip install openai`

标准请求示例：
```bash
python tools/server/synthesize_via_api.py \
  --base-url http://127.0.0.1:8080 \
  --model cosyvoice-3 \
  --voice alloy \
  --text “你好，这里是 API” \
  --response-format wav \
  --output out.wav
```

批量生成（单进程顺序请求）：
```bash
python tools/server/synthesize_via_api.py \
  --base-url http://127.0.0.1:8080 \
  --model cosyvoice-3 \
  --voice alloy \
  --text “批量请求示例” \
  --response-format wav \
  --output samples/out.wav \
  --requests 5
```

带扩展字段示例：
```bash
python tools/server/synthesize_via_api.py \
  --base-url http://127.0.0.1:8080 \
  --model cosyvoice-3 \
  --voice alloy \
  --text “带采样扩展参数的请求” \
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

并发压测脚本，会并发发送多次请求并保留全部输出音频文件。

```bash
python tools/server/batch_tts_stress_test.py \
  --base-url http://127.0.0.1:8080 \
  --model cosyvoice-3 \
  --workers 4 \
  --repeat 8 \
  --out-dir build/bin/Release/server_batch
```

脚本会输出每个请求对应的音频文件路径。

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

- `/play [code]`：播放缓存音频（需要音频支持；`COSYVOICE_CLI_NO_PLAYBACK=ON` 时不可用）。按 `Ctrl+C` 可停止播放。
- `/save <file> [code]`：保存缓存音频到文件。
- `/list`：列出缓存音频。
- `/query [code]`：查看音频详情。
- `/delete [code]`：删除缓存音频。
- `/clear`：清空缓存。
- `/seed [value]`：显示或设置下一次 seed。
- `/seed-policy <fixed|random>`：显示或设置 seed 策略。
- `/help`：显示命令列表。
- `/exit`：退出交互模式，`Ctrl+C` 也可以退出。

当省略 `[code]` 时，默认使用最近一次合成的音频。

### 参数说明

核心参数：
- `--help, -h`：显示帮助并退出。
- `--interactive`：进入交互式 REPL 模式。
- `--model, -m <file>`：TTS 使用的 CosyVoice 模型文件（`.gguf`）。
- `--backend-path <dir>`：GGML backend 所在目录。如果不指定此选项，将默认加载程序所在目录的 GGML 后端。
- `--backend <name>`：GGML 后端名称（如 `cpu`、`cuda0`、`vulkan`、`metal`）。默认 `auto`（自动选择最佳后端）。与 `--cpu`/`--cuda` 互斥。
- `--cpu`：使用 CPU 后端（等价于 `--backend cpu`）。与 `--cuda`/`--backend` 互斥。
- `--cuda`：使用 CUDA 后端（等价于 `--backend cuda0`）。与 `--cpu`/`--backend` 互斥。
- `--text, -t <text>`：待合成文本。
- `--output, -o <file>`：输出音频文件路径。
  - 常规构建：输出格式由文件扩展名决定。
  - `COSYVOICE_NO_AUDIO=ON`：输出始终为 WAV。
- `--speed, -s <value>`：语速倍率，默认 `1.0`，必须大于 `0`。
- `--seed <value>`：采样与内部噪声生成的随机种子，必须是无符号 32 位整数；默认随机。
- `--max-llm-len <value>`：LLM 最大输入 token 数（`n_max_seq`），默认 `2048`，必须为正整数。
- `--threads, -j <value>`：模型推理使用的 CPU 线程数，必须是无符号 32 位整数；默认 `0`（使用当前硬件并发数）。
- `--llm-kv-cache-type <f32|f16|q8_0|q5_1|q5_0|q4_1|q4_0|k=<type>,v=<type>[,fallback=<type>]>`：LLM KV cache 类型。单一类型（如 `q8_0`）为 K 和 V 使用相同格式。可使用独立 K/V 类型（如 `k=q8_0,v=q4_0`）指定不同格式。默认 `k=q8_0,v=q4_0,fallback=q8_0`。
- `--inference-buffer-policy <shared|balanced|dedicated>`：推理缓冲区策略。仅在交互模式下生效；非交互模式始终使用 `shared` 以最小化内存占用并获得最快的单次合成速度。默认 `balanced`。
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
- 后端信息现在包含 UMA（统一内存架构）检测结果和生效的缓冲策略。
- 当检测到 UMA 且请求的缓冲策略为 `balanced` 时，库会自动切换为 `dedicated` 以获得更好性能；此时会打印日志提示。
- 默认输出为简洁分区样式。
- `--verbose` 显示完整运行细节（含上下文/内存明细与完整阶段耗时）。
- `--quiet` 不显示运行信息和耗时（错误信息仍会输出）。
- 采样参数会在同一行显示来源，例如：`temperature : 1.0000 (model default)`。
- `--verbose` 下内存显示的是生成后值，并附带相对生成前快照的增量（delta）。

必填与可选：
- 常规 TTS 必填：`--model`、`--text`、`--output`，以及一个提示源（`--prompt-speech` 或前端输入）。
- `--frontend-only` 必填：`--speech-tokenizer`、`--campplus`、音频输入、`--prompt-speech-output`。
- 可选参数默认值来自 CLI 或模型元数据：
  - CLI 默认：`--speed=1.0`、`--max-llm-len=2048`、`--threads=0`（硬件并发数）、`--llm-kv-cache-type=k=q8_0,v=q4_0,fallback=q8_0`、`--mode=auto`。
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
| `--llm-kv-cache-type` | `k=q8_0,v=q4_0,fallback=q8_0` | CLI |
| `--inference-buffer-policy` | `balanced`（交互模式）/ `shared`（非交互模式） | CLI |
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
