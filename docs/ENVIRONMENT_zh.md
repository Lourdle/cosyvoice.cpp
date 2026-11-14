# 环境变量

运行时库（`cosyvoice`）及其工具会读取少量环境变量。所有变量均为可选，读取时采用尽力而为策略；非法值会被忽略（适用时给出警告）。

| 变量 | 读取时机 | 作用 |
|---|---|---|
| `COSYVOICE_SIMD_LEVEL` | 库加载时 | CPU DSP SIMD 分派层级封顶（仅 x86-64） |
| `COSYVOICE_MAX_WEIGHT_BUFFER_SIZE` | 模型加载时 | 覆盖单块权重缓冲的最大分配字节数 |
| `COSYVOICE_HIFT_PHASE_LOG` | 流式 TTS | 输出 HiFT 流式帧统计日志 |
| `COSYVOICE_DUMP_MEL` | 流式 TTS | 将新增 mel 帧追加写入文件，用于连续性调试 |

## `COSYVOICE_SIMD_LEVEL`

库的所有接入方（包括 `cosyvoice-cli`、`cosyvoice-server` 与 `quantize`）在**库加载时读取一次**。用于在 x86-64 上给 CPU DSP 运行时分派层级封顶。完整语义见 [SIMD_zh.md — 运行时控制 API](SIMD_zh.md#运行时控制-api)。

接受的取值（不区分大小写）：

- `auto` — 不设上限，使用 CPU 支持的最快层级（默认）。
- `scalar`（别名 `none`）— 运行时禁用 SIMD。
- `sse42`（别名 `sse4.2`）— 封顶到 SSE4.2 层。
- `avx` — 封顶到 AVX 层。
- `avx2` — 封顶到 AVX2 层。
- `avx10-256`（别名 `avx10_256`、`avx10.1-256`、`avx10_1_256`）— 隐藏两个 512 位层级；AVX10-256 层仍可用。
- `avx512`（别名 `avx-512`）— 放行全部层级（与 `auto` 等价）。

非法值会被静默忽略。库加载完成后修改该变量不再生效。优先级：显式调用 `cosyvoice_set_simd_level()`（如 CLI/server 的 `--simd-level`）> 环境变量 > `AUTO`。

示例：

```bash
COSYVOICE_SIMD_LEVEL=avx2 cosyvoice-cli --model model.gguf --prompt-speech prompt_speech.gguf --text "hello" --output out.wav
```

仅适用于 x86-64。非 x86 构建（包括 ARM64 上的 SIMDe 模拟）在编译期固定层级，忽略此变量。

## `COSYVOICE_MAX_WEIGHT_BUFFER_SIZE`

在模型加载时读取。以字节为单位覆盖加载器向后端请求的单块权重缓冲最大分配量。用于触发多缓冲权重分配路径，以及规避某些驱动对单次分配设限的后端。取值按十进制字节数解析；`0` 表示移除上限。

- 若取值小于最大的单个张量，会报错并提示缓冲上限过小。
- 若取值为*很小的*数值（低于 64 MiB），会警告将创建大量设备缓冲区，可能拖慢加载并导致设备显存碎片化。
- 非法的（非数字）取值会被忽略并给出警告。

它与后端通过 `ggml_backend_get_max_size` 报告的分配上限互补；对于不报告上限的后端（如 CPU 与 CUDA，均返回 `SIZE_MAX`），这是触发多缓冲分配路径的唯一手段。

示例：

```bash
# 强制使用 256 MiB 的权重缓冲
COSYVOICE_MAX_WEIGHT_BUFFER_SIZE=268435456 cosyvoice-cli --model model.gguf --prompt-speech prompt_speech.gguf --text "hello" --output out.wav
```

## `COSYVOICE_HIFT_PHASE_LOG`

该变量被设置为任意值（**包括空值**）都会启用 HiFT 流式帧统计——代码只检查变量是否*被设置*（`getenv` 返回非空），因此要关闭必须*不设置*该变量。流式输出期间，每个块都会向 GGML 日志回调与 stderr 输出一行统计，包含 `prompt_frames`、`input_frames`、`new_frames`、`cache_frames`、`emitted_frames` 与 `emitted_total`。用于调试 HiFT 缓存/流式接缝。

示例：

```bash
COSYVOICE_HIFT_PHASE_LOG=1 cosyvoice-cli --model model.gguf --prompt-speech prompt_speech.gguf --text "hello" --output out.wav --stream
```

## `COSYVOICE_DUMP_MEL`

设置为文件路径。流式 TTS 期间，每个块会把自己新生成的 mel 帧以简单二进制格式追加到该文件，便于检查流式拼接处的 mel 序列是否连续：

```
[uint32 n_new_frames][uint32 mel_dimension][float32 * (n_new_frames * mel_dimension)]
```

文件以追加模式打开，因此会跨多次调用/多次运行累积。要关闭必须**不设置**该变量——空字符串仍算“已设置”（`getenv` 返回非空），此时代码会尝试打开空路径并静默失败。

示例：

```bash
COSYVOICE_DUMP_MEL=/tmp/mel.bin cosyvoice-cli --model model.gguf --prompt-speech prompt_speech.gguf --text "hello" --output out.wav --stream
```

这是调试辅助手段；该格式并非稳定 API，可能随时变更。
