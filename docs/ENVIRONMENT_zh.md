# 环境变量

运行时库（`cosyvoice`）及其工具会读取少量环境变量。所有变量均为可选，读取时采用尽力而为策略；非法值会被忽略（适用时给出警告）。

| 变量 | 读取时机 | 作用 |
|---|---|---|
| `COSYVOICE_SIMD_LEVEL` | 库加载时 | CPU DSP SIMD 分派层级封顶（仅 x86-64） |
| `COSYVOICE_MAX_WEIGHT_BUFFER_SIZE` | 模型加载时 | 覆盖单块权重缓冲的最大分配字节数 |
| `COSYVOICE_NO_MMAP_ZEROCOPY` | 模型加载时 | 关闭从文件映射直接使用权重（CPU 后端） |
| `COSYVOICE_LOG_VERBOSE` | 日志 | 在 Release 构建中输出 INFO/DEBUG 日志 |
| `COSYVOICE_HIFT_PHASE_LOG` | 流式 TTS | 输出 HiFT 流式帧统计日志 |
| `COSYVOICE_DUMP_MEL` | 流式 TTS | 将新增 mel 帧追加写入文件，用于连续性调试 |
| `COSYVOICE_STAGE_TIMING` | TTS 推理 | 输出各推理阶段的耗时 |
| `COSYVOICE_SCHED_RESERVE` | 模型加载时 | 为 Flow 缓冲按最坏 mel 长度预预留 |
| `COSYVOICE_MEMORY_POOL_BYTES` | 运行时初始化 | 为内部 C++ 容器预分配进程级内存池，单位为字节（0/未设置=关闭） |

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

## `COSYVOICE_NO_MMAP_ZEROCOPY`

在模型加载时读取。当模型从文件加载时，CPU 后端会直接使用文件映射：权重张量指向映射的 GGUF，而不是复制到单独的 CPU 缓冲区。这样可大约省下一份模型大小的常驻内存，并加快加载。将该变量设为除 `0` 以外的任意值可强制回退到旧的拷贝行为。

- 仅在当前设备为 CPU 后端且支持 `buffer_from_host_ptr` 时生效；GPU 加载始终把权重拷贝到设备。
- 上下文存活期间映射文件保持打开。Windows 上该文件以 `FILE_SHARE_DELETE` 打开，因此仍可被改名或删除；删除会在上下文释放时生效，在此之前原映射仍可正常读取。

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

## `COSYVOICE_LOG_VERBOSE`

控制库的 INFO/DEBUG 消息是否输出。Debug 构建（`_DEBUG`）默认开启；Release 构建默认关闭，除非设置该变量（任意值，包括空值），或设置了下面的诊断变量（`COSYVOICE_STAGE_TIMING`、`COSYVOICE_HIFT_PHASE_LOG`）。警告与错误始终输出。

所有输出都经过当前生效的 GGML 日志回调；默认回调把这些级别写到 stdout。

## `COSYVOICE_HIFT_PHASE_LOG`

该变量被设置为任意值（**包括空值**）都会启用 HiFT 流式帧统计——代码只检查变量是否*被设置*（`getenv` 返回非空），因此要关闭必须*不设置*该变量。流式输出期间，每个块都会通过 GGML 日志回调（INFO）输出一行统计，包含 `prompt_frames`、`input_frames`、`new_frames`、`cache_frames`、`emitted_frames` 与 `emitted_total`。设置该变量也会启用 verbose（见 `COSYVOICE_LOG_VERBOSE`）。用于调试 HiFT 缓存/流式接缝。

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

## `COSYVOICE_STAGE_TIMING`

该变量被设置为任意值（**包括空值**）都会启用分阶段推理计时。库会统计流水线各阶段耗时，并通过 GGML 日志回调（INFO）输出每行；设置该变量也会启用 verbose，因此 Release 构建下同样可见。

- `stage timing: text: ...` — 每次高层 TTS 调用一行：emoji 移除、文本规范化与分词。
- `stage timing: ...` — 每个合成轮次（一次 LLM 任务加一次声码器）一行：LLM prefill/decode/epilogue、flow encode、DiT 扩散、mel 组装与 HiFT，并附带每 token/每 step 均值和图重建次数。流式模式下带 chunk 序号前缀；首 chunk 的耗时即首包延迟。`buffers=`/`t2w=` 为当前调度器计算缓冲与 Flow 中间缓冲的大小，可看出跨轮次是否增长。
- `stage timing: stream: ...` — 整个流式请求的汇总：chunk 数、总墙钟时间与各阶段累计。
- `stage timing: sched re-reserved ...` — 图调度器需要重新预留计算缓冲（会同步所有后端）；该行包含重新预留前的缓冲大小，轮次行中计入 `sched_reallocs`。
- `stage timing: token2wav re-allocated (dit context|mel): X -> Y MiB` — Flow 中间缓冲在内容增长时被释放并按新尺寸重新分配；只会在未启用 `COSYVOICE_SCHED_RESERVE` 或超出预留长度时出现。轮次行中计入 `t2w_reallocs`。

未设置该变量时，插桩在每个阶段边界只多一次分支判断，且不读取时钟。

示例：

```bash
COSYVOICE_STAGE_TIMING=1 cosyvoice-cli --model model.gguf --prompt-speech prompt_speech.gguf --text "hello" --output out.wav --stream
```


## `COSYVOICE_SCHED_RESERVE`

在模型加载时为 Flow（DiT）按最坏 mel 长度预先预留缓冲：包括调度器计算缓冲、在 step 之间承载 DiT 状态与累积 mel 的中间缓冲，以及宿主侧 mel 缓存。后端调度器在图的布局变化时会重新规划分配，且缓冲只增不减；因此在默认（不带 KV cache）的 Flow 路径上，流式请求会随着累积 mel 增长而在每个 chunk 重新分配缓冲。提前预留可以把这部分分配挪到加载期，不再落在推理路径上。只有 Flow 路径被预留；LLM 与 HiFT 图保持原有行为。

取值是 mel 帧数：

- 未设置、为负数或非数字——关闭（非法值会打印警告）。
- `0`——使用由上下文参数推导的长度：`n_max_seq * token_mel_ratio`（模型可能产出的最长 mel）。
- `N > 0`——按 `N` 个 mel 帧预留，并截断到推导出的上限。

预留长度是计算内存的固定开销：即使短请求，调度器缓冲也会预先增长到该最坏情况（参考模型约每帧 0.7 MiB）。请按预期的最长音频来设置；超过预留的请求仍可正常工作，只是像以前一样继续增长缓冲。配置了 DiT KV cache（`dit_kv_fixed_slots`/`dit_kv_offloadable_slots`）时会跳过预留，因为该路径下每 chunk 的图尺寸本就有界。

未设置该变量时不进行预留，这也是默认行为。

示例：

```bash
COSYVOICE_SCHED_RESERVE=2048 cosyvoice-cli --model model.gguf --prompt-speech prompt_speech.gguf --text "hello" --output out.wav --stream
```

## `COSYVOICE_MEMORY_POOL_BYTES`

为库内部的 C++ 容器（分词器、文本分块、prompt/TTS 状态以及其它 `src/` 运行时结构）预分配一个进程级内存池。池由 `std::pmr::synchronized_pool_resource` 支持，初始化时预留 arena。不超过 32 KiB 的请求由池服务；更大或要求超对齐的请求直接走系统分配器。arena 用尽后池会继续从系统堆分配，因此该值不是硬性内存上限。

取值是字节数：

- 未设置或 `0`——关闭（默认）。
- `N > 0`——启用池，初始 arena 为 `N` 字节；会按 arena 对齐（64 KiB）向上取整。
- 负数或非数字——关闭并打印警告。

池是进程级的，只能初始化一次。环境变量在库被加载时、任何库调用之前生效，因此必须在进程启动前设置。也可以改用 `cosyvoice_init_runtime()`；由于变量在加载时就已生效，它优先于之后的显式调用——后者会以“已初始化”拒绝。如果池化容器已经分配过内存，再启用池会被警告拒绝，系统分配器保持生效。

池的创建与释放是进程生命周期的：arena 不会归还给操作系统，而是在进程存续期间复用。该池不覆盖 ggml/后端张量、ICU 内部的内存分配，也不覆盖 `tools/`/`tests/` 代码。启动横幅与警告信息在库被加载时打印一次，此时尚未安装 verbose 日志过滤，因此即使不设 `COSYVOICE_LOG_VERBOSE` 也能看到。

示例：

```bash
COSYVOICE_MEMORY_POOL_BYTES=33554432 COSYVOICE_LOG_VERBOSE=1 cosyvoice-cli --model model.gguf --prompt-speech prompt_speech.gguf --text "hello" --output out.wav --stream
```
