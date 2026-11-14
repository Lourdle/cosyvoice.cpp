# 流式 TTS、DiT KV 缓存与 Buffer 策略

## 流式 TTS

流式 TTS 在合成过程中通过回调函数逐段交付音频，无需等待完整语句生成完毕即可开始播放，从而实现实时播放与更低的主观延迟。

流式输出通过 CLI/server 的 `--stream` 标志启用；分块粒度由 `--chunk-tokens` 控制。

## DiT KV 缓存

每个块都要跑完 DiT 的扩散步（默认 10 步）——若无缓存，每个新块的每一步都会对此前已产出的整个音频序列重算注意力。**DiT KV 缓存**按扩散步保存已产出位置的注意力 key/value（每个扩散步一个缓存槽位），新块只计算自己新增的位置：每个位置在每个步上只计算一次，之后的所有块直接复用。

> DiT KV 缓存**仅在流式 TTS 时使用**；非流式调用忽略此缓存。

### 槽位组织

DiT KV 缓存按 **槽位（slot）** 组织，每个槽位对应一个扩散步的 KV 缓存。默认 10 个扩散步，因此至多 10 个设备槽位（每步一个）。

槽位分为三类：

| 类别 | 内存位置 | 行为 |
|----------|--------|------|
| **固定** | 常驻设备（GPU） | 最快，从不卸载 |
| **可卸载** | 不使用时卸载到 CPU | 节省设备显存，但增加传输开销 |
| **不缓存** | 不存储 | 每步全量重算注意力，无额外内存开销 |

**步-槽位映射。** 扩散步按顺序排列——不缓存步在最前，然后是可卸载步，最后是固定步。固定步各独占一个设备槽位；可卸载步共享同一个设备临时槽（设备槽 0），并各自持有一个 CPU 缓冲区来拷贝 KV。可卸载槽数恰好为 1 时会被归一化为固定槽位（`offloadable=1 → fixed+1`），且两个数量都会被裁剪，保证 `fixed + offloadable` 不超过扩散步数。总缓存步数 = `固定 + 可卸载`，其余步全量重算。内部槽位/调度布局详见 [API_zh_cosyvoice.md — DiT KV 缓存概念](API_zh_cosyvoice.md#dit-kv-缓存概念)。

相邻扩散步也可通过 `--dit-kv-actual-fixed-slots` / `--dit-kv-actual-offloadable-slots` **共享**一份物理缓存：固定步与可卸载步各自被划分为该数量的分组，每组共享一个设备槽位 / CPU 缓冲区，缓存缩小为名义值的 `实际/名义`。组内各步读取的历史为组内最后写入者的结果（一种很小的近似），且 CPU 往返从每步一次变为每组一次。

KV 缓存占用较大，因此默认 0 个槽位（全部 10 步全量重算）。启用缓存后，若序列长度超过配置的缓存长度，会丢弃部分位置——推理可正常继续，但输出质量可能下降。可卸载槽位需要和设备与 CPU 间传输数据，可能无法带来速度提升，甚至比全量重算更慢。

### 配置

DiT KV 缓存参数通过 CLI/server 的 `--dit-kv-*` 参数配置（context 参数——CLI 侧作用于交互/流式模式）：

- `--dit-kv-cache-type`：缓存存储格式——`f32`/`f16`/`q8_0`/`q5_1`/`q5_0`/`q4_1`/`q4_0`，或 K/V 独立指定的非对称格式（`k=<type>,v=<type>[,fallback=<type>]`，与 `--llm-kv-cache-type` 同风格）。
- `--dit-kv-fixed-slots`：常驻设备内存的槽位数。默认：`0`。
- `--dit-kv-offloadable-slots`：可卸载到 CPU 的槽位数。默认：`0`。
- `--dit-kv-actual-fixed-slots`：支撑固定槽位的物理设备槽位数（相邻步共享一份）。`0` = 不共享。默认：`0`。
- `--dit-kv-actual-offloadable-slots`：支撑可卸载槽位的物理 CPU 缓冲区数（相邻步共享一个）。`0` = 不共享。默认：`0`。
- `--dit-kv-cache-length`：缓存保留的最大序列位置数。默认：`0` = 最大 LLM 长度 × 10。

扩散步数来自模型的 `decoder.diffusion_steps` GGUF 元数据（缺失时默认为 10）。可在加载时通过 `cosyvoice_context_params_v4_t::diffusion_steps` 覆盖——CLI/Server 对应 `--diffusion-steps`（传 `<= 0` 表示沿用元数据值）；任何取值都会被裁剪到运行时上限 50。实际生效值可用 `cosyvoice_get_diffusion_steps()` 查询。

建议的起点（共 10 个扩散步）：

| 目标 | `--dit-kv-fixed-slots` | `--dit-kv-offloadable-slots` |
|------|------------------------|------------------------------|
| 禁用缓存（默认） | `0` | `0` |
| 最快，显存充足 | `10` | `0` |
| 速度/显存均衡 | `2` | `0` |
| 提速并降低显存占用 | `1` | `9` |
| 显存占用最小（较慢） | `0` | `9` |

## 推理 Buffer 策略

推理引擎使用 buffer 策略控制中间张量的分配方式（默认 `dedicated`；CLI 非交互单次合成路径始终使用 `shared`）：

- `shared`：LLM KV 缓存与 DiT 部分中间 buffer 共享内存。每次推理 LLM 模块都会完整运行一次。节省内存，但在未启用 Flash Attention 时 CUDA 上运行不稳定。
- `balanced`：与 `shared` 类似，但在 LLM 推理完成后将下次可复用的 LLM KV 缓存卸载到 CPU。
- `dedicated`：完全独立分配。LLM KV 缓存驻留设备可快速复用——流式推理建议使用此模式。

Buffer 策略是 context 参数，可通过 `cosyvoice_context_params_v4_t` 配置（CLI/server 对应 `--inference-buffer-policy`，WebUI 亦可设置）。
