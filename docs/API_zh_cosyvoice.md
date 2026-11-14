# cosyvoice.h API 参考

本文档覆盖 `include/cosyvoice.h` 中声明的全部公开符号。
运行时支持通过多个 worker 槽实现并发推理；复制上下文后，不同线程可以绑定不同 worker，同时共享已加载的模型资源。

## COSYVOICE_API

### 语法

```c
#ifndef COSYVOICE_API
    #ifdef COSYVOICE_STATIC
        #define COSYVOICE_API
    #else
        #ifdef _WIN32
            #define COSYVOICE_API __declspec(dllimport)
        #else
            #define COSYVOICE_API __attribute__((visibility("default")))
        #endif
    #endif
#endif
```

### 说明

控制 C API 符号在不同链接方式和平台下的可见性。

### 备注

静态链接时定义 `COSYVOICE_STATIC` 可关闭导入属性；Windows 默认使用 `__declspec(dllimport)`。

## cosyvoice_kv_cache_type_t

### 语法

```c
typedef enum cosyvoice_kv_cache_type
{
    COSYVOICE_KV_CACHE_TYPE_F32,
    COSYVOICE_KV_CACHE_TYPE_F16,
    COSYVOICE_KV_CACHE_TYPE_Q8_0,
    COSYVOICE_KV_CACHE_TYPE_Q5_1,
    COSYVOICE_KV_CACHE_TYPE_Q5_0,
    COSYVOICE_KV_CACHE_TYPE_Q4_1,
    COSYVOICE_KV_CACHE_TYPE_Q4_0,
    COSYVOICE_KV_CACHE_TYPE_COUNT
} cosyvoice_kv_cache_type_t, cosyvoice_llm_kv_cache_type_t;
```

#### 向后兼容别名

旧的 `cosyvoice_llm_kv_cache_type_t` 和 `COSYVOICE_LLM_KV_CACHE_TYPE_*` 常量以后向兼容宏提供：

```c
#define COSYVOICE_LLM_KV_CACHE_TYPE_F32   COSYVOICE_KV_CACHE_TYPE_F32
#define COSYVOICE_LLM_KV_CACHE_TYPE_F16   COSYVOICE_KV_CACHE_TYPE_F16
#define COSYVOICE_LLM_KV_CACHE_TYPE_Q8_0  COSYVOICE_KV_CACHE_TYPE_Q8_0
#define COSYVOICE_LLM_KV_CACHE_TYPE_Q5_1  COSYVOICE_KV_CACHE_TYPE_Q5_1
#define COSYVOICE_LLM_KV_CACHE_TYPE_Q5_0  COSYVOICE_KV_CACHE_TYPE_Q5_0
#define COSYVOICE_LLM_KV_CACHE_TYPE_Q4_1  COSYVOICE_KV_CACHE_TYPE_Q4_1
#define COSYVOICE_LLM_KV_CACHE_TYPE_Q4_0  COSYVOICE_KV_CACHE_TYPE_Q4_0
#define COSYVOICE_LLM_KV_CACHE_TYPE_COUNT COSYVOICE_KV_CACHE_TYPE_COUNT
```

### 说明

指定 KV 缓存存储格式。同时用于 LLM 和 DiT 模块。

### 枚举值

- `COSYVOICE_KV_CACHE_TYPE_F32`：32 位浮点格式。
- `COSYVOICE_KV_CACHE_TYPE_F16`：16 位浮点格式。
- `COSYVOICE_KV_CACHE_TYPE_Q8_0`：GGML `Q8_0` 量化格式。
- `COSYVOICE_KV_CACHE_TYPE_Q5_1`：GGML `Q5_1` 量化格式。
- `COSYVOICE_KV_CACHE_TYPE_Q5_0`：GGML `Q5_0` 量化格式。
- `COSYVOICE_KV_CACHE_TYPE_Q4_1`：GGML `Q4_1` 量化格式。
- `COSYVOICE_KV_CACHE_TYPE_Q4_0`：GGML `Q4_0` 量化格式。
- `COSYVOICE_KV_CACHE_TYPE_COUNT`：哨兵值，不用于运行配置。

### 分离 K/V 缓存宏

`cosyvoice_kv_cache_type_t` 可以通过位打包（bit-packing）编码 K 和 V 缓存的不同类型，
从而为 K 和 V 张量使用不同的量化格式（例如 K 用 `Q8_0`、V 用 `Q4_0`），在质量与内存之间灵活取舍。

#### 打包格式（bit 31 = 1 表示分离模式）

| 位     | 字段           |
|--------|----------------|
|  0–4   | K 缓存类型     |
|  5–9   | V 缓存类型     |
| 10–14  | 回退缓存类型   |
| 15–30  | 保留           |
| 31     | 分离标志位     |

当 bit 31 为 0 时，值被当作普通的 `cosyvoice_kv_cache_type_t`，K 和 V 使用同一类型（向后兼容）。

```c
#define COSYVOICE_MAKE_SEPARATE_KV_CACHE(k_type, v_type, fallback_type)
#define COSYVOICE_IS_SEPARATE_KV_CACHE(t)
#define COSYVOICE_K_CACHE_TYPE(t)
#define COSYVOICE_V_CACHE_TYPE(t)
#define COSYVOICE_KV_CACHE_FALLBACK(t)
```

- `COSYVOICE_MAKE_SEPARATE_KV_CACHE(k_type, v_type, fallback_type)` — 将独立的 K、V 和回退类型打包为一个值。
- `COSYVOICE_IS_SEPARATE_KV_CACHE(t)` — 若 `t` 为分离模式则返回非零。
- `COSYVOICE_K_CACHE_TYPE(t)` — 从打包值中提取 K 缓存类型。
- `COSYVOICE_V_CACHE_TYPE(t)` — 从打包值中提取 V 缓存类型。
- `COSYVOICE_KV_CACHE_FALLBACK(t)` — 从打包值中提取回退缓存类型。

当后端不支持首选 K 或 V 类型时，会先尝试回退类型；如果回退也失败，则会自动继续回退。

#### 在参数中使用

可以直接使用 `cosyvoice_context_params_t` 或 `cosyvoice_context_params_v3_t` 中的位域成员，或通过 `COSYVOICE_MAKE_SEPARATE_KV_CACHE` 打包后赋值给 `llm_kv_cache_type` / `dit_kv_cache_type`：

```c
params.llm_k_cache_type = COSYVOICE_KV_CACHE_TYPE_Q8_0;
params.llm_v_cache_type = COSYVOICE_KV_CACHE_TYPE_Q4_0;
params.llm_kv_cache_separate_buffers = true;
```

或等价地：

```c
params.llm_kv_cache_type = COSYVOICE_MAKE_SEPARATE_KV_CACHE(
    COSYVOICE_KV_CACHE_TYPE_Q8_0,
    COSYVOICE_KV_CACHE_TYPE_Q4_0,
    COSYVOICE_KV_CACHE_TYPE_Q8_0);
```

## cosyvoice_inference_buffer_policy_t

### 语法

```c
typedef enum cosyvoice_inference_buffer_policy
{
    COSYVOICE_INFERENCE_BUFFER_POLICY_SHARED,
    COSYVOICE_INFERENCE_BUFFER_POLICY_BALANCED,
    COSYVOICE_INFERENCE_BUFFER_POLICY_DEDICATED,
    COSYVOICE_INFERENCE_BUFFER_POLICY_COUNT
} cosyvoice_inference_buffer_policy_t;
```

### 说明

控制推理阶段缓冲区的分配与复用策略。

### 枚举值

- `COSYVOICE_INFERENCE_BUFFER_POLICY_SHARED`：共享 KV 缓存与 token2wav 中间缓冲，优先降低内存。
- `COSYVOICE_INFERENCE_BUFFER_POLICY_BALANCED`：兼顾内存占用与下次恢复速度。
- `COSYVOICE_INFERENCE_BUFFER_POLICY_DEDICATED`：使用独立缓冲，优先性能。
- `COSYVOICE_INFERENCE_BUFFER_POLICY_COUNT`：哨兵值。

## cosyvoice_builtin_sampler_rng_policy_t

### 语法

```c
typedef enum cosyvoice_builtin_sampler_rng_policy
{
    COSYVOICE_BUILTIN_SAMPLER_RNG_POLICY_RESET_PER_SESSION,
    COSYVOICE_BUILTIN_SAMPLER_RNG_POLICY_CONTINUE_ACROSS_SESSIONS,
    COSYVOICE_BUILTIN_SAMPLER_RNG_POLICY_COUNT
} cosyvoice_builtin_sampler_rng_policy_t;
```

### 说明

定义内置采样器随机数状态在会话之间的推进方式。

### 枚举值

- `COSYVOICE_BUILTIN_SAMPLER_RNG_POLICY_RESET_PER_SESSION`：每个会话重置到种子值，便于复现。
- `COSYVOICE_BUILTIN_SAMPLER_RNG_POLICY_CONTINUE_ACROSS_SESSIONS`：跨会话连续推进，结果更具随机性。
- `COSYVOICE_BUILTIN_SAMPLER_RNG_POLICY_COUNT`：哨兵值。

## cosyvoice_sampling_params_t

### 语法

```c
typedef struct cosyvoice_sampling_params
{
    int   top_k;
    float top_p;
    int   win_size;
    float tau_r;
} cosyvoice_sampling_params_t;
```

### 说明

内置采样器使用的采样参数。

### 成员

- `top_k`：候选 token 仅保留前 K 项。
- `top_p`：nucleus 采样累计概率阈值。
- `win_size`：重复控制窗口长度。
- `tau_r`：重复控制系数。

## cosyvoice_generation_config_t

### 语法

```c
typedef struct cosyvoice_generation_config
{
    float                       temperature;
    cosyvoice_sampling_params_t sampling;
    float                       min_token_text_ratio;
    float                       max_token_text_ratio;
} cosyvoice_generation_config_t;
```

### 说明

语音生成过程中的运行时配置。

### 成员

- `temperature`：softmax 温度。
- `sampling`：采样器参数集合。
- `min_token_text_ratio`：声学 token 与输入文本长度比值下限。
- `max_token_text_ratio`：声学 token 与输入文本长度比值上限。

## cosyvoice_llm_token_prob_t

### 语法

```c
typedef struct cosyvoice_llm_token_prob
{
    int   token_id;
    float prob;
} cosyvoice_llm_token_prob_t;
```

### 说明

在自定义采样流程中表示候选 token 及其概率。

### 成员

- `token_id`：词表 token ID。
- `prob`：当前候选概率。

## cosyvoice_generated_speech_ptr

### 语法

```c
typedef struct cosyvoice_generated_speech
{
    float*   data;
    uint32_t length;
} *cosyvoice_generated_speech_ptr;
```

### 说明

生成语音结果结构体指针类型。

### 成员

- `data`：32 位浮点 PCM 缓冲区。
- `length`：`data` 中采样点数量。

## cosyvoice_context_t

### 语法

```c
typedef struct cosyvoice_context* cosyvoice_context_t;
```

### 说明

模型运行时上下文的不透明句柄。

## cosyvoice_prompt_speech_t

### 语法

```c
typedef struct cosyvoice_prompt_speech* cosyvoice_prompt_speech_t;
```

### 说明

提示语音特征对象的不透明句柄。

## cosyvoice_prompt_t

### 语法

```c
typedef struct cosyvoice_prompt* cosyvoice_prompt_t;
```

### 说明

提示对象的不透明句柄。

## cosyvoice_tts_context_t

### 语法

```c
typedef struct cosyvoice_tts_context* cosyvoice_tts_context_t;
```

### 说明

可复用 TTS 会话的不透明句柄。

## cosyvoice_sampler_t

### 语法

```c
typedef int (*cosyvoice_sampler_t)(
    cosyvoice_llm_token_prob_t*        nucleus_probs,
    int                                k,
    float*                             probs,
    uint32_t                           size,
    const cosyvoice_sampling_params_t* sampling_params,
    int*                               accepted_tokens,
    uint32_t                           n_accepted_tokens,
    void*                              sampler_ctx
);
```

### 说明

自定义 token 采样回调。

### 参数

- `nucleus_probs`：nucleus 过滤后的候选列表。
- `k`：`nucleus_probs` 元素个数。
- `probs`：完整词表概率缓冲区。
- `size`：`probs` 元素数量。
- `sampling_params`：当前采样参数。
- `accepted_tokens`：当前序列已接受 token 列表。
- `n_accepted_tokens`：已接受 token 数量。
- `sampler_ctx`：用户上下文指针。

### 返回值

被选中的 token ID。

## cosyvoice_context_params_t

### 语法

```c
typedef struct cosyvoice_context_params
{
    bool llm_use_flash_attn;
    bool flow_use_flash_attn;

    union
    {
        struct
        {
            cosyvoice_kv_cache_type_t llm_k_cache_type : 5;              ///< K 缓存数据类型。
            cosyvoice_kv_cache_type_t llm_v_cache_type : 5;              ///< V 缓存数据类型。
            cosyvoice_kv_cache_type_t llm_kv_cache_fallback : 5;         ///< 首选类型不受支持时的回退类型。
            cosyvoice_kv_cache_type_t : 16;                               ///< 保留位。
            uint32_t                  llm_kv_cache_separate_buffers : 1; ///< 是否分别为 K 和 V 分配独立缓存。
        };
        cosyvoice_kv_cache_type_t      llm_kv_cache_type;                 ///< 向后兼容的统一类型。
    };
    bool                                llm_allow_kv_cache_fallback;
    cosyvoice_inference_buffer_policy_t inference_buffer_policy;

    uint32_t n_batch;
    uint32_t n_max_seq;
    uint32_t seed;
    cosyvoice_builtin_sampler_rng_policy_t builtin_sampler_rng_policy;

    union
    {
        cosyvoice_sampler_t     sampler;      // 普通自定义采样器
        cosyvoice_sampler_ext_t sampler_ext;  // 带 worker 编号的自定义采样器（同一存储）
    };
    void*                     sampler_ctx;
} cosyvoice_context_params_t;
```

### 说明

上下文创建与推理行为控制参数集合。

### 成员

- `llm_use_flash_attn`：是否启用 LLM Flash Attention。
- `flow_use_flash_attn`：是否启用 Flow Flash Attention。
- `llm_k_cache_type`、`llm_v_cache_type`：当 `llm_kv_cache_separate_buffers` 为 true 时，K 和 V 缓存各自使用的数据类型。
- `llm_kv_cache_fallback`：当首选 K 或 V 类型不被后端支持时的回退类型。
- `llm_kv_cache_separate_buffers`：为 true 时，K 和 V 缓存使用独立缓冲区，分别由 `llm_k_cache_type` 和 `llm_v_cache_type` 指定类型。当统一 `llm_kv_cache_type` 的 bit 31 未设置时此字段被忽略。
- `llm_kv_cache_type`：KV 缓存数据类型。可以是普通的枚举值（统一类型，K 和 V 使用相同格式），也可以是 `COSYVOICE_MAKE_SEPARATE_KV_CACHE` 创建的打包值（分别编码 K、V 和回退类型）。
- `llm_allow_kv_cache_fallback`：不兼容时是否允许回退 KV 类型。
- `inference_buffer_policy`：推理缓冲策略，默认 `COSYVOICE_INFERENCE_BUFFER_POLICY_DEDICATED`。
- `n_batch`：推理批大小。
- `n_max_seq`：最大序列长度。
- `seed`：随机种子。
- `builtin_sampler_rng_policy`：内置采样器 RNG 策略。
- `sampler`：自定义采样器，`NULL` 表示使用内置采样器。
- `sampler_ext`：带 worker 编号的自定义采样器，与 `sampler` 共用同一存储。运行时始终以扩展签名调用回调（末位多一个 `worker_no` 参数）；在此槽位放一个普通 `cosyvoice_sampler_t` 也可以，多余参数会被忽略。
- `sampler_ctx`：传入采样回调的用户上下文。

## cosyvoice_init_backend

### 语法

```c
COSYVOICE_API void cosyvoice_init_backend();
```

### 说明

初始化默认后端运行时。

### 备注

当选定的后端需要显式初始化时，在创建上下文之前调用。

## cosyvoice_init_backend_from_path

### 语法

```c
COSYVOICE_API void cosyvoice_init_backend_from_path(const char* dir_path);
```

### 说明

使用指定目录初始化后端运行时。

### 参数

- `dir_path`：后端初始化目录路径。

### 备注

会从指定目录加载后端资源并完成运行时初始化。

## COSYVOICE_SIMD_CONTROL_SUPPORTED

### 语法

```c
#if defined(__x86_64__) || defined(_M_X64)
#define COSYVOICE_SIMD_CONTROL_SUPPORTED 1
#endif
```

### 说明

SIMD 检测与控制 API 的特性检测宏。仅在 x86-64 构建中定义；非 x86 目标不导出
本节任何符号，调用方需用该宏保护相关调用。

## COSYVOICE_SIMD_CAP_SSE42 / _AVX / _FMA3 / _AVX2 / _AVX512 / _AVX10_1_256 / _AVX10_1_512

### 语法

```c
#define COSYVOICE_SIMD_CAP_SSE42       (1u << 0)
#define COSYVOICE_SIMD_CAP_AVX         (1u << 1)
#define COSYVOICE_SIMD_CAP_FMA3        (1u << 2)
#define COSYVOICE_SIMD_CAP_AVX2        (1u << 3)
#define COSYVOICE_SIMD_CAP_AVX512      (1u << 4)
#define COSYVOICE_SIMD_CAP_AVX10_1_256 (1u << 5)
#define COSYVOICE_SIMD_CAP_AVX10_1_512 (1u << 6)
```

### 说明

CPU DSP SIMD 能力类的位标志，用于 `cosyvoice_simd_info_t`，与
`src/simd-dispatch.h` 中的内部 `simd_caps` 位一一对应。
`COSYVOICE_SIMD_CAP_AVX512` 同样服务于 AVX10-512 机器（分派将其引导到
AVX-512 层内核）。

## cosyvoice_simd_level_t

### 语法

```c
typedef enum cosyvoice_simd_level
{
    COSYVOICE_SIMD_LEVEL_AUTO        = 0,
    COSYVOICE_SIMD_LEVEL_SCALAR      = 1,
    COSYVOICE_SIMD_LEVEL_SSE42       = 2,
    COSYVOICE_SIMD_LEVEL_AVX         = 3,
    COSYVOICE_SIMD_LEVEL_AVX2        = 4,
    COSYVOICE_SIMD_LEVEL_AVX10_1_256 = 5,
    COSYVOICE_SIMD_LEVEL_AVX512      = 6,
    COSYVOICE_SIMD_LEVEL_COUNT
} cosyvoice_simd_level_t;
```

### 说明

CPU DSP 分派可选中 SIMD 层级的运行时上限。分派链按能力从高到低尝试
（AVX-512/AVX10-512 → AVX10.1-256 → AVX2 → AVX → SSE4.2 → 标量）；设置某层级
即允许该层及以下所有层、禁用其以上所有层。

### 取值

- `COSYVOICE_SIMD_LEVEL_AUTO`：不设上限，使用可用的最快层级（默认）。
- `COSYVOICE_SIMD_LEVEL_SCALAR`：运行时禁用 SIMD，强制标量内核。
- `COSYVOICE_SIMD_LEVEL_SSE42`：封顶到 SSE4.2 层。
- `COSYVOICE_SIMD_LEVEL_AVX`：封顶到 AVX 层。
- `COSYVOICE_SIMD_LEVEL_AVX2`：封顶到 AVX2+FMA3 层。
- `COSYVOICE_SIMD_LEVEL_AVX10_1_256`：禁用 512 位层，AVX10-256 层仍可用。
- `COSYVOICE_SIMD_LEVEL_AVX512`：显式放行全部层级（与 `AUTO` 等价）。

## cosyvoice_simd_info_t

### 语法

```c
typedef struct cosyvoice_simd_info
{
    uint32_t                 supported;
    uint32_t                 built;
    uint32_t                 current;
    cosyvoice_simd_level_t   level;
    bool                     scalar_only;
} cosyvoice_simd_info_t;
```

### 说明

由 `cosyvoice_get_simd_info()` 填充的 SIMD 检测与运行时封顶状态快照。

### 字段

- `supported`：CPU 检测到的能力（`COSYVOICE_SIMD_CAP_*` 位）。纯标量
  （`COSYVOICE_NO_SIMD`）构建中恒为 0。
- `built`：本构建编译进的层级。x86 上标量兜底层始终可用，不以位表示。
- `current`：分派当前实际选中的能力，即 `supported` ∩ `built` ∩ 层级上限。
- `level`：当前设置的层级上限（无上限时为 `COSYVOICE_SIMD_LEVEL_AUTO`）。
- `scalar_only`：为 true 表示构建时启用了 `COSYVOICE_NO_SIMD`。

## cosyvoice_get_simd_info

### 语法

```c
COSYVOICE_API void cosyvoice_get_simd_info(cosyvoice_simd_info_t* info);
```

### 说明

查询 SIMD 检测与运行时封顶状态。

### 参数

- `info`：输出结构体。传入 NULL 安全（空操作）。

## cosyvoice_get_simd_level

### 语法

```c
COSYVOICE_API cosyvoice_simd_level_t cosyvoice_get_simd_level(void);
```

### 说明

获取当前设置的 SIMD 层级上限。

### 返回值

当前层级；未设置上限时返回 `COSYVOICE_SIMD_LEVEL_AUTO`。

## cosyvoice_set_simd_level

### 语法

```c
COSYVOICE_API bool cosyvoice_set_simd_level(cosyvoice_simd_level_t level);
```

### 说明

为后续所有 CPU DSP 内核调用设置 SIMD 层级上限。进程级全局、线程安全
（原子操作）；进行中的内核保持其进入时的层级。

### 参数

- `level`：`COSYVOICE_SIMD_LEVEL_*` 之一。`COSYVOICE_SIMD_LEVEL_AUTO` 恢复无上限
  分派。高于 CPU 支持或构建包含范围的层级无害——分派会自动钳制到该层级及以下
  可用的最佳层。

### 返回值

层级合法（且在纯标量构建中有意义——该构建仅接受 `AUTO` 与 `SCALAR`）时
返回 true，否则 false。

### 备注

可用于调试（如对标验证标量层、二分定位某指令集相关问题），或在混合负载下
规避 AVX-512 降频。检测部分基于 CPUID、启动时一次性完成；分派架构详见
[docs/SIMD_zh.md](SIMD_zh.md)。

## cosyvoice_init_default_context_params

### 语法

```c
COSYVOICE_API void cosyvoice_init_default_context_params(cosyvoice_context_params_t* params);
```

### 说明

将上下文参数结构体填充为默认值。

### 参数

- `params`：输出参数结构体。

## cosyvoice_load_from_file

### 语法

```c
COSYVOICE_API cosyvoice_context_t cosyvoice_load_from_file(const char* filename);
```

### 说明

使用默认参数从 GGUF 文件加载模型。

### 参数

- `filename`：模型文件路径。

### 返回值

成功返回上下文句柄，失败返回 `NULL`。

### 备注

使用默认后端和默认线程设置加载模型。

等价于使用默认初始化后的 `params` 调用 `cosyvoice_load_from_file_ext(filename, &params, NULL, 0, 0)`。

## cosyvoice_load_from_file_with_params

### 语法

```c
COSYVOICE_API cosyvoice_context_t cosyvoice_load_from_file_with_params(
    const char*                       filename,
    const cosyvoice_context_params_t* params
);
```

### 说明

使用显式参数从文件加载模型。

### 参数

- `filename`：模型文件路径。
- `params`：上下文参数。

### 返回值

成功返回上下文句柄，失败返回 `NULL`。

### 备注

等价于调用 `cosyvoice_load_from_file_ext(filename, params, NULL, 0, 0)`。

## cosyvoice_context_params_v2_t

### 语法

```c
typedef struct cosyvoice_context_params_v2
{
    cosyvoice_context_params_t base_params;
    uint32_t n_workers;
} cosyvoice_context_params_v2_t;

#ifdef __cplusplus
// 供 C++ 版 cosyvoice_load_from_file_ext 模板使用的扁平变体。
// 与上方 C typedef 布局兼容：唯一区别是把结构体尾部的对齐间隙
// 显式化为一个保留字段，使派生的 v3 结构体保持完全相同的成员偏移。
struct cosyvoice_context_params_v2_cpp : cosyvoice_context_params_t
{
    uint32_t n_workers;
    uint32_t reserved_tail_padding;
};
#endif
```

### 说明

在 `cosyvoice_context_params_t` 基础上增加 worker 数量配置。

### 成员

- `base_params`：基础上下文参数。
- `n_workers`：要创建的 worker 槽数量。
- `reserved_tail_padding`（仅 C++ 变体）：占位保留字段，占据结构体尾部的对齐间隙——请置 0。

## cosyvoice_load_from_file_with_params_v2

### 语法

```c
COSYVOICE_API cosyvoice_context_t cosyvoice_load_from_file_with_params_v2(
    const char*                          filename,
    const cosyvoice_context_params_v2_t* params
);
```

### 说明

使用扩展参数加载模型上下文，包含 worker 数量配置。

### 参数

- `filename`：模型文件路径。
- `params`：扩展上下文参数。

### 返回值

成功返回上下文句柄，失败返回 `NULL`。

## cosyvoice_context_params_v3_t

### 语法

```c
typedef struct cosyvoice_context_params_v3
{
    cosyvoice_context_params_v2_t base_params;

    union
    {
        struct
        {
#if defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__) || defined(_BYTE_ORDER) && (_BYTE_ORDER == _BIG_ENDIAN) || defined(__BIG_ENDIAN__) && !defined(__LITTLE_ENDIAN__) || defined(__ARMEB__) || defined(__MIPSEB__) || defined(__sparc__)
            uint32_t                  dit_kv_cache_separate_buffers : 1;
            cosyvoice_kv_cache_type_t : 16;
            cosyvoice_kv_cache_type_t dit_kv_cache_fallback : 5;
            cosyvoice_kv_cache_type_t dit_v_cache_type : 5;
            cosyvoice_kv_cache_type_t dit_k_cache_type : 5;
#else
            cosyvoice_kv_cache_type_t dit_k_cache_type : 5;
            cosyvoice_kv_cache_type_t dit_v_cache_type : 5;
            cosyvoice_kv_cache_type_t dit_kv_cache_fallback : 5;
            cosyvoice_kv_cache_type_t : 16;
            uint32_t                  dit_kv_cache_separate_buffers : 1;
#endif
        };
        cosyvoice_kv_cache_type_t dit_kv_cache_type;
    };
    bool     dit_allow_kv_cache_fallback;
    uint32_t dit_kv_fixed_slots;
    uint32_t dit_kv_offloadable_slots;
    uint32_t dit_kv_cache_length;
} cosyvoice_context_params_v3_t;

#ifdef __cplusplus
struct cosyvoice_context_params_v3_cpp : cosyvoice_context_params_v2_cpp
{
    union
    {
        struct
        {
#if defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__) || defined(_BYTE_ORDER) && (_BYTE_ORDER == _BIG_ENDIAN) || defined(__BIG_ENDIAN__) && !defined(__LITTLE_ENDIAN__) || defined(__ARMEB__) || defined(__MIPSEB__) || defined(__sparc__)
            uint32_t                  dit_kv_cache_separate_buffers : 1;
            cosyvoice_kv_cache_type_t : 16;
            cosyvoice_kv_cache_type_t dit_kv_cache_fallback : 5;
            cosyvoice_kv_cache_type_t dit_v_cache_type : 5;
            cosyvoice_kv_cache_type_t dit_k_cache_type : 5;
#else
            cosyvoice_kv_cache_type_t dit_k_cache_type : 5;
            cosyvoice_kv_cache_type_t dit_v_cache_type : 5;
            cosyvoice_kv_cache_type_t dit_kv_cache_fallback : 5;
            cosyvoice_kv_cache_type_t : 16;
            uint32_t                  dit_kv_cache_separate_buffers : 1;
#endif
        };
        cosyvoice_kv_cache_type_t dit_kv_cache_type;
    };
    bool     dit_allow_kv_cache_fallback;
    uint32_t dit_kv_fixed_slots;
    uint32_t dit_kv_offloadable_slots;
    uint32_t dit_kv_cache_length;
    uint32_t reserved_tail_padding;
};
#endif
```

### 说明

在 `cosyvoice_context_params_v2_t` 基础上增加 DiT（扩散模型）KV 缓存配置。流式 TTS 时每个块都要跑完 DiT 的全部扩散步——没有缓存时，每个新块的每一步都会对已产出的整个音频序列重算自注意力。KV 缓存按扩散步保存已产出位置的注意力 key/value（默认每个扩散步一个缓存槽位，可通过 V4 的 `dit_kv_actual_*_slots` 让多个步共享），新块只需计算自己新增的位置——但缓存本身非常大（合计最多 `sequence_length × n_diffusion_steps` 个 key-value 对）。

### 成员

- `base_params`：V2 基础参数。
- `dit_k_cache_type`：DiT 模块 K 缓存数据类型。
- `dit_v_cache_type`：DiT 模块 V 缓存数据类型。
- `dit_kv_cache_separate_buffers`：若为 true，则分别为 K 和 V 分配独立缓存。
- `dit_kv_cache_fallback`：首选类型不受支持时的回退类型。
- `dit_kv_cache_type`：快捷方式——指定统一类型（不分离 K/V）。
- `dit_allow_kv_cache_fallback`：若为 true，不支持时回退到 flash attention 兼容类型。
- `dit_kv_fixed_slots`：固定（设备内存，从不卸载）DiT KV 槽位数。每个固定槽位对应一个扩散步的 KV 缓存；默认每步独占一个设备槽位，可通过 V4 的 `dit_kv_actual_fixed_slots` 让多个步共享以减少占用。
- `dit_kv_offloadable_slots`：可卸载（CPU 卸载）DiT KV 槽位数。所有可卸载步共享同一个设备临时槽位，默认每槽各对应一个 CPU 缓冲区来拷贝 KV，可通过 V4 的 `dit_kv_actual_offloadable_slots` 让多个步共享一个缓冲区。
- `dit_kv_cache_length`：DiT KV 缓存最大序列长度。0 表示使用默认值（`n_max_seq × 10`）。

### DiT KV 缓存概念

面向用户的概述见 [STREAMING_zh.md — DiT KV 缓存](STREAMING_zh.md#dit-kv-缓存)；以下说明内部布局。

**槽位布局。** 设备缓存分配 `n_slots = actual_fixed_slots + (offloadable_slots > 0 ? 1 : 0)` 个物理槽位，其中 `actual_fixed_slots` 在未启用 KV 共享时等于 `fixed_slots`（见 `cosyvoice_context_params_v4_t`）。启用卸载时，槽位 0 是专用 **临时（scratch）槽位**，由所有可卸载步共享；固定组占用槽位 `1..actual_fixed_slots`。未启用卸载时，固定组占用槽位 `0..actual_fixed_slots-1`。

**步-槽位调度。** `diffusion_steps`（默认 10 步）按顺序划分：`n_nocache = diffusion_steps − fixed − offloadable` 个不缓存步、可卸载步、最后是固定步：

- **不缓存步**每步全量重算注意力，不接触任何 KV 槽位。
- **可卸载步**全部写入槽位 0。每个 CPU 分组的第一个步把该组缓冲区拷贝回设备，最后一个步把结果拷贝回去；每个可卸载分组分配一个 CPU 缓冲区。
- **固定步**按相邻分组映射到设备槽位（`槽位 = 首个固定槽位 + (步 − 组起始步) × actual_fixed / fixed`），每组共享一个槽位，组内后写者覆盖前者；各组 KV 常驻设备，因此最后一个固定组会为下一个流式 chunk 播种缓存。

**Flash Attention 路径。** 启用 `flow_use_flash_attn` 时图只构建一次，通过 `slide_kv_slot` 沿槽位链滑动画图视图。最后一个可卸载步执行从临时槽位到第一个固定槽位的边界滑动，把 KV 移交给固定链；固定组内部各步共享同一槽位，无需滑动，滑动只在组边界触发。固定步从不重新绑定，其链位置完全由滑动索引决定。

**非 FA 路径。** 每个缓存步重新构建图，并在构建时显式绑定槽位——可卸载步 `bind_slot(0)`，固定步绑定其分组对应的物理槽位。

**归一化与裁剪。** 加载时若只有一个可卸载槽位（`offloadable == 1`），会转为固定槽位（`fixed++`、`offloadable = 0`）；名义槽位数先被裁剪，保证 `fixed ≤ diffusion_steps` 且 `fixed + offloadable ≤ diffusion_steps`；随后物理槽位数裁剪到 `[1, fixed]` / `[1, offloadable]`（`0` 表示不共享，取名义值）。构建器还会为 `actual_offloadable` 分配对应的 CPU KV 缓冲区，用于往返卸载状态。

**缓存长度。** `dit_kv_cache_length` 限制每个槽位保留的最大序列位置数。流式长度超过该值时只直接截断长度、丢弃后面的部分上下文——推理不会崩溃，但音频质量可能下降。

## cosyvoice_context_params_v4_t

### 语法

```c
typedef struct cosyvoice_context_params_v4
{
    cosyvoice_context_params_v3_t base_params;
    int32_t diffusion_steps;
    uint32_t dit_kv_actual_fixed_slots;
    uint32_t dit_kv_actual_offloadable_slots;
    bool     strict_seed_mode;
} cosyvoice_context_params_v4_t;

#ifdef __cplusplus
struct cosyvoice_context_params_v4_cpp : cosyvoice_context_params_v3_cpp
{
    int32_t  diffusion_steps;
    uint32_t dit_kv_actual_fixed_slots;
    uint32_t dit_kv_actual_offloadable_slots;
    bool     strict_seed_mode;
};
#endif
```

### 说明

在 `cosyvoice_context_params_v3_t` 基础上增加每个 chunk 内 DiT 运行的流匹配（flow matching）扩散步数、让相邻扩散步共享 DiT KV 缓存（减少内存占用）的配置，以及严格 seed 复现模式。

### 字段

- `base_params`：V3 基础参数（含 DiT KV 缓存配置）。
- `diffusion_steps`：流匹配扩散步数。当 `<= 0` 时取模型的 `decoder.diffusion_steps` GGUF 元数据（该元数据缺失时默认为 10）；任何正数都会被裁剪到运行时上限（50）。DiT KV 槽位数会依据最终生效的步数再次裁剪。
- `dit_kv_actual_fixed_slots`：支撑 `dit_kv_fixed_slots` 个逻辑固定槽位的**物理**设备槽位数。相邻固定步被划分为该数量的分组，每组共享一个设备槽位：常驻 DiT KV 缓存缩小为原来的 `actual_fixed / fixed`，代价是组内各步读取的历史为组内最后写入者的结果（一种很小的近似）。`0` 表示不共享（每个固定步一个槽位）；正数被裁剪到 `[1, dit_kv_fixed_slots]`。
- `dit_kv_actual_offloadable_slots`：支撑 `dit_kv_offloadable_slots` 个逻辑可卸载槽位的**物理** CPU 缓冲区数量。相邻可卸载步被划分为该数量的分组，每组共享一个 CPU 缓冲区，且只在组首步拷入、组尾步拷出，同时减少内存与 CPU 往返。`0` 表示不共享（每个可卸载步一个缓冲区）；正数被裁剪到 `[1, dit_kv_offloadable_slots]`。
- `strict_seed_mode`：为 true 时严格保证相同 sampler seed 的重复生成结果一致；为 false 时省去一个小的 prefill 通道、速度略快，但不再严格保证 seed 可复现。库默认值为 `true`：使用 V1–V3 参数结构体加载时恒为严格模式。传 V4 结构体时该标志按原值取用（bool 没有"0 = 默认"的哨兵语义），因此零初始化的 `cosyvoice_context_params_v4_t params = {}` 表示严格模式**关闭**——请显式设置。

## cosyvoice_load_from_file_with_params_v3

### 语法

```c
COSYVOICE_API cosyvoice_context_t cosyvoice_load_from_file_with_params_v3(
    const char*                          filename,
    const cosyvoice_context_params_v3_t* params
);
```

### 说明

使用 V3 扩展参数加载模型上下文，包含 DiT KV 缓存配置。

### 参数

- `filename`：模型文件路径。
- `params`：V3 上下文参数。

### 返回值

成功返回上下文句柄，失败返回 `NULL`。

## cosyvoice_load_from_file_with_params_v4

### 语法

```c
COSYVOICE_API cosyvoice_context_t cosyvoice_load_from_file_with_params_v4(
    const char*                          filename,
    const cosyvoice_context_params_v4_t* params
);
```

### 说明

使用 V4 扩展参数加载模型上下文，在 V3 的 DiT KV 缓存配置之上增加了自定义扩散步数。

### 参数

- `filename`：模型文件路径。
- `params`：V4 上下文参数。

### 返回值

成功返回上下文句柄，失败返回 `NULL`。

## cosyvoice_duplicate_context

### 语法

```c
COSYVOICE_API cosyvoice_context_t cosyvoice_duplicate_context(cosyvoice_context_t ctx);
```

### 说明

创建一个新的上下文，新上下文与源上下文共享已加载的模型资源。

### 参数

- `ctx`：源上下文。

### 返回值

成功返回复制后的上下文句柄，失败返回 `NULL`。

### 备注

新上下文会继承源上下文当前绑定的 worker，之后可通过 `cosyvoice_set_worker_no()` 独立切换。

## cosyvoice_get_n_workers

### 语法

```c
COSYVOICE_API uint32_t cosyvoice_get_n_workers(cosyvoice_context_t ctx);
```

### 说明

获取当前上下文可用的 worker 槽总数。

### 参数

- `ctx`：模型上下文。

### 返回值

worker 槽数量。

## cosyvoice_get_worker_no

### 语法

```c
COSYVOICE_API uint32_t cosyvoice_get_worker_no(cosyvoice_context_t ctx);
```

### 说明

获取当前激活的 worker 槽编号。

### 参数

- `ctx`：模型上下文。

### 返回值

当前 worker 编号。

## cosyvoice_set_worker_no

### 语法

```c
COSYVOICE_API bool cosyvoice_set_worker_no(cosyvoice_context_t ctx, uint32_t worker_no);
```

### 说明

设置后续推理使用的 worker 槽。

### 参数

- `ctx`：模型上下文。
- `worker_no`：worker 槽编号。

### 返回值

成功返回 `true`，否则返回 `false`。

### 备注

当两个线程需要在共享同一份模型资源的前提下并发推理时，建议先复制上下文，再分别绑定不同 worker。

## cosyvoice_free

### 语法

```c
COSYVOICE_API void cosyvoice_free(cosyvoice_context_t ctx);
```

### 说明

释放模型上下文及其关联资源。

### 参数

- `ctx`：待释放上下文。

## cosyvoice_get_context_params

### 语法

```c
COSYVOICE_API void cosyvoice_get_context_params(
    cosyvoice_context_t         ctx,
    cosyvoice_context_params_t* params
);
```

### 说明

读取当前上下文的生效参数。

### 参数

- `ctx`：模型上下文。
- `params`：输出参数结构体。

## cosyvoice_get_default_generation_config

### 语法

```c
COSYVOICE_API void cosyvoice_get_default_generation_config(
    cosyvoice_context_t            ctx,
    cosyvoice_generation_config_t* config
);
```

### 说明

获取模型文件中加载的默认生成配置，尚未应用 worker 级覆盖。

### 参数

- `ctx`：模型上下文。
- `config`：输出默认生成配置。

## cosyvoice_get_architecture

### 语法

```c
COSYVOICE_API const char* cosyvoice_get_architecture(cosyvoice_context_t ctx);
```

### 说明

获取已加载模型的架构标识。

### 参数

- `ctx`：模型上下文。

### 返回值

以 `\0` 结尾的 UTF-8 架构字符串，例如 `cosyvoice3-2512`。

## cosyvoice_is_backend_uma

### 语法

```c
COSYVOICE_API bool cosyvoice_is_backend_uma(cosyvoice_context_t ctx);
```

### 说明

查询后端是否使用统一内存架构（UMA）。

### 参数

- `ctx`：模型上下文。

### 返回值

检测到 UMA 内存时返回 `true`，否则返回 `false`。

### 备注

UMA 判定在模型加载阶段完成：运行时通过对比后端 tensor 写入带宽与主机 `memcpy` 带宽来推断。Apple Silicon（`__aarch64__`）默认视为 UMA。该结果仅作信息展示，不会改变任何缓冲策略或其他库行为。调用方可通过此接口展示后端特性或据此调整缓冲策略。

> **注意**：UMA 检测基于带宽探测的启发式方法，结果可能因硬件、驱动版本和探测时系统负载不同而有偏差，请将其视为粗略参考而非确定性的硬件能力判断。

## cosyvoice_get_generation_config

### 语法

```c
COSYVOICE_API void cosyvoice_get_generation_config(
    cosyvoice_context_t            ctx,
    cosyvoice_generation_config_t* config
);
```

### 说明

获取当前生成配置。

### 参数

- `ctx`：模型上下文。
- `config`：输出配置结构体。

## cosyvoice_get_sample_rate

### 语法

```c
COSYVOICE_API uint32_t cosyvoice_get_sample_rate(cosyvoice_context_t ctx);
```

### 说明

获取模型输出采样率。

### 参数

- `ctx`：模型上下文。

### 返回值

采样率（Hz）。

## cosyvoice_get_diffusion_steps

### 语法

```c
COSYVOICE_API int cosyvoice_get_diffusion_steps(cosyvoice_context_t ctx);
```

### 说明

返回模型 DiT 解码器实际使用的流匹配扩散步数。该值在加载时由 `cosyvoice_context_params_v4_t::diffusion_steps` 覆盖项、`decoder.diffusion_steps` GGUF 元数据、或内置默认值 10 解析得到，并已裁剪到运行时上限。

### 参数

- `ctx`：模型上下文。

### 返回值

实际生效的扩散步数（恒 `>= 1`）。

## cosyvoice_set_generation_config

### 语法

```c
COSYVOICE_API bool cosyvoice_set_generation_config(
    cosyvoice_context_t                  ctx,
    const cosyvoice_generation_config_t* config
);
```

### 说明

校验并应用新的生成配置。

### 参数

- `ctx`：模型上下文。
- `config`：待应用配置。

### 返回值

成功返回 `true`，失败返回 `false`。

### 备注

该接口不会修改采样率。

## cosyvoice_set_sampler

### 语法

```c
COSYVOICE_API void cosyvoice_set_sampler(cosyvoice_context_t ctx, cosyvoice_sampler_t sampler, void* sampler_ctx);
```

### 说明

设置自定义采样器回调。

### 参数

- `ctx`：模型上下文。
- `sampler`：采样器函数，`NULL` 表示恢复内置采样器。
- `sampler_ctx`：用户上下文。

## cosyvoice_sampler_ext_t

### 语法

```c
typedef int (_cdecl * cosyvoice_sampler_ext_t)(
    cosyvoice_llm_token_prob_t*        nucleus_probs,
    int                                k,
    float*                             probs,
    uint32_t                           size,
    const cosyvoice_sampling_params_t* sampling_params,
    int*                               accepted_tokens,
    uint32_t                           n_accepted_tokens,
    void*                              sampler_ctx,
    uint32_t                           worker_no
);
```

### 说明

扩展采样器回调，会额外收到 worker 编号。

### 参数

- `worker_no`：当前激活的 worker 槽编号。

## cosyvoice_get_sampler

### 语法

```c
COSYVOICE_API void cosyvoice_get_sampler(cosyvoice_context_t ctx, cosyvoice_sampler_t* sampler, void** sampler_ctx);
```

### 说明

获取当前采样器与其上下文。

### 参数

- `ctx`：模型上下文。
- `sampler`：输出采样器指针。
- `sampler_ctx`：输出上下文指针。

## cosyvoice_get_builtin_sampler_rng_policy

### 语法

```c
COSYVOICE_API cosyvoice_builtin_sampler_rng_policy_t cosyvoice_get_builtin_sampler_rng_policy(cosyvoice_context_t ctx);
```

### 说明

获取内置采样器 RNG 策略。

### 参数

- `ctx`：模型上下文。

### 返回值

当前 RNG 策略值。

## cosyvoice_set_builtin_sampler_rng_policy

### 语法

```c
COSYVOICE_API bool cosyvoice_set_builtin_sampler_rng_policy(cosyvoice_context_t ctx, cosyvoice_builtin_sampler_rng_policy_t policy);
```

### 说明

设置内置采样器 RNG 策略。

### 参数

- `ctx`：模型上下文。
- `policy`：策略值。

### 返回值

设置成功返回 `true`，失败返回 `false`。

## cosyvoice_set_sampler_seed

### 语法

```c
COSYVOICE_API bool cosyvoice_set_sampler_seed(cosyvoice_context_t ctx, uint32_t seed);
```

### 说明

设置采样器随机种子。

### 参数

- `ctx`：模型上下文。
- `seed`：种子值。

### 返回值

当前内置采样器可用时返回 `true`，否则返回 `false`。

### 备注

该种子仅对当前 worker 生效。即使使用自定义采样器，种子也会被保存，并在当前 worker 切回内置采样器时生效。

## cosyvoice_get_sampler_seed

### 语法

```c
COSYVOICE_API uint32_t cosyvoice_get_sampler_seed(cosyvoice_context_t ctx);
```

### 说明

获取当前 worker 的内置采样器 seed。

### 参数

- `ctx`：模型上下文。

### 返回值

当前 worker 的采样器种子。

## cosyvoice_generate_random_seed

### 语法

```c
COSYVOICE_API uint32_t cosyvoice_generate_random_seed();
```

### 说明

生成 32 位随机种子。

### 返回值

适合用于 `cosyvoice_set_sampler_seed` 的伪确定性随机 32 位无符号整数。

## cosyvoice_prompt_speech_load_from_file

### 语法

```c
COSYVOICE_API cosyvoice_prompt_speech_t cosyvoice_prompt_speech_load_from_file(const char* filename);
```

### 说明

从文件加载提示语音特征。

### 参数

- `filename`：文件路径。

### 返回值

成功返回句柄，失败返回 `NULL`。

## cosyvoice_prompt_speech_load

### 语法

```c
COSYVOICE_API cosyvoice_prompt_speech_t cosyvoice_prompt_speech_load(const void* data, size_t size);
```

### 说明

从内存缓冲区加载提示语音特征。

### 参数

- `data`：内存中的提示语音 GGUF 数据指针。
- `size`：数据字节大小。

### 返回值

成功返回提示语音句柄，失败返回 `NULL`。

### 备注

- 数据缓冲区必须包含 `feat`、`embedding`、`tokens`、`text` 等张量的提示语音 GGUF 格式。
- 如果元数据中包含 CRC32 校验和，则会自动验证。
- 加载完成后，数据缓冲区可由调用方释放。

## cosyvoice_prompt_speech_save_to_file

### 语法

```c
COSYVOICE_API bool cosyvoice_prompt_speech_save_to_file(cosyvoice_prompt_speech_t prompt_speech, const char* filename);
```

### 说明

将提示语音对象保存到文件。

### 参数

- `prompt_speech`：提示语音对象。
- `filename`：输出路径。

### 返回值

保存成功返回 `true`，否则返回 `false`。

## cosyvoice_prompt_init_from_prompt_speech

### 语法

```c
COSYVOICE_API cosyvoice_prompt_t cosyvoice_prompt_init_from_prompt_speech(cosyvoice_context_t ctx, cosyvoice_prompt_speech_t prompt_speech);
```

### 说明

根据提示语音对象构建提示句柄。

### 参数

- `ctx`：模型上下文。
- `prompt_speech`：提示语音对象。

### 返回值

成功返回提示句柄，失败返回 `NULL`。

## cosyvoice_prompt_speech_free

### 语法

```c
COSYVOICE_API void cosyvoice_prompt_speech_free(cosyvoice_prompt_speech_t prompt_speech);
```

### 说明

释放提示语音对象。

### 参数

- `prompt_speech`：待释放对象。

## cosyvoice_prompt_free

### 语法

```c
COSYVOICE_API void cosyvoice_prompt_free(cosyvoice_prompt_t prompt);
```

### 说明

释放提示对象。

### 参数

- `prompt`：待释放对象。

## cosyvoice_tts_context_new

### 语法

```c
COSYVOICE_API cosyvoice_tts_context_t cosyvoice_tts_context_new(cosyvoice_context_t ctx, cosyvoice_prompt_t prompt);
```

### 说明

创建可复用 TTS 会话。

### 参数

- `ctx`：模型上下文。
- `prompt`：初始提示对象。

### 返回值

成功返回会话句柄，失败返回 `NULL`。

### 备注

该会话用于多次合成时复用上下文状态。

## cosyvoice_tts_context_free

### 语法

```c
COSYVOICE_API void cosyvoice_tts_context_free(cosyvoice_tts_context_t ctx);
```

### 说明

销毁 TTS 会话。

### 参数

- `ctx`：会话句柄。

## cosyvoice_tts_context_set_prompt

### 语法

```c
COSYVOICE_API void cosyvoice_tts_context_set_prompt(cosyvoice_tts_context_t ctx, cosyvoice_prompt_t prompt);
```

### 说明

更新 TTS 会话绑定的提示对象。

### 参数

- `ctx`：会话句柄。
- `prompt`：新提示对象。

### 备注

会重置该会话缓存的指令文本。

## cosyvoice_tts_context_set_text_normalization_enabled

### 语法

```c
COSYVOICE_API bool cosyvoice_tts_context_set_text_normalization_enabled(cosyvoice_tts_context_t ctx, bool enabled);
```

### 说明

为指定 TTS 会话开启或关闭前端文本标准化。

### 参数

- `ctx`：会话句柄。
- `enabled`：`true` 表示开启文本标准化；`false` 表示跳过标准化，直接对原始文本分词。

### 返回值

成功返回 `true`，如果标准化不可用（例如编译时未启用 ICU）则返回 `false`。

### 备注

新创建的 TTS 会话默认开启文本标准化。

## cosyvoice_tts_context_get_text_normalization_enabled

### 语法

```c
COSYVOICE_API bool cosyvoice_tts_context_get_text_normalization_enabled(cosyvoice_tts_context_t ctx);
```

### 说明

查询指定 TTS 会话当前是否开启前端文本标准化。

### 参数

- `ctx`：会话句柄。

### 返回值

开启时返回 `true`，关闭时返回 `false`。

## cosyvoice_tts_context_set_split_text_enabled

### 语法

```c
COSYVOICE_API bool cosyvoice_tts_context_set_split_text_enabled(cosyvoice_tts_context_t ctx, bool enabled);
```

### 说明

为指定 TTS 会话开启或关闭分片处理。

### 参数

- `ctx`：会话句柄。
- `enabled`：`true` 表示先将文本切分为片段再合成；`false` 表示整段文本一次性合成。

### 返回值

成功返回 `true`。

### 备注

新创建的 TTS 会话默认开启分片处理。

## cosyvoice_tts_context_get_split_text_enabled

### 语法

```c
COSYVOICE_API bool cosyvoice_tts_context_get_split_text_enabled(cosyvoice_tts_context_t ctx);
```

### 说明

查询指定 TTS 会话当前是否开启分片处理。

### 参数

- `ctx`：会话句柄。

### 返回值

开启时返回 `true`，关闭时返回 `false`。

## cosyvoice_tts_context_set_fast_split_text_enabled

### 语法

```c
COSYVOICE_API bool cosyvoice_tts_context_set_fast_split_text_enabled(cosyvoice_tts_context_t ctx, bool enabled);
```

### 说明

为指定 TTS 会话开启或关闭快速分片处理。

### 参数

- `ctx`：会话句柄。
- `enabled`：`true` 表示启用基于 token 的快速分片；`false` 表示使用较慢的文本重组路径。

### 返回值

成功返回 `true`。

### 备注

新创建的 TTS 会话默认开启快速分片处理。

## cosyvoice_tts_context_get_fast_split_text_enabled

### 语法

```c
COSYVOICE_API bool cosyvoice_tts_context_get_fast_split_text_enabled(cosyvoice_tts_context_t ctx);
```

### 说明

查询指定 TTS 会话当前是否开启快速分片处理。

### 参数

- `ctx`：会话句柄。

### 返回值

开启时返回 `true`，关闭时返回 `false`。

## cosyvoice_tts_context_set_fade_in_enabled

### 语法

```c
COSYVOICE_API bool cosyvoice_tts_context_set_fade_in_enabled(cosyvoice_tts_context_t ctx, bool enabled);
```

### 说明

为指定 TTS 会话开启或关闭默认输出淡入。

### 参数

- `ctx`：会话句柄。
- `enabled`：`true` 表示对生成结果应用 20 ms 淡入；`false` 表示返回未经淡入处理的原始 PCM。

### 返回值

成功返回 `true`。

### 备注

新创建的 TTS 会话默认开启淡入。

## cosyvoice_tts_context_get_fade_in_enabled

### 语法

```c
COSYVOICE_API bool cosyvoice_tts_context_get_fade_in_enabled(cosyvoice_tts_context_t ctx);
```

### 说明

查询指定 TTS 会话当前是否开启输出淡入。

### 参数

- `ctx`：会话句柄。

### 返回值

开启时返回 `true`，关闭时返回 `false`。

## cosyvoice_tts_context_get_flags

### 语法

```c
COSYVOICE_API uint32_t cosyvoice_tts_context_get_flags(cosyvoice_tts_context_t ctx);
```

### 说明

获取当前 TTS 会话的标志位掩码。

### 参数

- `ctx`：会话句柄。

### 返回值

由 `COSYVOICE_TTS_FLAG_*` 组成的位掩码。

## cosyvoice_tts_context_set_flags

### 语法

```c
COSYVOICE_API uint32_t cosyvoice_tts_context_set_flags(cosyvoice_tts_context_t ctx, uint32_t flags);
```

### 说明

设置当前 TTS 会话的标志位掩码。

### 参数

- `ctx`：会话句柄。
- `flags`：请求设置的标志位掩码。

### 返回值

返回过滤掉不支持标志后的实际生效值。

## cosyvoice_tts_zero_shot

### 语法

```c
COSYVOICE_API bool cosyvoice_tts_zero_shot(
    cosyvoice_tts_context_t        ctx,
    const char*                    text,
    float                          speed,
    cosyvoice_generated_speech_ptr result
);
```

### 说明

以 zero-shot 模式生成语音。

### 参数

- `ctx`：TTS 会话。
- `text`：输入文本。
- `speed`：语速系数。
- `result`：输出波形容器。

### 返回值

成功返回 `true`，失败返回 `false`。

### 备注

直接使用当前会话的 prompt 从文本生成语音。

## cosyvoice_tts_instruct

### 语法

```c
COSYVOICE_API bool cosyvoice_tts_instruct(
    cosyvoice_tts_context_t        ctx,
    const char*                    text,
    const char*                    instruction,
    float                          speed,
    cosyvoice_generated_speech_ptr result
);
```

### 说明

以 instruct 模式生成语音。

### 参数

- `ctx`：TTS 会话。
- `text`：输入文本。
- `instruction`：指令文本。
- `speed`：语速系数。
- `result`：输出波形容器。

### 返回值

成功返回 `true`，失败返回 `false`。

## cosyvoice_tts_cross_lingual

### 语法

```c
COSYVOICE_API bool cosyvoice_tts_cross_lingual(
    cosyvoice_tts_context_t        ctx,
    const char*                    text,
    float                          speed,
    cosyvoice_generated_speech_ptr result
);
```

### 说明

以 cross-lingual 模式生成语音。

### 参数

- `ctx`：TTS 会话。
- `text`：输入文本。
- `speed`：语速系数。
- `result`：输出波形容器。

### 返回值

成功返回 `true`，失败返回 `false`。

## cosyvoice_tts_audio_callback_t

### 语法

```c
typedef bool (*cosyvoice_tts_audio_callback_t)(const float* audio, uint32_t n_samples, void* user_data);
```

### 说明

流式 TTS API 用于逐段交付音频的回调函数类型。

### 参数

- `audio`：32 位浮点 PCM 采样数据，单声道。
- `n_samples`：本次回调的采样点数。
- `user_data`：注册回调时传入的不透明上下文。

### 返回值

返回 `true` 继续流式合成，返回 `false` 中止合成。

## cosyvoice_tts_zero_shot_stream

### 语法

```c
COSYVOICE_API bool cosyvoice_tts_zero_shot_stream(
    cosyvoice_tts_context_t        ctx,
    const char*                    text,
    float                          speed,
    cosyvoice_tts_audio_callback_t callback,
    void*                          user_data
);
```

### 说明

以 zero-shot 模式流式生成语音，通过回调逐段交付音频。

### 参数

- `ctx`：TTS 会话。
- `text`：输入文本。
- `speed`：语速系数。
- `callback`：接收每段音频的回调函数。
- `user_data`：传入回调的不透明上下文。

### 返回值

成功返回 `true`，失败返回 `false`。

## cosyvoice_tts_instruct_stream

### 语法

```c
COSYVOICE_API bool cosyvoice_tts_instruct_stream(
    cosyvoice_tts_context_t        ctx,
    const char*                    text,
    const char*                    instruction,
    float                          speed,
    cosyvoice_tts_audio_callback_t callback,
    void*                          user_data
);
```

### 说明

以 instruct 模式流式生成语音。

### 参数

- `ctx`：TTS 会话。
- `text`：输入文本。
- `instruction`：指令文本。
- `speed`：语速系数。
- `callback`：接收每段音频的回调函数。
- `user_data`：传入回调的不透明上下文。

### 返回值

成功返回 `true`，失败返回 `false`。

## cosyvoice_tts_cross_lingual_stream

### 语法

```c
COSYVOICE_API bool cosyvoice_tts_cross_lingual_stream(
    cosyvoice_tts_context_t        ctx,
    const char*                    text,
    float                          speed,
    cosyvoice_tts_audio_callback_t callback,
    void*                          user_data
);
```

### 说明

以 cross-lingual 模式流式生成语音。

### 参数

- `ctx`：TTS 会话。
- `text`：输入文本。
- `speed`：语速系数。
- `callback`：接收每段音频的回调函数。
- `user_data`：传入回调的不透明上下文。

### 返回值

成功返回 `true`，失败返回 `false`。

## cosyvoice_save_wav

### 语法

```c
COSYVOICE_API bool cosyvoice_save_wav(const char* filename, const float* data, uint32_t data_len, uint32_t sample_rate);
```

### 说明

将浮点 PCM 保存为 WAV 文件。

### 参数

- `filename`：输出路径。
- `data`：PCM 数据。
- `data_len`：采样点数量。
- `sample_rate`：采样率。

### 返回值

写入成功返回 `true`，否则返回 `false`。

### 备注

输出为单声道 32 位浮点 WAV PCM 数据。

## cosyvoice_memory_usage_t

### 语法

```c
typedef struct cosyvoice_memory_usage
{
    size_t parameters;
    size_t kv_cache;
    size_t token2wav;
    size_t buffers;
    size_t cpu_buffers;
    size_t offloaded_kv_cache;
    size_t random_noise;
} cosyvoice_memory_usage_t;
```

### 说明

描述模型上下文的内存占用明细。

### 成员

- `parameters`：模型参数占用。
- `kv_cache`：KV 缓存占用。
- `token2wav`：token2wav 中间缓冲占用。
- `buffers`：内部缓冲占用。
- `cpu_buffers`：CPU 缓冲占用。
- `offloaded_kv_cache`：下放到 CPU 的 KV 缓存占用。
- `random_noise`：随机噪声缓冲占用。

## cosyvoice_get_memory_usage

### 语法

```c
COSYVOICE_API void cosyvoice_get_memory_usage(cosyvoice_context_t ctx, cosyvoice_memory_usage_t* usage);
```

### 说明

获取当前内存占用快照。

### 参数

- `ctx`：模型上下文。
- `usage`：输出内存明细结构体。

## cosyvoice_get_total_memory_usage

### 语法

```c
COSYVOICE_API void cosyvoice_get_total_memory_usage(cosyvoice_context_t ctx, cosyvoice_memory_usage_t* usage);
```

### 说明

获取所有 worker 的总内存占用快照。

### 参数

- `ctx`：模型上下文。
- `usage`：输出内存明细结构体。

## cosyvoice_empty_buffer_cache

### 语法

```c
COSYVOICE_API void cosyvoice_empty_buffer_cache(cosyvoice_context_t ctx);
```

### 说明

清理上下文缓存的可复用推理缓冲区。

### 参数

- `ctx`：模型上下文。

## cosyvoice_request_stop

### 语法

```c
COSYVOICE_API void cosyvoice_request_stop(cosyvoice_context_t ctx);
```

### 说明

请求当前 worker 的任务停止并等待操作取消。清理内部上下文状态后返回。**必须在另一条线程调用**——在 stream 回调中调用会导致死锁。

### 参数

- `ctx`：模型上下文。

## cosyvoice_is_stop_requested

### 语法

```c
COSYVOICE_API bool cosyvoice_is_stop_requested(cosyvoice_context_t ctx);
```

### 说明

检查当前 worker 是否有停止请求。立即返回。不会清除停止标志。

### 参数

- `ctx`：模型上下文。

### 返回值

有停止请求则返回 `true`，否则返回 `false`。
