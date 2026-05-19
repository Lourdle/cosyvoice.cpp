# cosyvoice.h API 参考

本文档覆盖 `include/cosyvoice.h` 中声明的全部公开符号。

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

## cosyvoice_llm_kv_cache_type_t

### 语法

```c
typedef enum cosyvoice_llm_kv_cache_type
{
    COSYVOICE_LLM_KV_CACHE_TYPE_F32,
    COSYVOICE_LLM_KV_CACHE_TYPE_F16,
    COSYVOICE_LLM_KV_CACHE_TYPE_Q8_0,
    COSYVOICE_LLM_KV_CACHE_TYPE_Q5_1,
    COSYVOICE_LLM_KV_CACHE_TYPE_Q5_0,
    COSYVOICE_LLM_KV_CACHE_TYPE_Q4_1,
    COSYVOICE_LLM_KV_CACHE_TYPE_Q4_0,
    COSYVOICE_LLM_KV_CACHE_TYPE_COUNT
} cosyvoice_llm_kv_cache_type_t;
```

### 说明

指定 LLM 的 KV 缓存存储格式。

### 枚举值

- `COSYVOICE_LLM_KV_CACHE_TYPE_F32`：32 位浮点格式。
- `COSYVOICE_LLM_KV_CACHE_TYPE_F16`：16 位浮点格式。
- `COSYVOICE_LLM_KV_CACHE_TYPE_Q8_0`：GGML `Q8_0` 量化格式。
- `COSYVOICE_LLM_KV_CACHE_TYPE_Q5_1`：GGML `Q5_1` 量化格式。
- `COSYVOICE_LLM_KV_CACHE_TYPE_Q5_0`：GGML `Q5_0` 量化格式。
- `COSYVOICE_LLM_KV_CACHE_TYPE_Q4_1`：GGML `Q4_1` 量化格式。
- `COSYVOICE_LLM_KV_CACHE_TYPE_Q4_0`：GGML `Q4_0` 量化格式。
- `COSYVOICE_LLM_KV_CACHE_TYPE_COUNT`：哨兵值，不用于运行配置。

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

    cosyvoice_llm_kv_cache_type_t       llm_kv_cache_type;
    bool                                llm_allow_kv_cache_fallback;
    cosyvoice_inference_buffer_policy_t inference_buffer_policy;

    uint32_t n_batch;
    uint32_t n_max_seq;
    uint32_t seed;
    cosyvoice_builtin_sampler_rng_policy_t builtin_sampler_rng_policy;

    cosyvoice_sampler_t sampler;
    void*               sampler_ctx;
} cosyvoice_context_params_t;
```

### 说明

上下文创建与推理行为控制参数集合。

### 成员

- `llm_use_flash_attn`：是否启用 LLM Flash Attention。
- `flow_use_flash_attn`：是否启用 Flow Flash Attention。
- `llm_kv_cache_type`：KV 缓存数据类型。
- `llm_allow_kv_cache_fallback`：不兼容时是否允许回退 KV 类型。
- `inference_buffer_policy`：推理缓冲策略。
- `n_batch`：推理批大小。
- `n_max_seq`：最大序列长度。
- `seed`：随机种子。
- `builtin_sampler_rng_policy`：内置采样器 RNG 策略。
- `sampler`：自定义采样器，`NULL` 表示使用内置采样器。
- `sampler_ctx`：传入采样回调的用户上下文。

## cosyvoice_init_backend

### 语法

```c
COSYVOICE_API void cosyvoice_init_backend();
```

### 说明

初始化默认后端运行时。

### 参数

无。

### 返回值

无返回值。

### 备注

会从指定目录加载后端资源并完成运行时初始化。

## cosyvoice_init_backend_from_path

### 语法

```c
COSYVOICE_API void cosyvoice_init_backend_from_path(const char* dir_path);
```

### 说明

使用指定目录初始化后端运行时。

### 参数

- `dir_path`：后端初始化目录路径。

### 返回值

无返回值。

## cosyvoice_init_default_context_params

### 语法

```c
COSYVOICE_API void cosyvoice_init_default_context_params(cosyvoice_context_params_t* params);
```

### 说明

将上下文参数结构体填充为默认值。

### 参数

- `params`：输出参数结构体。

### 返回值

无返回值。

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

## cosyvoice_free

### 语法

```c
COSYVOICE_API void cosyvoice_free(cosyvoice_context_t ctx);
```

### 说明

释放模型上下文及其关联资源。

### 参数

- `ctx`：待释放上下文。

### 返回值

无返回值。

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

### 返回值

无返回值。

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

### 返回值

无返回值。

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

### 返回值

无返回值。

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

### 返回值

无返回值。

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

即使使用自定义采样器，种子也会被保存。

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

### 返回值

无返回值。

## cosyvoice_prompt_free

### 语法

```c
COSYVOICE_API void cosyvoice_prompt_free(cosyvoice_prompt_t prompt);
```

### 说明

释放提示对象。

### 参数

- `prompt`：待释放对象。

### 返回值

无返回值。

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

### 返回值

无返回值。

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

### 返回值

无返回值。

### 备注

会重置该会话缓存的指令文本。

## cosyvoice_tts_context_set_text_normalization_enabled

### 语法

```c
COSYVOICE_API void cosyvoice_tts_context_set_text_normalization_enabled(cosyvoice_tts_context_t ctx, bool enabled);
```

### 说明

为指定 TTS 会话开启或关闭前端文本标准化。

### 参数

- `ctx`：会话句柄。
- `enabled`：`true` 表示开启文本标准化；`false` 表示跳过标准化，直接对原始文本分词。

### 返回值

无返回值。

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

### 返回值

无返回值。

## cosyvoice_empty_buffer_cache

### 语法

```c
COSYVOICE_API void cosyvoice_empty_buffer_cache(cosyvoice_context_t ctx);
```

### 说明

清理上下文缓存的可复用推理缓冲区。

### 参数

- `ctx`：模型上下文。

### 返回值

无返回值。
