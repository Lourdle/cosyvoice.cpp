# cosyvoice-lowlevel.h API 参考

本文档覆盖 `include/cosyvoice-lowlevel.h` 中声明的全部公开符号。

## cosyvoice_tokenization_result_t

### 语法

```c
typedef struct cosyvoice_tokenization_result* cosyvoice_tokenization_result_t;
```

### 说明

分词结果容器的不透明句柄。

## cosyvoice_tokenizer_context_t

### 语法

```c
typedef struct cosyvoice_tokenizer_context*   cosyvoice_tokenizer_context_t;
```

### 说明

分词器上下文的不透明句柄。

## cosyvoice_inference_mode_t

### 语法

```c
typedef enum cosyvoice_inference_mode
{
    COSYVOICE_INFERENCE_MODE_NULL = -1,
    COSYVOICE_INFERENCE_MODE_ZERO_SHOT,
    COSYVOICE_INFERENCE_MODE_INSTRUCT,
    COSYVOICE_INFERENCE_MODE_CROSS_LINGUAL,
    COSYVOICE_INFERENCE_MODE_COUNT
} cosyvoice_inference_mode_t;
```

### 说明

低层提示设置接口使用的推理模式。

### 枚举值

- `COSYVOICE_INFERENCE_MODE_NULL`：不修改提示内容。
- `COSYVOICE_INFERENCE_MODE_ZERO_SHOT`：zero-shot 模式。
- `COSYVOICE_INFERENCE_MODE_INSTRUCT`：instruct 模式。
- `COSYVOICE_INFERENCE_MODE_CROSS_LINGUAL`：cross-lingual 模式。
- `COSYVOICE_INFERENCE_MODE_COUNT`：哨兵值。

## cosyvoice_noise_callback_stage_t

### 语法

```c
typedef enum cosyvoice_noise_callback_stage
{
    COSYVOICE_NOISE_CALLBACK_STAGE_BEFORE_FLOW,
    COSYVOICE_NOISE_CALLBACK_STAGE_AFTER_FLOW,
    COSYVOICE_NOISE_CALLBACK_STAGE_BEFORE_HIFT,
    COSYVOICE_NOISE_CALLBACK_STAGE_AFTER_HIFT
} cosyvoice_noise_callback_stage_t;
```

### 说明

指示噪声回调触发于 Flow/HiFT 执行前后哪个阶段。

### 枚举值

- `COSYVOICE_NOISE_CALLBACK_STAGE_BEFORE_FLOW`：Flow 前回调，需要返回噪声缓冲。
- `COSYVOICE_NOISE_CALLBACK_STAGE_AFTER_FLOW`：Flow 后回调，返回值忽略。
- `COSYVOICE_NOISE_CALLBACK_STAGE_BEFORE_HIFT`：HiFT 前回调，需要返回噪声缓冲。
- `COSYVOICE_NOISE_CALLBACK_STAGE_AFTER_HIFT`：HiFT 后回调，返回值忽略。

## cosyvoice_noise_callback_t

### 语法

```c
typedef float* (*cosyvoice_noise_callback_t)(
    cosyvoice_noise_callback_stage_t stage,
    uint32_t                         length,
    float*                           noise,
    void*                            ctx
);
```

### 说明

用于接管或观测噪声缓冲区的回调。

### 参数

- `stage`：当前阶段。
- `length`：期望采样点数量。
- `noise`：前置阶段为 `NULL`，后置阶段为之前缓冲区。
- `ctx`：用户上下文。

### 返回值

前置阶段返回噪声缓冲区指针；后置阶段返回值忽略。

## cosyvoice_log_callback_default

### 语法

```c
COSYVOICE_API void cosyvoice_log_callback_default(enum ggml_log_level level, const char* text, void* user_data);
```

### 说明

默认 GGML 日志回调。

### 参数

- `level`：日志级别。
- `text`：日志文本。
- `user_data`：用户数据。

### 返回值

无返回值。

## cosyvoice_load_from_file_ext

### 语法

```c
COSYVOICE_API cosyvoice_context_t cosyvoice_load_from_file_ext(
    const char*                       filename,
    const cosyvoice_context_params_t* params,
    ggml_backend_t                    backend,
    uint32_t                          n_threads,
    uint32_t                          reserved
);
```

### 说明

使用显式后端与线程配置加载模型。

### 参数

- `filename`：模型文件路径。
- `params`：上下文参数。
- `backend`：可选后端句柄。
- `n_threads`：CPU 线程数；传 `0` 时在可用情况下使用硬件并发数。
- `reserved`：保留参数，传 `0`。

### 返回值

成功返回上下文句柄，失败返回 `NULL`。

### 备注

- 该接口在 GGML 动态库与静态库构建下都可用。
- `backend == NULL` 表示自动选择后端。
- 如果 `backend != NULL`，其所有权会转移给创建出的上下文，并在 `cosyvoice_free()` 时自动释放。

## cosyvoice_get_last_status

### 语法

```c
COSYVOICE_API enum ggml_status cosyvoice_get_last_status(cosyvoice_context_t ctx);
```

### 说明

获取最近一次后端操作状态码。

### 参数

- `ctx`：模型上下文。

### 返回值

`ggml_status` 枚举值。

## cosyvoice_get_word_token_embed_weight

### 语法

```c
COSYVOICE_API const ggml_tensor* cosyvoice_get_word_token_embed_weight(cosyvoice_context_t ctx);
```

### 说明

获取词 token embedding 权重张量。

### 参数

- `ctx`：模型上下文。

### 返回值

只读张量指针。

## cosyvoice_get_speech_token_embed_weight

### 语法

```c
COSYVOICE_API const ggml_tensor* cosyvoice_get_speech_token_embed_weight(cosyvoice_context_t ctx);
```

### 说明

获取语音 token embedding 权重张量。

### 参数

- `ctx`：模型上下文。

### 返回值

只读张量指针。

## cosyvoice_llm_prefill

### 语法

```c
COSYVOICE_API bool cosyvoice_llm_prefill(
    cosyvoice_context_t ctx,
    enum ggml_type      type,
    const void*         data,
    uint32_t            n_tokens
);
```

### 说明

使用 embedding 序列进行 LLM 预填充。

### 参数

- `ctx`：模型上下文。
- `type`：输入元素类型。
- `data`：embedding 数据。
- `n_tokens`：token 数量。

### 返回值

成功返回 `true`，失败返回 `false`。

### 备注

该操作不计算下一个 token 的 logits。

## cosyvoice_llm_decode

### 语法

```c
COSYVOICE_API bool cosyvoice_llm_decode(
    cosyvoice_context_t ctx,
    enum ggml_type      type,
    const void*         data
);
```

### 说明

执行一次 LLM 解码步骤。

### 参数

- `ctx`：模型上下文。
- `type`：输入元素类型。
- `data`：当前步 embedding。

### 返回值

成功返回 `true`，失败返回 `false`。

### 备注

该函数只推进解码状态。在调用 `cosyvoice_llm_sample_token()` 前，需要先调用 `cosyvoice_llm_prepare_probs()`。

## cosyvoice_llm_prepare_probs

### 语法

```c
COSYVOICE_API void cosyvoice_llm_prepare_probs(cosyvoice_context_t ctx, bool allow_stop_tokens);
```

### 说明

基于最近一次解码结果准备采样概率。

### 参数

- `ctx`：模型上下文。
- `allow_stop_tokens`：为 `false` 时，将 stop token 的概率置为 0。

### 返回值

无返回值。

### 备注

每次 `cosyvoice_llm_decode()` 成功后，都应在 `cosyvoice_llm_sample_token()` 前调用本函数。

## cosyvoice_llm_get_kv_cache_len

### 语法

```c
COSYVOICE_API uint32_t cosyvoice_llm_get_kv_cache_len(cosyvoice_context_t ctx);
```

### 说明

获取当前 KV 缓存长度。

### 参数

- `ctx`：模型上下文。

### 返回值

当前 KV 缓存 token 数量。

## cosyvoice_llm_set_kv_cache_len

### 语法

```c
COSYVOICE_API bool cosyvoice_llm_set_kv_cache_len(cosyvoice_context_t ctx, uint32_t len);
```

### 说明

将 KV 缓存长度裁剪到指定值。

### 参数

- `ctx`：模型上下文。
- `len`：目标长度。

### 返回值

成功返回 `true`，失败返回 `false`。

### 备注

`len` 不得大于当前长度。

## cosyvoice_llm_sample_token

### 语法

```c
COSYVOICE_API int cosyvoice_llm_sample_token(cosyvoice_context_t ctx);
```

### 说明

从当前 logits 中采样下一个 token。

### 参数

- `ctx`：模型上下文。

### 返回值

采样得到的 token ID。

## cosyvoice_llm_is_stop_token

### 语法

```c
COSYVOICE_API bool cosyvoice_llm_is_stop_token(cosyvoice_context_t ctx, int token_id);
```

### 说明

判断 token 是否为停止标记。

### 参数

- `ctx`：模型上下文。
- `token_id`：待判断 token ID。

### 返回值

是停止 token 返回 `true`，否则返回 `false`。

## cosyvoice_llm_accept_token

### 语法

```c
COSYVOICE_API void cosyvoice_llm_accept_token(cosyvoice_context_t ctx, int token_id);
```

### 说明

将 token 记入已接受序列。

### 参数

- `ctx`：模型上下文。
- `token_id`：token ID。

### 返回值

无返回值。

## cosyvoice_llm_clear_accepted_tokens

### 语法

```c
COSYVOICE_API void cosyvoice_llm_clear_accepted_tokens(cosyvoice_context_t ctx);
```

### 说明

清空已接受 token 历史。

### 参数

- `ctx`：模型上下文。

### 返回值

无返回值。

## cosyvoice_llm_get_n_accepted_tokens

### 语法

```c
COSYVOICE_API uint32_t cosyvoice_llm_get_n_accepted_tokens(cosyvoice_context_t ctx);
```

### 说明

获取已接受 token 数量。

### 参数

- `ctx`：模型上下文。

### 返回值

已接受 token 数量。

## cosyvoice_llm_get_accepted_tokens

### 语法

```c
COSYVOICE_API const int* cosyvoice_llm_get_accepted_tokens(cosyvoice_context_t ctx);
```

### 说明

获取已接受 token 缓冲区指针。

### 参数

- `ctx`：模型上下文。

### 返回值

只读 token ID 数组指针。

## cosyvoice_llm_job

### 语法

```c
COSYVOICE_API bool cosyvoice_llm_job(
    cosyvoice_context_t ctx,
    const int*          text,
    uint32_t            text_len,
    cosyvoice_prompt_t  prompt
);
```

### 说明

执行低层 LLM 推理流程。

### 参数

- `ctx`：模型上下文。
- `text`：输入文本 token 数组。
- `text_len`：输入 token 数量。
- `prompt`：提示对象。

### 返回值

成功返回 `true`，失败返回 `false`。

## cosyvoice_token2wav

### 语法

```c
COSYVOICE_API bool cosyvoice_token2wav(
    cosyvoice_context_t            ctx,
    const int*                     token_ids,
    uint32_t                       n_tokens,
    float                          speed,
    cosyvoice_prompt_t             prompt,
    cosyvoice_generated_speech_ptr generated_speech
);
```

### 说明

将声学 token 转换为波形。

### 参数

- `ctx`：模型上下文。
- `token_ids`：声学 token 数组。
- `n_tokens`：token 数量。
- `speed`：语速系数。
- `prompt`：提示对象。
- `generated_speech`：输出波形容器。

### 返回值

成功返回 `true`，失败返回 `false`。

## cosyvoice_tts

### 语法

```c
COSYVOICE_API bool cosyvoice_tts(
    cosyvoice_context_t            ctx,
    const int*                     text,
    uint32_t                       text_len,
    float                          speed,
    cosyvoice_prompt_t             prompt,
    cosyvoice_generated_speech_ptr result
);
```

### 说明

执行完整低层 TTS 管线。

### 参数

- `ctx`：模型上下文。
- `text`：输入文本 token 数组。
- `text_len`：token 数量。
- `speed`：语速系数。
- `prompt`：提示对象。
- `result`：输出波形容器。

### 返回值

成功返回 `true`，失败返回 `false`。

### 备注

该接口将 LLM 生成与波形转换封装为一次调用。

## cosyvoice_get_tokenizer

### 语法

```c
COSYVOICE_API cosyvoice_tokenizer_context_t cosyvoice_get_tokenizer(cosyvoice_context_t ctx);
```

### 说明

获取模型上下文持有的分词器（借用句柄）。

### 参数

- `ctx`：模型上下文。

### 返回值

分词器句柄。

### 备注

该句柄为模型上下文借用句柄。

## cosyvoice_tokenizer_load_from_file

### 语法

```c
COSYVOICE_API cosyvoice_tokenizer_context_t cosyvoice_tokenizer_load_from_file(const char* filename);
```

### 说明

从文件独立加载分词器。

### 参数

- `filename`：文件路径。

### 返回值

成功返回分词器句柄，失败返回 `NULL`。

## cosyvoice_tokenizer_free

### 语法

```c
COSYVOICE_API void cosyvoice_tokenizer_free(cosyvoice_tokenizer_context_t ctx);
```

### 说明

释放独立分词器上下文。

### 参数

- `ctx`：分词器句柄。

### 返回值

无返回值。

### 备注

仅用于释放由 `cosyvoice_tokenizer_load_from_file` 创建的独立分词器。

## cosyvoice_tokenization_result_create

### 语法

```c
COSYVOICE_API cosyvoice_tokenization_result_t cosyvoice_tokenization_result_create();
```

### 说明

创建空分词结果容器。

### 参数

无。

### 返回值

成功返回结果句柄，失败返回 `NULL`。

## cosyvoice_tokenization_result_free

### 语法

```c
COSYVOICE_API void cosyvoice_tokenization_result_free(cosyvoice_tokenization_result_t result);
```

### 说明

释放分词结果容器。

### 参数

- `result`：结果句柄。

### 返回值

无返回值。

## cosyvoice_tokenization_result_get_tokens

### 语法

```c
COSYVOICE_API int* cosyvoice_tokenization_result_get_tokens(cosyvoice_tokenization_result_t result);
```

### 说明

获取可写 token 缓冲区。

### 参数

- `result`：结果句柄。

### 返回值

token 数组指针。

## cosyvoice_tokenization_result_get_n_tokens

### 语法

```c
COSYVOICE_API uint32_t cosyvoice_tokenization_result_get_n_tokens(cosyvoice_tokenization_result_t result);
```

### 说明

获取结果中的 token 数量。

### 参数

- `result`：结果句柄。

### 返回值

token 数量。

## cosyvoice_tokenize

### 语法

```c
COSYVOICE_API uint32_t cosyvoice_tokenize(
    cosyvoice_tokenizer_context_t   ctx,
    const char*                     text,
    cosyvoice_tokenization_result_t result,
    bool                            parse_special
);
```

### 说明

对以空字符结尾的 UTF-8 字符串分词。

### 参数

- `ctx`：分词器句柄。
- `text`：输入文本。
- `result`：输出结果容器。
- `parse_special`：是否解析特殊 token。

### 返回值

写入的 token 数量。

## cosyvoice_tokenize_ext

### 语法

```c
COSYVOICE_API uint32_t cosyvoice_tokenize_ext(
    cosyvoice_tokenizer_context_t   ctx,
    const char*                     text,
    uint32_t                        text_len,
    cosyvoice_tokenization_result_t result,
    bool                            parse_special
);
```

### 说明

按显式字节长度对 UTF-8 文本分词。

### 参数

- `ctx`：分词器句柄。
- `text`：输入文本。
- `text_len`：输入字节长度。
- `result`：输出结果容器。
- `parse_special`：是否解析特殊 token。

### 返回值

写入的 token 数量。

## cosyvoice_set_noise_callback

### 语法

```c
COSYVOICE_API void cosyvoice_set_noise_callback(cosyvoice_context_t ctx, cosyvoice_noise_callback_t callback, void* callback_ctx);
```

### 说明

设置噪声回调。

### 参数

- `ctx`：模型上下文。
- `callback`：回调函数。
- `callback_ctx`：回调上下文。

### 返回值

无返回值。

## cosyvoice_get_noise_callback

### 语法

```c
COSYVOICE_API void cosyvoice_get_noise_callback(cosyvoice_context_t ctx, cosyvoice_noise_callback_t* callback, void** callback_ctx);
```

### 说明

获取当前噪声回调与上下文。

### 参数

- `ctx`：模型上下文。
- `callback`：输出回调指针。
- `callback_ctx`：输出上下文指针。

### 返回值

无返回值。

## cosyvoice_get_hift_rand_ini_len

### 语法

```c
COSYVOICE_API uint32_t cosyvoice_get_hift_rand_ini_len(cosyvoice_context_t ctx);
```

### 说明

获取 HiFT 初始化噪声缓冲区所需长度。

### 参数

- `ctx`：模型上下文。

### 返回值

所需采样点数量。

## cosyvoice_set_hift_rand_ini

### 语法

```c
COSYVOICE_API void cosyvoice_set_hift_rand_ini(cosyvoice_context_t ctx, const float* data);
```

### 说明

覆盖 HiFT 初始化噪声缓冲区。

### 参数

- `ctx`：模型上下文。
- `data`：噪声数据指针。

### 返回值

无返回值。

## cosyvoice_prompt_speech_get_crc32

### 语法

```c
COSYVOICE_API uint32_t cosyvoice_prompt_speech_get_crc32(cosyvoice_prompt_speech_t prompt_speech);
```

### 说明

计算提示语音对象的 CRC32。

### 参数

- `prompt_speech`：提示语音对象。

### 返回值

CRC32 值。

## cosyvoice_prompt_get_crc32

### 语法

```c
COSYVOICE_API uint32_t cosyvoice_prompt_get_crc32(cosyvoice_prompt_t prompt);
```

### 说明

计算提示对象的 CRC32。

### 参数

- `prompt`：提示对象。

### 返回值

CRC32 值。

## cosyvoice_get_instruction_prefix

### 语法

```c
COSYVOICE_API const char* cosyvoice_get_instruction_prefix(cosyvoice_context_t ctx);
```

### 说明

获取模型要求的指令前缀。

### 参数

- `ctx`：模型上下文。

### 返回值

UTF-8 前缀字符串指针。

## cosyvoice_prompt_set

### 语法

```c
COSYVOICE_API cosyvoice_prompt_t cosyvoice_prompt_set(
    cosyvoice_context_t        ctx,
    cosyvoice_prompt_t         prompt,
    cosyvoice_inference_mode_t mode,
    const char*                instruction,
#ifdef __cplusplus
    uint32_t                   instruction_length = 0xFFFFFFFFU,
    bool                       inplace = true
#else
    uint32_t                   instruction_length,
    bool                       inplace
#endif
);
```

### 说明

使用原始文本设置提示模式与指令内容。

### 参数

- `ctx`：模型上下文。
- `prompt`：待更新提示对象。
- `mode`：推理模式。
- `instruction`：指令文本。
- `instruction_length`：指令字节长度。
- `inplace`：是否原地更新。

### 返回值

更新后的提示句柄。

### 备注

该接口不会自动拼接模型指令前缀。

## cosyvoice_prompt_set_ext

### 语法

```c
COSYVOICE_API cosyvoice_prompt_t cosyvoice_prompt_set_ext(
    cosyvoice_context_t        ctx,
    cosyvoice_prompt_t         prompt,
    cosyvoice_inference_mode_t mode,
    const int*                 instruction,
    uint32_t                   instruction_length,
    bool                       inplace
);
```

### 说明

使用 token 化指令设置提示模式与内容。

### 参数

- `ctx`：模型上下文。
- `prompt`：待更新提示对象。
- `mode`：推理模式。
- `instruction`：指令 token 数组。
- `instruction_length`：指令 token 数量。
- `inplace`：是否原地更新。

### 返回值

更新后的提示句柄。

### 备注

不会自动拼接指令前缀，也不会自动追加 `<|endofprompt|>`。
