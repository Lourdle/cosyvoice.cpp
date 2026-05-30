# cosyvoice-interface.h API 参考

本文档覆盖 `include/cosyvoice-interface.h` 中声明的全部符号，包含 C++ 接口方法。
运行时支持通过多个 worker 槽实现并发推理；不同上下文可以绑定到不同 worker，同时共享已加载的模型资源。
这些接口属于内部实现细节，不承诺 ABI 稳定，未来版本可能会调整。

## cosyvoice_model_context

### 语法

```cpp
struct cosyvoice_model_context
{
    virtual uint32_t get_sample_rate() = 0;
    virtual void get_default_generation_config(cosyvoice_generation_config_t* config) = 0;
    virtual void get_generation_config(cosyvoice_generation_config_t* config) = 0;
    virtual bool set_generation_config(const cosyvoice_generation_config_t* config) = 0;
    virtual void get_context_params(cosyvoice_context_params_t* params) = 0;
    virtual const char* get_architecture() = 0;
    virtual const char* get_instruction_prefix() = 0;
    virtual bool set_worker_no(uint32_t worker_no) = 0;
    virtual uint32_t get_worker_no() = 0;
    virtual uint32_t get_n_workers() = 0;

    virtual void get_sampler(cosyvoice_sampler_t* sampler, void** sampler_ctx) = 0;
    virtual void set_sampler(cosyvoice_sampler_t sampler, void* sampler_ctx) = 0;
    virtual cosyvoice_builtin_sampler_rng_policy_t get_builtin_sampler_rng_policy() = 0;
    virtual bool set_builtin_sampler_rng_policy(cosyvoice_builtin_sampler_rng_policy_t policy) = 0;
    virtual bool set_sampler_seed(uint32_t seed) = 0;

    virtual bool llm_prefill(ggml_type type, const void* data, uint32_t seq_len) = 0;
    virtual bool llm_decode(ggml_type type, const void* data) = 0;
    virtual void llm_prepare_probs(bool allow_stop_tokens) = 0;

    virtual const ggml_tensor* get_word_token_embed_weight() = 0;
    virtual const ggml_tensor* get_speech_token_embed_weight() = 0;

    virtual uint32_t llm_get_kv_cache_len() = 0;
    virtual bool llm_set_kv_cache_len(uint32_t len) = 0;

    virtual int llm_sample_token() = 0;
    virtual bool llm_is_stop_token(int token_id) = 0;
    virtual void llm_accept_token(int token_id) = 0;
    virtual void llm_clear_accepted_tokens() = 0;
    virtual uint32_t llm_get_n_accepted_tokens() = 0;
    virtual const int* llm_get_accepted_tokens() = 0;

    virtual bool llm_job(const int* text, uint32_t text_len, cosyvoice_prompt_t prompt) = 0;
    virtual bool token2wav(
        const int*                 token_ids,
        uint32_t                   n_tokens,
        float                      speed,
        cosyvoice_prompt_t         prompt,
        cosyvoice_generated_speech_ptr result
    ) = 0;

    virtual ggml_status get_last_status() = 0;

    virtual void set_prompt(
        cosyvoice_prompt_t         prompt,
        cosyvoice_inference_mode_t mode,
        const int*                 instruction,
        uint32_t                   instruction_length
    ) = 0;

    virtual void get_memory_usage(cosyvoice_memory_usage_t* usage) = 0;
    virtual void empty_buffer_cache() = 0;

    virtual void set_noise_callback(cosyvoice_noise_callback_t callback, void* callback_ctx) = 0;
    virtual void get_noise_callback(cosyvoice_noise_callback_t* callback, void** callback_ctx) = 0;
    virtual uint32_t get_hift_rand_ini_len() = 0;
    virtual void set_hift_rand_ini(const float* data) = 0;
};
```

### 说明

模型运行时实现的核心抽象接口。

## cosyvoice_model_context::get_sample_rate

### 语法

```cpp
virtual uint32_t get_sample_rate() = 0;
```

### 说明

获取模型输出采样率。

### 参数

无。

### 返回值

采样率（Hz）。

## cosyvoice_model_context::get_generation_config

### 语法

```cpp
virtual void get_generation_config(cosyvoice_generation_config_t* config) = 0;
```

### 说明

读取当前生成配置。

### 参数

- `config`：输出配置结构体。

### 返回值

无返回值。

## cosyvoice_model_context::get_default_generation_config

### 语法

```cpp
virtual void get_default_generation_config(cosyvoice_generation_config_t* config) = 0;
```

### 说明

读取模型文件中的默认生成配置，尚未应用 worker 级覆盖。

### 参数

- `config`：输出配置结构体。

### 返回值

无返回值。

## cosyvoice_model_context::set_generation_config

### 语法

```cpp
virtual bool set_generation_config(const cosyvoice_generation_config_t* config) = 0;
```

### 说明

校验并应用生成配置。

### 参数

- `config`：待应用配置。

### 返回值

成功返回 `true`，失败返回 `false`。

## cosyvoice_model_context::get_context_params

### 语法

```cpp
virtual void get_context_params(cosyvoice_context_params_t* params) = 0;
```

### 说明

读取当前生效上下文参数。

### 参数

- `params`：输出参数结构体。

### 返回值

无返回值。

## cosyvoice_model_context::set_worker_no

### 语法

```cpp
virtual bool set_worker_no(uint32_t worker_no) = 0;
```

### 说明

设置后续操作使用的 worker 槽。

### 参数

- `worker_no`：worker 槽编号。

### 返回值

成功返回 `true`，否则返回 `false`。

### 备注

当两个线程需要在不同 worker 上并发推理时，应先复制上下文，再分别绑定 worker。

## cosyvoice_model_context::get_worker_no

### 语法

```cpp
virtual uint32_t get_worker_no() = 0;
```

### 说明

获取当前激活的 worker 槽编号。

### 返回值

当前 worker 编号。

## cosyvoice_model_context::get_n_workers

### 语法

```cpp
virtual uint32_t get_n_workers() = 0;
```

### 说明

获取可用 worker 槽总数。

### 返回值

worker 槽数量。

## cosyvoice_model_context::get_architecture

### 语法

```cpp
virtual const char* get_architecture() = 0;
```

### 说明

获取已加载模型的架构标识。

### 返回值

以 `\0` 结尾的 UTF-8 架构字符串，例如 `cosyvoice3-2512`。

## cosyvoice_model_context::get_instruction_prefix

### 语法

```cpp
virtual const char* get_instruction_prefix() = 0;
```

### 说明

获取模型要求的指令前缀。

### 参数

无。

### 返回值

UTF-8 字符串指针。

## cosyvoice_model_context::get_sampler

### 语法

```cpp
virtual void get_sampler(cosyvoice_sampler_t* sampler, void** sampler_ctx) = 0;
```

### 说明

获取当前采样器与上下文。

### 参数

- `sampler`：输出采样器函数。
- `sampler_ctx`：输出用户上下文。

### 返回值

无返回值。

## cosyvoice_model_context::set_sampler

### 语法

```cpp
virtual void set_sampler(cosyvoice_sampler_t sampler, void* sampler_ctx) = 0;
```

### 说明

设置采样器与上下文。

### 参数

- `sampler`：采样器函数。
- `sampler_ctx`：用户上下文。

### 返回值

无返回值。

## cosyvoice_model_context::get_builtin_sampler_rng_policy

### 语法

```cpp
virtual cosyvoice_builtin_sampler_rng_policy_t get_builtin_sampler_rng_policy() = 0;
```

### 说明

获取内置采样器 RNG 策略。

### 参数

无。

### 返回值

当前策略值。

## cosyvoice_model_context::set_builtin_sampler_rng_policy

### 语法

```cpp
virtual bool set_builtin_sampler_rng_policy(cosyvoice_builtin_sampler_rng_policy_t policy) = 0;
```

### 说明

设置内置采样器 RNG 策略。

### 参数

- `policy`：策略值。

### 返回值

成功返回 `true`，失败返回 `false`。

## cosyvoice_model_context::set_sampler_seed

### 语法

```cpp
virtual bool set_sampler_seed(uint32_t seed) = 0;
```

### 说明

设置采样器随机种子。

### 参数

- `seed`：种子值。

### 返回值

成功返回 `true`，失败返回 `false`。

## cosyvoice_model_context::get_sampler_seed

### 语法

```cpp
virtual uint32_t get_sampler_seed() = 0;
```

### 说明

获取当前 worker 的采样器种子。

### 返回值

当前 worker 的种子值。

## cosyvoice_model_context::llm_prefill

### 语法

```cpp
virtual bool llm_prefill(ggml_type type, const void* data, uint32_t seq_len) = 0;
```

### 说明

对 LLM 执行预填充。

### 参数

- `type`：输入元素类型。
- `data`：embedding 数据。
- `seq_len`：序列长度。

### 返回值

成功返回 `true`，失败返回 `false`。

## cosyvoice_model_context::llm_decode

### 语法

```cpp
virtual bool llm_decode(ggml_type type, const void* data) = 0;
```

### 说明

执行一次 LLM 解码。

### 参数

- `type`：输入元素类型。
- `data`：embedding 向量。

### 返回值

成功返回 `true`，失败返回 `false`。

### 备注

该方法只推进解码状态，采样前需要调用 `llm_prepare_probs()`。

## cosyvoice_model_context::llm_prepare_probs

### 语法

```cpp
virtual void llm_prepare_probs(bool allow_stop_tokens) = 0;
```

### 说明

基于最近一次解码结果准备用于采样的概率分布。

### 参数

- `allow_stop_tokens`：为 `false` 时，将 stop token 的概率置为 0。

### 返回值

无返回值。

## cosyvoice_model_context::get_word_token_embed_weight

### 语法

```cpp
virtual const ggml_tensor* get_word_token_embed_weight() = 0;
```

### 说明

获取词 token embedding 张量。

### 参数

无。

### 返回值

只读张量指针。

## cosyvoice_model_context::get_speech_token_embed_weight

### 语法

```cpp
virtual const ggml_tensor* get_speech_token_embed_weight() = 0;
```

### 说明

获取语音 token embedding 张量。

### 参数

无。

### 返回值

只读张量指针。

## cosyvoice_model_context::llm_get_kv_cache_len

### 语法

```cpp
virtual uint32_t llm_get_kv_cache_len() = 0;
```

### 说明

获取 KV 缓存长度。

### 参数

无。

### 返回值

KV 缓存 token 数量。

## cosyvoice_model_context::llm_set_kv_cache_len

### 语法

```cpp
virtual bool llm_set_kv_cache_len(uint32_t len) = 0;
```

### 说明

设置 KV 缓存长度。

### 参数

- `len`：目标长度。

### 返回值

成功返回 `true`，失败返回 `false`。

## cosyvoice_model_context::llm_sample_token

### 语法

```cpp
virtual int llm_sample_token() = 0;
```

### 说明

采样下一个 token。

### 参数

无。

### 返回值

采样 token ID。

## cosyvoice_model_context::llm_is_stop_token

### 语法

```cpp
virtual bool llm_is_stop_token(int token_id) = 0;
```

### 说明

判断是否为停止 token。

### 参数

- `token_id`：待判断 token ID。

### 返回值

是停止 token 返回 `true`，否则返回 `false`。

## cosyvoice_model_context::llm_accept_token

### 语法

```cpp
virtual void llm_accept_token(int token_id) = 0;
```

### 说明

将 token 记入已接受序列。

### 参数

- `token_id`：token ID。

### 返回值

无返回值。

## cosyvoice_model_context::llm_clear_accepted_tokens

### 语法

```cpp
virtual void llm_clear_accepted_tokens() = 0;
```

### 说明

清空已接受 token 历史。

### 参数

无。

### 返回值

无返回值。

## cosyvoice_model_context::llm_get_n_accepted_tokens

### 语法

```cpp
virtual uint32_t llm_get_n_accepted_tokens() = 0;
```

### 说明

获取已接受 token 数量。

### 参数

无。

### 返回值

token 数量。

## cosyvoice_model_context::llm_get_accepted_tokens

### 语法

```cpp
virtual const int* llm_get_accepted_tokens() = 0;
```

### 说明

获取已接受 token 缓冲区指针。

### 参数

无。

### 返回值

只读 token 数组指针。

## cosyvoice_model_context::llm_job

### 语法

```cpp
virtual bool llm_job(const int* text, uint32_t text_len, cosyvoice_prompt_t prompt) = 0;
```

### 说明

执行低层 LLM 推理任务。

### 参数

- `text`：输入 token 数组。
- `text_len`：token 数量。
- `prompt`：提示对象。

### 返回值

成功返回 `true`，失败返回 `false`。

## cosyvoice_model_context::token2wav

### 语法

```cpp
virtual bool token2wav(
    const int*                 token_ids,
    uint32_t                   n_tokens,
    float                      speed,
    cosyvoice_prompt_t         prompt,
    cosyvoice_generated_speech_ptr result
) = 0;
```

### 说明

将声学 token 转换为波形。

### 参数

- `token_ids`：声学 token 数组。
- `n_tokens`：token 数量。
- `speed`：语速系数。
- `prompt`：提示对象。
- `result`：输出波形容器。

### 返回值

成功返回 `true`，失败返回 `false`。

## cosyvoice_model_context::get_last_status

### 语法

```cpp
virtual ggml_status get_last_status() = 0;
```

### 说明

获取最近一次后端操作状态。

### 参数

无。

### 返回值

`ggml_status` 值。

## cosyvoice_model_context::set_prompt

### 语法

```cpp
virtual void set_prompt(
    cosyvoice_prompt_t         prompt,
    cosyvoice_inference_mode_t mode,
    const int*                 instruction,
    uint32_t                   instruction_length
) = 0;
```

### 说明

设置提示模式与 token 化指令。

### 参数

- `prompt`：提示对象。
- `mode`：推理模式。
- `instruction`：指令 token 数组。
- `instruction_length`：指令 token 数量。

### 返回值

无返回值。

## cosyvoice_model_context::get_memory_usage

### 语法

```cpp
virtual void get_memory_usage(cosyvoice_memory_usage_t* usage) = 0;
```

### 说明

获取内存占用快照。

### 参数

- `usage`：输出内存统计结构体。

### 返回值

无返回值。

## cosyvoice_model_context::empty_buffer_cache

### 语法

```cpp
virtual void empty_buffer_cache() = 0;
```

### 说明

清理可复用缓冲缓存。

### 参数

无。

### 返回值

无返回值。

## cosyvoice_model_context::set_noise_callback

### 语法

```cpp
virtual void set_noise_callback(cosyvoice_noise_callback_t callback, void* callback_ctx) = 0;
```

### 说明

设置噪声回调。

### 参数

- `callback`：回调函数。
- `callback_ctx`：回调上下文。

### 返回值

无返回值。

## cosyvoice_model_context::get_noise_callback

### 语法

```cpp
virtual void get_noise_callback(cosyvoice_noise_callback_t* callback, void** callback_ctx) = 0;
```

### 说明

获取当前噪声回调与上下文。

### 参数

- `callback`：输出回调指针。
- `callback_ctx`：输出上下文指针。

### 返回值

无返回值。

## cosyvoice_model_context::get_hift_rand_ini_len

### 语法

```cpp
virtual uint32_t get_hift_rand_ini_len() = 0;
```

### 说明

获取 HiFT 初始化噪声长度。

### 参数

无。

### 返回值

所需采样点数量。

## cosyvoice_model_context::set_hift_rand_ini

### 语法

```cpp
virtual void set_hift_rand_ini(const float* data) = 0;
```

### 说明

设置 HiFT 初始化噪声缓冲区。

### 参数

- `data`：噪声数据指针。

### 返回值

无返回值。

## cosyvoice_tokenization_result

### 语法

```cpp
struct cosyvoice_tokenization_result
{
    virtual int* get_tokens() = 0;
    virtual uint32_t get_n_tokens() = 0;
};
```

### 说明

分词结果抽象容器接口。

## cosyvoice_tokenization_result::get_tokens

### 语法

```cpp
virtual int* get_tokens() = 0;
```

### 说明

获取可写 token 缓冲区。

### 参数

无。

### 返回值

token 数组指针。

## cosyvoice_tokenization_result::get_n_tokens

### 语法

```cpp
virtual uint32_t get_n_tokens() = 0;
```

### 说明

获取 token 数量。

### 参数

无。

### 返回值

token 数量。

## cosyvoice_tokenizer_context

### 语法

```cpp
struct cosyvoice_tokenizer_context
{
    virtual uint32_t tokenize(const char* text, uint32_t text_len, cosyvoice_tokenization_result_t result, bool parse_special = true) = 0;

    inline uint32_t tokenize(const char* text, cosyvoice_tokenization_result_t result, bool parse_special = true)
    {
        auto p = text;
        while (*p) ++p;
        return tokenize(text, static_cast<uint32_t>(p - text), result, parse_special);
    }
};
```

### 说明

模型与独立分词器共享的抽象分词接口。

## cosyvoice_tokenizer_context::tokenize（长度重载）

### 语法

```cpp
virtual uint32_t tokenize(const char* text, uint32_t text_len, cosyvoice_tokenization_result_t result, bool parse_special = true) = 0;
```

### 说明

按显式字节长度对 UTF-8 文本分词。

### 参数

- `text`：输入 UTF-8 缓冲区。
- `text_len`：输入字节长度。
- `result`：输出结果容器。
- `parse_special`：是否解析特殊 token。

### 返回值

写入 token 数量。

## cosyvoice_tokenizer_context::tokenize（空终止重载）

### 语法

```cpp
inline uint32_t tokenize(const char* text, cosyvoice_tokenization_result_t result, bool parse_special = true)
```

### 说明

便捷重载，会先计算空终止字符串长度再转调长度重载。

### 参数

- `text`：空终止 UTF-8 字符串。
- `result`：输出结果容器。
- `parse_special`：是否解析特殊 token。

### 返回值

写入 token 数量。

## cosyvoice_context

### 语法

```cpp
struct cosyvoice_context : virtual cosyvoice_model_context,
                           virtual cosyvoice_tokenizer_context {};
```

### 说明

组合模型接口与分词接口的统一上下文抽象。
