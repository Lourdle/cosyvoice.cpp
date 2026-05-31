# cosyvoice-interface.h API Reference

This page documents all symbols declared in `include/cosyvoice-interface.h`, including C++ interface methods.
The runtime supports concurrent inference through multiple worker slots; separate contexts can bind to different workers while sharing loaded model resources.
These interfaces are internal implementation details and do not guarantee ABI stability; they may change in future releases.

## cosyvoice_model_context

### Syntax

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
    virtual bool is_backend_uma() = 0;
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

### Description

Abstract interface implemented by runtime model contexts.

## cosyvoice_model_context::get_sample_rate

### Syntax

```cpp
virtual uint32_t get_sample_rate() = 0;
```

### Description

Gets model output sample rate.

### Returns

Sample rate in Hz.

## cosyvoice_model_context::get_generation_config

### Syntax

```cpp
virtual void get_generation_config(cosyvoice_generation_config_t* config) = 0;
```

### Description

Copies current generation configuration.

### Parameters

- `config`: Output configuration structure.

### Returns

No return value.

## cosyvoice_model_context::get_default_generation_config

### Syntax

```cpp
virtual void get_default_generation_config(cosyvoice_generation_config_t* config) = 0;
```

### Description

Copies the model-file default generation configuration before worker-level overrides.

### Parameters

- `config`: Output configuration structure.

### Returns

No return value.

## cosyvoice_model_context::set_generation_config

### Syntax

```cpp
virtual bool set_generation_config(const cosyvoice_generation_config_t* config) = 0;
```

### Description

Validates and applies generation configuration.

### Parameters

- `config`: Configuration to apply.

### Returns

`true` on success; otherwise `false`.

## cosyvoice_model_context::get_context_params

### Syntax

```cpp
virtual void get_context_params(cosyvoice_context_params_t* params) = 0;
```

### Description

Copies effective context parameters.

### Parameters

- `params`: Output parameter structure.

### Returns

No return value.

## cosyvoice_model_context::set_worker_no

### Syntax

```cpp
virtual bool set_worker_no(uint32_t worker_no) = 0;
```

### Description

Selects the active worker slot for subsequent operations.

### Parameters

- `worker_no`: Worker-slot index.

### Returns

`true` on success; otherwise `false`.

### Remarks

Use a duplicated context when two threads need to run concurrently on different workers.

## cosyvoice_model_context::get_worker_no

### Syntax

```cpp
virtual uint32_t get_worker_no() = 0;
```

### Description

Gets the current active worker slot number.

### Returns

Current worker-slot index.

## cosyvoice_model_context::get_n_workers

### Syntax

```cpp
virtual uint32_t get_n_workers() = 0;
```

### Description

Gets the total number of worker slots available.

### Returns

Worker-slot count.

## cosyvoice_model_context::get_architecture

### Syntax

```cpp
virtual const char* get_architecture() = 0;
```

### Description

Gets the architecture identifier of the loaded model.

### Returns

Null-terminated UTF-8 architecture string, for example `cosyvoice3-2512`.

## cosyvoice_model_context::get_instruction_prefix

### Syntax

```cpp
virtual const char* get_instruction_prefix() = 0;
```

### Description

Gets instruction prefix required by model.

### Returns

Null-terminated UTF-8 string pointer.

## cosyvoice_model_context::is_backend_uma

### Syntax

```cpp
virtual bool is_backend_uma() = 0;
```

### Description

Queries whether the backend appears to use unified memory architecture (UMA).

### Returns

`true` if the backend memory is detected as UMA; otherwise `false`.

### Remarks

The result is determined at model load time. The runtime compares backend tensor-write bandwidth against host `memcpy` bandwidth; if the backend achieves at least 70% of host memcpy throughput, UMA is assumed. On Apple Silicon (`__aarch64__`), UMA is always reported.

> **Note**: UMA detection is a heuristic based on bandwidth probing. Results may be inaccurate depending on hardware, driver version, and system load at probe time. Treat the result as a rough hint rather than a definitive hardware capability.

## cosyvoice_model_context::get_sampler

### Syntax

```cpp
virtual void get_sampler(cosyvoice_sampler_t* sampler, void** sampler_ctx) = 0;
```

### Description

Gets current sampler and user context.

### Parameters

- `sampler`: Output callback pointer.
- `sampler_ctx`: Output user context pointer.

### Returns

No return value.

## cosyvoice_model_context::set_sampler

### Syntax

```cpp
virtual void set_sampler(cosyvoice_sampler_t sampler, void* sampler_ctx) = 0;
```

### Description

Sets sampler callback and context.

### Parameters

- `sampler`: Callback pointer.
- `sampler_ctx`: User context pointer.

### Returns

No return value.

## cosyvoice_model_context::get_builtin_sampler_rng_policy

### Syntax

```cpp
virtual cosyvoice_builtin_sampler_rng_policy_t get_builtin_sampler_rng_policy() = 0;
```

### Description

Gets RNG policy of built-in sampler.

### Returns

Current RNG policy.

## cosyvoice_model_context::set_builtin_sampler_rng_policy

### Syntax

```cpp
virtual bool set_builtin_sampler_rng_policy(cosyvoice_builtin_sampler_rng_policy_t policy) = 0;
```

### Description

Sets RNG policy of built-in sampler.

### Parameters

- `policy`: Policy value.

### Returns

`true` on success; otherwise `false`.

## cosyvoice_model_context::set_sampler_seed

### Syntax

```cpp
virtual bool set_sampler_seed(uint32_t seed) = 0;
```

### Description

Sets sampler seed.

### Parameters

- `seed`: Seed value.

### Returns

`true` on success; otherwise `false`.

## cosyvoice_model_context::get_sampler_seed

### Syntax

```cpp
virtual uint32_t get_sampler_seed() = 0;
```

### Description

Gets the active worker's sampler seed.

### Returns

Current seed value for the active worker.

## cosyvoice_model_context::llm_prefill

### Syntax

```cpp
virtual bool llm_prefill(ggml_type type, const void* data, uint32_t seq_len) = 0;
```

### Description

Prefills LLM with embedding sequence.

### Parameters

- `type`: Embedding element type.
- `data`: Embedding buffer.
- `seq_len`: Sequence length.

### Returns

`true` on success; otherwise `false`.

## cosyvoice_model_context::llm_decode

### Syntax

```cpp
virtual bool llm_decode(ggml_type type, const void* data) = 0;
```

### Description

Runs one LLM decode step.

### Parameters

- `type`: Embedding element type.
- `data`: Embedding vector.

### Returns

`true` on success; otherwise `false`.

### Remarks

This method only advances decode state. Call `llm_prepare_probs()` before sampling.

## cosyvoice_model_context::llm_prepare_probs

### Syntax

```cpp
virtual void llm_prepare_probs(bool allow_stop_tokens) = 0;
```

### Description

Prepares probabilities from the latest decode output for token sampling.

### Parameters

- `allow_stop_tokens`: If `false`, stop tokens are masked to zero probability.

### Returns

No return value.

## cosyvoice_model_context::get_word_token_embed_weight

### Syntax

```cpp
virtual const ggml_tensor* get_word_token_embed_weight() = 0;
```

### Description

Gets word-token embedding tensor.

### Returns

Read-only tensor pointer.

## cosyvoice_model_context::get_speech_token_embed_weight

### Syntax

```cpp
virtual const ggml_tensor* get_speech_token_embed_weight() = 0;
```

### Description

Gets speech-token embedding tensor.

### Returns

Read-only tensor pointer.

## cosyvoice_model_context::llm_get_kv_cache_len

### Syntax

```cpp
virtual uint32_t llm_get_kv_cache_len() = 0;
```

### Description

Gets current KV-cache length.

### Returns

Token count in KV cache.

## cosyvoice_model_context::llm_set_kv_cache_len

### Syntax

```cpp
virtual bool llm_set_kv_cache_len(uint32_t len) = 0;
```

### Description

Trims KV-cache length.

### Parameters

- `len`: Target length.

### Returns

`true` on success; otherwise `false`.

## cosyvoice_model_context::llm_sample_token

### Syntax

```cpp
virtual int llm_sample_token() = 0;
```

### Description

Samples next token.

### Returns

Token id.

## cosyvoice_model_context::llm_is_stop_token

### Syntax

```cpp
virtual bool llm_is_stop_token(int token_id) = 0;
```

### Description

Checks stop-token condition.

### Parameters

- `token_id`: Token id to check.

### Returns

`true` if stop token; otherwise `false`.

## cosyvoice_model_context::llm_accept_token

### Syntax

```cpp
virtual void llm_accept_token(int token_id) = 0;
```

### Description

Accepts token into sequence.

### Parameters

- `token_id`: Token id.

### Returns

No return value.

## cosyvoice_model_context::llm_clear_accepted_tokens

### Syntax

```cpp
virtual void llm_clear_accepted_tokens() = 0;
```

### Description

Clears accepted-token history.

### Returns

No return value.

## cosyvoice_model_context::llm_get_n_accepted_tokens

### Syntax

```cpp
virtual uint32_t llm_get_n_accepted_tokens() = 0;
```

### Description

Gets number of accepted tokens.

### Returns

Accepted-token count.

## cosyvoice_model_context::llm_get_accepted_tokens

### Syntax

```cpp
virtual const int* llm_get_accepted_tokens() = 0;
```

### Description

Gets accepted-token buffer pointer.

### Returns

Pointer to token id array.

## cosyvoice_model_context::llm_job

### Syntax

```cpp
virtual bool llm_job(const int* text, uint32_t text_len, cosyvoice_prompt_t prompt) = 0;
```

### Description

Runs low-level LLM generation.

### Parameters

- `text`: Input text tokens.
- `text_len`: Number of tokens.
- `prompt`: Prompt handle.

### Returns

`true` on success; otherwise `false`.

## cosyvoice_model_context::token2wav

### Syntax

```cpp
virtual bool token2wav(
    const int*                 token_ids,
    uint32_t                   n_tokens,
    float                      speed,
    cosyvoice_prompt_t         prompt,
    cosyvoice_generated_speech_ptr result
) = 0;
```

### Description

Converts speech tokens to waveform.

### Parameters

- `token_ids`: Speech token ids.
- `n_tokens`: Number of tokens.
- `speed`: Speed multiplier.
- `prompt`: Prompt handle.
- `result`: Output waveform container.

### Returns

`true` on success; otherwise `false`.

## cosyvoice_model_context::get_last_status

### Syntax

```cpp
virtual ggml_status get_last_status() = 0;
```

### Description

Gets status of latest backend operation.

### Returns

`ggml_status` value.

## cosyvoice_model_context::set_prompt

### Syntax

```cpp
virtual void set_prompt(
    cosyvoice_prompt_t         prompt,
    cosyvoice_inference_mode_t mode,
    const int*                 instruction,
    uint32_t                   instruction_length
) = 0;
```

### Description

Sets prompt mode and tokenized instruction.

### Parameters

- `prompt`: Prompt handle.
- `mode`: Inference mode.
- `instruction`: Instruction token ids.
- `instruction_length`: Number of instruction tokens.

### Returns

No return value.

## cosyvoice_model_context::get_memory_usage

### Syntax

```cpp
virtual void get_memory_usage(cosyvoice_memory_usage_t* usage) = 0;
```

### Description

Gets memory-usage snapshot.

### Parameters

- `usage`: Output usage structure.

### Returns

No return value.

## cosyvoice_model_context::empty_buffer_cache

### Syntax

```cpp
virtual void empty_buffer_cache() = 0;
```

### Description

Releases reusable cached buffers.

### Returns

No return value.

## cosyvoice_model_context::set_noise_callback

### Syntax

```cpp
virtual void set_noise_callback(cosyvoice_noise_callback_t callback, void* callback_ctx) = 0;
```

### Description

Registers noise callback.

### Parameters

- `callback`: Callback function.
- `callback_ctx`: Callback context.

### Returns

No return value.

## cosyvoice_model_context::get_noise_callback

### Syntax

```cpp
virtual void get_noise_callback(cosyvoice_noise_callback_t* callback, void** callback_ctx) = 0;
```

### Description

Gets current noise callback and context.

### Parameters

- `callback`: Output callback pointer.
- `callback_ctx`: Output context pointer.

### Returns

No return value.

## cosyvoice_model_context::get_hift_rand_ini_len

### Syntax

```cpp
virtual uint32_t get_hift_rand_ini_len() = 0;
```

### Description

Gets required HiFT initialization-noise length.

### Returns

Required sample count.

## cosyvoice_model_context::set_hift_rand_ini

### Syntax

```cpp
virtual void set_hift_rand_ini(const float* data) = 0;
```

### Description

Sets HiFT initialization-noise buffer.

### Parameters

- `data`: Noise-buffer pointer.

### Returns

No return value.

## cosyvoice_tokenization_result

### Syntax

```cpp
struct cosyvoice_tokenization_result
{
    virtual int* get_tokens() = 0;
    virtual uint32_t get_n_tokens() = 0;
};
```

### Description

Abstract container for tokenizer output.

## cosyvoice_tokenization_result::get_tokens

### Syntax

```cpp
virtual int* get_tokens() = 0;
```

### Description

Gets mutable token buffer.

### Returns

Pointer to token array.

## cosyvoice_tokenization_result::get_n_tokens

### Syntax

```cpp
virtual uint32_t get_n_tokens() = 0;
```

### Description

Gets token count.

### Returns

Number of tokens in buffer.

## cosyvoice_tokenizer_context

### Syntax

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

### Description

Abstract tokenizer interface shared by runtime and standalone tokenizers.

## cosyvoice_tokenizer_context::tokenize (length overload)

### Syntax

```cpp
virtual uint32_t tokenize(const char* text, uint32_t text_len, cosyvoice_tokenization_result_t result, bool parse_special = true) = 0;
```

### Description

Tokenizes UTF-8 text using explicit byte length.

### Parameters

- `text`: UTF-8 text buffer.
- `text_len`: Text length in bytes.
- `result`: Output tokenization container.
- `parse_special`: Whether to parse special tokens.

### Returns

Number of tokens written.

## cosyvoice_tokenizer_context::tokenize (null-terminated overload)

### Syntax

```cpp
inline uint32_t tokenize(const char* text, cosyvoice_tokenization_result_t result, bool parse_special = true)
```

### Description

Convenience overload for null-terminated UTF-8 input.

### Parameters

- `text`: Null-terminated UTF-8 string.
- `result`: Output tokenization container.
- `parse_special`: Whether to parse special tokens.

### Returns

Number of tokens written.

## cosyvoice_context

### Syntax

```cpp
struct cosyvoice_context : virtual cosyvoice_model_context,
                           virtual cosyvoice_tokenizer_context {};
```

### Description

Combined interface implemented by concrete runtime contexts.
