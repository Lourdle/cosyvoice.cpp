# cosyvoice-lowlevel.h API Reference

This page documents all public symbols declared in `include/cosyvoice-lowlevel.h`.

## cosyvoice_tokenization_result_t

### Syntax

```c
typedef struct cosyvoice_tokenization_result* cosyvoice_tokenization_result_t;
```

### Description

Opaque handle to a tokenization-result container.

## cosyvoice_tokenizer_context_t

### Syntax

```c
typedef struct cosyvoice_tokenizer_context*   cosyvoice_tokenizer_context_t;
```

### Description

Opaque handle to a tokenizer context.

## cosyvoice_inference_mode_t

### Syntax

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

### Description

Prompt-update mode used by low-level prompt APIs.

### Values

- `COSYVOICE_INFERENCE_MODE_NULL`: No update; useful when duplicating prompts.
- `COSYVOICE_INFERENCE_MODE_ZERO_SHOT`: Zero-shot generation.
- `COSYVOICE_INFERENCE_MODE_INSTRUCT`: Instruction-following generation.
- `COSYVOICE_INFERENCE_MODE_CROSS_LINGUAL`: Cross-lingual generation that ignores prompt instruction text.
- `COSYVOICE_INFERENCE_MODE_COUNT`: Sentinel value.

## cosyvoice_noise_callback_stage_t

### Syntax

```c
typedef enum cosyvoice_noise_callback_stage
{
    COSYVOICE_NOISE_CALLBACK_STAGE_BEFORE_FLOW,
    COSYVOICE_NOISE_CALLBACK_STAGE_AFTER_FLOW,
    COSYVOICE_NOISE_CALLBACK_STAGE_BEFORE_HIFT,
    COSYVOICE_NOISE_CALLBACK_STAGE_AFTER_HIFT
} cosyvoice_noise_callback_stage_t;
```

### Description

Indicates which Flow/HiFT stage triggered the noise callback.

### Values

- `COSYVOICE_NOISE_CALLBACK_STAGE_BEFORE_FLOW`: Called before Flow; callback must provide noise buffer.
- `COSYVOICE_NOISE_CALLBACK_STAGE_AFTER_FLOW`: Called after Flow; return value ignored.
- `COSYVOICE_NOISE_CALLBACK_STAGE_BEFORE_HIFT`: Called before HiFT; callback must provide noise buffer.
- `COSYVOICE_NOISE_CALLBACK_STAGE_AFTER_HIFT`: Called after HiFT; return value ignored.

## cosyvoice_noise_callback_t

### Syntax

```c
typedef float* (*cosyvoice_noise_callback_t)(
    cosyvoice_noise_callback_stage_t stage,
    uint32_t                         length,
    float*                           noise,
    void*                            ctx
);
```

### Description

Callback for observing or overriding random-noise buffers.

### Parameters

- `stage`: Callback stage.
- `length`: Required number of float samples.
- `noise`: Null for `BEFORE_*`; previous buffer for `AFTER_*`.
- `ctx`: User context.

### Returns

Buffer to use for `BEFORE_*` calls; ignored for `AFTER_*` calls.

## cosyvoice_log_callback_default

### Syntax

```c
COSYVOICE_API void cosyvoice_log_callback_default(enum ggml_log_level level, const char* text, void* user_data);
```

### Description

Default GGML log callback used by runtime.

### Parameters

- `level`: GGML log level.
- `text`: Log message.
- `user_data`: Caller-provided context.

### Returns

No return value.

## cosyvoice_load_from_file_ext

### Syntax

```c
COSYVOICE_API cosyvoice_context_t cosyvoice_load_from_file_ext(
    const char*                       filename,
    const cosyvoice_context_params_t* params,
    ggml_backend_t                    backend,
    uint32_t                          n_threads,
    uint32_t                          reserved
);
```

### Description

Loads model context with explicit backend and threading parameters.

### Parameters

- `filename`: Model path.
- `params`: Context parameters.
- `backend`: Backend handle.
- `n_threads`: Thread count.
- `reserved`: Reserved field.

### Returns

Context handle on success; `NULL` on failure.

### Remarks

Available only when `GGML_SHARED` is enabled.

## cosyvoice_get_last_status

### Syntax

```c
COSYVOICE_API enum ggml_status cosyvoice_get_last_status(cosyvoice_context_t ctx);
```

### Description

Gets status code from the most recent backend operation.

### Parameters

- `ctx`: Context handle.

### Returns

`enum ggml_status` value.

## cosyvoice_get_word_token_embed_weight

### Syntax

```c
COSYVOICE_API const ggml_tensor* cosyvoice_get_word_token_embed_weight(cosyvoice_context_t ctx);
```

### Description

Returns word-token embedding tensor.

### Parameters

- `ctx`: Context handle.

### Returns

Read-only tensor pointer.

## cosyvoice_get_speech_token_embed_weight

### Syntax

```c
COSYVOICE_API const ggml_tensor* cosyvoice_get_speech_token_embed_weight(cosyvoice_context_t ctx);
```

### Description

Returns speech-token embedding tensor.

### Parameters

- `ctx`: Context handle.

### Returns

Read-only tensor pointer.

## cosyvoice_llm_prefill

### Syntax

```c
COSYVOICE_API bool cosyvoice_llm_prefill(
    cosyvoice_context_t ctx,
    enum ggml_type      type,
    const void*         data,
    uint32_t            n_tokens
);
```

### Description

Prefills LLM with a sequence of token embeddings.

### Parameters

- `ctx`: Context handle.
- `type`: Input element type.
- `data`: Embedding buffer.
- `n_tokens`: Token count.

### Returns

`true` on success; otherwise `false`.

### Remarks

Does not compute next-token logits.

## cosyvoice_llm_decode

### Syntax

```c
COSYVOICE_API bool cosyvoice_llm_decode(
    cosyvoice_context_t ctx,
    enum ggml_type      type,
    const void*         data
);
```

### Description

Runs one decode step and updates internal logits.

### Parameters

- `ctx`: Context handle.
- `type`: Input element type.
- `data`: Embedding vector.

### Returns

`true` on success; otherwise `false`.

## cosyvoice_llm_get_kv_cache_len

### Syntax

```c
COSYVOICE_API uint32_t cosyvoice_llm_get_kv_cache_len(cosyvoice_context_t ctx);
```

### Description

Gets current KV-cache token length.

### Parameters

- `ctx`: Context handle.

### Returns

Current KV length.

## cosyvoice_llm_set_kv_cache_len

### Syntax

```c
COSYVOICE_API bool cosyvoice_llm_set_kv_cache_len(cosyvoice_context_t ctx, uint32_t len);
```

### Description

Trims current KV-cache length.

### Parameters

- `ctx`: Context handle.
- `len`: Target length.

### Returns

`true` on success; otherwise `false`.

### Remarks

`len` must be less than or equal to the current length.

## cosyvoice_llm_sample_token

### Syntax

```c
COSYVOICE_API int cosyvoice_llm_sample_token(cosyvoice_context_t ctx);
```

### Description

Samples next token from current logits.

### Parameters

- `ctx`: Context handle.

### Returns

Sampled token id.

## cosyvoice_llm_is_stop_token

### Syntax

```c
COSYVOICE_API bool cosyvoice_llm_is_stop_token(cosyvoice_context_t ctx, int token_id);
```

### Description

Checks whether token id is a stop token.

### Parameters

- `ctx`: Context handle.
- `token_id`: Token id.

### Returns

`true` if stop token; otherwise `false`.

## cosyvoice_llm_accept_token

### Syntax

```c
COSYVOICE_API void cosyvoice_llm_accept_token(cosyvoice_context_t ctx, int token_id);
```

### Description

Accepts token into generated sequence.

### Parameters

- `ctx`: Context handle.
- `token_id`: Token id.

### Returns

No return value.

## cosyvoice_llm_clear_accepted_tokens

### Syntax

```c
COSYVOICE_API void cosyvoice_llm_clear_accepted_tokens(cosyvoice_context_t ctx);
```

### Description

Clears accepted-token history.

### Parameters

- `ctx`: Context handle.

### Returns

No return value.

## cosyvoice_llm_get_n_accepted_tokens

### Syntax

```c
COSYVOICE_API uint32_t cosyvoice_llm_get_n_accepted_tokens(cosyvoice_context_t ctx);
```

### Description

Gets number of accepted tokens.

### Parameters

- `ctx`: Context handle.

### Returns

Accepted-token count.

## cosyvoice_llm_get_accepted_tokens

### Syntax

```c
COSYVOICE_API const int* cosyvoice_llm_get_accepted_tokens(cosyvoice_context_t ctx);
```

### Description

Gets pointer to accepted-token buffer.

### Parameters

- `ctx`: Context handle.

### Returns

Read-only pointer to token id array.

## cosyvoice_llm_job

### Syntax

```c
COSYVOICE_API bool cosyvoice_llm_job(
    cosyvoice_context_t ctx,
    const int*          text,
    uint32_t            text_len,
    cosyvoice_prompt_t  prompt
);
```

### Description

Runs low-level LLM generation for tokenized text and prompt.

### Parameters

- `ctx`: Context handle.
- `text`: Text token ids.
- `text_len`: Number of tokens in `text`.
- `prompt`: Prompt handle.

### Returns

`true` on success; otherwise `false`.

## cosyvoice_token2wav

### Syntax

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

### Description

Converts speech tokens to waveform.

### Parameters

- `ctx`: Context handle.
- `token_ids`: Speech token ids.
- `n_tokens`: Number of speech tokens.
- `speed`: Speech speed multiplier.
- `prompt`: Prompt handle.
- `generated_speech`: Output waveform container.

### Returns

`true` on success; otherwise `false`.

## cosyvoice_tts

### Syntax

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

### Description

Runs full low-level text-to-speech pipeline.

### Parameters

- `ctx`: Context handle.
- `text`: Text token ids.
- `text_len`: Number of text tokens.
- `speed`: Speech speed multiplier.
- `prompt`: Prompt handle.
- `result`: Output waveform container.

### Returns

`true` on success; otherwise `false`.

### Remarks

Runs LLM generation and waveform conversion as a single convenience pipeline.

## cosyvoice_get_tokenizer

### Syntax

```c
COSYVOICE_API cosyvoice_tokenizer_context_t cosyvoice_get_tokenizer(cosyvoice_context_t ctx);
```

### Description

Borrows tokenizer owned by model context.

### Parameters

- `ctx`: Context handle.

### Returns

Tokenizer handle owned by context.

### Remarks

This handle is borrowed from the model context.

## cosyvoice_tokenizer_load_from_file

### Syntax

```c
COSYVOICE_API cosyvoice_tokenizer_context_t cosyvoice_tokenizer_load_from_file(const char* filename);
```

### Description

Loads standalone tokenizer from file.

### Parameters

- `filename`: Model/tokenizer file path.

### Returns

Tokenizer handle on success; `NULL` on failure.

## cosyvoice_tokenizer_free

### Syntax

```c
COSYVOICE_API void cosyvoice_tokenizer_free(cosyvoice_tokenizer_context_t ctx);
```

### Description

Frees standalone tokenizer context.

### Parameters

- `ctx`: Tokenizer handle.

### Returns

No return value.

### Remarks

Use this only for tokenizers created by `cosyvoice_tokenizer_load_from_file`.

## cosyvoice_tokenization_result_create

### Syntax

```c
COSYVOICE_API cosyvoice_tokenization_result_t cosyvoice_tokenization_result_create();
```

### Description

Creates empty tokenization-result container.

### Returns

Result handle on success; `NULL` on failure.

## cosyvoice_tokenization_result_free

### Syntax

```c
COSYVOICE_API void cosyvoice_tokenization_result_free(cosyvoice_tokenization_result_t result);
```

### Description

Frees tokenization-result container.

### Parameters

- `result`: Result handle.

### Returns

No return value.

## cosyvoice_tokenization_result_get_tokens

### Syntax

```c
COSYVOICE_API int* cosyvoice_tokenization_result_get_tokens(cosyvoice_tokenization_result_t result);
```

### Description

Gets mutable token buffer.

### Parameters

- `result`: Result handle.

### Returns

Pointer to token id array.

## cosyvoice_tokenization_result_get_n_tokens

### Syntax

```c
COSYVOICE_API uint32_t cosyvoice_tokenization_result_get_n_tokens(cosyvoice_tokenization_result_t result);
```

### Description

Gets number of tokens in a tokenization result.

### Parameters

- `result`: Result handle.

### Returns

Token count.

## cosyvoice_tokenize

### Syntax

```c
COSYVOICE_API uint32_t cosyvoice_tokenize(
    cosyvoice_tokenizer_context_t   ctx,
    const char*                     text,
    cosyvoice_tokenization_result_t result,
    bool                            parse_special
);
```

### Description

Tokenizes null-terminated UTF-8 string.

### Parameters

- `ctx`: Tokenizer handle.
- `text`: Input UTF-8 string.
- `result`: Output result container.
- `parse_special`: Whether to parse special tokens.

### Returns

Number of tokens written.

## cosyvoice_tokenize_ext

### Syntax

```c
COSYVOICE_API uint32_t cosyvoice_tokenize_ext(
    cosyvoice_tokenizer_context_t   ctx,
    const char*                     text,
    uint32_t                        text_len,
    cosyvoice_tokenization_result_t result,
    bool                            parse_special
);
```

### Description

Tokenizes UTF-8 text with explicit byte length.

### Parameters

- `ctx`: Tokenizer handle.
- `text`: Input UTF-8 data.
- `text_len`: Byte length of `text`.
- `result`: Output result container.
- `parse_special`: Whether to parse special tokens.

### Returns

Number of tokens written.

## cosyvoice_set_noise_callback

### Syntax

```c
COSYVOICE_API void cosyvoice_set_noise_callback(cosyvoice_context_t ctx, cosyvoice_noise_callback_t callback, void* callback_ctx);
```

### Description

Registers callback for noise-buffer observation/override.

### Parameters

- `ctx`: Context handle.
- `callback`: Callback function.
- `callback_ctx`: Callback context pointer.

### Returns

No return value.

## cosyvoice_get_noise_callback

### Syntax

```c
COSYVOICE_API void cosyvoice_get_noise_callback(cosyvoice_context_t ctx, cosyvoice_noise_callback_t* callback, void** callback_ctx);
```

### Description

Gets currently registered noise callback and context.

### Parameters

- `ctx`: Context handle.
- `callback`: Output callback pointer.
- `callback_ctx`: Output context pointer.

### Returns

No return value.

## cosyvoice_get_hift_rand_ini_len

### Syntax

```c
COSYVOICE_API uint32_t cosyvoice_get_hift_rand_ini_len(cosyvoice_context_t ctx);
```

### Description

Gets required length of HiFT initialization noise buffer.

### Parameters

- `ctx`: Context handle.

### Returns

Required sample count.

## cosyvoice_set_hift_rand_ini

### Syntax

```c
COSYVOICE_API void cosyvoice_set_hift_rand_ini(cosyvoice_context_t ctx, const float* data);
```

### Description

Overrides HiFT initialization noise buffer.

### Parameters

- `ctx`: Context handle.
- `data`: Noise buffer pointer.

### Returns

No return value.

## cosyvoice_prompt_speech_get_crc32

### Syntax

```c
COSYVOICE_API uint32_t cosyvoice_prompt_speech_get_crc32(cosyvoice_prompt_speech_t prompt_speech);
```

### Description

Computes CRC32 for prompt-speech object.

### Parameters

- `prompt_speech`: Prompt-speech handle.

### Returns

CRC32 value.

## cosyvoice_prompt_get_crc32

### Syntax

```c
COSYVOICE_API uint32_t cosyvoice_prompt_get_crc32(cosyvoice_prompt_t prompt);
```

### Description

Computes CRC32 for prompt object.

### Parameters

- `prompt`: Prompt handle.

### Returns

CRC32 value.

## cosyvoice_get_instruction_prefix

### Syntax

```c
COSYVOICE_API const char* cosyvoice_get_instruction_prefix(cosyvoice_context_t ctx);
```

### Description

Gets instruction prefix expected by current model.

### Parameters

- `ctx`: Context handle.

### Returns

Null-terminated UTF-8 prefix string.

## cosyvoice_prompt_set

### Syntax

```c
COSYVOICE_API cosyvoice_prompt_t cosyvoice_prompt_set(
    cosyvoice_context_t        ctx,
    cosyvoice_prompt_t         prompt,
    cosyvoice_inference_mode_t mode,
    const char*                instruction,
#ifdef __cplusplus
    uint32_t                   instruction_length = 0xFFFFFFFFU,
    const char*                locale = nullptr,
    bool                       inplace = true
#else
    uint32_t                   instruction_length,
    const char*                locale,
    bool                       inplace
#endif
);
```

### Description

Sets prompt mode and instruction text using raw text input.

### Parameters

- `ctx`: Context handle.
- `prompt`: Prompt handle to update.
- `mode`: Inference mode.
- `instruction`: Instruction text.
- `instruction_length`: Instruction length in bytes.
- `locale`: Optional locale hint for normalization.
- `inplace`: Whether to modify input prompt in place.

### Returns

Updated prompt handle.

### Remarks

Instruction prefix is not prepended automatically.

## cosyvoice_prompt_set_ext

### Syntax

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

### Description

Sets prompt mode and instruction content from tokenized input.

### Parameters

- `ctx`: Context handle.
- `prompt`: Prompt handle to update.
- `mode`: Inference mode.
- `instruction`: Tokenized instruction ids.
- `instruction_length`: Number of instruction tokens.
- `inplace`: Whether to update existing prompt.

### Returns

Updated prompt handle.

### Remarks

This API does not prepend instruction prefix and does not append `<|endofprompt|>` automatically.
