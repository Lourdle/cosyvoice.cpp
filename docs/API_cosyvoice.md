# cosyvoice.h API Reference

This page documents all public symbols declared in `include/cosyvoice.h`.
The runtime supports concurrent inference through multiple worker slots; context duplication lets separate threads bind to different workers while sharing loaded model resources.

## COSYVOICE_API

### Syntax

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

### Description

Controls symbol import/export visibility for the C API. Define `COSYVOICE_STATIC` when linking statically to suppress import attributes.

### Remarks

On Windows, the macro expands to `__declspec(dllimport)` by default. On non-Windows targets, default symbol visibility is used.

## cosyvoice_llm_kv_cache_type_t

### Syntax

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

### Description

Specifies supported KV-cache storage formats for the LLM module.

### Values

- `COSYVOICE_LLM_KV_CACHE_TYPE_F32`: 32-bit floating-point cache.
- `COSYVOICE_LLM_KV_CACHE_TYPE_F16`: 16-bit floating-point cache.
- `COSYVOICE_LLM_KV_CACHE_TYPE_Q8_0`: GGML `Q8_0` quantized cache.
- `COSYVOICE_LLM_KV_CACHE_TYPE_Q5_1`: GGML `Q5_1` quantized cache.
- `COSYVOICE_LLM_KV_CACHE_TYPE_Q5_0`: GGML `Q5_0` quantized cache.
- `COSYVOICE_LLM_KV_CACHE_TYPE_Q4_1`: GGML `Q4_1` quantized cache.
- `COSYVOICE_LLM_KV_CACHE_TYPE_Q4_0`: GGML `Q4_0` quantized cache.
- `COSYVOICE_LLM_KV_CACHE_TYPE_COUNT`: Sentinel value, not a runtime mode.

## cosyvoice_inference_buffer_policy_t

### Syntax

```c
typedef enum cosyvoice_inference_buffer_policy
{
    COSYVOICE_INFERENCE_BUFFER_POLICY_SHARED,
    COSYVOICE_INFERENCE_BUFFER_POLICY_BALANCED,
    COSYVOICE_INFERENCE_BUFFER_POLICY_DEDICATED,
    COSYVOICE_INFERENCE_BUFFER_POLICY_COUNT
} cosyvoice_inference_buffer_policy_t;
```

### Description

Defines memory allocation and reuse behavior for inference buffers.

### Values

- `COSYVOICE_INFERENCE_BUFFER_POLICY_SHARED`: Maximizes sharing between KV-cache and token2wav intermediates to reduce memory usage.
- `COSYVOICE_INFERENCE_BUFFER_POLICY_BALANCED`: Shares buffers while preserving reusable sequence segments for faster restoration.
- `COSYVOICE_INFERENCE_BUFFER_POLICY_DEDICATED`: Allocates separate buffers to prioritize throughput.
- `COSYVOICE_INFERENCE_BUFFER_POLICY_COUNT`: Sentinel value.

## cosyvoice_builtin_sampler_rng_policy_t

### Syntax

```c
typedef enum cosyvoice_builtin_sampler_rng_policy
{
    COSYVOICE_BUILTIN_SAMPLER_RNG_POLICY_RESET_PER_SESSION,
    COSYVOICE_BUILTIN_SAMPLER_RNG_POLICY_CONTINUE_ACROSS_SESSIONS,
    COSYVOICE_BUILTIN_SAMPLER_RNG_POLICY_COUNT
} cosyvoice_builtin_sampler_rng_policy_t;
```

### Description

Controls how the built-in sampler random state evolves across LLM sessions.

### Values

- `COSYVOICE_BUILTIN_SAMPLER_RNG_POLICY_RESET_PER_SESSION`: Resets to configured seed per session for reproducible runs.
- `COSYVOICE_BUILTIN_SAMPLER_RNG_POLICY_CONTINUE_ACROSS_SESSIONS`: Keeps advancing RNG state between sessions.
- `COSYVOICE_BUILTIN_SAMPLER_RNG_POLICY_COUNT`: Sentinel value.

## cosyvoice_sampling_params_t

### Syntax

```c
typedef struct cosyvoice_sampling_params
{
    int   top_k;
    float top_p;
    int   win_size;
    float tau_r;
} cosyvoice_sampling_params_t;
```

### Description

Defines nucleus-sampling parameters used by the built-in sampler.

### Fields

- `top_k`: Limits candidate set size before sampling.
- `top_p`: Cumulative probability threshold for nucleus truncation.
- `win_size`: Sliding-window length used by repetition-aware logic.
- `tau_r`: Repetition control coefficient.

## cosyvoice_generation_config_t

### Syntax

```c
typedef struct cosyvoice_generation_config
{
    float                       temperature;
    cosyvoice_sampling_params_t sampling;
    float                       min_token_text_ratio;
    float                       max_token_text_ratio;
} cosyvoice_generation_config_t;
```

### Description

Holds runtime generation controls for token sampling and output-length limits.

### Fields

- `temperature`: Softmax temperature applied to logits.
- `sampling`: Built-in sampler configuration.
- `min_token_text_ratio`: Lower bound for generated acoustic-token count relative to input text length.
- `max_token_text_ratio`: Upper bound for generated acoustic-token count relative to input text length.

## cosyvoice_llm_token_prob_t

### Syntax

```c
typedef struct cosyvoice_llm_token_prob
{
    int   token_id;
    float prob;
} cosyvoice_llm_token_prob_t;
```

### Description

Represents one candidate token and its probability, typically passed to custom samplers.

### Fields

- `token_id`: Vocabulary token identifier.
- `prob`: Probability assigned after filtering.

## cosyvoice_generated_speech_ptr

### Syntax

```c
typedef struct cosyvoice_generated_speech
{
    float*   data;
    uint32_t length;
} *cosyvoice_generated_speech_ptr;
```

### Description

Pointer to generated waveform metadata and PCM sample buffer.

### Fields

- `data`: Pointer to 32-bit float PCM samples.
- `length`: Number of samples in `data`.

## cosyvoice_context_t

### Syntax

```c
typedef struct cosyvoice_context* cosyvoice_context_t;
```

### Description

Opaque handle to a loaded model context.

## cosyvoice_prompt_speech_t

### Syntax

```c
typedef struct cosyvoice_prompt_speech* cosyvoice_prompt_speech_t;
```

### Description

Opaque handle to prompt-speech features loaded or extracted for inference.

## cosyvoice_prompt_t

### Syntax

```c
typedef struct cosyvoice_prompt* cosyvoice_prompt_t;
```

### Description

Opaque handle to a prepared prompt object bound to a model context.

## cosyvoice_tts_context_t

### Syntax

```c
typedef struct cosyvoice_tts_context* cosyvoice_tts_context_t;
```

### Description

Opaque handle to a reusable text-to-speech session context.

## cosyvoice_sampler_t

### Syntax

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

### Description

Callback used to override token selection from LLM logits.

### Parameters

- `nucleus_probs`: Candidate tokens after nucleus filtering.
- `k`: Number of elements in `nucleus_probs`.
- `probs`: Full probability distribution buffer.
- `size`: Number of entries in `probs`.
- `sampling_params`: Active sampling configuration.
- `accepted_tokens`: Tokens already accepted in current sequence.
- `n_accepted_tokens`: Number of accepted tokens.
- `sampler_ctx`: User context pointer registered with sampler.

### Returns

Selected token id.

## cosyvoice_context_params_t

### Syntax

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

### Description

Groups context creation options that affect backend behavior, memory planning, and sampler setup.

### Fields

- `llm_use_flash_attn`: Enables flash attention for the LLM when backend supports it.
- `flow_use_flash_attn`: Enables flash attention for Flow module when available.
- `llm_kv_cache_type`: Requested KV-cache storage type.
- `llm_allow_kv_cache_fallback`: Allows fallback to flash-attention-compatible KV type when unsupported.
- `inference_buffer_policy`: Strategy for inference-buffer allocation and reuse.
- `n_batch`: Kernel batch size.
- `n_max_seq`: Maximum sequence length.
- `seed`: RNG seed for built-in sampler and noise generation.
- `builtin_sampler_rng_policy`: Built-in sampler RNG evolution policy; ignored when `sampler` is non-null.
- `sampler`: Optional custom sampler; set to `NULL` to use built-in sampler.
- `sampler_ctx`: User pointer passed to `sampler`.

## cosyvoice_init_backend

### Syntax

```c
COSYVOICE_API void cosyvoice_init_backend();
```

### Description

Initializes the default backend runtime.

### Remarks

Call before creating contexts when the selected backend requires explicit initialization.

## cosyvoice_init_backend_from_path

### Syntax

```c
COSYVOICE_API void cosyvoice_init_backend_from_path(const char* dir_path);
```

### Description

Initializes backend runtime using resources located in a custom directory.

### Parameters

- `dir_path`: Directory path used during backend initialization.

### Remarks

Loads backend resources from the specified directory and initializes the runtime.

## cosyvoice_init_default_context_params

### Syntax

```c
COSYVOICE_API void cosyvoice_init_default_context_params(cosyvoice_context_params_t* params);
```

### Description

Fills a context-parameter struct with library defaults.

### Parameters

- `params`: Output parameter block to initialize.

## cosyvoice_load_from_file

### Syntax

```c
COSYVOICE_API cosyvoice_context_t cosyvoice_load_from_file(const char* filename);
```

### Description

Loads a model context from a GGUF file using default context parameters.

### Parameters

- `filename`: Path to the model file.

### Returns

Loaded context handle on success; `NULL` on failure.

### Remarks

Uses default backend and thread settings.

Equivalent to calling `cosyvoice_load_from_file_ext(filename, &params, NULL, 0, 0)` with default-initialized `params`.

## cosyvoice_load_from_file_with_params

### Syntax

```c
COSYVOICE_API cosyvoice_context_t cosyvoice_load_from_file_with_params(
    const char*                       filename,
    const cosyvoice_context_params_t* params
);
```

### Description

Loads a model context from file with explicit context parameters.

### Parameters

- `filename`: Path to the model file.
- `params`: Context-parameter block to apply.

### Returns

Loaded context handle on success; `NULL` on failure.

### Remarks

Equivalent to calling `cosyvoice_load_from_file_ext(filename, params, NULL, 0, 0)`.

## cosyvoice_context_params_v2_t

### Syntax

```c
typedef struct cosyvoice_context_params_v2
{
    cosyvoice_context_params_t base_params;
    uint32_t n_workers;
} cosyvoice_context_params_v2_t;
```

### Description

Extends `cosyvoice_context_params_t` with a worker count for concurrent inference.

### Fields

- `base_params`: Base context parameters.
- `n_workers`: Number of worker slots to create.

## cosyvoice_load_from_file_with_params_v2

### Syntax

```c
COSYVOICE_API cosyvoice_context_t cosyvoice_load_from_file_with_params_v2(
    const char*                          filename,
    const cosyvoice_context_params_v2_t* params
);
```

### Description

Loads a model context with extended parameters, including worker-count configuration.

### Parameters

- `filename`: Path to the model file.
- `params`: Extended context-parameter block.

### Returns

Loaded context handle on success; `NULL` on failure.

## cosyvoice_duplicate_context

### Syntax

```c
COSYVOICE_API cosyvoice_context_t cosyvoice_duplicate_context(cosyvoice_context_t ctx);
```

### Description

Creates a new context that shares the loaded model resources with the original context.

### Parameters

- `ctx`: Source context.

### Returns

Duplicated context handle on success; `NULL` on failure.

### Remarks

The new context starts with the same active worker binding as the source context, then can be rebound independently with `cosyvoice_set_worker_no()`.

## cosyvoice_get_n_workers

### Syntax

```c
COSYVOICE_API uint32_t cosyvoice_get_n_workers(cosyvoice_context_t ctx);
```

### Description

Gets the total number of worker slots available in the context.

### Parameters

- `ctx`: Context handle.

### Returns

Worker-slot count.

## cosyvoice_get_worker_no

### Syntax

```c
COSYVOICE_API uint32_t cosyvoice_get_worker_no(cosyvoice_context_t ctx);
```

### Description

Gets the active worker slot number.

### Parameters

- `ctx`: Context handle.

### Returns

Current worker slot number.

## cosyvoice_set_worker_no

### Syntax

```c
COSYVOICE_API bool cosyvoice_set_worker_no(cosyvoice_context_t ctx, uint32_t worker_no);
```

### Description

Sets the active worker slot used by subsequent inference calls.

### Parameters

- `ctx`: Context handle.
- `worker_no`: Worker-slot index.

### Returns

`true` on success; otherwise `false`.

### Remarks

Use this on duplicated contexts when two threads need to run inference concurrently while sharing the same loaded model.

## cosyvoice_free

### Syntax

```c
COSYVOICE_API void cosyvoice_free(cosyvoice_context_t ctx);
```

### Description

Releases a model context and all resources owned by it.

### Parameters

- `ctx`: Context handle to destroy.

## cosyvoice_get_context_params

### Syntax

```c
COSYVOICE_API void cosyvoice_get_context_params(
    cosyvoice_context_t         ctx,
    cosyvoice_context_params_t* params
);
```

### Description

Retrieves effective context parameters currently active in a loaded context.

### Parameters

- `ctx`: Context handle.
- `params`: Output structure receiving parameters.

## cosyvoice_get_default_generation_config

### Syntax

```c
COSYVOICE_API void cosyvoice_get_default_generation_config(
    cosyvoice_context_t            ctx,
    cosyvoice_generation_config_t* config
);
```

### Description

Gets the generation configuration loaded from the model file before any worker-specific overrides are applied.

### Parameters

- `ctx`: Context handle.
- `config`: Output structure receiving the default configuration.

## cosyvoice_get_architecture

### Syntax

```c
COSYVOICE_API const char* cosyvoice_get_architecture(cosyvoice_context_t ctx);
```

### Description

Returns the architecture identifier of the loaded model.

### Parameters

- `ctx`: Context handle.

### Returns

Null-terminated UTF-8 architecture string, such as `cosyvoice3-2512`.

## cosyvoice_get_generation_config

### Syntax

```c
COSYVOICE_API void cosyvoice_get_generation_config(
    cosyvoice_context_t            ctx,
    cosyvoice_generation_config_t* config
);
```

### Description

Gets current generation configuration.

### Parameters

- `ctx`: Context handle.
- `config`: Output structure receiving current configuration.

## cosyvoice_get_sample_rate

### Syntax

```c
COSYVOICE_API uint32_t cosyvoice_get_sample_rate(cosyvoice_context_t ctx);
```

### Description

Returns output sample rate of the loaded model.

### Parameters

- `ctx`: Context handle.

### Returns

Sample rate in Hz.

## cosyvoice_set_generation_config

### Syntax

```c
COSYVOICE_API bool cosyvoice_set_generation_config(
    cosyvoice_context_t                  ctx,
    const cosyvoice_generation_config_t* config
);
```

### Description

Validates and applies generation configuration, overriding model defaults.

### Parameters

- `ctx`: Context handle.
- `config`: Configuration to apply.

### Returns

`true` when configuration is valid and accepted; otherwise `false`.

### Remarks

The sample-rate field is not part of this configuration and is not changed by this API.

## cosyvoice_set_sampler

### Syntax

```c
COSYVOICE_API void cosyvoice_set_sampler(cosyvoice_context_t ctx, cosyvoice_sampler_t sampler, void* sampler_ctx);
```

### Description

Registers a custom sampler callback for token selection.

### Parameters

- `ctx`: Context handle.
- `sampler`: Callback function; set `NULL` to restore built-in sampler.
- `sampler_ctx`: User context pointer passed to callback.

## cosyvoice_sampler_ext_t

### Syntax

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

### Description

Extended sampler callback that receives the worker slot number.

### Parameters

- `worker_no`: Active worker slot index.

## cosyvoice_get_sampler

### Syntax

```c
COSYVOICE_API void cosyvoice_get_sampler(cosyvoice_context_t ctx, cosyvoice_sampler_t* sampler, void** sampler_ctx);
```

### Description

Returns currently configured sampler callback and context pointer.

### Parameters

- `ctx`: Context handle.
- `sampler`: Output callback pointer.
- `sampler_ctx`: Output user context pointer.

## cosyvoice_get_builtin_sampler_rng_policy

### Syntax

```c
COSYVOICE_API cosyvoice_builtin_sampler_rng_policy_t cosyvoice_get_builtin_sampler_rng_policy(cosyvoice_context_t ctx);
```

### Description

Gets RNG policy used by the built-in sampler.

### Parameters

- `ctx`: Context handle.

### Returns

Current built-in sampler RNG policy.

## cosyvoice_set_builtin_sampler_rng_policy

### Syntax

```c
COSYVOICE_API bool cosyvoice_set_builtin_sampler_rng_policy(cosyvoice_context_t ctx, cosyvoice_builtin_sampler_rng_policy_t policy);
```

### Description

Sets built-in sampler RNG policy.

### Parameters

- `ctx`: Context handle.
- `policy`: Policy to apply.

### Returns

`true` when `policy` is valid and built-in sampler is active; otherwise `false`.

## cosyvoice_set_sampler_seed

### Syntax

```c
COSYVOICE_API bool cosyvoice_set_sampler_seed(cosyvoice_context_t ctx, uint32_t seed);
```

### Description

Sets seed used by sampler RNG.

### Parameters

- `ctx`: Context handle.
- `seed`: Seed value.

### Returns

`true` if built-in sampler is currently active; otherwise `false`.

### Remarks

The seed applies to the active worker only. The seed value is stored even when a custom sampler is active and takes effect when built-in sampler is re-enabled on that worker.

## cosyvoice_get_sampler_seed

### Syntax

```c
COSYVOICE_API uint32_t cosyvoice_get_sampler_seed(cosyvoice_context_t ctx);
```

### Description

Gets the active worker's built-in sampler seed.

### Parameters

- `ctx`: Context handle.

### Returns

Current sampler seed for the active worker.

## cosyvoice_generate_random_seed

### Syntax

```c
COSYVOICE_API uint32_t cosyvoice_generate_random_seed();
```

### Description

Generates a random 32-bit seed.

### Returns

A deterministic-looking random 32-bit unsigned integer suitable for use with `cosyvoice_set_sampler_seed`.

## cosyvoice_prompt_speech_load_from_file

### Syntax

```c
COSYVOICE_API cosyvoice_prompt_speech_t cosyvoice_prompt_speech_load_from_file(const char* filename);
```

### Description

Loads prompt-speech features from disk.

### Parameters

- `filename`: Prompt-speech file path.

### Returns

Prompt-speech handle on success; `NULL` on failure.

## cosyvoice_prompt_speech_save_to_file

### Syntax

```c
COSYVOICE_API bool cosyvoice_prompt_speech_save_to_file(cosyvoice_prompt_speech_t prompt_speech, const char* filename);
```

### Description

Saves a prompt-speech object to disk.

### Parameters

- `prompt_speech`: Prompt-speech handle to save.
- `filename`: Output file path.

### Returns

`true` on success; otherwise `false`.

## cosyvoice_prompt_init_from_prompt_speech

### Syntax

```c
COSYVOICE_API cosyvoice_prompt_t cosyvoice_prompt_init_from_prompt_speech(cosyvoice_context_t ctx, cosyvoice_prompt_speech_t prompt_speech);
```

### Description

Builds a prompt object for a model context from prompt-speech features.

### Parameters

- `ctx`: Context handle.
- `prompt_speech`: Source prompt-speech handle.

### Returns

Prompt handle on success; `NULL` on failure.

## cosyvoice_prompt_speech_free

### Syntax

```c
COSYVOICE_API void cosyvoice_prompt_speech_free(cosyvoice_prompt_speech_t prompt_speech);
```

### Description

Releases a prompt-speech handle.

### Parameters

- `prompt_speech`: Handle to free.

## cosyvoice_prompt_free

### Syntax

```c
COSYVOICE_API void cosyvoice_prompt_free(cosyvoice_prompt_t prompt);
```

### Description

Releases a prompt handle.

### Parameters

- `prompt`: Handle to free.

## cosyvoice_tts_context_new

### Syntax

```c
COSYVOICE_API cosyvoice_tts_context_t cosyvoice_tts_context_new(cosyvoice_context_t ctx, cosyvoice_prompt_t prompt);
```

### Description

Creates a reusable TTS session bound to a model context and initial prompt.

### Parameters

- `ctx`: Context handle.
- `prompt`: Initial prompt handle.

### Returns

TTS context handle on success; `NULL` on failure.

### Remarks

The created session keeps reusable runtime state tied to the specified model context and prompt, so repeated synthesis calls can avoid rebuilding the full session each time.

## cosyvoice_tts_context_free

### Syntax

```c
COSYVOICE_API void cosyvoice_tts_context_free(cosyvoice_tts_context_t ctx);
```

### Description

Destroys a TTS session context.

### Parameters

- `ctx`: TTS context handle.

## cosyvoice_tts_context_set_prompt

### Syntax

```c
COSYVOICE_API void cosyvoice_tts_context_set_prompt(cosyvoice_tts_context_t ctx, cosyvoice_prompt_t prompt);
```

### Description

Updates the prompt used by an existing TTS session.

### Parameters

- `ctx`: TTS context handle.
- `prompt`: New prompt handle.

### Remarks

Calling this API resets cached instruction text associated with the session.

## cosyvoice_tts_context_set_text_normalization_enabled

### Syntax

```c
COSYVOICE_API bool cosyvoice_tts_context_set_text_normalization_enabled(cosyvoice_tts_context_t ctx, bool enabled);
```

### Description

Enables or disables frontend text normalization for a TTS session.

### Parameters

- `ctx`: TTS context handle.
- `enabled`: `true` to enable text normalization; `false` to bypass normalization and tokenize raw input text directly.

### Returns

`true` on success, `false` if normalization is unavailable (for example when compiled without ICU).

### Remarks

Text normalization is enabled by default for newly created TTS contexts.

## cosyvoice_tts_context_get_text_normalization_enabled

### Syntax

```c
COSYVOICE_API bool cosyvoice_tts_context_get_text_normalization_enabled(cosyvoice_tts_context_t ctx);
```

### Description

Queries whether frontend text normalization is enabled for a TTS session.

### Parameters

- `ctx`: TTS context handle.

### Returns

`true` when text normalization is enabled; otherwise `false`.

## cosyvoice_tts_context_set_split_text_enabled

### Syntax

```c
COSYVOICE_API bool cosyvoice_tts_context_set_split_text_enabled(cosyvoice_tts_context_t ctx, bool enabled);
```

### Description

Enables or disables fragment splitting for a TTS session.

### Parameters

- `ctx`: TTS context handle.
- `enabled`: `true` to split text into fragments before synthesis; `false` to synthesize the full text in one pass.

### Returns

`true` on success.

### Remarks

Text splitting is enabled by default for newly created TTS contexts.

## cosyvoice_tts_context_get_split_text_enabled

### Syntax

```c
COSYVOICE_API bool cosyvoice_tts_context_get_split_text_enabled(cosyvoice_tts_context_t ctx);
```

### Description

Queries whether fragment splitting is enabled for a TTS session.

### Parameters

- `ctx`: TTS context handle.

### Returns

`true` when text splitting is enabled; otherwise `false`.

## cosyvoice_tts_context_set_fast_split_text_enabled

### Syntax

```c
COSYVOICE_API bool cosyvoice_tts_context_set_fast_split_text_enabled(cosyvoice_tts_context_t ctx, bool enabled);
```

### Description

Enables or disables fast token-based splitting for a TTS session.

### Parameters

- `ctx`: TTS context handle.
- `enabled`: `true` to enable fast split; `false` to use the slower text reassembly path.

### Returns

`true` on success.

### Remarks

Fast split is enabled by default for newly created TTS contexts.

## cosyvoice_tts_context_get_fast_split_text_enabled

### Syntax

```c
COSYVOICE_API bool cosyvoice_tts_context_get_fast_split_text_enabled(cosyvoice_tts_context_t ctx);
```

### Description

Queries whether fast token-based splitting is enabled for a TTS session.

### Parameters

- `ctx`: TTS context handle.

### Returns

`true` when fast split is enabled; otherwise `false`.

## cosyvoice_tts_context_set_fade_in_enabled

### Syntax

```c
COSYVOICE_API bool cosyvoice_tts_context_set_fade_in_enabled(cosyvoice_tts_context_t ctx, bool enabled);
```

### Description

Enables or disables the default output fade-in for a TTS session.

### Parameters

- `ctx`: TTS context handle.
- `enabled`: `true` to apply a 20 ms fade-in to generated output; `false` to return raw PCM without fade-in.

### Returns

`true` on success.

### Remarks

Fade-in is enabled by default for newly created TTS contexts.

## cosyvoice_tts_context_get_fade_in_enabled

### Syntax

```c
COSYVOICE_API bool cosyvoice_tts_context_get_fade_in_enabled(cosyvoice_tts_context_t ctx);
```

### Description

Queries whether output fade-in is enabled for a TTS session.

### Parameters

- `ctx`: TTS context handle.

### Returns

`true` when fade-in is enabled; otherwise `false`.

## cosyvoice_tts_context_get_flags

### Syntax

```c
COSYVOICE_API uint32_t cosyvoice_tts_context_get_flags(cosyvoice_tts_context_t ctx);
```

### Description

Gets the current TTS context flag bitmask.

### Parameters

- `ctx`: TTS context handle.

### Returns

A bitmask composed of `COSYVOICE_TTS_FLAG_*` values.

## cosyvoice_tts_context_set_flags

### Syntax

```c
COSYVOICE_API uint32_t cosyvoice_tts_context_set_flags(cosyvoice_tts_context_t ctx, uint32_t flags);
```

### Description

Sets the TTS context flag bitmask.

### Parameters

- `ctx`: TTS context handle.
- `flags`: Requested flag bitmask.

### Returns

The effective flags after masking unsupported bits.

## cosyvoice_tts_zero_shot

### Syntax

```c
COSYVOICE_API bool cosyvoice_tts_zero_shot(
    cosyvoice_tts_context_t        ctx,
    const char*                    text,
    float                          speed,
    cosyvoice_generated_speech_ptr result
);
```

### Description

Generates speech in zero-shot mode.

### Parameters

- `ctx`: TTS context handle.
- `text`: Input text.
- `speed`: Speech speed multiplier.
- `result`: Output waveform container.

### Returns

`true` on success; otherwise `false`.

### Remarks

Generates speech directly from text using the current session prompt.

## cosyvoice_tts_instruct

### Syntax

```c
COSYVOICE_API bool cosyvoice_tts_instruct(
    cosyvoice_tts_context_t        ctx,
    const char*                    text,
    const char*                    instruction,
    float                          speed,
    cosyvoice_generated_speech_ptr result
);
```

### Description

Generates speech in instruct mode using explicit instruction text.

### Parameters

- `ctx`: TTS context handle.
- `text`: Input text.
- `instruction`: Instruction text guiding style or behavior.
- `speed`: Speech speed multiplier.
- `result`: Output waveform container.

### Returns

`true` on success; otherwise `false`.

## cosyvoice_tts_cross_lingual

### Syntax

```c
COSYVOICE_API bool cosyvoice_tts_cross_lingual(
    cosyvoice_tts_context_t        ctx,
    const char*                    text,
    float                          speed,
    cosyvoice_generated_speech_ptr result
);
```

### Description

Generates speech in cross-lingual mode.

### Parameters

- `ctx`: TTS context handle.
- `text`: Input text.
- `speed`: Speech speed multiplier.
- `result`: Output waveform container.

### Returns

`true` on success; otherwise `false`.

## cosyvoice_save_wav

### Syntax

```c
COSYVOICE_API bool cosyvoice_save_wav(const char* filename, const float* data, uint32_t data_len, uint32_t sample_rate);
```

### Description

Writes floating-point PCM samples to a WAV file.

### Parameters

- `filename`: Output WAV path.
- `data`: Input PCM sample buffer.
- `data_len`: Number of samples.
- `sample_rate`: Sample rate in Hz.

### Returns

`true` on success; otherwise `false`.

### Remarks

Outputs mono 32-bit float WAV PCM data.

## cosyvoice_memory_usage_t

### Syntax

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

### Description

Reports memory usage components for a loaded context.

### Fields

- `parameters`: Memory for model parameters on main device.
- `kv_cache`: Memory used by KV cache on main device.
- `token2wav`: Memory used by token2wav intermediates on main device.
- `buffers`: Internal buffer memory on main device.
- `cpu_buffers`: Host-side buffer memory.
- `offloaded_kv_cache`: KV-cache memory offloaded to CPU.
- `random_noise`: CPU memory used by random-noise buffers.

## cosyvoice_get_memory_usage

### Syntax

```c
COSYVOICE_API void cosyvoice_get_memory_usage(cosyvoice_context_t ctx, cosyvoice_memory_usage_t* usage);
```

### Description

Retrieves a snapshot of current memory usage for the context.

### Parameters

- `ctx`: Context handle.
- `usage`: Output structure receiving usage values.

## cosyvoice_empty_buffer_cache

### Syntax

```c
COSYVOICE_API void cosyvoice_empty_buffer_cache(cosyvoice_context_t ctx);
```

### Description

Releases reusable inference buffers cached inside the context.

### Parameters

- `ctx`: Context handle.
