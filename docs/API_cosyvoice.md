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

## cosyvoice_kv_cache_type_t

### Syntax

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

#### Backward Compatibility Aliases

The old `cosyvoice_llm_kv_cache_type_t` and `COSYVOICE_LLM_KV_CACHE_TYPE_*` constants
are provided as backward-compatibility aliases:

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

### Description

Specifies supported KV-cache storage formats. Used by both the LLM and DiT modules.

### Values

- `COSYVOICE_KV_CACHE_TYPE_F32`: 32-bit floating-point cache.
- `COSYVOICE_KV_CACHE_TYPE_F16`: 16-bit floating-point cache.
- `COSYVOICE_KV_CACHE_TYPE_Q8_0`: GGML `Q8_0` quantized cache.
- `COSYVOICE_KV_CACHE_TYPE_Q5_1`: GGML `Q5_1` quantized cache.
- `COSYVOICE_KV_CACHE_TYPE_Q5_0`: GGML `Q5_0` quantized cache.
- `COSYVOICE_KV_CACHE_TYPE_Q4_1`: GGML `Q4_1` quantized cache.
- `COSYVOICE_KV_CACHE_TYPE_Q4_0`: GGML `Q4_0` quantized cache.
- `COSYVOICE_KV_CACHE_TYPE_COUNT`: Sentinel value, not a runtime mode.

### Separate K/V Cache Macros

The `cosyvoice_kv_cache_type_t` value can encode separate types for the K and V cache
buffers via bit-packing.  This allows using different quantization formats for the K and V
tensors (e.g. K in `Q8_0` and V in `Q4_0`) to trade off quality vs. memory.

#### Packing format (bit 31 = 1 signals separate mode)

| Bits   | Field                |
|--------|----------------------|
|  0–4   | K cache type         |
|  5–9   | V cache type         |
| 10–14  | Fallback cache type  |
| 15–30  | Reserved             |
| 31     | Separate-K/V flag    |

When bit 31 is 0, the value is a plain `cosyvoice_kv_cache_type_t` that applies
to both K and V (backward compatible).

```c
#define COSYVOICE_MAKE_SEPARATE_KV_CACHE(k_type, v_type, fallback_type)
#define COSYVOICE_IS_SEPARATE_KV_CACHE(t)
#define COSYVOICE_K_CACHE_TYPE(t)
#define COSYVOICE_V_CACHE_TYPE(t)
#define COSYVOICE_KV_CACHE_FALLBACK(t)
```

- `COSYVOICE_MAKE_SEPARATE_KV_CACHE(k_type, v_type, fallback_type)` — Pack separate K, V and fallback types into a single value.
- `COSYVOICE_IS_SEPARATE_KV_CACHE(t)` — Returns non-zero if `t` is in separate-K/V mode.
- `COSYVOICE_K_CACHE_TYPE(t)` — Extracts the K cache type from a packed value.
- `COSYVOICE_V_CACHE_TYPE(t)` — Extracts the V cache type from a packed value.
- `COSYVOICE_KV_CACHE_FALLBACK(t)` — Extracts the fallback cache type from a packed value.

When the preferred K or V type is not supported by the backend, the fallback type is
tried first; if that also fails, auto-fallback applies.

#### Application in parameters

Use the bitfield members of `cosyvoice_context_params_t` or `cosyvoice_context_params_v3_t`
directly, or pack a value with `COSYVOICE_MAKE_SEPARATE_KV_CACHE` and assign it to
`llm_kv_cache_type` / `dit_kv_cache_type`:

```c
params.llm_k_cache_type = COSYVOICE_KV_CACHE_TYPE_Q8_0;
params.llm_v_cache_type = COSYVOICE_KV_CACHE_TYPE_Q4_0;
params.llm_kv_cache_separate_buffers = true;
```

or equivalently:

```c
params.llm_kv_cache_type = COSYVOICE_MAKE_SEPARATE_KV_CACHE(
    COSYVOICE_KV_CACHE_TYPE_Q8_0,
    COSYVOICE_KV_CACHE_TYPE_Q4_0,
    COSYVOICE_KV_CACHE_TYPE_Q8_0);
```

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

    union
    {
        struct
        {
            cosyvoice_kv_cache_type_t llm_k_cache_type : 5;              ///< K cache data type.
            cosyvoice_kv_cache_type_t llm_v_cache_type : 5;              ///< V cache data type.
            cosyvoice_kv_cache_type_t llm_kv_cache_fallback : 5;         ///< Fallback type when preferred type unsupported.
            cosyvoice_kv_cache_type_t : 16;                               ///< Reserved.
            uint32_t                  llm_kv_cache_separate_buffers : 1; ///< Allocate separate K & V buffers.
        };
        cosyvoice_kv_cache_type_t      llm_kv_cache_type;                 ///< Backward-compatible unified type.
    };
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
- `llm_k_cache_type`, `llm_v_cache_type`: Separate data types for K and V cache when `llm_kv_cache_separate_buffers` is true.
- `llm_kv_cache_fallback`: Fallback type when the preferred K or V type is unsupported by the backend.
- `llm_kv_cache_separate_buffers`: If true, allocate separate buffers for K and V caches using the types from `llm_k_cache_type` and `llm_v_cache_type`. Ignored when the unified `llm_kv_cache_type` does not have bit 31 set.
- `llm_kv_cache_type`: Requested KV-cache storage type. Can be a plain enum value (unified, applied to both K and V) or a packed value created with `COSYVOICE_MAKE_SEPARATE_KV_CACHE` that encodes separate K, V, and fallback types.
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

## cosyvoice_context_params_v3_t

### Syntax

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
};
#endif
```

### Description

Extends `cosyvoice_context_params_v2_t` with DiT (diffusion) KV cache configuration. The DiT module runs multiple diffusion steps during streaming TTS — each step computes self-attention over the full audio sequence. A KV cache can avoid redundant attention recomputation across steps, but the cache is very large (up to `sequence_length × n_diffusion_steps` key-value pairs).

### Fields

- `base_params`: V2 base parameters.
- `dit_k_cache_type`: Data type of the K cache in the DiT module.
- `dit_v_cache_type`: Data type of the V cache in the DiT module.
- `dit_kv_cache_separate_buffers`: If true, allocate separate buffers for K and V.
- `dit_kv_cache_fallback`: Fallback type when preferred K/V type is unsupported.
- `dit_kv_cache_type`: Shorthand — assigns a unified type (no separate K/V).
- `dit_allow_kv_cache_fallback`: If true, fall back to a flash-attention-compatible type.
- `dit_kv_fixed_slots`: Number of fixed (device memory, never offloaded) DiT KV slots. Each slot holds the KV cache for one diffusion step.
- `dit_kv_offloadable_slots`: Number of offloadable (CPU offload) DiT KV slots.
- `dit_kv_cache_length`: Maximum sequence length for the DiT KV cache. 0 to use default (`n_max_seq × 10`).

### DiT KV Cache Concept

See [README.md — Streaming TTS](#streaming-tts--dit-kv-cache) for a detailed explanation of the DiT KV cache concept.

## cosyvoice_load_from_file_with_params_v3

### Syntax

```c
COSYVOICE_API cosyvoice_context_t cosyvoice_load_from_file_with_params_v3(
    const char*                          filename,
    const cosyvoice_context_params_v3_t* params
);
```

### Description

Loads a model context with V3 extended parameters, including DiT KV cache configuration.

### Parameters

- `filename`: Path to the model file.
- `params`: V3 context-parameter block.

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

## cosyvoice_is_backend_uma

### Syntax

```c
COSYVOICE_API bool cosyvoice_is_backend_uma(cosyvoice_context_t ctx);
```

### Description

Queries whether the backend appears to use unified memory architecture (UMA).

### Parameters

- `ctx`: Context handle.

### Returns

`true` if the backend memory is detected as UMA; otherwise `false`.

### Remarks

The result is determined at model load time by probing the backend memory bandwidth. On Apple Silicon (`__aarch64__`), UMA is assumed by default. On other platforms, the runtime compares backend tensor-set bandwidth against host `memcpy` bandwidth. When UMA is detected and the requested buffer policy is `balanced`, the library automatically switches to `dedicated` to avoid redundant buffer sharing. This query is useful for callers that want to display backend characteristics or make policy decisions based on the memory architecture.

> **Note**: UMA detection is a heuristic based on bandwidth probing. Results may be inaccurate depending on hardware, driver version, and system load at probe time. Treat the result as a rough hint rather than a definitive hardware capability.

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

## cosyvoice_prompt_speech_load

### Syntax

```c
COSYVOICE_API cosyvoice_prompt_speech_t cosyvoice_prompt_speech_load(const void* data, size_t size);
```

### Description

Loads prompt-speech features from a memory buffer.

### Parameters

- `data`: Pointer to the prompt-speech GGUF data in memory.
- `size`: Size of the prompt-speech data in bytes.

### Returns

Prompt-speech handle on success; `NULL` on failure.

### Remarks

- The data buffer must contain prompt-speech GGUF data with `feat`, `embedding`, `tokens`, and `text` tensors.
- A CRC32 checksum is verified if present in the metadata.
- After loading, the data buffer is no longer needed and can be freed by the caller.

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

## cosyvoice_tts_audio_callback_t

### Syntax

```c
typedef bool (*cosyvoice_tts_audio_callback_t)(const float* audio, uint32_t n_samples, void* user_data);
```

### Description

Callback invoked by the streaming TTS API for each generated audio chunk.

### Parameters

- `audio`: PCM samples in 32-bit floating point format, 1 channel.
- `n_samples`: Number of samples in this chunk.
- `user_data`: Opaque context passed when registering the callback.

### Returns

`true` to continue streaming, `false` to abort synthesis.

## cosyvoice_tts_zero_shot_stream

### Syntax

```c
COSYVOICE_API bool cosyvoice_tts_zero_shot_stream(
    cosyvoice_tts_context_t        ctx,
    const char*                    text,
    float                          speed,
    cosyvoice_tts_audio_callback_t callback,
    void*                          user_data
);
```

### Description

Generates speech with streaming output in zero-shot mode. Audio chunks are delivered incrementally via the callback.

### Parameters

- `ctx`: TTS context handle.
- `text`: Input text.
- `speed`: Speech speed multiplier.
- `callback`: Callback receiving each audio chunk.
- `user_data`: Opaque context passed to the callback.

### Returns

`true` on success; otherwise `false`.

## cosyvoice_tts_instruct_stream

### Syntax

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

### Description

Generates speech with streaming output in instruct mode.

### Parameters

- `ctx`: TTS context handle.
- `text`: Input text.
- `instruction`: Instruction text guiding style or behavior.
- `speed`: Speech speed multiplier.
- `callback`: Callback receiving each audio chunk.
- `user_data`: Opaque context passed to the callback.

### Returns

`true` on success; otherwise `false`.

## cosyvoice_tts_cross_lingual_stream

### Syntax

```c
COSYVOICE_API bool cosyvoice_tts_cross_lingual_stream(
    cosyvoice_tts_context_t        ctx,
    const char*                    text,
    float                          speed,
    cosyvoice_tts_audio_callback_t callback,
    void*                          user_data
);
```

### Description

Generates speech with streaming output in cross-lingual mode.

### Parameters

- `ctx`: TTS context handle.
- `text`: Input text.
- `speed`: Speech speed multiplier.
- `callback`: Callback receiving each audio chunk.
- `user_data`: Opaque context passed to the callback.

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

## cosyvoice_get_total_memory_usage

### Syntax

```c
COSYVOICE_API void cosyvoice_get_total_memory_usage(cosyvoice_context_t ctx, cosyvoice_memory_usage_t* usage);
```

### Description

Retrieves a snapshot of total memory usage across all workers in the context.

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
