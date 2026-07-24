#ifndef COSYVOICE_H
#define COSYVOICE_H

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

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#else
    #include <stdbool.h>
#endif

// ----------------------------------------------------------------------------
// Enumerations
// ----------------------------------------------------------------------------

/**
 * @brief Supported KV-cache storage formats.
 */
typedef enum cosyvoice_kv_cache_type
{
    COSYVOICE_KV_CACHE_TYPE_F32,   ///< Store KV cache in 32-bit floating point format.
    COSYVOICE_KV_CACHE_TYPE_F16,   ///< Store KV cache in 16-bit floating point format.
    COSYVOICE_KV_CACHE_TYPE_Q8_0,  ///< Store KV cache in GGML Q8_0 quantized format.
    COSYVOICE_KV_CACHE_TYPE_Q5_1,  ///< Store KV cache in GGML Q5_1 quantized format.
    COSYVOICE_KV_CACHE_TYPE_Q5_0,  ///< Store KV cache in GGML Q5_0 quantized format.
    COSYVOICE_KV_CACHE_TYPE_Q4_1,  ///< Store KV cache in GGML Q4_1 quantized format.
    COSYVOICE_KV_CACHE_TYPE_Q4_0,  ///< Store KV cache in GGML Q4_0 quantized format.
    COSYVOICE_KV_CACHE_TYPE_COUNT  // Sentinel value.
} cosyvoice_kv_cache_type_t, cosyvoice_llm_kv_cache_type_t;

#define COSYVOICE_LLM_KV_CACHE_TYPE_F32 COSYVOICE_KV_CACHE_TYPE_F32
#define COSYVOICE_LLM_KV_CACHE_TYPE_F16 COSYVOICE_KV_CACHE_TYPE_F16
#define COSYVOICE_LLM_KV_CACHE_TYPE_Q8_0 COSYVOICE_KV_CACHE_TYPE_Q8_0
#define COSYVOICE_LLM_KV_CACHE_TYPE_Q5_1 COSYVOICE_KV_CACHE_TYPE_Q5_1
#define COSYVOICE_LLM_KV_CACHE_TYPE_Q5_0 COSYVOICE_KV_CACHE_TYPE_Q5_0
#define COSYVOICE_LLM_KV_CACHE_TYPE_Q4_1 COSYVOICE_KV_CACHE_TYPE_Q4_1
#define COSYVOICE_LLM_KV_CACHE_TYPE_Q4_0 COSYVOICE_KV_CACHE_TYPE_Q4_0
#define COSYVOICE_LLM_KV_CACHE_TYPE_COUNT COSYVOICE_KV_CACHE_TYPE_COUNT

/**
 * @brief Buffer allocation strategies for inference workloads.
 */
typedef enum cosyvoice_inference_buffer_policy
{
    COSYVOICE_INFERENCE_BUFFER_POLICY_SHARED,    ///< Share buffers between the LLM KV cache and token2wav intermediates to minimize memory usage.
    COSYVOICE_INFERENCE_BUFFER_POLICY_BALANCED,  ///< Share buffers, but keep reusable sequence segments in memory for faster restoration on the next run.
    COSYVOICE_INFERENCE_BUFFER_POLICY_DEDICATED, ///< Use separate buffers for the LLM KV cache and token2wav intermediates to prioritize speed.
    COSYVOICE_INFERENCE_BUFFER_POLICY_COUNT      // Sentinel value.
} cosyvoice_inference_buffer_policy_t;

/**
 * @brief Policies controlling how the built-in sampler RNG state evolves.
 */
typedef enum cosyvoice_builtin_sampler_rng_policy
{
    COSYVOICE_BUILTIN_SAMPLER_RNG_POLICY_RESET_PER_SESSION,        ///< Reset the built-in sampler RNG to `seed` for each LLM session so identical sessions produce identical results.
    COSYVOICE_BUILTIN_SAMPLER_RNG_POLICY_CONTINUE_ACROSS_SESSIONS, ///< Keep advancing the built-in sampler RNG across LLM sessions so identical sessions can produce different results.
    COSYVOICE_BUILTIN_SAMPLER_RNG_POLICY_COUNT                     // Sentinel value.
} cosyvoice_builtin_sampler_rng_policy_t;

// ----------------------------------------------------------------------------
// Basic Structures
// ----------------------------------------------------------------------------

/**
 * @brief Sampling parameters used by the built-in nucleus sampler.
 */
typedef struct cosyvoice_sampling_params
{
    int   top_k;    ///< Limit sampling to the top-k candidate tokens.
    float top_p;    ///< Keep the smallest token set whose cumulative probability reaches this threshold.
    int   win_size; ///< Sliding-window size used by the repetition-aware sampler.
    float tau_r;    ///< Repetition control coefficient used by the sampler.
} cosyvoice_sampling_params_t;

/**
 * @brief Runtime generation controls for speech synthesis.
 */
typedef struct cosyvoice_generation_config
{
    float                       temperature;          ///< Softmax temperature applied during token sampling.
    cosyvoice_sampling_params_t sampling;             ///< Parameters for the built-in sampler.
    float                       min_token_text_ratio; ///< Minimum allowed ratio of generated acoustic tokens to input text length.
    float                       max_token_text_ratio; ///< Maximum allowed ratio of generated acoustic tokens to input text length.
} cosyvoice_generation_config_t;

/**
 * @brief Candidate token and probability pair exposed to custom samplers.
 */
typedef struct cosyvoice_llm_token_prob
{
    int   token_id; ///< Token identifier in the model vocabulary.
    float prob;     ///< Probability assigned to the token after filtering.
} cosyvoice_llm_token_prob_t;

/**
 * @brief Generated waveform buffer returned by synthesis APIs.
 */
typedef struct cosyvoice_generated_speech
{
    float*   data;   ///< Pointer to PCM samples in 32-bit floating point format.
    uint32_t length; ///< Number of samples in `data`.
} *cosyvoice_generated_speech_ptr;

// ----------------------------------------------------------------------------
// Opaque Type Declarations
// ----------------------------------------------------------------------------

typedef struct cosyvoice_context*       cosyvoice_context_t;            // Handle to a loaded model context.
typedef struct cosyvoice_prompt_speech* cosyvoice_prompt_speech_t;      // Handle to a prompt-speech asset loaded from disk.
typedef struct cosyvoice_prompt*        cosyvoice_prompt_t;             // Handle to a prompt object prepared for inference.
typedef struct cosyvoice_tts_context*   cosyvoice_tts_context_t;        // Handle to a reusable text-to-speech session context.

// ----------------------------------------------------------------------------
// Callback Types
// ----------------------------------------------------------------------------

/**
 * @brief Custom token sampler callback.
 * @param nucleus_probs Candidate tokens after nucleus filtering.
 * @param k Number of items in `nucleus_probs`.
 * @param probs Full probability buffer for the current vocabulary distribution.
 * @param size Number of entries in `probs`.
 * @param sampling_params Active sampling parameters.
 * @param accepted_tokens Tokens already accepted in the current sequence.
 * @param n_accepted_tokens Number of accepted tokens.
 * @param sampler_ctx User context passed to the callback.
 * @return The selected token id.
 */
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

/**
 * @brief Extended custom token sampler callback that includes the worker number for multi-worker contexts.
 * @param nucleus_probs Candidate tokens after nucleus filtering.
 * @param k Number of items in `nucleus_probs`.
 * @param probs Full probability buffer for the current vocabulary distribution.
 * @param size Number of entries in `probs`.
 * @param sampling_params Active sampling parameters.
 * @param accepted_tokens Tokens already accepted in the current sequence.
 * @param n_accepted_tokens Number of accepted tokens.
 * @param sampler_ctx User context passed to the callback.
 * @param worker_no The index of the worker.
 * @return The selected token id.
 */
typedef int (*cosyvoice_sampler_ext_t)(
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

// ----------------------------------------------------------------------------
// Context Parameters
// ----------------------------------------------------------------------------

/**
 * @brief Pack separate K, V, and fallback cache types into a single value.
 *
 * Layout (bit 31 = 1 means separate mode):
 *   bits  0–4  K cache type (5 bits)
 *   bits  5–9  V cache type (5 bits)
 *   bits 10–14 fallback cache type (5 bits)
 *   bits 15–30 unused
 *   bit      31 separate-K/V flag (must be 1)
 *
 * When bit 31 is 0, the value is a plain `cosyvoice_kv_cache_type_t`
 * that applies to both K and V (backward compatible).
 *
 * When the preferred K or V type is not supported by the backend, the
 * fallback type is tried first; if that also fails, auto-fallback applies.
 */
#define COSYVOICE_MAKE_SEPARATE_KV_CACHE(k_type, v_type, fallback_type) \
    ((cosyvoice_kv_cache_type_t)((k_type) | ((v_type) << 5) | ((fallback_type) << 10) | (1U << 31)))

/**
 * @brief Test whether llm_kv_cache_type is in separate-K/V mode.
 */
#define COSYVOICE_IS_SEPARATE_KV_CACHE(t)  (((t) & (1U << 31)) != 0)

/**
 * @brief Extract the K cache type from a packed value.
 */
#define COSYVOICE_K_CACHE_TYPE(t)          ((cosyvoice_kv_cache_type_t)((t) & 0x1F))

/**
 * @brief Extract the V cache type from a packed value.
 */
#define COSYVOICE_V_CACHE_TYPE(t)          ((cosyvoice_kv_cache_type_t)(((t) >> 5) & 0x1F))

/**
 * @brief Extract the fallback cache type from a packed value.
 */
#define COSYVOICE_KV_CACHE_FALLBACK(t)     ((cosyvoice_kv_cache_type_t)(((t) >> 10) & 0x1F))

/**
 * @brief Parameters used when creating a model context.
 */
typedef struct cosyvoice_context_params
{
    bool llm_use_flash_attn;  ///< Use flash attention if supported by the backend. Falls back to regular attention otherwise.
    bool flow_use_flash_attn; ///< Use flash attention for the Flow module if supported.

    union
    {
        struct
        {
#if defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__) || defined(_BYTE_ORDER) && (_BYTE_ORDER == _BIG_ENDIAN) || defined(__BIG_ENDIAN__) && !defined(__LITTLE_ENDIAN__) || defined(__ARMEB__) || defined(__MIPSEB__) || defined(__sparc__)
            uint32_t                    llm_kv_cache_separate_buffers : 1; ///< If true, allocate separate buffers for the K and V caches in the LLM module. Ignored when `llm_kv_cache_type` is set to a unified format.
            cosyvoice_kv_cache_type_t : 16;                               ///< Unused bits.
            cosyvoice_kv_cache_type_t   llm_kv_cache_fallback : 5;         ///< Fallback type when the preferred K or V type is unsupported.
            cosyvoice_kv_cache_type_t   llm_v_cache_type : 5;              ///< The data type of the V cache in the LLM module.
            cosyvoice_kv_cache_type_t   llm_k_cache_type : 5;              ///< The data type of the K cache in the LLM module.
#else
            cosyvoice_kv_cache_type_t   llm_k_cache_type : 5;              ///< The data type of the K cache in the LLM module.
            cosyvoice_kv_cache_type_t   llm_v_cache_type : 5;              ///< The data type of the V cache in the LLM module.
            cosyvoice_kv_cache_type_t   llm_kv_cache_fallback : 5;         ///< Fallback type when the preferred K or V type is unsupported.
            cosyvoice_kv_cache_type_t : 16;                               ///< Unused bits.
            uint32_t                    llm_kv_cache_separate_buffers : 1; ///< If true, allocate separate buffers for the K and V caches in the LLM module. Ignored when `llm_kv_cache_type` is set to a unified format.
#endif
        };
        cosyvoice_kv_cache_type_t       llm_kv_cache_type;                 ///< The data type of the KV cache in the LLM module.
    };
    bool                                llm_allow_kv_cache_fallback; ///< If true, fall back to a Flash Attention-compatible KV cache type when the requested one is unsupported.
    cosyvoice_inference_buffer_policy_t inference_buffer_policy;     ///< Controls how inference buffers are allocated and reused, which affects performance and memory usage.

    uint32_t n_batch;   ///< Batch size used by inference kernels.
    uint32_t n_max_seq; ///< Maximum supported sequence length.
    uint32_t seed;      ///< Seed used by the built-in sampler and noise generator RNG.
    cosyvoice_builtin_sampler_rng_policy_t builtin_sampler_rng_policy; ///< Controls how the built-in sampler RNG evolves across LLM sessions. Ignored when `sampler` is not null.

    // Sampling overrides
    union
    {
        cosyvoice_sampler_t sampler;
        cosyvoice_sampler_ext_t sampler_ext;
    }; ///< Optional custom sampler. Pass null to use the built-in sampler.
    void* sampler_ctx; ///< User context for `sampler`. Ignored when using the built-in sampler.
} cosyvoice_context_params_t;

typedef struct cosyvoice_context_params_v2
{
    cosyvoice_context_params_t base_params; ///< Base context parameters for backward compatibility.
    uint32_t n_workers;                      ///< Number of workers to use for inference.
} cosyvoice_context_params_v2_t;

/**
 * @brief Extended context parameters that add DiT KV cache configuration options.
 */
typedef struct cosyvoice_context_params_v3
{
    cosyvoice_context_params_v2_t base_params; ///< V2 base parameters.

    union
    {
        struct
        {
#if defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__) || defined(_BYTE_ORDER) && (_BYTE_ORDER == _BIG_ENDIAN) || defined(__BIG_ENDIAN__) && !defined(__LITTLE_ENDIAN__) || defined(__ARMEB__) || defined(__MIPSEB__) || defined(__sparc__)
            uint32_t                  dit_kv_cache_separate_buffers : 1; ///< If true, allocate separate buffers for the K and V caches in the DiT module. Ignored when `dit_kv_cache_type` is set to a unified format.
            cosyvoice_kv_cache_type_t : 16;                              ///< Unused bits.
            cosyvoice_kv_cache_type_t dit_kv_cache_fallback : 5;         ///< Fallback type when the preferred K or V type is unsupported.
            cosyvoice_kv_cache_type_t dit_v_cache_type : 5;              ///< The data type of the V cache in the DiT module.
            cosyvoice_kv_cache_type_t dit_k_cache_type : 5;              ///< The data type of the K cache in the DiT module.
#else
            cosyvoice_kv_cache_type_t dit_k_cache_type : 5;              ///< The data type of the K cache in the DiT module.
            cosyvoice_kv_cache_type_t dit_v_cache_type : 5;              ///< The data type of the V cache in the DiT module.
            cosyvoice_kv_cache_type_t dit_kv_cache_fallback : 5;         ///< Fallback type when the preferred K or V type is unsupported.
            cosyvoice_kv_cache_type_t : 16;                              ///< Unused bits.
            uint32_t                  dit_kv_cache_separate_buffers : 1; ///< If true, allocate separate buffers for the K and V caches in the DiT module. Ignored when `dit_kv_cache_type` is set to a unified format.
#endif
        };
        cosyvoice_kv_cache_type_t dit_kv_cache_type;                     ///< The data type of the KV cache in the DiT module.
    };
    bool     dit_allow_kv_cache_fallback; ///< If true, fall back to a Flash Attention-compatible KV cache type when the requested one is unsupported.
    uint32_t dit_kv_fixed_slots;          ///< Number of fixed (non-offloadable) DiT KV slots.
    uint32_t dit_kv_offloadable_slots;    ///< Number of offloadable DiT KV slots.
    uint32_t dit_kv_cache_length;         ///< Maximum sequence length for the DiT KV cache. 0 to use default (n_max_seq * 10).
} cosyvoice_context_params_v3_t;
#ifdef __cplusplus
struct cosyvoice_context_params_v2_cpp : cosyvoice_context_params_t
{
    uint32_t n_workers; ///< Number of workers to use for inference.
};

struct cosyvoice_context_params_v3_cpp : cosyvoice_context_params_v2_cpp
{
    union
    {
        struct
        {
#if defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__) || defined(_BYTE_ORDER) && (_BYTE_ORDER == _BIG_ENDIAN) || defined(__BIG_ENDIAN__) && !defined(__LITTLE_ENDIAN__) || defined(__ARMEB__) || defined(__MIPSEB__) || defined(__sparc__)
            uint32_t                  dit_kv_cache_separate_buffers : 1; ///< If true, allocate separate buffers for the K and V caches in the DiT module. Ignored when `dit_kv_cache_type` is set to a unified format.
            cosyvoice_kv_cache_type_t : 16;                              ///< Unused bits.
            cosyvoice_kv_cache_type_t dit_kv_cache_fallback : 5;         ///< Fallback type when the preferred K or V type is unsupported.
            cosyvoice_kv_cache_type_t dit_v_cache_type : 5;              ///< The data type of the V cache in the DiT module.
            cosyvoice_kv_cache_type_t dit_k_cache_type : 5;              ///< The data type of the K cache in the DiT module.
#else
            cosyvoice_kv_cache_type_t dit_k_cache_type : 5;              ///< The data type of the K cache in the DiT module.
            cosyvoice_kv_cache_type_t dit_v_cache_type : 5;              ///< The data type of the V cache in the DiT module.
            cosyvoice_kv_cache_type_t dit_kv_cache_fallback : 5;         ///< Fallback type when the preferred K or V type is unsupported.
            cosyvoice_kv_cache_type_t : 16;                              ///< Unused bits.
            uint32_t                  dit_kv_cache_separate_buffers : 1; ///< If true, allocate separate buffers for the K and V caches in the DiT module. Ignored when `dit_kv_cache_type` is set to a unified format.
#endif
        };
        cosyvoice_kv_cache_type_t dit_kv_cache_type;                     ///< The data type of the KV cache in the DiT module.
    };
    bool     dit_allow_kv_cache_fallback; ///< If true, fall back to a Flash Attention-compatible KV cache type when the requested one is unsupported.
    uint32_t dit_kv_fixed_slots;          ///< Number of fixed (non-offloadable) DiT KV slots.
    uint32_t dit_kv_offloadable_slots;    ///< Number of offloadable DiT KV slots.
    uint32_t dit_kv_cache_length;          ///< Maximum sequence length for the DiT KV cache. 0 to use default (n_max_seq * 10).
};
#endif

// ----------------------------------------------------------------------------
// Random Seed Utility
// ----------------------------------------------------------------------------

/**
 * @brief Generate a random 32-bit seed.
 * @return A random uint32_t value.
 */
COSYVOICE_API uint32_t cosyvoice_generate_random_seed();

// ----------------------------------------------------------------------------
// Backend Initialization API
// ----------------------------------------------------------------------------

/**
 * @brief Initialize the default backend runtime.
 * @note Call this before creating contexts when the selected backend requires explicit initialization.
 */
COSYVOICE_API void cosyvoice_init_backend();

/**
 * @brief Initialize the backend runtime using a custom directory.
 * @param dir_path Path to the backend directory used during initialization.
 */
COSYVOICE_API void cosyvoice_init_backend_from_path(const char* dir_path);

// ----------------------------------------------------------------------------
// Context Initialization & Management API
// ----------------------------------------------------------------------------

/**
 * @brief Fill a parameter structure with default values.
 */
COSYVOICE_API void                cosyvoice_init_default_context_params(cosyvoice_context_params_t* params);

/**
 * @brief Load a model context from a GGUF file using default context parameters.
 */
COSYVOICE_API cosyvoice_context_t cosyvoice_load_from_file(const char* filename);

/**
 * @brief Load a model context from a GGUF file using explicit context parameters.
 */
COSYVOICE_API cosyvoice_context_t cosyvoice_load_from_file_with_params(
    const char*                       filename,
    const cosyvoice_context_params_t* params
);

/**
 * @brief Load a model context from a GGUF file using extended context parameters with concurrency support.
 */
COSYVOICE_API cosyvoice_context_t cosyvoice_load_from_file_with_params_v2(
    const char*                          filename,
    const cosyvoice_context_params_v2_t* params
);

/**
 * @brief Load a model context from a GGUF file using V3 context parameters with DiT KV cache configuration.
 */
COSYVOICE_API cosyvoice_context_t cosyvoice_load_from_file_with_params_v3(
    const char*                          filename,
    const cosyvoice_context_params_v3_t* params
);

/**
 * @brief Duplicate a loaded model context handle.
 * @note The duplicate shares the loaded model resources with the original context. It starts with the same active worker binding as the original context, and can then be rebound independently with `cosyvoice_set_worker_no()`.
 */
COSYVOICE_API cosyvoice_context_t cosyvoice_duplicate_context(cosyvoice_context_t ctx);

/**
 * @brief Destroy a model context and release all associated resources.
 */
COSYVOICE_API void                cosyvoice_free(cosyvoice_context_t ctx);

/**
 * @brief Retrieve the effective context parameters of a loaded model.
 */
COSYVOICE_API void                cosyvoice_get_context_params(
    cosyvoice_context_t         ctx,
    cosyvoice_context_params_t* params
);

/**
 * @brief Get the total number of worker slots available for a model context.
 */
COSYVOICE_API uint32_t			  cosyvoice_get_n_workers(cosyvoice_context_t ctx);

/**
 * @brief Get the current active worker slot number for a model context.
 */
COSYVOICE_API uint32_t            cosyvoice_get_worker_no(cosyvoice_context_t ctx);

/**
 * @brief Set the active worker slot number for a model context.
 * @note Use this on a duplicated context to bind that context instance to a specific worker.
 */
COSYVOICE_API bool                cosyvoice_set_worker_no(
    cosyvoice_context_t ctx,
    uint32_t worker_no
);

/**
 * @brief Retrieve the architecture of the loaded model.
 */
COSYVOICE_API const char*         cosyvoice_get_architecture(cosyvoice_context_t ctx);

/**
 * @brief Query whether the backend appears to use unified memory (UMA).
 */
COSYVOICE_API bool                cosyvoice_is_backend_uma(cosyvoice_context_t ctx);

// ----------------------------------------------------------------------------
// Generation Configuration API
// ----------------------------------------------------------------------------

/**
 * @brief Retrieve the default generation configuration loaded from the model file.
*/
COSYVOICE_API void     cosyvoice_get_default_generation_config(
    cosyvoice_context_t            ctx,
    cosyvoice_generation_config_t* config
);

/**
 * @brief Retrieve the current generation configuration of the active worker.
 */
COSYVOICE_API void     cosyvoice_get_generation_config(
    cosyvoice_context_t            ctx,
    cosyvoice_generation_config_t* config
);

/**
 * @brief Set the generation config for the active worker. This overrides the default config loaded from the model file.
 * @note This function does not change the sample rate; any sample-rate field in the config is ignored.
 * @return True if the configuration is valid, otherwise false.
 */
COSYVOICE_API bool     cosyvoice_set_generation_config(
    cosyvoice_context_t                  ctx,
    const cosyvoice_generation_config_t* config
);

/**
 * @brief Retrieve the output sample rate of the loaded model.
 */
COSYVOICE_API uint32_t cosyvoice_get_sample_rate(cosyvoice_context_t ctx);

// ----------------------------------------------------------------------------
// Sampler API
// ----------------------------------------------------------------------------

/**
 * @brief Set a custom sampler for token sampling based on LLM logits.
 * @note The sampler is stored on the active worker.
 * @note If the sampler is null, the default nucleus sampler is used.
 */
COSYVOICE_API void cosyvoice_set_sampler(cosyvoice_context_t ctx, cosyvoice_sampler_t sampler, void* sampler_ctx);

/**
 * @brief Get the current active worker's sampler and its context.
 */
COSYVOICE_API void cosyvoice_get_sampler(cosyvoice_context_t ctx, cosyvoice_sampler_t* sampler, void** sampler_ctx);

/**
 * @brief Get the RNG policy used by the active worker's built-in sampler.
 */
COSYVOICE_API cosyvoice_builtin_sampler_rng_policy_t cosyvoice_get_builtin_sampler_rng_policy(cosyvoice_context_t ctx);

/**
 * @brief Set the RNG policy used by the active worker's built-in sampler.
 * @return True if `policy` is valid and the built-in sampler is active; false if `policy` is invalid or a custom sampler is in use.
 */
COSYVOICE_API bool cosyvoice_set_builtin_sampler_rng_policy(cosyvoice_context_t ctx, cosyvoice_builtin_sampler_rng_policy_t policy);

/**
 * @brief Set the seed used by the sampler RNG of the active worker.
 * @note This applies to the active worker only.
 * @note The seed is always stored, even when a custom sampler is active.
 *       If the function returns false, the stored seed will still take effect
 *       after switching back to the built-in sampler.
 * @return True when the built-in sampler is currently active; otherwise false.
 */
COSYVOICE_API bool cosyvoice_set_sampler_seed(cosyvoice_context_t ctx, uint32_t seed);

/**
 * @brief Get the seed used by the built-in sampler on the active worker.
 */
COSYVOICE_API uint32_t cosyvoice_get_sampler_seed(cosyvoice_context_t ctx);

// ----------------------------------------------------------------------------
// Prompt API
// ----------------------------------------------------------------------------

/**
 * @brief Load prompt speech features from a file.
 * @param filename Prompt-speech file path.
 * @return Prompt-speech handle on success; NULL on failure.
 */
COSYVOICE_API cosyvoice_prompt_speech_t cosyvoice_prompt_speech_load_from_file(const char* filename);

/**
 * @brief Load prompt speech features from a memory buffer.
 * @param data Pointer to prompt-speech GGUF data.
 * @param size Size of the data buffer.
 * @return Prompt-speech handle on success; NULL on failure.
 * @note The data buffer can be freed after loading.
 */
COSYVOICE_API cosyvoice_prompt_speech_t cosyvoice_prompt_speech_load(const void* data, size_t size);

/**
 * @brief Save a prompt-speech object to disk.
 * @return True on success, otherwise false.
 */
COSYVOICE_API bool cosyvoice_prompt_speech_save_to_file(cosyvoice_prompt_speech_t prompt_speech, const char* filename);

/**
 * @brief Create a prompt object from prompt speech for a given model context.
 */
COSYVOICE_API cosyvoice_prompt_t cosyvoice_prompt_init_from_prompt_speech(cosyvoice_context_t ctx, cosyvoice_prompt_speech_t prompt_speech);

/**
 * @brief Free a prompt-speech object.
 */
COSYVOICE_API void cosyvoice_prompt_speech_free(cosyvoice_prompt_speech_t prompt_speech);

/**
 * @brief Free a prompt object.
 */
COSYVOICE_API void cosyvoice_prompt_free(cosyvoice_prompt_t prompt);

// ----------------------------------------------------------------------------
// TTS Context Flags
// ----------------------------------------------------------------------------

#define COSYVOICE_TTS_FLAG_TEXT_NORMALIZATION (1u << 0)
#define COSYVOICE_TTS_FLAG_SPLIT_TEXT         (1u << 1)
#define COSYVOICE_TTS_FLAG_FAST_SPLIT         (1u << 2)
#define COSYVOICE_TTS_FLAG_FADE_IN            (1u << 3)

// ----------------------------------------------------------------------------
// Text-to-Speech Generation API
// ----------------------------------------------------------------------------

/**
 * @brief Create a reusable TTS session bound to a model context and prompt.
 */
COSYVOICE_API cosyvoice_tts_context_t cosyvoice_tts_context_new(cosyvoice_context_t ctx, cosyvoice_prompt_t prompt);

/**
 * @brief Destroy a TTS session context.
 */
COSYVOICE_API void                    cosyvoice_tts_context_free(cosyvoice_tts_context_t ctx);

/**
 * @brief Set the prompt for the TTS context. This also resets the cached instruction text.
 */
COSYVOICE_API void cosyvoice_tts_context_set_prompt(cosyvoice_tts_context_t ctx, cosyvoice_prompt_t prompt);

/**
 * @brief Enable or disable frontend text normalization for this TTS context.
 * @return True on success, false if normalization is unavailable (e.g. compiled without ICU).
 * @note Enabled by default.
 */
COSYVOICE_API bool cosyvoice_tts_context_set_text_normalization_enabled(cosyvoice_tts_context_t ctx, bool enabled);

/**
 * @brief Query whether frontend text normalization is enabled for this TTS context.
 */
COSYVOICE_API bool cosyvoice_tts_context_get_text_normalization_enabled(cosyvoice_tts_context_t ctx);

/**
 * @brief Enable or disable text splitting for this TTS context.
 * @note Enabled by default.
 */
COSYVOICE_API bool cosyvoice_tts_context_set_split_text_enabled(cosyvoice_tts_context_t ctx, bool enabled);

/**
 * @brief Query whether text splitting is enabled for this TTS context.
 */
COSYVOICE_API bool cosyvoice_tts_context_get_split_text_enabled(cosyvoice_tts_context_t ctx);

/**
 * @brief Enable or disable fast text splitting for this TTS context.
 * @note Enabled by default.
 */
COSYVOICE_API bool cosyvoice_tts_context_set_fast_split_text_enabled(cosyvoice_tts_context_t ctx, bool enabled);

/**
 * @brief Query whether fast text splitting is enabled for this TTS context.
 */
COSYVOICE_API bool cosyvoice_tts_context_get_fast_split_text_enabled(cosyvoice_tts_context_t ctx);

/**
 * @brief Enable or disable output fade-in for this TTS context.
 * @note Enabled by default.
 */
COSYVOICE_API bool cosyvoice_tts_context_set_fade_in_enabled(cosyvoice_tts_context_t ctx, bool enabled);

/**
 * @brief Query whether output fade-in is enabled for this TTS context.
 */
COSYVOICE_API bool cosyvoice_tts_context_get_fade_in_enabled(cosyvoice_tts_context_t ctx);

/**
 * @brief Get the current TTS context flags bitmask.
 * @return A combination of COSYVOICE_TTS_FLAG_* values.
 */
COSYVOICE_API uint32_t cosyvoice_tts_context_get_flags(cosyvoice_tts_context_t ctx);

/**
 * @brief Set the TTS context flags bitmask.
 * @details Unrecognized bits are silently masked out. Unavailable flags
 *          (e.g. COSYVOICE_TTS_FLAG_TEXT_NORMALIZATION without ICU) are also removed.
 * @return The effective flags after masking.
 */
COSYVOICE_API uint32_t cosyvoice_tts_context_set_flags(cosyvoice_tts_context_t ctx, uint32_t flags);

/**
 * @brief Generate speech in zero-shot mode.
 */
COSYVOICE_API bool cosyvoice_tts_zero_shot(
    cosyvoice_tts_context_t        ctx,
    const char*                    text,
    float                          speed,
    cosyvoice_generated_speech_ptr result
);

/**
 * @brief Generate speech in instruct mode with a custom instruction.
 */
COSYVOICE_API bool cosyvoice_tts_instruct(
    cosyvoice_tts_context_t        ctx,
    const char*                    text,
    const char*                    instruction,
    float                          speed,
    cosyvoice_generated_speech_ptr result
);

/**
 * @brief Generate speech in cross-lingual mode.
 */
COSYVOICE_API bool cosyvoice_tts_cross_lingual(
    cosyvoice_tts_context_t        ctx,
    const char*                    text,
    float                          speed,
    cosyvoice_generated_speech_ptr result
);

/**
 * @brief Callback invoked by the streaming TTS API for each generated audio chunk.
 * @param audio PCM samples in 32-bit floating point format, 1 channel.
 * @param n_samples Number of samples in this chunk.
 * @param user_data Opaque context passed when registering the callback.
 * @return True to continue streaming, false to abort the synthesis.
 */
typedef bool (*cosyvoice_tts_audio_callback_t)(const float* audio, uint32_t n_samples, void* user_data);

/**
 * @brief Generate speech with streaming output, delivering audio chunks via a callback.
 * @details The function synthesizes speech incrementally and invokes @p callback for
 *          each chunk as it becomes available. The callback receives the PCM data
 *          sequentially; the previous chunk's data is no longer valid after the
 *          callback returns.
 * @param ctx The TTS context.
 * @param text Input text to synthesize.
 * @param instruction Optional instruction (used in instruct mode). Pass NULL for zero-shot or cross-lingual.
 * @param speed Speed multiplier (1.0 = normal).
 * @param callback Callback receiving each audio chunk.
 * @param user_data Opaque context passed to @p callback.
 * @return True on success, false on failure.
 */
COSYVOICE_API bool cosyvoice_tts_zero_shot_stream(
    cosyvoice_tts_context_t        ctx,
    const char*                    text,
    float                          speed,
    cosyvoice_tts_audio_callback_t callback,
    void*                          user_data
);

/**
 * @brief Generate speech with streaming output in instruct mode.
 * @see cosyvoice_tts_zero_shot_stream
 */
COSYVOICE_API bool cosyvoice_tts_instruct_stream(
    cosyvoice_tts_context_t        ctx,
    const char*                    text,
    const char*                    instruction,
    float                          speed,
    cosyvoice_tts_audio_callback_t callback,
    void*                          user_data
);

/**
 * @brief Generate speech with streaming output in cross-lingual mode.
 * @see cosyvoice_tts_zero_shot_stream
 */
COSYVOICE_API bool cosyvoice_tts_cross_lingual_stream(
    cosyvoice_tts_context_t        ctx,
    const char*                    text,
    float                          speed,
    cosyvoice_tts_audio_callback_t callback,
    void*                          user_data
);

// ----------------------------------------------------------------------------
// Audio Output Utilities
// ----------------------------------------------------------------------------

/**
 * @brief Save floating-point PCM data as a WAV file.
 * @return True on success, otherwise false.
 */
COSYVOICE_API bool cosyvoice_save_wav(const char* filename, const float* data, uint32_t data_len, uint32_t sample_rate);

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// Memory Usage API
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

/**
 * @brief Memory usage breakdown for a loaded model context.
 */
typedef struct cosyvoice_memory_usage
{
    size_t parameters;         ///< Memory used for model parameters. Shared across workers, on the main device.
    size_t kv_cache;           ///< Memory used for the active worker's KV cache. On the main device.
    size_t token2wav;          ///< Memory used for the active worker's token2wav intermediates. On the main device.
    size_t buffers;            ///< Memory used for the active worker's internal buffers. On the main device.
    size_t cpu_buffers;        ///< Memory used for the active worker's CPU buffers. On CPU.
    size_t offloaded_kv_cache; ///< Memory offloaded for the active worker's KV cache. On CPU.
    size_t random_noise;       ///< Memory used for shared random-noise buffers. On CPU.
} cosyvoice_memory_usage_t;

/**
 * @brief Retrieve a memory usage snapshot for the active worker in the current context.
 */
COSYVOICE_API void cosyvoice_get_memory_usage(cosyvoice_context_t ctx, cosyvoice_memory_usage_t* usage);

/**
 * @brief Retrieve the total memory usage across all workers in the current context.
 */
COSYVOICE_API void cosyvoice_get_total_memory_usage(cosyvoice_context_t ctx, cosyvoice_memory_usage_t* usage);

/**
 * @brief Release reusable inference buffers cached by the context.
 */
COSYVOICE_API void cosyvoice_empty_buffer_cache(cosyvoice_context_t ctx);

#ifdef __cplusplus
}
#endif

#endif // COSYVOICE_H
