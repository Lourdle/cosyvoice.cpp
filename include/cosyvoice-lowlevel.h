#ifndef COSYVOICE_LOWLEVEL_H
#define COSYVOICE_LOWLEVEL_H

#ifndef COSYVOICE_H
    #error "This header is not meant to be included directly. Please include cosyvoice.h before including this header."
#endif

#include <ggml.h>
#include <ggml-alloc.h>

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

// ----------------------------------------------------------------------------
// Low-Level Types
// ----------------------------------------------------------------------------

typedef struct cosyvoice_tokenization_result* cosyvoice_tokenization_result_t; // Handle to a tokenization result object.
typedef struct cosyvoice_tokenizer_context*   cosyvoice_tokenizer_context_t;   // Handle to a tokenizer context.

/**
 * @brief Prompt update modes used by low-level prompt APIs.
 */
typedef enum cosyvoice_inference_mode
{
    COSYVOICE_INFERENCE_MODE_NULL = -1,      ///< Do nothing. Useful when duplicating a prompt without modifying it.
    COSYVOICE_INFERENCE_MODE_ZERO_SHOT,      ///< Generate speech from the input text without any additional context.
    COSYVOICE_INFERENCE_MODE_INSTRUCT,       ///< Generate speech from the input text and follow the prompt instruction text as closely as possible.
    COSYVOICE_INFERENCE_MODE_CROSS_LINGUAL,  ///< Generate speech from the input text and ignore the prompt instruction text.
    COSYVOICE_INFERENCE_MODE_COUNT           ///< Sentinel value.
} cosyvoice_inference_mode_t;

/**
 * @brief Stages reported to the noise callback around Flow and HiFT execution.
 */
typedef enum cosyvoice_noise_callback_stage
{
    COSYVOICE_NOISE_CALLBACK_STAGE_BEFORE_FLOW, ///< Called before Flow runs. `noise` is null and the callback should return a buffer of `length` samples.
    COSYVOICE_NOISE_CALLBACK_STAGE_AFTER_FLOW,  ///< Called after Flow finishes. `noise` is the buffer returned at `BEFORE_FLOW` and the callback return value is ignored.
    COSYVOICE_NOISE_CALLBACK_STAGE_BEFORE_HIFT, ///< Called before HiFT runs. `noise` is null and the callback should return a buffer of `length` samples.
    COSYVOICE_NOISE_CALLBACK_STAGE_AFTER_HIFT   ///< Called after HiFT finishes. `noise` is the buffer returned at `BEFORE_HIFT` and the callback return value is ignored.
} cosyvoice_noise_callback_stage_t;

// ----------------------------------------------------------------------------
// Callbacks
// ----------------------------------------------------------------------------

/**
 * @brief Callback used to provide and observe random-noise buffers for Flow and HiFT.
 * @param stage Stage indicating whether the callback is invoked before or after a module run.
 * @param length Required number of float samples for the noise buffer.
 * @param noise Noise buffer pointer associated with `stage`.
 *              For `COSYVOICE_NOISE_CALLBACK_STAGE_BEFORE_*`, this is null and the callback should return a buffer of `length` samples.
 *              For `COSYVOICE_NOISE_CALLBACK_STAGE_AFTER_*`, this is the same buffer pointer that was previously returned for the matching BEFORE stage.
 * @param ctx User context passed when registering the callback.
 * @return Buffer to use for the corresponding `COSYVOICE_NOISE_CALLBACK_STAGE_BEFORE_*` call.
 *         The return value is ignored for `COSYVOICE_NOISE_CALLBACK_STAGE_AFTER_*` calls.
 */
typedef float* (*cosyvoice_noise_callback_t)(
    cosyvoice_noise_callback_stage_t stage,
    uint32_t                         length,
    float*                           noise,
    void*                            ctx
);

/**
 * @brief Callback invoked by the backend scheduler around tensor evaluation.
 * @param t Graph node about to be evaluated (when `ask` is true) or just evaluated (when `ask` is false).
 * @param ask If true, the callback is asked whether it needs the result of `t`; if the callback returns
 *            true, the scheduler synchronizes and calls it again with `ask` set to false.
 *            If false, `t` has just been evaluated and its data is readable.
 * @param ctx User context passed when registering the callback.
 * @return When `ask` is true, whether the result of `t` is needed (the scheduler then evaluates `t`,
 *         synchronizes, and invokes the callback again with `ask` set to false).
 *         When `ask` is false, whether evaluation should continue; returning false aborts the
 *         remaining nodes of the current graph, so always return true unless aborting is intended.
 * @note When `ask` is true the callback must not read tensor data; it should only inspect the node.
 * @note Returning true for a node makes the scheduler evaluate graphs node-by-node up to that node,
 *       which slows down inference. Only request the nodes that are actually needed.
 */
typedef bool (*cosyvoice_sched_eval_callback_t)(
    struct ggml_tensor* t,
    bool                ask,
    void*               ctx
);

// ----------------------------------------------------------------------------
// Context Parameters Versioning
// ----------------------------------------------------------------------------

#define COSYVOICE_CONTEXT_PARAMS_VERSION     (0ul)
#define COSYVOICE_CONTEXT_PARAMS_V2_VERSION  (1ul)
#define COSYVOICE_CONTEXT_PARAMS_V3_VERSION  (2ul)
#define COSYVOICE_CONTEXT_PARAMS_V4_VERSION  (3ul)

// ----------------------------------------------------------------------------
// Logging Utilities
// ----------------------------------------------------------------------------

/**
 * @brief Default GGML log callback used by the runtime.
 * @param level GGML log severity.
 * @param text Log message text.
 * @param user_data Optional user data supplied by the caller.
 */
COSYVOICE_API void cosyvoice_log_callback_default(enum ggml_log_level level, const char* text, void* user_data);

// ----------------------------------------------------------------------------
// Context Loading and Status
// ----------------------------------------------------------------------------

/**
 * @brief Load a model context with explicit backend, threading, and context parameters.
 * @note `backend == NULL` selects the default backend.
 * @note If `backend != NULL`, backend ownership is transferred to the created context and released by `cosyvoice_free()`.
 * @note Set `n_threads` to 0 to use hardware concurrency when available.
 */
COSYVOICE_API cosyvoice_context_t cosyvoice_load_from_file_ext(
    const char*                       filename,
    const cosyvoice_context_params_t* params,
    ggml_backend_t                    backend,
    uint32_t                          n_threads,
    uint32_t                          params_version
);

#ifdef __cplusplus
}

namespace cosyvoice::internal_utils
{
template<typename T, typename U>
struct _is_same { static constexpr bool value = false; };
template<typename T>
struct _is_same<T, T> { static constexpr bool value = true; };
}

/**
 * @brief C++ wrapper for `cosyvoice_load_from_file_ext` that deduces the context parameter version from the type of `params`.
 * @note This overload is only available in C++ and requires the full definition of all context parameter types.
 */
template<typename params_t>
constexpr cosyvoice_context_t cosyvoice_load_from_file_ext(
    const char*                       filename,
    const params_t*                   params,
    ggml_backend_t                    backend,
    uint32_t                          n_threads
)
{
    using namespace cosyvoice::internal_utils;

    if constexpr (_is_same<params_t, cosyvoice_context_params_t>::value)
        return cosyvoice_load_from_file_ext(filename, params, backend, n_threads, COSYVOICE_CONTEXT_PARAMS_VERSION);
    else if constexpr (_is_same<params_t, cosyvoice_context_params_v2_t>::value)
        return cosyvoice_load_from_file_ext(filename, &params->base_params, backend, n_threads, COSYVOICE_CONTEXT_PARAMS_V2_VERSION);
    else if constexpr (_is_same<params_t, cosyvoice_context_params_v3_t>::value)
        return cosyvoice_load_from_file_ext(filename, &params->base_params.base_params, backend, n_threads, COSYVOICE_CONTEXT_PARAMS_V3_VERSION);
    else if constexpr (_is_same<params_t, cosyvoice_context_params_v4_t>::value)
        return cosyvoice_load_from_file_ext(filename, &params->base_params.base_params.base_params, backend, n_threads, COSYVOICE_CONTEXT_PARAMS_V4_VERSION);
    else
    {
        static_assert(_is_same<params_t, cosyvoice_context_params_v2_cpp>::value || _is_same<params_t, cosyvoice_context_params_v3_cpp>::value || _is_same<params_t, cosyvoice_context_params_v4_cpp>::value, "Unsupported context parameter type");
        constexpr auto version =
            _is_same<params_t, cosyvoice_context_params_v4_cpp>::value ? COSYVOICE_CONTEXT_PARAMS_V4_VERSION :
            _is_same<params_t, cosyvoice_context_params_v3_cpp>::value ? COSYVOICE_CONTEXT_PARAMS_V3_VERSION :
            COSYVOICE_CONTEXT_PARAMS_V2_VERSION;
        return cosyvoice_load_from_file_ext(filename, reinterpret_cast<const cosyvoice_context_params_t*>(params), backend, n_threads, version);
    }
}

extern "C" {
#endif

/**
 * @brief Load a model context with explicit backend, threading, and context parameters.
 * @note `backend == NULL` selects the default backend.
 * @note If `backend != NULL`, backend ownership is transferred to the created context and released by `cosyvoice_free()`.
 * @note Set `n_threads` to 0 to use hardware concurrency when available.
 * @note After loading, the data buffer is no longer needed and can be freed by the caller.
 */
COSYVOICE_API cosyvoice_context_t cosyvoice_load_ext(
    const void*                       data,
    size_t                            size,
    const cosyvoice_context_params_t* params,
    ggml_backend_t                    backend,
    uint32_t                          n_threads,
    uint32_t                          params_version
);

/**
 * @brief Get the status code of the last backend operation.
 */
COSYVOICE_API enum ggml_status cosyvoice_get_last_status(cosyvoice_context_t ctx);

// ----------------------------------------------------------------------------
// Scheduler Evaluation
// ----------------------------------------------------------------------------

/**
 * @brief Register a scheduler evaluation callback for the active worker.
 * @note Use cosyvoice_set_worker_no() to select the worker before registering. Each worker keeps
 *       its own callback and user context; registering on one worker leaves the others unchanged.
 * @note Passing a null callback removes the callback from the active worker.
 * @note The callback is invoked from the worker's inference thread. Registering or removing the
 *       callback is not thread-safe while inference is in flight on that worker.
 * @note Enabling a callback that requests nodes forces node-by-node graph evaluation and reduces
 *       inference performance. The callback persists across buffer-cache resets of that worker.
 */
COSYVOICE_API void cosyvoice_set_sched_eval_callback(cosyvoice_context_t ctx, cosyvoice_sched_eval_callback_t callback, void* callback_ctx);

// ----------------------------------------------------------------------------
// Pipeline Stage Notifications
// ----------------------------------------------------------------------------

/**
 * @brief Pipeline stages reported through @ref cosyvoice_stage_callback_t.
 */
typedef enum cosyvoice_stage_t
{
    COSYVOICE_STAGE_LLM_BEGIN,       ///< The LLM stage is about to start. Return false to cancel the call.
    COSYVOICE_STAGE_LLM_TOKEN,       ///< A speech token was accepted. `value` is the number of accepted tokens in the current sequence. Return false to stop generation and continue with token2wav.
    COSYVOICE_STAGE_LLM_END,         ///< The LLM stage finished. `value` is the number of accepted tokens. Return false to cancel the call.
    COSYVOICE_STAGE_TOKEN2WAV_BEGIN, ///< The token2wav stage is about to start. Return false to cancel the call.
    COSYVOICE_STAGE_FLOW_STEP,       ///< A flow-matching DiT step is about to run. `value` is the 0-based step index. Return false to skip this step (non-streaming) or cancel the call (streaming).
    COSYVOICE_STAGE_HIFT_BEGIN,      ///< The HiFT vocoder is about to run. Return false to cancel the call.
    COSYVOICE_STAGE_TOKEN2WAV_END,   ///< The token2wav stage finished. Return false to cancel the call.
} cosyvoice_stage_t;

/**
 * @brief Callback invoked at pipeline stage boundaries.
 * @param stage The stage event.
 * @param value Event-specific integer value (see @ref cosyvoice_stage_t).
 * @param worker_no Index of the worker whose pipeline reported the event.
 * @param user_data User context passed to @ref cosyvoice_set_stage_callback.
 * @return false to cancel the call, stop LLM generation, or skip a flow step, depending on the stage.
 * @note The callback runs on the inference thread while the worker is in use. Do not call context
 *       APIs that wait for the worker from the callback (they will deadlock); read-only getters
 *       are safe.
 * @note The callback is stored at context level and shared by all workers; @p worker_no identifies
 *       the source worker.
 */
typedef bool (*cosyvoice_stage_callback_t)(cosyvoice_stage_t stage, int32_t value, uint32_t worker_no, void* user_data);

/**
 * @brief Register a pipeline stage callback at context level, shared by all workers.
 * @note Passing a null callback removes the callback.
 */
COSYVOICE_API void cosyvoice_set_stage_callback(cosyvoice_context_t ctx, cosyvoice_stage_callback_t callback, void* user_data);

// ----------------------------------------------------------------------------
// Embedding Tensor Access
// ----------------------------------------------------------------------------

/**
 * @brief Get the word-token embedding table used by the LLM.
 */
COSYVOICE_API const ggml_tensor* cosyvoice_get_word_token_embed_weight(cosyvoice_context_t ctx);

/**
 * @brief Get the speech-token embedding table used by the LLM.
 */
COSYVOICE_API const ggml_tensor* cosyvoice_get_speech_token_embed_weight(cosyvoice_context_t ctx);

// ----------------------------------------------------------------------------
// LLM Evaluation & KV Cache Configuration
// ----------------------------------------------------------------------------

/**
 * @brief Prefill the LLM module with the given token embeddings.
 * @note This does not compute the logits of the next token.
 */
COSYVOICE_API bool cosyvoice_llm_prefill(
    cosyvoice_context_t ctx,
    enum ggml_type      type,
    const void*         data,
    uint32_t            n_tokens
);

/**
 * @brief Decode the LLM module output with the given token embeddings.
 * @note Logits are stored internally. Use the sampler API to sample the next token.
 */
COSYVOICE_API bool cosyvoice_llm_decode(
    cosyvoice_context_t ctx,
    enum ggml_type      type,
    const void*         data
);

/**
 * @brief Prefill the LLM module with the given token embeddings and compute the logits of the next token.
 * @note Logits are stored internally. Use the sampler API to sample the next token.
 * @note Unlike @ref cosyvoice_llm_prefill followed by @ref cosyvoice_llm_decode, this evaluates
 *       the attention output of only the last prefill position in a single graph computation.
 */
COSYVOICE_API bool cosyvoice_llm_prefill_logits(
    cosyvoice_context_t ctx,
    enum ggml_type      type,
    const void*         data,
    uint32_t            n_tokens
);

/**
 * @brief Prepare the current LLM probabilities for sampling.
 * @param allow_stop_tokens If false, stop tokens are masked to zero probability.
 */
COSYVOICE_API void cosyvoice_llm_prepare_probs(cosyvoice_context_t ctx, bool allow_stop_tokens);

/**
 * @brief Get the current length of the KV cache (tokens fed into the LLM).
 */
COSYVOICE_API uint32_t cosyvoice_llm_get_kv_cache_len(cosyvoice_context_t ctx);

/**
 * @brief Set the current KV cache length.
 * @note The new length must be less than or equal to the current length.
 */
COSYVOICE_API bool cosyvoice_llm_set_kv_cache_len(cosyvoice_context_t ctx, uint32_t len);

/**
* @brief Offload the KV cache to CPU memory.
*/
COSYVOICE_API void cosyvoice_llm_offload_kv_cache(cosyvoice_context_t ctx);

/**
* @brief Load the KV cache from CPU memory back to the backend.
*/
COSYVOICE_API void cosyvoice_llm_load_kv_cache(cosyvoice_context_t ctx);

// ----------------------------------------------------------------------------
// Sampling and Token Management
// ----------------------------------------------------------------------------

/**
 * @brief Sample a token from the current LLM logits.
 */
COSYVOICE_API int  cosyvoice_llm_sample_token(cosyvoice_context_t ctx);

/**
 * @brief Check whether the given token is a stop token.
 */
COSYVOICE_API bool cosyvoice_llm_is_stop_token(cosyvoice_context_t ctx, int token_id);

/**
 * @brief Accept the given token as the next token in the generated sequence.
 */
COSYVOICE_API void cosyvoice_llm_accept_token(cosyvoice_context_t ctx, int token_id);

/**
 * @brief Clear all accepted tokens.
 */
COSYVOICE_API void cosyvoice_llm_clear_accepted_tokens(cosyvoice_context_t ctx);

/**
 * @brief Get the number of tokens accepted in the current sequence.
 */
COSYVOICE_API uint32_t   cosyvoice_llm_get_n_accepted_tokens(cosyvoice_context_t ctx);

/**
 * @brief Get a pointer to the accepted-token buffer.
 */
COSYVOICE_API const int* cosyvoice_llm_get_accepted_tokens(cosyvoice_context_t ctx);

/**
 * @brief Run a full LLM forward pass over the given embedding rows without sampling.
 * @param type Type of the embedding rows in @p data (for example F32, or the speech-token embedding table's quantized type).
 * @param data @p n_tokens embedding rows; the row length is the speech-token embedding width.
 * @param n_tokens Number of rows. Must not exceed the context batch size (@ref cosyvoice_context_params_t::n_batch).
 * @param logits Optional output buffer for the teacher-forced logits in `[n_tokens][n_vocab]` row-major order.
 * @param logits_capacity Capacity of @p logits in floats. When @p logits is non-null it must be at least `n_tokens * n_vocab`.
 * @param n_vocab Output parameter for the number of logits per token.
 * @note All LLM layers are evaluated on every row, including the final layer, and the output
 *       projection is computed. The rows are processed on top of the active worker's current
 *       LLM KV-cache contents; truncate the cache first with @ref cosyvoice_llm_set_kv_cache_len
 *       to replay a suffix of a longer sequence, and call again with the next slice for batches
 *       larger than @p n_batch. Pipeline stage callbacks are not fired by this function.
 * @note This is also usable as a warmup pass on an empty cache.
 */
COSYVOICE_API bool cosyvoice_llm_full_forward(
    cosyvoice_context_t ctx,
    enum ggml_type      type,
    const void*         data,
    uint32_t            n_tokens,
    float*              logits,
    uint64_t            logits_capacity,
    uint32_t*           n_vocab
);

// ----------------------------------------------------------------------------
// Low-Level Inference Routines
// ----------------------------------------------------------------------------

/**
 * @brief Run the LLM with the given input tokens and prompt.
 * @note Generated speech tokens are stored internally and can be queried with the accepted-token APIs.
 */
COSYVOICE_API bool cosyvoice_llm_job(
    cosyvoice_context_t ctx,
    const int*          text,
    uint32_t            text_len,
    cosyvoice_prompt_t  prompt
);

/**
* @brief Run the LLM with the given input tokens and prompt, with additional options.
 * @param max_new_tokens Maximum number of new tokens to generate. If 0, no new tokens are generated.
 * @param final Output parameter indicating whether the generation is complete (true) or more tokens can be generated (false).
 * @note @p text must not be nullptr. Use @ref cosyvoice_llm_get_n_accepted_tokens to check whether this is a fresh run or a continuation.
 */
COSYVOICE_API bool cosyvoice_llm_job_ext(
    cosyvoice_context_t ctx,
    const int*          text,
    uint32_t            text_len,
    cosyvoice_prompt_t  prompt,
    uint32_t            max_new_tokens,
    bool*               final
);

/**
 * @brief Convert generated speech tokens to waveform data.
 */
COSYVOICE_API bool cosyvoice_token2wav(
    cosyvoice_context_t            ctx,
    const int*                     token_ids,
    uint32_t                       n_tokens,
    float                          speed,
    cosyvoice_prompt_t             prompt,
    cosyvoice_generated_speech_ptr generated_speech
);

/**
* @brief Convert generated speech tokens to waveform data with additional options.
*/
COSYVOICE_API bool cosyvoice_token2wav_ext(
    cosyvoice_context_t            ctx,
    const int*                     token_ids,
    uint32_t                       n_tokens,
    float                          speed,
    cosyvoice_prompt_t             prompt,
    bool                           streaming,
    bool                           finalize,
    cosyvoice_generated_speech_ptr result
);

/**
 * @brief Run the full TTS pipeline from input tokens to waveform output.
 */
COSYVOICE_API bool cosyvoice_tts(
    cosyvoice_context_t            ctx,
    const int*                     text,
    uint32_t                       text_len,
    float                          speed,
    cosyvoice_prompt_t             prompt,
    cosyvoice_generated_speech_ptr result
);

/**
 * @brief Run the full TTS pipeline from input tokens to waveform output with streaming support.
 * @param callback Callback function to receive generated audio chunks.
 * @param user_data User-defined data passed to the callback.
 */
COSYVOICE_API bool cosyvoice_tts_stream(
    cosyvoice_context_t            ctx,
    const int*                     text,
    uint32_t                       text_len,
    float                          speed,
    cosyvoice_prompt_t             prompt,
    cosyvoice_tts_audio_callback_t callback,
    void*                          user_data
);

/**
* @brief Get the number of tokens processed in each chunk during streaming inference.
*/
COSYVOICE_API uint32_t cosyvoice_get_chunk_tokens(cosyvoice_context_t ctx);

/**
* @brief Set the number of tokens processed in each chunk during streaming inference.
 */
COSYVOICE_API void cosyvoice_set_chunk_tokens(cosyvoice_context_t ctx, uint32_t n_tokens);

// ----------------------------------------------------------------------------
// Tokenizer Operations
// ----------------------------------------------------------------------------

/**
 * @brief Borrow the tokenizer owned by a model context.
 */
COSYVOICE_API cosyvoice_tokenizer_context_t   cosyvoice_get_tokenizer(cosyvoice_context_t ctx);

/**
 * @brief Load a standalone tokenizer from a model file.
 * @param filename Tokenizer file path.
 * @return Tokenizer handle on success; NULL on failure.
 */
COSYVOICE_API cosyvoice_tokenizer_context_t   cosyvoice_tokenizer_load_from_file(const char* filename);

/**
 * @brief Load a standalone tokenizer from a memory buffer.
 * @param data Pointer to model GGUF data.
 * @param size Size of the data buffer.
 * @return Tokenizer handle on success; NULL on failure.
 * @note The data buffer can be freed after loading.
 */
COSYVOICE_API cosyvoice_tokenizer_context_t   cosyvoice_tokenizer_load(const void* data, size_t size);

/**
 * @brief Free a tokenizer context loaded independently.
 */
COSYVOICE_API void                            cosyvoice_tokenizer_free(cosyvoice_tokenizer_context_t ctx);

/**
 * @brief Create an empty tokenization result container.
 */
COSYVOICE_API cosyvoice_tokenization_result_t cosyvoice_tokenization_result_create();

/**
 * @brief Free a tokenization result container.
 */
COSYVOICE_API void                            cosyvoice_tokenization_result_free(cosyvoice_tokenization_result_t result);

/**
 * @brief Get the mutable token buffer stored in a tokenization result.
 */
COSYVOICE_API int*                            cosyvoice_tokenization_result_get_tokens(cosyvoice_tokenization_result_t result);

/**
 * @brief Get the number of tokens stored in a tokenization result.
 */
COSYVOICE_API uint32_t                        cosyvoice_tokenization_result_get_n_tokens(cosyvoice_tokenization_result_t result);

/**
 * @brief Tokenize a null-terminated UTF-8 string.
 * @return Number of tokens written to `result`.
 */
COSYVOICE_API uint32_t cosyvoice_tokenize(
    cosyvoice_tokenizer_context_t   ctx,
    const char*                     text,
    cosyvoice_tokenization_result_t result,
    bool                            parse_special
);

/**
 * @brief Tokenize a UTF-8 string with an explicit byte length.
 * @return Number of tokens written to `result`.
 */
COSYVOICE_API uint32_t cosyvoice_tokenize_ext(
    cosyvoice_tokenizer_context_t   ctx,
    const char*                     text,
    uint32_t                        text_len,
    cosyvoice_tokenization_result_t result,
    bool                            parse_special
);

// ----------------------------------------------------------------------------
// Shared Noise Management
// ----------------------------------------------------------------------------

/**
 * @brief Register a shared callback for inspecting or overriding random-noise buffers.
 */
COSYVOICE_API void     cosyvoice_set_noise_callback(cosyvoice_context_t ctx, cosyvoice_noise_callback_t callback, void* callback_ctx);

/**
 * @brief Get the currently registered shared noise callback and its context.
 */
COSYVOICE_API void     cosyvoice_get_noise_callback(cosyvoice_context_t ctx, cosyvoice_noise_callback_t* callback, void** callback_ctx);

/**
 * @brief Get the required length of the shared HiFT initialization-noise buffer.
 */
COSYVOICE_API uint32_t cosyvoice_get_hift_rand_ini_len(cosyvoice_context_t ctx);

/**
 * @brief Override the shared HiFT initialization-noise buffer.
 */
COSYVOICE_API void     cosyvoice_set_hift_rand_ini(cosyvoice_context_t ctx, const float* data);

// ----------------------------------------------------------------------------
// Prompt Utilities
// ----------------------------------------------------------------------------

/**
 * @brief Get a CRC32 hash for a prompt-speech object.
 */
COSYVOICE_API uint32_t cosyvoice_prompt_speech_get_crc32(cosyvoice_prompt_speech_t prompt_speech);

/**
 * @brief Get a CRC32 hash for a prompt object.
 */
COSYVOICE_API uint32_t cosyvoice_prompt_get_crc32(cosyvoice_prompt_t prompt);

/**
 * @brief Get the instruction prefix expected by the current model.
 */
COSYVOICE_API const char* cosyvoice_get_instruction_prefix(cosyvoice_context_t ctx);

/**
 * @brief Set the inference mode and instruction text of a prompt from raw input.
 * @note The text is normalized and tokenized internally if the inference mode is instruct.
 * @note This API does not prepend the model's instruction prefix automatically.
 *       If needed, prepend the value returned by `cosyvoice_get_instruction_prefix()` yourself.
 * @note When `inplace` is true, the input prompt is updated in place and may be reused directly.
 */
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

/**
 * @brief Set the inference mode and instruction text of a prompt from tokenized input.
 * @note This API does not prepend the model's instruction prefix automatically.
 * @note This API does not append the `<|endofprompt|>` special token automatically.
 * @note If `inplace` is false, the original prompt is not freed.
 */
COSYVOICE_API cosyvoice_prompt_t cosyvoice_prompt_set_ext(
    cosyvoice_context_t        ctx,
    cosyvoice_prompt_t         prompt,
    cosyvoice_inference_mode_t mode,
    const int*                 instruction,
    uint32_t                   instruction_length,
    bool                       inplace
);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // COSYVOICE_LOWLEVEL_H
