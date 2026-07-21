#pragma once

#ifndef COSYVOICE_H
#error "This header is not meant to be included directly. Please include cosyvoice.h before including this header."
#endif

#ifndef COSYVOICE_LOWLEVEL_H
#error "This header is not meant to be included directly. Please include cosyvoice-lowlevel.h before including this header."
#endif

// ----------------------------------------------------------------------------
// Model Core Interfaces
// ----------------------------------------------------------------------------

/**
 * @brief Internal abstract interface implemented by model contexts.
 */
struct cosyvoice_model_context
{
    // Configuration
    virtual uint32_t get_sample_rate() = 0; ///< Get the model output sample rate.
    virtual void get_default_generation_config(cosyvoice_generation_config_t* config) = 0; ///< Copy the model-default generation configuration into `config`.
    virtual void get_generation_config(cosyvoice_generation_config_t* config) = 0; ///< Copy the current active worker's generation configuration into `config`.
    virtual bool set_generation_config(const cosyvoice_generation_config_t* config) = 0; ///< Validate and apply a generation configuration to the active worker.
    virtual void get_context_params(cosyvoice_context_params_t* params) = 0; ///< Copy the effective context parameters into `params`.
    virtual const char* get_instruction_prefix() = 0; ///< Get the instruction prefix expected by the current model.
    virtual const char* get_architecture() = 0; ///< Get a string identifying the current model architecture, e.g. "cosyvoice3-2512".
    virtual bool is_backend_uma() = 0; ///< Query whether the backend appears to use unified memory (UMA).
    virtual bool set_worker_no(uint32_t worker_no) = 0; ///< Select the active worker slot for subsequent operations. Use a duplicated context when running different workers concurrently. Returns false if the worker number is out of range.
    virtual uint32_t get_worker_no() = 0; ///< Get the current active worker slot number.
    virtual uint32_t get_n_workers() = 0; ///< Get the total number of worker slots available.

    // Sampling Configuration
    virtual void get_sampler(cosyvoice_sampler_t* sampler, void** sampler_ctx) = 0; ///< Get the current active worker's sampler callback and user context.
    virtual void set_sampler(cosyvoice_sampler_t sampler, void* sampler_ctx) = 0; ///< Set the sampler callback and user context for the active worker.
    virtual cosyvoice_builtin_sampler_rng_policy_t get_builtin_sampler_rng_policy() = 0; ///< Get the RNG policy used by the active worker's built-in sampler.
    virtual bool set_builtin_sampler_rng_policy(cosyvoice_builtin_sampler_rng_policy_t policy) = 0; ///< Set the RNG policy used by the active worker's built-in sampler.
    virtual bool set_sampler_seed(uint32_t seed) = 0; ///< Set the built-in sampler seed for the active worker. The seed is stored even if a custom sampler is active.
    virtual uint32_t get_sampler_seed() = 0; ///< Get the built-in sampler seed for the active worker.

    // LLM Operations
    virtual bool llm_prefill(ggml_type type, const void* data, uint32_t seq_len) = 0; ///< Prefill the LLM with a sequence of embeddings.
    virtual bool llm_decode(ggml_type type, const void* data) = 0; ///< Decode one step from an embedding vector.
    virtual void llm_prepare_probs(bool allow_stop_tokens) = 0; ///< Prepare probabilities for sampling and optionally mask stop tokens.

    // Embedding Access
    virtual const ggml_tensor* get_word_token_embed_weight() = 0; ///< Get the word-token embedding tensor.
    virtual const ggml_tensor* get_speech_token_embed_weight() = 0; ///< Get the speech-token embedding tensor.

    // KV Cache
    virtual uint32_t llm_get_kv_cache_len() = 0; ///< Get the current KV-cache sequence length.
    virtual bool llm_set_kv_cache_len(uint32_t len) = 0; ///< Trim the KV-cache sequence length.
    virtual void llm_offload_kv_cache() = 0; ///< Offload the KV cache to CPU memory.
    virtual void llm_load_kv_cache() = 0; ///< Load the KV cache from CPU memory back to the backend.

    // Token Sampling and Acceptance
    virtual int llm_sample_token() = 0; ///< Sample the next token from the current logits.
    virtual bool llm_is_stop_token(int token_id) = 0; ///< Check whether a token should stop generation.
    virtual void llm_accept_token(int token_id) = 0; ///< Accept a sampled token into the current sequence.
    virtual void llm_clear_accepted_tokens() = 0; ///< Clear the accepted-token history.
    virtual uint32_t llm_get_n_accepted_tokens() = 0; ///< Get the number of accepted tokens.
    virtual const int* llm_get_accepted_tokens() = 0; ///< Get the accepted-token buffer.

    // Inference
    virtual bool llm_job(
        const int*          text,
        uint32_t            text_len,
        cosyvoice_prompt_t  prompt
    ) = 0; ///< Run low-level LLM inference for a prompt and tokenized text.
    
    virtual bool token2wav(
        const int*                     token_ids,
        uint32_t                       n_tokens,
        float                          speed,
        cosyvoice_prompt_t             prompt,
        cosyvoice_generated_speech_ptr result
    ) = 0; ///< Convert speech tokens into waveform samples.
    
    // Extended Inference
    virtual bool llm_job_ext(
        const int*          text,
        uint32_t            text_len,
        cosyvoice_prompt_t  prompt,
        uint32_t            max_new_tokens,
        bool*               final
    ) = 0; ///< Run low-level LLM inference for a prompt and tokenized text with additional options.
   
    virtual bool token2wav_ext(
        const int*                     token_ids,
        uint32_t                       n_tokens,
        float                          speed,
        cosyvoice_prompt_t             prompt,
        uint32_t*                      offset,
        bool                           streaming,
        bool                           finalize,
        cosyvoice_generated_speech_ptr result
    ) = 0; ///< Convert speech tokens into waveform samples with additional options.

    virtual uint32_t get_chunk_tokens() = 0; ///< Get the number of tokens processed in each chunk during streaming inference.
    virtual void set_chunk_tokens(uint32_t n_tokens) = 0; ///< Set the number of tokens processed in each chunk during streaming inference.

    // Status
    virtual ggml_status get_last_status() = 0; ///< Get the status of the most recent backend operation.

    // Prompt Management
    virtual void set_prompt(
        cosyvoice_prompt_t         prompt,
        cosyvoice_inference_mode_t mode,
        const int*                 instruction,
        uint32_t                   instruction_length
    ) = 0;

    // Memory
    virtual void get_memory_usage(cosyvoice_memory_usage_t* usage) = 0; ///< Get the current memory-usage snapshot
    virtual void get_total_memory_usage(cosyvoice_memory_usage_t* usage) = 0; ///< Get the total memory usage.
    virtual void empty_buffer_cache() = 0; ///< Release cached reusable buffers.

    // Noise Control
    virtual void set_noise_callback(cosyvoice_noise_callback_t callback, void* callback_ctx) = 0; ///< Register the shared noise callback used by all workers.
    virtual void get_noise_callback(cosyvoice_noise_callback_t* callback, void** callback_ctx) = 0; ///< Query the shared noise callback.
    virtual uint32_t get_hift_rand_ini_len() = 0; ///< Get the required shared HiFT initialization-noise length.
    virtual void set_hift_rand_ini(const float* data) = 0; ///< Set the shared HiFT initialization-noise buffer.

    // Stop-request API
    virtual void request_stop() = 0; ///< Request that the active worker's current job stop as soon as possible.
    virtual bool stop_requested() = 0; ///< Check and atomically clear the stop-requested flag. Returns true if a stop was requested.
};

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// Tokenizer Interfaces
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

/**
 * @brief Internal abstraction for storing tokenizer output.
 */
struct cosyvoice_tokenization_result
{
    virtual int* get_tokens() = 0; ///< Get the mutable token buffer.
    virtual uint32_t get_n_tokens() = 0; ///< Get the number of stored tokens.
};

/**
 * @brief Internal tokenizer interface shared by model and standalone tokenizers.
 */
struct cosyvoice_tokenizer_context
{
    virtual uint32_t tokenize(const char* text, uint32_t text_len, cosyvoice_tokenization_result_t result, bool parse_special = true) = 0; ///< Tokenize UTF-8 text with an explicit byte length.

    inline uint32_t tokenize(const char* text, cosyvoice_tokenization_result_t result, bool parse_special = true)
    {
        auto p = text;
        while (*p) ++p;
        return tokenize(text, static_cast<uint32_t>(p - text), result, parse_special);
    }
};

// ----------------------------------------------------------------------------
// Combined Context Interface
// ----------------------------------------------------------------------------

/**
 * @brief Internal combined interface implemented by concrete runtime contexts.
 */
struct cosyvoice_context : virtual cosyvoice_model_context,
                           virtual cosyvoice_tokenizer_context {};
