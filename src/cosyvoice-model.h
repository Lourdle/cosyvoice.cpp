#pragma once

#include "cosyvoice-modules.h"
#include "cosyvoice-lowlevel.h"
#include "cosyvoice-interface.h"
#include "cosyvoice-kv-cache.h"

#include <ggml-cpp.h>

#include <set>
#include <random>
#include <memory>
#include <shared_mutex>

struct cosyvoice_model_shared
{
    cosyvoice_model_shared(const cosyvoice_context_params_v2_cpp& params);

    ggml_context_ptr ctx;

    ggml_backend_buffer_ptr buffer;
    ggml_backend_buffer_ptr cpu_buffer;

    cosyvoice_context_params_v2_cpp params;
    ggml_backend_op_capabilities op_caps;
    bool backend_uma;

    std::mt19937 noise_rng;
    std::shared_mutex noise_mutex;
    std::unique_ptr<float[]> rand_noise;
    uint32_t rand_noise_len;
    cosyvoice_generation_config_t config;
    cosyvoice_noise_callback_t noise_callback;
    void* noise_callback_ctx;

    std::unique_ptr<char[]> architecture;
    std::unique_ptr<char[]> instruction_prefix;
};

struct cosyvoice_worker_context
{
    cosyvoice_worker_context(ggml_backend_t backend);
    ~cosyvoice_worker_context() = default;

    ggml_backend_ptr backend;
    ggml_backend_ptr cpu_backend;
    ggml_backend_sched_ptr sched;

    ggml_cgraph* gf;
    ggml_context_ptr ctx0;

    ggml_tensor* llm_input;
    ggml_tensor* llm_probs;
    ggml_tensor* position_ids;
    ggml_tensor* causal_mask;

    std::mt19937 sampler_rng, noise_rng;

    std::vector<int> tokens;
    std::unique_ptr<int[]> full_position_ids;
    std::unique_ptr<ggml_fp16_t[]> causal_mask_buffer;

    cosyvoice_kv_cache llm_kv_cache;
    cosyvoice_kv_cache dit_kv_cache;

    ggml_status status;
    uint32_t prompt_crc32;
    uint32_t sampler_seed;
    cosyvoice_generation_config_t config;
    cosyvoice_sampler_t sampler;
    void* sampler_ctx;
    cosyvoice_builtin_sampler_rng_policy_t builtin_sampler_rng_policy;

    std::vector<float> flow_cache;
    std::unique_ptr<char[]> batch_buffer;
    std::unique_ptr<float[]> nucleus_probs;
    uint32_t nucleus_probs_capacity;
    int nucleus_probs_len;
    std::unique_ptr<float[]> probs;
    ggml_backend_buffer_ptr llm_kv_buffer;
    ggml_backend_buffer_ptr dit_kv_buffer;
};

struct cosyvoice_model : virtual cosyvoice_model_context, virtual cosyvoice_object_ref_counter
{
    cosyvoice_model(ggml_backend_t backend, const cosyvoice_context_params_v2_cpp& params);
    ~cosyvoice_model();

    virtual void load(gguf_loader& loader) = 0;

    uint32_t get_worker_no();
    uint32_t get_n_workers();

    uint32_t llm_get_kv_cache_len();
    bool llm_set_kv_cache_len(uint32_t len);
    void llm_offload_kv_cache();
    void llm_load_kv_cache();

    int llm_sample_token();
    void llm_accept_token(int token);
    void llm_clear_accepted_tokens();
    uint32_t llm_get_n_accepted_tokens();
    const int* llm_get_accepted_tokens();

    ggml_status get_last_status();
    virtual void empty_buffer_cache();

    void set_prompt(
        cosyvoice_prompt_t         prompt,
        cosyvoice_inference_mode_t mode,
        const int*                 instruction,
        uint32_t                   instruction_length
    );

    void get_default_generation_config(cosyvoice_generation_config_t* config);
    void get_generation_config(cosyvoice_generation_config_t* config);
    bool set_generation_config(const cosyvoice_generation_config_t* config);
    const char* get_instruction_prefix();
    void get_context_params(cosyvoice_context_params_t* params);
    const char* get_architecture();
    bool is_backend_uma();

    void get_sampler(cosyvoice_sampler_t* sampler, void** sampler_ctx);
    void set_sampler(cosyvoice_sampler_t sampler, void* sampler_ctx);
    bool using_builtin_sampler() const;
    bool reset_builtin_sampler_rng();
    cosyvoice_builtin_sampler_rng_policy_t get_builtin_sampler_rng_policy();
    bool set_builtin_sampler_rng_policy(cosyvoice_builtin_sampler_rng_policy_t policy);
    bool set_sampler_seed(uint32_t seed);
    uint32_t get_sampler_seed();

    void set_noise_callback(cosyvoice_noise_callback_t callback, void* callback_ctx);
    void get_noise_callback(cosyvoice_noise_callback_t* callback, void** callback_ctx);

    cosyvoice_model_shared* shared;
    cosyvoice_worker_context* workers;
    cosyvoice_worker_context* worker;
};

struct cosyvoice_model_3_shared
{
    CausalMaskedDiffWithDiT flow;
    CausalHiFTGenerator hift;
    CosyVoice3LM llm;

    ggml_type k_type;
    ggml_type v_type;

    std::set<int> stop_tokens;
    std::set<int> silent_tokens;
};

struct cosyvoice_3_worker_context
{
    cosyvoice_3_worker_context();
    ~cosyvoice_3_worker_context() = default;

    ggml_context_ptr ctx1;
    ggml_backend_buffer_ptr token2wav_buffer;

    uint32_t orig_max_seq_len;
};

struct cosyvoice_model_3 : cosyvoice_model
{
    cosyvoice_model_3(ggml_backend_t backend, const cosyvoice_context_params_v2_cpp& params);
    ~cosyvoice_model_3();

    void load(gguf_loader& loader);

    bool set_worker_no(uint32_t worker_no);

    bool llm_decode(ggml_type type, const void* data);
    void llm_prepare_probs(bool allow_stop_tokens);
    bool llm_prefill(ggml_type type, const void* data, uint32_t seq_len);

    bool llm_is_stop_token(int token_id);

    uint32_t get_sample_rate();

    const ggml_tensor* get_word_token_embed_weight();
    const ggml_tensor* get_speech_token_embed_weight();

    uint32_t get_hift_rand_ini_len();
    void set_hift_rand_ini(const float* data);

    bool llm_job(const int* text, uint32_t text_len, cosyvoice_prompt_t prompt);
    bool llm_job_ext(const int* text, uint32_t text_len, cosyvoice_prompt_t prompt, uint32_t max_new_tokens, bool* final);
    bool token2wav(const int* token_ids, uint32_t n_tokens, float speed, cosyvoice_prompt_t prompt, cosyvoice_generated_speech_ptr result);
    bool token2wav_ext(const int* token_ids, uint32_t n_tokens, float speed, cosyvoice_prompt_t prompt, uint32_t* offset, bool streaming, bool finalize, cosyvoice_generated_speech_ptr result);
    uint32_t get_chunk_tokens();
    uint32_t get_flow_overlap_tokens();
    uint32_t get_hift_overlap_tokens();

    void empty_buffer_cache();
    void get_memory_usage(cosyvoice_memory_usage_t* usage);
    void get_total_memory_usage(cosyvoice_memory_usage_t* usage);
    void reset_shared_buffer(ggml_backend_buffer* new_buffer);

    cosyvoice_model_3_shared* cv3_shared;
    cosyvoice_3_worker_context* cv3_workers;
    cosyvoice_3_worker_context* cv3_worker;
};
