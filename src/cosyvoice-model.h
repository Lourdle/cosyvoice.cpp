#pragma once

#include "cosyvoice-modules.h"
#include "cosyvoice-lowlevel.h"
#include "cosyvoice-interface.h"
#include "cosyvoice-kv-cache.h"
#include "cosyvoice-memory.h"
#include "cosyvoice-stage-timing.h"
#include "common.h"

#include <ggml-cpp.h>

#include <mutex>
#include <condition_variable>
#include <random>
#include <memory>
#include <shared_mutex>

struct cosyvoice_model_shared
{
    cosyvoice_model_shared(const cosyvoice_context_params_v4_cpp& params);

    ggml_context_ptr ctx;

    // Keeps the model file mapping alive when weights are used zero-copy from it.
    file_mmap mmap;
    // Read-only view over `mmap`; does not own the memory. Declared after `mmap` so it is
    // destroyed first.
    ggml_backend_buffer_ptr mmap_buffer;

    vector<ggml_backend_buffer_ptr> buffers;
    ggml_backend_buffer_ptr cpu_buffer;

    cosyvoice_context_params_v4_cpp params;
    ggml_backend_op_capabilities op_caps;
    bool backend_uma;
    int hift_overlap;
    // COSYVOICE_SCHED_RESERVE: mel frames to pre-reserve for the Flow compute
    // buffers at load time, or -1 when the feature is disabled.
    int64_t sched_reserve_frames = -1;

    std::mt19937 noise_rng;
    std::shared_mutex noise_mutex;
    std::unique_ptr<float[]> rand_noise;
    uint32_t rand_noise_len;
    cosyvoice_generation_config_t config;
    cosyvoice_noise_callback_t noise_callback;
    void* noise_callback_ctx;
    std::unique_ptr<int[]> full_position_ids;

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

    vector<int> tokens;

    // Consecutive-silent-token counter for the FSQ silence filtering. It must
    // survive across llm_job_ext() calls that continue the same generation
    // (streaming chunks), so it lives in the worker instead of the job call.
    uint32_t silent_token_num{0};

    cosyvoice_kv_cache llm_kv_cache;
    cosyvoice_kv_cache dit_kv_cache;

    ggml_status status;
    uint32_t prompt_crc32;
    uint32_t sampler_seed;
    cosyvoice_generation_config_t config;
    cosyvoice_sampler_t sampler;
    void* sampler_ctx;
    cosyvoice_builtin_sampler_rng_policy_t builtin_sampler_rng_policy;

    uint32_t chunk_size;
    vector<float> flow_cache;
    // Streaming re-renders the whole accumulated mel each chunk, matching the
    // reference implementation, and crops away the already-emitted prefix at
    // this sample offset before handing the result to the caller.
    uint32_t wav_offset{0};
    vector<uint32_t> chunk_boundaries;
    uint32_t offset{0};

    // Stage diagnostics (COSYVOICE_STAGE_TIMING): per-round values are reset at
    // the start of every LLM job and printed after the vocoder; `stage_stream`
    // accumulates the whole streaming request and `stage_chunk_index` counts
    // its chunks, reset where the streaming state above is reset.
    stage_timing_round stage_round;
    stage_timing_round stage_stream;
    int64_t stage_round_begin{0};
    uint32_t stage_chunk_index{0};

    std::unique_ptr<char[]> batch_buffer;
    std::unique_ptr<float[]> nucleus_probs;
    // Immutable copy of the top-k candidates of the last computed logits, so
    // preparing the nucleus is idempotent and can be repeated for one logits
    // state without re-filtering already normalized probabilities.
    std::unique_ptr<float[]> nucleus_raw;
    uint32_t nucleus_probs_capacity;
    int nucleus_probs_len;
    bool raw_probs_masked;
    std::unique_ptr<float[]> probs;
    ggml_backend_buffer_ptr llm_kv_buffer;
    ggml_backend_buffer_ptr dit_kv_buffer;

    // Scheduler evaluation callback registered on this worker's scheduler.
    cosyvoice_sched_eval_callback_t sched_eval_callback{ nullptr };
    void* sched_eval_callback_ctx{ nullptr };

    std::atomic<bool> stop_flag{false};
    std::atomic<int>  use_count{0};
    std::mutex        cv_mutex;
    std::condition_variable cv;
};

struct cosyvoice_model : virtual cosyvoice_model_context, virtual cosyvoice_object_ref_counter
{
    cosyvoice_model(ggml_backend_t backend, const cosyvoice_context_params_v4_cpp& params);
    ~cosyvoice_model();

    virtual void load(gguf_loader& loader) = 0;

    uint32_t get_worker_no();
    uint32_t get_n_workers();
    void request_stop();
    bool is_stop_requested();

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
    uint32_t get_chunk_tokens();
    void set_chunk_tokens(uint32_t n_tokens);

    void set_noise_callback(cosyvoice_noise_callback_t callback, void* callback_ctx);
    void get_noise_callback(cosyvoice_noise_callback_t* callback, void** callback_ctx);

    void set_sched_eval_callback(cosyvoice_sched_eval_callback_t callback, void* callback_ctx);

    // Pipeline Stage Notifications
    void set_stage_callback(cosyvoice_stage_callback_t callback, void* user_data);
    bool invoke_stage_callback(cosyvoice_stage_t stage, int32_t value);

    cosyvoice_model_shared* shared;
    cosyvoice_worker_context* workers;
    cosyvoice_worker_context* worker;

    // Pipeline stage callback, shared by all workers of the context.
    cosyvoice_stage_callback_t stage_callback{ nullptr };
    void* stage_callback_user_data{ nullptr };
};

struct cosyvoice_model_3_shared
{
    CausalMaskedDiffWithDiT flow;
    CausalHiFTGenerator hift;
    CosyVoice3LM llm;

    set<int> stop_tokens;
    set<int> silent_tokens;
};

struct cosyvoice_3_worker_context
{
    cosyvoice_3_worker_context();
    ~cosyvoice_3_worker_context() = default;

    ggml_context_ptr ctx1;
    ggml_backend_buffer_ptr token2wav_buffer;
};

struct cosyvoice_model_3 : cosyvoice_model
{
    cosyvoice_model_3(ggml_backend_t backend, const cosyvoice_context_params_v4_cpp& params);
    ~cosyvoice_model_3();

    void load(gguf_loader& loader);

    bool set_worker_no(uint32_t worker_no);

    bool llm_decode(ggml_type type, const void* data);
    void llm_prepare_probs(bool allow_stop_tokens);
    bool llm_prefill(ggml_type type, const void* data, uint32_t seq_len);
    bool llm_prefill_logits(ggml_type type, const void* data, uint32_t seq_len);

    // Sequence Replay
    bool llm_full_forward(ggml_type type, const void* data, uint32_t n_tokens, float* logits, uint64_t logits_capacity, uint32_t* n_vocab);

    bool llm_is_stop_token(int token_id);

    uint32_t get_sample_rate();
    int get_diffusion_steps();

    const ggml_tensor* get_word_token_embed_weight();
    const ggml_tensor* get_speech_token_embed_weight();

    uint32_t get_hift_rand_ini_len();
    void set_hift_rand_ini(const float* data);

    bool llm_job(const int* text, uint32_t text_len, cosyvoice_prompt_t prompt);
    bool llm_job_ext(const int* text, uint32_t text_len, cosyvoice_prompt_t prompt, uint32_t max_new_tokens, bool* final);
    bool token2wav(const int* token_ids, uint32_t n_tokens, float speed, cosyvoice_prompt_t prompt, cosyvoice_generated_speech_ptr result);
    bool token2wav_ext(const int* token_ids, uint32_t n_tokens, float speed, cosyvoice_prompt_t prompt, bool streaming, bool finalize, cosyvoice_generated_speech_ptr result);

    bool reserve_token2wav_sched(uint32_t frames);
    void empty_buffer_cache() override;

    void get_memory_usage(cosyvoice_memory_usage_t* usage);
    void get_total_memory_usage(cosyvoice_memory_usage_t* usage);
    void reset_shared_buffer(ggml_backend_buffer* new_buffer);

    cosyvoice_model_3_shared* cv3_shared;
    cosyvoice_3_worker_context* cv3_workers;
    cosyvoice_3_worker_context* cv3_worker;
};
