#include "cosyvoice-model.h"
#include "common.h"

#include <cstring>
#include <span>

void cosyvoice_init_default_context_params(cosyvoice_context_params_t* params)
{
    params->flow_use_flash_attn = true;
    params->llm_use_flash_attn = true;

    params->llm_kv_cache_type = COSYVOICE_LLM_KV_CACHE_TYPE_Q8_0;
    params->llm_allow_kv_cache_fallback = true;
    params->inference_buffer_policy = COSYVOICE_INFERENCE_BUFFER_POLICY_BALANCED;

    params->n_batch = 256;
    params->n_max_seq = COSYVOICE_DEFAULT_LLM_MAX_SEQ_LEN;

    params->seed = cosyvoice_generate_random_seed();
    params->builtin_sampler_rng_policy = COSYVOICE_BUILTIN_SAMPLER_RNG_POLICY_RESET_PER_SESSION;

    params->sampler = nullptr;
    params->sampler_ctx = nullptr;
}

cosyvoice_model_shared::cosyvoice_model_shared(const cosyvoice_context_params_t& params)
    : params(params), ctx(nullptr) {}

cosyvoice_worker_context::cosyvoice_worker_context(ggml_backend_t backend)
    : backend(backend), cpu_backend(backend),
    ctx0(ggml_init(ggml_init_params{ .mem_size = ggml_graph_overhead() * kCosyVoiceGraphSize, .no_alloc = true })),
    gf(nullptr), llm_input(nullptr), llm_probs(nullptr), position_ids(nullptr), causal_mask(nullptr), kv_cache(),
    status(GGML_STATUS_SUCCESS), prompt_crc32(0), rand_noise_len(0) {}

cosyvoice_model::cosyvoice_model(ggml_backend_t backend, const cosyvoice_context_params_t& params)
    : shared(new cosyvoice_model_shared(params)), worker(new cosyvoice_worker_context(backend))
{
    if (GGML_BACKEND_DEVICE_TYPE_CPU != ggml_backend_dev_type(ggml_backend_get_device(backend)))
    {
        worker->cpu_backend.release();
        worker->cpu_backend.reset(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr));
    }

    empty_buffer_cache();
    set_sampler(params.sampler, params.sampler_ctx);
    set_noise_callback(nullptr, nullptr);

    auto ctx0 = worker->ctx0.get();
    auto& op_caps = shared->op_caps;
    ggml_tensor* a = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
    a = ggml_concat(ctx0, a, a, 0);
    op_caps.concat_i32 = ggml_backend_supports_op(backend, a);

    a = ggml_new_tensor_3d(ctx0, GGML_TYPE_F16, 11, 4, 51);
    a = ggml_repeat_4d(ctx0, a, 11, 4, 51, 4);
    op_caps.repeat_f16 = ggml_backend_supports_op(backend, a);

    a = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 16, 4);
    a = ggml_pad(ctx0, a, 4, 0, 0, 0);
    op_caps.pad = ggml_backend_supports_op(backend, a);

    a = ggml_fill(ctx0, a, 0.0f);
    op_caps.fill = ggml_backend_supports_op(backend, a);

    a = ggml_cumsum(ctx0, a);
    op_caps.cumsum = ggml_backend_supports_op(backend, a);

    a = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 16);
    a = ggml_pad_reflect_1d(ctx0, a, 1, 0);
    op_caps.pad_reflect_1d = ggml_backend_supports_op(backend, a);

    {
        ggml_tensor* w = ggml_new_tensor_3d(ctx0, GGML_TYPE_F16, 3, 4, 8);
        a = ggml_new_tensor_3d(ctx0, GGML_TYPE_F16, 16, 4, 1);
        a = ggml_im2col(ctx0, w, a, 1, 0, 0, 0, 1, 0, false, GGML_TYPE_F16);
        op_caps.im2col_f16 = ggml_backend_supports_op(backend, a);
    }
}

bool cosyvoice_model::using_builtin_sampler() const
{
    return shared->params.sampler == reinterpret_cast<cosyvoice_sampler_t>(cosyvoice_llm_sampler);
}

bool cosyvoice_model::reset_builtin_sampler_rng()
{
    if (using_builtin_sampler())
    {
        worker->sampler_rng.seed(shared->params.seed);
        worker->noise_rng.seed(worker->sampler_rng());
        return true;
    }
    return false;
}

cosyvoice_builtin_sampler_rng_policy_t cosyvoice_model::get_builtin_sampler_rng_policy()
{
    return shared->params.builtin_sampler_rng_policy;
}

bool cosyvoice_model::set_sampler_seed(uint32_t seed)
{
    shared->params.seed = seed;
    return reset_builtin_sampler_rng();
}

bool cosyvoice_model::set_builtin_sampler_rng_policy(cosyvoice_builtin_sampler_rng_policy_t policy)
{
    if (using_builtin_sampler())
        switch (policy)
        {
        case COSYVOICE_BUILTIN_SAMPLER_RNG_POLICY_RESET_PER_SESSION:
        case COSYVOICE_BUILTIN_SAMPLER_RNG_POLICY_CONTINUE_ACROSS_SESSIONS:
            shared->params.builtin_sampler_rng_policy = policy;
            return true;
        }
    return false;
}

cosyvoice_model::~cosyvoice_model()
{
    if (get_ref_count() == 1)
    {
        if (worker->backend == worker->cpu_backend)
            worker->cpu_backend.release();

        switch (shared->params.inference_buffer_policy)
        {
        case COSYVOICE_INFERENCE_BUFFER_POLICY_BALANCED:
        case COSYVOICE_INFERENCE_BUFFER_POLICY_SHARED:
            worker->kv_buffer.release();
        }

        delete shared;
        delete worker;
    }
}

void cosyvoice_model::empty_buffer_cache()
{
    ggml_backend_t backends[] = { worker->backend.get(), worker->cpu_backend.get() };
    worker->sched.reset(ggml_backend_sched_new(
        backends,
        nullptr,
        worker->backend == worker->cpu_backend ? 1 : 2,
        kCosyVoiceSchedGraphSize,
        true,
        true
    ));

    worker->rand_noise.reset();
    worker->rand_noise_len = 0;

    worker->kv_cache.clear_offloaded_cache();
}

void cosyvoice_model_3::empty_buffer_cache()
{
    cosyvoice_model::empty_buffer_cache();

    if (cv3_worker->orig_max_seq_len != shared->params.n_max_seq)
    {
        shared->params.n_max_seq = cv3_worker->orig_max_seq_len;
        worker->kv_buffer.reset(
            worker->kv_cache.initialize_buffer(
                worker->backend.get(),
                static_cast<int>(cv3_shared->llm.layers[0].self_attn.k_proj.weight->ne[1] / cv3_shared->llm.num_key_value_heads),
                static_cast<int>(cv3_shared->llm.layers[0].self_attn.v_proj.weight->ne[1] / cv3_shared->llm.num_key_value_heads),
                cv3_shared->llm.num_attention_heads,
                cv3_shared->llm.num_key_value_heads,
                shared->params.n_max_seq,
                cv3_shared->kv_type,
                shared->params.llm_use_flash_attn));

        switch (shared->params.inference_buffer_policy)
        {
        case COSYVOICE_INFERENCE_BUFFER_POLICY_BALANCED:
        case COSYVOICE_INFERENCE_BUFFER_POLICY_SHARED:
            cv3_worker->token2wav_buffer.release();
            cv3_worker->token2wav_buffer.reset(worker->kv_buffer.get());
            break;
        case COSYVOICE_INFERENCE_BUFFER_POLICY_DEDICATED:
            cv3_worker->token2wav_buffer.reset();
            break;
        default:
            GGML_ABORT("unexpected policy");
        }
    }
}

void cosyvoice_model_3::get_memory_usage(cosyvoice_memory_usage_t* usage)
{
    usage->parameters = ggml_backend_buffer_get_size(shared->buffer.get());
    usage->buffers = ggml_backend_sched_get_buffer_size(worker->sched.get(), worker->backend.get());
    usage->cpu_buffers = ggml_backend_sched_get_buffer_size(worker->sched.get(), worker->cpu_backend.get());
    usage->offloaded_kv_cache = worker->kv_cache.get_offloaded_cache_size();
    usage->random_noise = sizeof(float) * worker->rand_noise_len;
    usage->kv_cache = worker->kv_buffer.get() ? ggml_backend_buffer_get_size(worker->kv_buffer.get()) : 0;
    usage->token2wav = cv3_worker->token2wav_buffer.get() ? ggml_backend_buffer_get_size(cv3_worker->token2wav_buffer.get()) : 0;
}

void cosyvoice_model_3::reset_shared_buffer(ggml_backend_buffer* new_buffer)
{
    cv3_worker->token2wav_buffer.reset(new_buffer);
    if (shared->params.inference_buffer_policy != COSYVOICE_INFERENCE_BUFFER_POLICY_DEDICATED)
    {
        worker->kv_buffer.release();
        worker->kv_buffer.reset(new_buffer);
        shared->params.n_max_seq = worker->kv_cache.reset_buffer(new_buffer);
    }
}

cosyvoice_3_worker_context::cosyvoice_3_worker_context() :
    ctx1(ggml_init(ggml_init_params{ .mem_size = ggml_tensor_overhead() * 4, .no_alloc = true })) {}

cosyvoice_model_3::cosyvoice_model_3(ggml_backend_t backend, const cosyvoice_context_params_t& params)
    : cosyvoice_model(backend, params), cv3_shared(new cosyvoice_model_3_shared), cv3_worker(new cosyvoice_3_worker_context()) {}

cosyvoice_model_3::~cosyvoice_model_3()
{
    if (get_ref_count() == 1)
    {
        delete cv3_shared;
        delete cv3_worker;
    }
}

void CausalHiFTGenerator::set_rand_ini(const float* data) const
{
    ggml_backend_tensor_set(m_source.l_sin_gen.rand_ini, data, 0, (nb_harmonics + 1) * sizeof(float));
}

bool cosyvoice_model_3::llm_is_stop_token(int token_id)
{
    return cv3_shared->stop_tokens.find(token_id) != cv3_shared->stop_tokens.end();
}

const ggml_tensor* cosyvoice_model_3::get_word_token_embed_weight()
{
    return cv3_shared->llm.embed_tokens_weight;
}

const ggml_tensor* cosyvoice_model_3::get_speech_token_embed_weight()
{
    return cv3_shared->llm.speech_embedding_weight;
}

uint32_t cosyvoice_model_3::get_hift_rand_ini_len()
{
    return cv3_shared->hift.nb_harmonics + 1;
}

void cosyvoice_model_3::set_hift_rand_ini(const float* data)
{
    cv3_shared->hift.set_rand_ini(data);
}

uint32_t cosyvoice_model_3::get_sample_rate()
{
    return cv3_shared->hift.sampling_rate;
}

void cosyvoice_model::get_generation_config(cosyvoice_generation_config_t* config)
{
    *config = shared->config;
}

bool cosyvoice_model::set_generation_config(const cosyvoice_generation_config_t* config)
{
    if (config->max_token_text_ratio < config->min_token_text_ratio
        || config->min_token_text_ratio < 0.f)
        return false;

    if (config->temperature <= 0.f)
        return false;

    if (config->sampling.win_size <= 0
        || config->sampling.top_k < 0
        || config->sampling.top_p < 0.f
        || config->sampling.top_p > 1.f
        || config->sampling.tau_r < 0.f)
        return false;

    shared->config = *config;
    worker->nucleus_probs.reset(new float[config->sampling.top_k * 2 + 1]);
    return true;
}

const char* cosyvoice_model::get_instruction_prefix()
{
    return shared->instruction_prefix.get();
}

void cosyvoice_model::get_context_params(cosyvoice_context_params_t* params)
{
    *params = shared->params;
    get_sampler(&params->sampler, &params->sampler_ctx);
}

const char* cosyvoice_model::get_architecture()
{
    return shared->architecture.get();
}

void cosyvoice_model::get_sampler(cosyvoice_sampler_t* sampler, void** sampler_ctx)
{
    if (using_builtin_sampler())
    {
        *sampler = nullptr;
        *sampler_ctx = nullptr;
    }
    else
    {
        *sampler = shared->params.sampler;
        *sampler_ctx = shared->params.sampler_ctx;
    }
}

void cosyvoice_model::set_sampler(cosyvoice_sampler_t sampler, void* sampler_ctx)
{
    if (sampler)
    {
        shared->params.sampler = sampler;
        shared->params.sampler_ctx = sampler_ctx;

    }
    else
    {
        shared->params.sampler = reinterpret_cast<cosyvoice_sampler_t>(cosyvoice_llm_sampler);
        shared->params.sampler_ctx = &worker->sampler_rng;
        reset_builtin_sampler_rng();
    }
}

static float* cosyvoice_default_noise_callback(
    cosyvoice_noise_callback_stage_t stage,
    uint32_t                         length,
    float*                           noise,
    cosyvoice_worker_context*        worker
)
{
    switch (stage)
    {
    case COSYVOICE_NOISE_CALLBACK_STAGE_BEFORE_FLOW:
    case COSYVOICE_NOISE_CALLBACK_STAGE_BEFORE_HIFT:
        if (length > worker->rand_noise_len)
        {
            auto new_noise = std::make_unique<float[]>(length);
            memcpy(new_noise.get(), worker->rand_noise.get(), worker->rand_noise_len * sizeof(float));

            std::normal_distribution<float> dist(0.0f, 1.0f);
            for (auto& i : std::span(new_noise.get() + worker->rand_noise_len, length - worker->rand_noise_len))
                i = dist(worker->noise_rng);
            new_noise.swap(worker->rand_noise);
            worker->rand_noise_len = length;
        }
        return worker->rand_noise.get();
    }
    return nullptr;
}

void cosyvoice_model::set_noise_callback(cosyvoice_noise_callback_t callback, void* callback_ctx)
{
    if (callback)
    {
        worker->noise_callback = callback;
        worker->noise_callback_ctx = callback_ctx;
    }
    else
    {
        worker->noise_callback = reinterpret_cast<cosyvoice_noise_callback_t>(cosyvoice_default_noise_callback);
        worker->noise_callback_ctx = worker;
    }
}

void cosyvoice_model::get_noise_callback(cosyvoice_noise_callback_t* callback, void** callback_ctx)
{
    if (worker->noise_callback == reinterpret_cast<cosyvoice_noise_callback_t>(cosyvoice_default_noise_callback))
    {
        *callback = nullptr;
        *callback_ctx = nullptr;
    }
    else
    {
        *callback = worker->noise_callback;
        *callback_ctx = worker->noise_callback_ctx;
    }
}

int cosyvoice_model::llm_sample_token()
{
    GGML_ASSERT(worker->llm_probs);
    return shared->params.sampler(
        reinterpret_cast<cosyvoice_llm_token_prob_t*>(worker->nucleus_probs.get() + 1),
        *reinterpret_cast<int*>(worker->nucleus_probs.get()),
        worker->probs.get(),
        static_cast<uint32_t>(get_speech_token_embed_weight()->ne[1]),
        &shared->config.sampling,
        worker->tokens.data(),
        static_cast<uint32_t>(worker->tokens.size()),
        shared->params.sampler_ctx);
}

void cosyvoice_model::llm_accept_token(int token)
{
    worker->tokens.push_back(token);
}

void cosyvoice_model::llm_clear_accepted_tokens()
{
    worker->tokens.clear();
}

uint32_t cosyvoice_model::llm_get_n_accepted_tokens()
{
    return static_cast<uint32_t>(worker->tokens.size());
}

const int* cosyvoice_model::llm_get_accepted_tokens()
{
    return worker->tokens.data();
}

ggml_status cosyvoice_model::get_last_status()
{
    return worker->status;
}

uint32_t cosyvoice_model::llm_get_kv_cache_len()
{
    return worker->kv_cache.cur_len;
}

bool cosyvoice_model::llm_set_kv_cache_len(uint32_t len)
{
    auto& cur_len = worker->kv_cache.cur_len;
    if (len <= cur_len)
    {
        cur_len = len;
        return true;
    }
    return false;
}
