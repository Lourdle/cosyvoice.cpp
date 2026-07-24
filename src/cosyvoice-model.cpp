#include "cosyvoice-model.h"
#include "common.h"

#include <cstring>
#include <span>
#include <mutex>

static float* cosyvoice_default_noise_callback(
    cosyvoice_noise_callback_stage_t stage,
    uint32_t                         length,
    float* noise,
    cosyvoice_model_shared* shared
)
{
    switch (stage)
    {
    case COSYVOICE_NOISE_CALLBACK_STAGE_BEFORE_FLOW:
    case COSYVOICE_NOISE_CALLBACK_STAGE_BEFORE_HIFT:
    {
        std::shared_lock read_lock(shared->noise_mutex);
        if (length <= shared->rand_noise_len && shared->rand_noise)
            return shared->rand_noise.get();
    }
    {
        std::unique_lock write_lock(shared->noise_mutex);
        if (length > shared->rand_noise_len)
        {
            std::unique_ptr<float[]> new_noise(new float[length]);
            if (shared->rand_noise)
                memcpy(new_noise.get(), shared->rand_noise.get(), shared->rand_noise_len * sizeof(float));

            std::normal_distribution<float> dist(0.0f, 1.0f);
            for (auto& i : std::span(new_noise.get() + shared->rand_noise_len, length - shared->rand_noise_len))
                i = dist(shared->noise_rng);
            shared->rand_noise.swap(new_noise);
            shared->rand_noise_len = length;
        }
        return shared->rand_noise.get();
    }
    }
    return nullptr;
}

void cosyvoice_init_default_context_params(cosyvoice_context_params_t* params)
{
    params->flow_use_flash_attn = true;
    params->llm_use_flash_attn = true;

    params->llm_kv_cache_type = COSYVOICE_KV_CACHE_TYPE_Q8_0;
    params->llm_allow_kv_cache_fallback = true;
    params->inference_buffer_policy = COSYVOICE_INFERENCE_BUFFER_POLICY_BALANCED;

    params->n_batch = 256;
    params->n_max_seq = COSYVOICE_DEFAULT_LLM_MAX_SEQ_LEN;

    params->seed = cosyvoice_generate_random_seed();
    params->builtin_sampler_rng_policy = COSYVOICE_BUILTIN_SAMPLER_RNG_POLICY_RESET_PER_SESSION;

    params->sampler = nullptr;
    params->sampler_ctx = nullptr;
}

cosyvoice_model_shared::cosyvoice_model_shared(const cosyvoice_context_params_v3_cpp& params)
    : params(params), ctx(nullptr), backend_uma(false), rand_noise_len(0), noise_callback(nullptr), noise_callback_ctx(nullptr) {}

cosyvoice_worker_context::cosyvoice_worker_context(ggml_backend_t backend)
    : backend(backend), cpu_backend(backend),
    ctx0(ggml_init(ggml_init_params{ .mem_size = ggml_graph_overhead() * kCosyVoiceGraphSize, .no_alloc = true })),
    gf(nullptr), llm_input(nullptr), llm_probs(nullptr), position_ids(nullptr), causal_mask(nullptr), llm_kv_cache(),
    status(GGML_STATUS_SUCCESS), prompt_crc32(0), sampler_seed(0), sampler(nullptr), sampler_ctx(nullptr), builtin_sampler_rng_policy(COSYVOICE_BUILTIN_SAMPLER_RNG_POLICY_RESET_PER_SESSION), nucleus_probs_capacity(0), nucleus_probs_len(0) {}

cosyvoice_model::cosyvoice_model(ggml_backend_t backend, const cosyvoice_context_params_v3_cpp& params)
    : shared(new cosyvoice_model_shared(params)), workers(reinterpret_cast<cosyvoice_worker_context*>(malloc(sizeof(cosyvoice_worker_context) * params.n_workers)))
{
    auto dev = ggml_backend_get_device(backend);
    bool dev_is_cpu = ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU;

    shared->noise_rng.seed(shared->params.seed);
    shared->noise_callback = reinterpret_cast<cosyvoice_noise_callback_t>(cosyvoice_default_noise_callback);
    shared->noise_callback_ctx = shared;

    auto init_worker = [&](cosyvoice_worker_context* cur)
    {
        worker = new (cur) cosyvoice_worker_context(backend);
        if (!dev_is_cpu)
        {
            worker->cpu_backend.release();
            worker->cpu_backend.reset(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr));
        }
        empty_buffer_cache();
        worker->config.temperature = 1.f;
        worker->config.max_token_text_ratio = 20.f;
        worker->config.min_token_text_ratio = 2.f;
        worker->builtin_sampler_rng_policy = shared->params.builtin_sampler_rng_policy;
        worker->sampler_seed = shared->noise_rng();
        worker->sampler_rng.seed(worker->sampler_seed);
        worker->sampler = reinterpret_cast<cosyvoice_sampler_t>(cosyvoice_llm_sampler);
        worker->sampler_ctx = workers;
    };

    init_worker(workers);
    for (size_t i = 1; i != params.n_workers; ++i)
    {
        backend = ggml_backend_dev_init(dev, nullptr);
        init_worker(workers + i);
    }

    set_sampler(params.sampler, params.sampler_ctx);
    worker->config.temperature = 1.f;
    worker->config.max_token_text_ratio = 20.f;
    worker->config.min_token_text_ratio = 2.f;

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

    op_caps.top_k = ggml_backend_supports_op(backend, ggml_top_k(ctx0, a, 5));

    op_caps.leaky_relu = ggml_backend_supports_op(backend, ggml_leaky_relu(ctx0, a, 0.01f, false));

    op_caps.sin = ggml_backend_supports_op(backend, ggml_sin(ctx0, a));
    op_caps.cos = ggml_backend_supports_op(backend, ggml_cos(ctx0, a));

    op_caps.arange = ggml_backend_supports_op(backend, ggml_arange(ctx0, 0.f, 10.f, 1.f));

    op_caps.elu = ggml_backend_supports_op(backend, ggml_elu(ctx0, a));
    op_caps.abs = ggml_backend_supports_op(backend, ggml_abs(ctx0, a));
    op_caps.floor = ggml_backend_supports_op(backend, ggml_floor(ctx0, a));

    {
        auto a2 = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 16, 4);
        auto b = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 16, 4);
        op_caps.acc = ggml_backend_supports_op(backend, ggml_acc(ctx0, a2, b, a2->nb[1], a2->nb[2], a2->nb[3], 0));
    }

    {
        ggml_tensor* w = ggml_new_tensor_3d(ctx0, GGML_TYPE_F16, 3, 4, 8);
        a = ggml_new_tensor_3d(ctx0, GGML_TYPE_F16, 16, 4, 1);
        a = ggml_im2col(ctx0, w, a, 1, 0, 0, 0, 1, 0, false, GGML_TYPE_F16);
        op_caps.im2col_f16 = ggml_backend_supports_op(backend, a);
    }

    worker = workers;
}

bool cosyvoice_model::using_builtin_sampler() const
{
    return worker->sampler == reinterpret_cast<cosyvoice_sampler_t>(cosyvoice_llm_sampler);
}

bool cosyvoice_model::reset_builtin_sampler_rng()
{
    if (using_builtin_sampler())
    {
        worker->sampler_rng.seed(worker->sampler_seed);
        return true;
    }
    return false;
}

cosyvoice_builtin_sampler_rng_policy_t cosyvoice_model::get_builtin_sampler_rng_policy()
{
    return worker->builtin_sampler_rng_policy;
}

uint32_t cosyvoice_model::get_chunk_tokens()
{
    return worker->chunk_size;
}

void cosyvoice_model::set_chunk_tokens(uint32_t n_tokens)
{
    worker->chunk_size = n_tokens;
}

bool cosyvoice_model::set_sampler_seed(uint32_t seed)
{
    worker->sampler_seed = seed;
    return reset_builtin_sampler_rng();
}

uint32_t cosyvoice_model::get_sampler_seed()
{
    return worker->sampler_seed;
}

bool cosyvoice_model::set_builtin_sampler_rng_policy(cosyvoice_builtin_sampler_rng_policy_t policy)
{
    if (using_builtin_sampler())
        switch (policy)
        {
        case COSYVOICE_BUILTIN_SAMPLER_RNG_POLICY_RESET_PER_SESSION:
        case COSYVOICE_BUILTIN_SAMPLER_RNG_POLICY_CONTINUE_ACROSS_SESSIONS:
            worker->builtin_sampler_rng_policy = policy;
            return true;
        }
    return false;
}

cosyvoice_model::~cosyvoice_model()
{
    if (get_ref_count() == 1)
    {
        for (uint32_t i = 0; i != shared->params.n_workers; ++i)
        {
            auto worker = workers + i;

            if (worker->backend == worker->cpu_backend)
                worker->cpu_backend.release();

            switch (shared->params.inference_buffer_policy)
            {
            case COSYVOICE_INFERENCE_BUFFER_POLICY_BALANCED:
            case COSYVOICE_INFERENCE_BUFFER_POLICY_SHARED:
                worker->llm_kv_buffer.release();
            }

            worker->~cosyvoice_worker_context();
        }

        delete shared;
        free(workers);
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

    {
        std::unique_lock lock(shared->noise_mutex);
        shared->rand_noise.reset();
        shared->rand_noise_len = 0;
    }

    worker->llm_kv_cache.clear_offloaded_cache();
    worker->prompt_crc32 = 0;
}

void cosyvoice_model_3::empty_buffer_cache()
{
    cosyvoice_model::empty_buffer_cache();

    if (cv3_worker->orig_max_seq_len != shared->params.n_max_seq)
    {
        shared->params.n_max_seq = cv3_worker->orig_max_seq_len;
        worker->llm_kv_buffer.reset(
            worker->llm_kv_cache.initialize_buffer(
                worker->backend.get(),
                static_cast<int>(cv3_shared->llm.layers[0].self_attn.k_proj.weight->ne[1] / cv3_shared->llm.num_key_value_heads),
                static_cast<int>(cv3_shared->llm.layers[0].self_attn.v_proj.weight->ne[1] / cv3_shared->llm.num_key_value_heads),
                shared->params.n_max_seq,
                1));

        switch (shared->params.inference_buffer_policy)
        {
        case COSYVOICE_INFERENCE_BUFFER_POLICY_BALANCED:
        case COSYVOICE_INFERENCE_BUFFER_POLICY_SHARED:
            cv3_worker->token2wav_buffer.release();
            cv3_worker->token2wav_buffer.reset(worker->llm_kv_buffer.get());
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
    usage->offloaded_kv_cache = worker->llm_kv_cache.get_offloaded_cache_size();
    usage->random_noise = sizeof(float) * shared->rand_noise_len;
    usage->kv_cache = worker->llm_kv_buffer.get() ? ggml_backend_buffer_get_size(worker->llm_kv_buffer.get()) : 0;
    usage->token2wav = cv3_worker->token2wav_buffer.get() ? ggml_backend_buffer_get_size(cv3_worker->token2wav_buffer.get()) : 0;
}

void cosyvoice_model_3::get_total_memory_usage(cosyvoice_memory_usage_t* usage)
{
    usage->parameters = ggml_backend_buffer_get_size(shared->buffer.get());
    for (uint32_t i = 0; i != shared->params.n_workers; ++i)
    {
        auto worker = workers + i;
        auto cv3_worker = cv3_workers + i;

        usage->buffers = ggml_backend_sched_get_buffer_size(worker->sched.get(), worker->backend.get());
        usage->cpu_buffers = ggml_backend_sched_get_buffer_size(worker->sched.get(), worker->cpu_backend.get());
        usage->offloaded_kv_cache = worker->llm_kv_cache.get_offloaded_cache_size();
        usage->random_noise = sizeof(float) * shared->rand_noise_len;
        usage->kv_cache = worker->llm_kv_buffer.get() ? ggml_backend_buffer_get_size(worker->llm_kv_buffer.get()) : 0;
        usage->token2wav = cv3_worker->token2wav_buffer.get() ? ggml_backend_buffer_get_size(cv3_worker->token2wav_buffer.get()) : 0;
    }
}

void cosyvoice_model_3::reset_shared_buffer(ggml_backend_buffer* new_buffer)
{
    cv3_worker->token2wav_buffer.reset(new_buffer);
    if (shared->params.inference_buffer_policy != COSYVOICE_INFERENCE_BUFFER_POLICY_DEDICATED)
    {
        worker->llm_kv_buffer.release();
        worker->llm_kv_buffer.reset(new_buffer);
        shared->params.n_max_seq = worker->llm_kv_cache.reset_buffer(new_buffer);
    }
}

cosyvoice_3_worker_context::cosyvoice_3_worker_context() :
    ctx1(ggml_init(ggml_init_params{ .mem_size = ggml_tensor_overhead() * 4, .no_alloc = true })) {}

cosyvoice_model_3::cosyvoice_model_3(ggml_backend_t backend, const cosyvoice_context_params_v3_cpp& params)
    : cosyvoice_model(backend, params), cv3_shared(new cosyvoice_model_3_shared), cv3_workers(new cosyvoice_3_worker_context[params.n_workers]())
{
    cv3_worker = cv3_workers;
}

cosyvoice_model_3::~cosyvoice_model_3()
{
    if (get_ref_count() == 1)
    {
        delete cv3_shared;
        delete[] cv3_workers;
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

void cosyvoice_model::get_default_generation_config(cosyvoice_generation_config_t* config)
{
    *config = shared->config;
}

void cosyvoice_model::get_generation_config(cosyvoice_generation_config_t* config)
{
    *config = worker->config;
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

    worker->config = *config;
    auto required_capacity = static_cast<uint32_t>(config->sampling.top_k * 2);
    if (worker->nucleus_probs_capacity < required_capacity)
    {
        worker->nucleus_probs.reset(new float[required_capacity]);
        worker->nucleus_probs_capacity = required_capacity;
    }
    worker->nucleus_probs_len = 0;
    return true;
}

const char* cosyvoice_model::get_instruction_prefix()
{
    return shared->instruction_prefix.get();
}

void cosyvoice_model::get_context_params(cosyvoice_context_params_t* params)
{
    *params = shared->params;
    params->seed = worker->sampler_seed;
    params->builtin_sampler_rng_policy = worker->builtin_sampler_rng_policy;
    get_sampler(&params->sampler, &params->sampler_ctx);
}

const char* cosyvoice_model::get_architecture()
{
    return shared->architecture.get();
}

bool cosyvoice_model::is_backend_uma()
{
    return shared->backend_uma;
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
        shared->params.sampler_ctx = workers;
        reset_builtin_sampler_rng();
    }
}

void cosyvoice_model::set_noise_callback(cosyvoice_noise_callback_t callback, void* callback_ctx)
{
    if (callback)
    {
        std::unique_lock lock(shared->noise_mutex);
        shared->noise_callback = callback;
        shared->noise_callback_ctx = callback_ctx;
    }
    else
    {
        std::unique_lock lock(shared->noise_mutex);
        shared->noise_callback = reinterpret_cast<cosyvoice_noise_callback_t>(cosyvoice_default_noise_callback);
        shared->noise_callback_ctx = shared;
    }
}

void cosyvoice_model::get_noise_callback(cosyvoice_noise_callback_t* callback, void** callback_ctx)
{
    std::shared_lock lock(shared->noise_mutex);
    if (shared->noise_callback == reinterpret_cast<cosyvoice_noise_callback_t>(cosyvoice_default_noise_callback))
    {
        *callback = nullptr;
        *callback_ctx = nullptr;
    }
    else
    {
        *callback = shared->noise_callback;
        *callback_ctx = shared->noise_callback_ctx;
    }
}


int cosyvoice_model::llm_sample_token()
{
    GGML_ASSERT(worker->llm_probs);
    return shared->params.sampler_ext(
        reinterpret_cast<cosyvoice_llm_token_prob_t*>(worker->nucleus_probs.get()),
        worker->nucleus_probs_len,
        worker->probs.get(),
        static_cast<uint32_t>(get_speech_token_embed_weight()->ne[1]),
        &worker->config.sampling,
        worker->tokens.data(),
        static_cast<uint32_t>(worker->tokens.size()),
        shared->params.sampler_ctx,
        static_cast<uint32_t>(worker - workers));
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

uint32_t cosyvoice_model::get_worker_no()
{
    return static_cast<uint32_t>(worker - workers);
}

uint32_t cosyvoice_model::get_n_workers()
{
    return shared->params.n_workers;
}

void cosyvoice_model::request_stop()
{
    if (worker->use_count.load(std::memory_order_acquire) == 0)
    {
        worker->stop_flag.store(false, std::memory_order_release);
        return;
    }

    worker->stop_flag.store(true, std::memory_order_release);

    std::unique_lock<std::mutex> lock(worker->cv_mutex);
    worker->cv.wait(lock, [this]() {
        return worker->use_count.load(std::memory_order_acquire) == 0;
    });

    worker->stop_flag.store(false, std::memory_order_release);
}

bool cosyvoice_model::stop_requested()
{
    return worker->stop_flag.exchange(false, std::memory_order_acq_rel);
}

uint32_t cosyvoice_model::llm_get_kv_cache_len()
{
    return worker->llm_kv_cache.cur_len;
}

bool cosyvoice_model::llm_set_kv_cache_len(uint32_t len)
{
    auto& cur_len = worker->llm_kv_cache.cur_len;
    if (len <= cur_len)
    {
        cur_len = len;
        return true;
    }
    return false;
}

void cosyvoice_model::llm_offload_kv_cache()
{
    auto sched = worker->sched.get();
    auto& kv_cache = worker->llm_kv_cache;
    ggml_backend_sched_reset(sched);
    kv_cache.offload_cache(worker->backend.get(), sched, kv_cache.cur_len);
}

void cosyvoice_model::llm_load_kv_cache()
{
    auto sched = worker->sched.get();
    auto& kv_cache = worker->llm_kv_cache;
    ggml_backend_sched_reset(sched);
    kv_cache.load_cache(worker->backend.get(), sched);
}
