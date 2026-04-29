#include "cosyvoice-model.h"
#include "cosyvoice-llm-kv-cache.h"

#include <cstring>
#include <span>
#include <chrono>

void cosyvoice_init_default_context_params(cosyvoice_context_params_t* params)
{
    params->flow_use_flash_attn = true;
    params->llm_use_flash_attn = true;

    params->llm_kv_cache_type = COSYVOICE_LLM_KV_CACHE_TYPE_Q8_0;
    params->llm_allow_kv_cache_fallback = true;
    params->inference_buffer_policy = COSYVOICE_INFERENCE_BUFFER_POLICY_BALANCED;

    params->n_batch = 256;
    params->n_max_seq = 2048;

    std::random_device rd;
    if (rd.entropy() == 0)
        params->seed = static_cast<uint32_t>(std::chrono::system_clock::now().time_since_epoch().count());
    else
        params->seed = rd();
    params->builtin_sampler_rng_policy = COSYVOICE_BUILTIN_SAMPLER_RNG_POLICY_RESET_PER_SESSION;

    params->sampler = nullptr;
    params->sampler_ctx = nullptr;
}

cosyvoice_model::cosyvoice_model(ggml_backend_t backend, const cosyvoice_context_params_t& params)
    : backend(backend), params(params), cpu_backend(backend),
    ctx(ggml_init(ggml_init_params{ .mem_size = ggml_graph_overhead() * GGML_DEFAULT_GRAPH_SIZE, .no_alloc = true })),
    ctx0(ggml_init(ggml_init_params{ .mem_size = ggml_graph_overhead() * (GGML_DEFAULT_GRAPH_SIZE + (params.flow_use_flash_attn ? 0 : GGML_DEFAULT_GRAPH_SIZE / 2)), .no_alloc = true })),
    gf(nullptr), llm_input(nullptr), llm_probs(nullptr), position_ids(nullptr), causal_mask(nullptr), kv_cache(nullptr),
    status(GGML_STATUS_SUCCESS), prompt_crc32(0), rand_noise_len(0)
{
    if (GGML_BACKEND_DEVICE_TYPE_CPU != ggml_backend_dev_type(ggml_backend_get_device(backend)))
    {
        cpu_backend.release();
        cpu_backend.reset(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr));
    }

    empty_buffer_cache();
    set_sampler(params.sampler, params.sampler_ctx);
    set_noise_callback(nullptr, nullptr);

    ggml_tensor* a = ggml_new_tensor_1d(ctx0.get(), GGML_TYPE_I32, 1);
    a = ggml_concat(ctx0.get(), a, a, 0);
    op_caps.concat_i32 = ggml_backend_supports_op(backend, a);

    a = ggml_new_tensor_3d(ctx0.get(), GGML_TYPE_F16, 11, 4, 51);
    a = ggml_repeat_4d(ctx0.get(), a, 11, 4, 51, 4);
    op_caps.repeat_f16 = ggml_backend_supports_op(backend, a);

    a = ggml_new_tensor_2d(ctx0.get(), GGML_TYPE_F32, 16, 4);
    a = ggml_pad(ctx0.get(), a, 4, 0, 0, 0);
    op_caps.pad = ggml_backend_supports_op(backend, a);

    a = ggml_new_tensor_1d(ctx0.get(), GGML_TYPE_F32, 16);
    a = ggml_pad_reflect_1d(ctx0.get(), a, 1, 0);
    op_caps.pad_reflect_1d = ggml_backend_supports_op(backend, a);

    {
        ggml_tensor* w = ggml_new_tensor_3d(ctx0.get(), GGML_TYPE_F16, 3, 4, 8);
        a = ggml_new_tensor_3d(ctx0.get(), GGML_TYPE_F16, 16, 4, 1);
        a = ggml_im2col(ctx0.get(), w, a, 1, 0, 0, 0, 1, 0, false, GGML_TYPE_F16);
        op_caps.im2col_f16 = ggml_backend_supports_op(backend, a);
    }
}

bool cosyvoice_model::using_builtin_sampler() const
{
    return params.sampler == reinterpret_cast<cosyvoice_sampler_t>(cosyvoice_llm_sampler);
}

bool cosyvoice_model::reset_builtin_sampler_rng()
{
    if (using_builtin_sampler())
    {
        sampler_rng.seed(params.seed);
        sampler_rng.seed(sampler_rng());
        return true;
    }
    return false;
}

cosyvoice_builtin_sampler_rng_policy_t cosyvoice_model::get_builtin_sampler_rng_policy()
{
    return params.builtin_sampler_rng_policy;
}

bool cosyvoice_model::set_sampler_seed(uint32_t seed)
{
    params.seed = seed;
    return reset_builtin_sampler_rng();
}

bool cosyvoice_model::set_builtin_sampler_rng_policy(cosyvoice_builtin_sampler_rng_policy_t policy)
{
    if (using_builtin_sampler())
        switch (policy)
        {
        case COSYVOICE_BUILTIN_SAMPLER_RNG_POLICY_RESET_PER_SESSION:
        case COSYVOICE_BUILTIN_SAMPLER_RNG_POLICY_CONTINUE_ACROSS_SESSIONS:
            params.builtin_sampler_rng_policy = policy;
            return true;
        }
    return false;
}

cosyvoice_model::~cosyvoice_model()
{
    delete kv_cache;

    if (backend == cpu_backend)
        cpu_backend.release();

    switch (params.inference_buffer_policy)
    {
    case COSYVOICE_INFERENCE_BUFFER_POLICY_BALANCED:
    case COSYVOICE_INFERENCE_BUFFER_POLICY_SHARED:
        kv_buffer.release();
    }
}

void cosyvoice_model::empty_buffer_cache()
{
    ggml_backend_t backends[] = { backend.get(), cpu_backend.get() };
    sched.reset(ggml_backend_sched_new(
        backends,
        nullptr,
        backend == cpu_backend ? 1 : 2,
        GGML_DEFAULT_GRAPH_SIZE * 2,
        true,
        true
    ));

    rand_noise.reset();
    rand_noise_len = 0;

    if (kv_cache)
        kv_cache->clear_offloaded_cache();
}

void cosyvoice_model_3::empty_buffer_cache()
{
    cosyvoice_model::empty_buffer_cache();

    if (orig_max_seq_len != params.n_max_seq)
    {
        params.n_max_seq = orig_max_seq_len;
        kv_buffer.reset(
            kv_cache->initialize_buffer(
                backend.get(),
                static_cast<int>(llm.layers[0].self_attn.k_proj.weight->ne[1] / llm.num_key_value_heads),
                static_cast<int>(llm.layers[0].self_attn.v_proj.weight->ne[1] / llm.num_key_value_heads),
                llm.num_attention_heads,
                llm.num_key_value_heads,
                params.n_max_seq,
                kv_type,
                this->params.llm_use_flash_attn));

        switch (params.inference_buffer_policy)
        {
        case COSYVOICE_INFERENCE_BUFFER_POLICY_BALANCED:
        case COSYVOICE_INFERENCE_BUFFER_POLICY_SHARED:
            token2wav_buffer.release();
            token2wav_buffer.reset(kv_buffer.get());
            break;
        case COSYVOICE_INFERENCE_BUFFER_POLICY_DEDICATED:
            token2wav_buffer.reset();
            break;
        default:
            GGML_ABORT("unexpected policy");
        }
    }
}

void cosyvoice_model_3::get_memory_usage(cosyvoice_memory_usage_t* usage)
{
    usage->parameters = ggml_backend_buffer_get_size(buffer.get());
    usage->buffers = ggml_backend_sched_get_buffer_size(sched.get(), backend.get());
    usage->cpu_buffers = ggml_backend_sched_get_buffer_size(sched.get(), cpu_backend.get());
    usage->offloaded_kv_cache = kv_cache->get_offloaded_cache_size();
    usage->random_noise = sizeof(float) * rand_noise_len;
    usage->kv_cache = kv_buffer.get() ? ggml_backend_buffer_get_size(kv_buffer.get()) : 0;
    usage->token2wav = token2wav_buffer.get() ? ggml_backend_buffer_get_size(token2wav_buffer.get()) : 0;
}

void cosyvoice_model_3::reset_shared_buffer(ggml_backend_buffer* new_buffer)
{
    token2wav_buffer.reset(new_buffer);
    if (params.inference_buffer_policy != COSYVOICE_INFERENCE_BUFFER_POLICY_DEDICATED)
    {
        kv_buffer.reset(new_buffer);
        params.n_max_seq = kv_cache->reset_buffer(new_buffer);
    }
}

cosyvoice_model_3::cosyvoice_model_3(ggml_backend_t backend, const cosyvoice_context_params_t& params)
    : cosyvoice_model(backend, params),
    ctx1(ggml_init(ggml_init_params{ .mem_size = ggml_tensor_overhead() * 4, .no_alloc = true }))
{}

void CausalHiFTGenerator::set_rand_ini(const float* data) const
{
    ggml_backend_tensor_set(m_source.l_sin_gen.rand_ini, data, 0, (nb_harmonics + 1) * sizeof(float));
}

bool cosyvoice_model_3::llm_is_stop_token(int token_id)
{
    return stop_tokens.find(token_id) != stop_tokens.end();
}

const ggml_tensor* cosyvoice_model_3::get_word_token_embed_weight()
{
    return llm.embed_tokens_weight;
}

const ggml_tensor* cosyvoice_model_3::get_speech_token_embed_weight()
{
    return llm.speech_embedding_weight;
}

uint32_t cosyvoice_model_3::get_hift_rand_ini_len()
{
    return hift.nb_harmonics + 1;
}

void cosyvoice_model_3::set_hift_rand_ini(const float* data)
{
    hift.set_rand_ini(data);
}

uint32_t cosyvoice_model_3::get_sample_rate()
{
    return hift.sampling_rate;
}

void cosyvoice_model::get_generation_config(cosyvoice_generation_config_t* config)
{
    *config = this->config;
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

    this->config = *config;
    nucleus_probs.reset(new float[config->sampling.top_k * 2 + 1]);
    return true;
}

const char* cosyvoice_model::get_instruction_prefix()
{
    return instruction_prefix.get();
}

void cosyvoice_model::get_context_params(cosyvoice_context_params_t* params)
{
    *params = this->params;
    get_sampler(&params->sampler, &params->sampler_ctx);
}

const char* cosyvoice_model::get_architecture()
{
    return architecture.get();
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
        *sampler = this->params.sampler;
        *sampler_ctx = this->params.sampler_ctx;
    }
}

void cosyvoice_model::set_sampler(cosyvoice_sampler_t sampler, void* sampler_ctx)
{
    if (sampler)
    {
        this->params.sampler = sampler;
        this->params.sampler_ctx = sampler_ctx;

    }
    else
    {
        this->params.sampler = reinterpret_cast<cosyvoice_sampler_t>(cosyvoice_llm_sampler);
        this->params.sampler_ctx = &sampler_rng;
        reset_builtin_sampler_rng();
    }
}

static float* cosyvoice_default_noise_callback(
    cosyvoice_noise_callback_stage_t stage,
    uint32_t                         length,
    float*                           noise,
    cosyvoice_model*                 model
)
{
    switch (stage)
    {
    case COSYVOICE_NOISE_CALLBACK_STAGE_BEFORE_FLOW:
    case COSYVOICE_NOISE_CALLBACK_STAGE_BEFORE_HIFT:
        if (length > model->rand_noise_len)
        {
            auto new_noise = std::make_unique<float[]>(length);
            memcpy(new_noise.get(), model->rand_noise.get(), model->rand_noise_len * sizeof(float));

            std::normal_distribution<float> dist(0.0f, 1.0f);
            for (auto& i : std::span(new_noise.get() + model->rand_noise_len, length - model->rand_noise_len))
                i = dist(model->noise_rng);
            new_noise.swap(model->rand_noise);
            model->rand_noise_len = length;
        }
        return model->rand_noise.get();
    }
    return nullptr;
}

void cosyvoice_model::set_noise_callback(cosyvoice_noise_callback_t callback, void* callback_ctx)
{
    if (callback)
    {
        this->noise_callback = callback;
        this->noise_callback_ctx = callback_ctx;
    }
    else
    {
        this->noise_callback = reinterpret_cast<cosyvoice_noise_callback_t>(cosyvoice_default_noise_callback);
        this->noise_callback_ctx = this;
    }
}

void cosyvoice_model::get_noise_callback(cosyvoice_noise_callback_t* callback, void** callback_ctx)
{
    if (noise_callback == reinterpret_cast<cosyvoice_noise_callback_t>(cosyvoice_default_noise_callback))
    {
        *callback = nullptr;
        *callback_ctx = nullptr;
    }
    else
    {
        *callback = noise_callback;
        *callback_ctx = noise_callback_ctx;
    }
}

int cosyvoice_model::llm_sample_token()
{
    GGML_ASSERT(llm_probs);
    return params.sampler(
        reinterpret_cast<cosyvoice_llm_token_prob_t*>(nucleus_probs.get() + 1),
        *reinterpret_cast<int*>(nucleus_probs.get()),
        probs.get(),
        static_cast<uint32_t>(get_speech_token_embed_weight()->ne[1]),
        &config.sampling,
        tokens.data(),
        static_cast<uint32_t>(tokens.size()),
        params.sampler_ctx);
}

void cosyvoice_model::llm_accept_token(int token)
{
    tokens.push_back(token);
}

void cosyvoice_model::llm_clear_accepted_tokens()
{
    tokens.clear();
}

uint32_t cosyvoice_model::llm_get_n_accepted_tokens()
{
    return static_cast<uint32_t>(tokens.size());
}

const int* cosyvoice_model::llm_get_accepted_tokens()
{
    return tokens.data();
}

ggml_status cosyvoice_model::get_last_status()
{
    return status;
}
