#include "cosyvoice-internal.h"
#include "cosyvoice-tokenizer.h"
#include "cosyvoice-model.h"
#include "cosyvoice-loader.h"
#include "common.h"

#include <ggml-backend.h>

#include <chrono>
#include <random>
#include <thread>

uint32_t cosyvoice_generate_random_seed()
{
    std::random_device rd;
    uint32_t seed;
    if (rd.entropy() == 0)
        seed = static_cast<uint32_t>(std::chrono::system_clock::now().time_since_epoch().count());
    else
        seed = rd();

    std::mt19937 gen(seed);
    std::uniform_int_distribution<uint32_t> dist(0, UINT32_MAX);
    gen.seed(dist(gen) ^ gen() ^ seed);
    return dist(gen);
}

use_count_guard::use_count_guard(cosyvoice_model* model) : worker(model->worker)
{
    worker->use_count.fetch_add(1, std::memory_order_relaxed);
}

use_count_guard::use_count_guard(cosyvoice_context* ctx)
    : use_count_guard(dynamic_cast<cosyvoice_model*>(ctx)) {}

use_count_guard::~use_count_guard()
{
    if (worker->use_count.fetch_sub(1, std::memory_order_release) == 1)
        worker->cv.notify_all();
}

#ifdef _WIN32
    #define NOMINMAX
    #include <Windows.h>

static void debug_output(const char* text)
{
    bool has_utf8 = false;
    int cbMultiByte = 0;
    for (auto p = text; *p; ++p)
    {
        if (static_cast<uint8_t>(*p) >= uint8_t(0x80))
            has_utf8 = true;
        ++cbMultiByte;
    }
    if (has_utf8)
    {
        int cchWideChar = MultiByteToWideChar(CP_UTF8, 0, text, cbMultiByte, nullptr, 0);
        auto OutputString = std::make_unique<WCHAR[]>(cchWideChar);
        MultiByteToWideChar(CP_UTF8, 0, text, cbMultiByte, OutputString.get(), cchWideChar);
        OutputDebugStringW(OutputString.get());
    }
    else
        OutputDebugStringA(text);
}
#else
static inline void debug_output(const char* text) {}
#endif

struct cosyvoice_internal_context : cosyvoice_context
{
    virtual ~cosyvoice_internal_context() = default;
};

struct cosyvoice_context_3 : cosyvoice_internal_context, cosyvoice_tokenizer, cosyvoice_model_3
{
    cosyvoice_context_3(const cosyvoice_context_params_v3_cpp& params, ggml_backend_t backend) :
        cosyvoice_model_3(backend, params) {}
};

void cosyvoice_init_backend()
{
    ggml_time_init();
    ggml_log_set(cosyvoice_log_callback_default, nullptr);

    ggml_init_params params{};
    ggml_context* ctx = ggml_init(params);
    ggml_free(ctx);
}

void cosyvoice_init_backend_from_path(const char* dir_path)
{
    ggml_backend_load_all_from_path(dir_path);
    cosyvoice_init_backend();
}

void cosyvoice_log_callback_default(ggml_log_level level, const char* text, void* user_data)
{
#ifdef _WIN32
    UINT uCodePage = GetConsoleOutputCP();
    if (uCodePage != CP_UTF8)
        SetConsoleCP(CP_UTF8);
#endif
    switch (level)
    {
    case GGML_LOG_LEVEL_NONE:
    case GGML_LOG_LEVEL_INFO:
    case GGML_LOG_LEVEL_DEBUG:
#ifdef _DEBUG
        fputs(text, stdout);
        fflush(stdout);
#endif
        if (level == GGML_LOG_LEVEL_DEBUG)
            debug_output(text);
        break;
    case GGML_LOG_LEVEL_WARN:
        fputs("\033[93m", stdout);
        fputs(text, stdout);
        fputs("\033[0m", stdout);
        debug_output(text);
        fflush(stdout);
        break;
    case GGML_LOG_LEVEL_ERROR:
        fputs("\033[31m", stderr);
        fputs(text, stderr);
        fputs("\033[0m", stderr);
        debug_output(text);
        fflush(stderr);
    }

#ifdef _WIN32
    if (uCodePage != CP_UTF8)
        SetConsoleCP(uCodePage);
#endif
}

cosyvoice_context_t cosyvoice_load_ext(const void* data, size_t size, const cosyvoice_context_params_t* params, ggml_backend_t backend, uint32_t n_threads, uint32_t version)
{
    gguf_parser parser;
    gguf_loader loader(parser, data, size);
    if (!loader) return nullptr;

    cosyvoice_context_params_v3_cpp params_v3 = {};
    params_v3.cosyvoice_context_params_t::operator=(*params);

    if (version >= COSYVOICE_CONTEXT_PARAMS_V2_VERSION)
        params_v3.n_workers = std::max(1u, reinterpret_cast<const cosyvoice_context_params_v2_t*>(params)->n_workers);
    else
        params_v3.n_workers = 1;

    if (version >= COSYVOICE_CONTEXT_PARAMS_V3_VERSION)
    {
        auto* p_v3 = reinterpret_cast<const cosyvoice_context_params_v3_t*>(params);
        if (params_v3.dit_kv_fixed_slots == 0 && params_v3.dit_kv_offloadable_slots != 0)
        {
            params_v3.dit_kv_fixed_slots = 1;
            --params_v3.dit_kv_offloadable_slots;
        }

        params_v3.dit_kv_cache_type = p_v3->dit_kv_cache_type;
        params_v3.dit_kv_fixed_slots = p_v3->dit_kv_fixed_slots;
        params_v3.dit_kv_offloadable_slots = p_v3->dit_kv_offloadable_slots;
        params_v3.dit_allow_kv_cache_fallback = p_v3->dit_allow_kv_cache_fallback;
        params_v3.dit_kv_cache_length = p_v3->dit_kv_cache_length;
        if (params_v3.dit_kv_cache_length == 0)
            params_v3.dit_kv_cache_length = params->n_max_seq * 10;
    }
    else
    {
        params_v3.dit_kv_fixed_slots = 0;
        params_v3.dit_kv_offloadable_slots = 0;
        params_v3.dit_kv_cache_length = params->n_max_seq * 10;
        params_v3.dit_allow_kv_cache_fallback = true;
        params_v3.dit_kv_cache_type = COSYVOICE_MAKE_SEPARATE_KV_CACHE(COSYVOICE_KV_CACHE_TYPE_Q8_0, COSYVOICE_KV_CACHE_TYPE_Q4_0, COSYVOICE_KV_CACHE_TYPE_Q8_0);
    }

    auto ctx = new cosyvoice_context_3(params_v3,
        backend ? backend : ggml_backend_init_best()
    );
    ctx->cosyvoice_model_3::load(loader);
    ctx->cosyvoice_tokenizer::load(loader);

    auto ggml_backend_set_n_threads = reinterpret_cast<ggml_backend_set_n_threads_t>(ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(ggml_backend_get_device(ctx->worker->cpu_backend.get())), "ggml_backend_set_n_threads"));
    if (n_threads == 0)
        n_threads = std::max<uint32_t>(1, std::thread::hardware_concurrency() / params_v3.n_workers);
    if (n_threads != 0)
        ggml_backend_set_n_threads(ctx->worker->cpu_backend.get(), n_threads);

    return ctx;
}

cosyvoice_context_t cosyvoice_load_from_file_ext(
    const char* filename,
    const cosyvoice_context_params_t* params,
    ggml_backend_t backend,
    uint32_t n_threads,
    uint32_t version)
{
    file_mmap mmap(filename);
    return mmap ? cosyvoice_load_ext(mmap.data(), mmap.size(), params, backend, n_threads, version) : nullptr;
}

cosyvoice_context_t cosyvoice_load_from_file(const char* filename)
{
    cosyvoice_context_params_t params;
    cosyvoice_init_default_context_params(&params);
    return cosyvoice_load_from_file_with_params(filename, &params);
}

cosyvoice_context_t cosyvoice_load_from_file_with_params(const char* filename, const cosyvoice_context_params_t* params)
{
    return cosyvoice_load_from_file_ext(filename, params, nullptr, 0);
}

cosyvoice_context_t cosyvoice_load_from_file_with_params_v2(const char* filename, const cosyvoice_context_params_v2_t* params)
{
    return cosyvoice_load_from_file_ext(filename, params, nullptr, 0);
}

cosyvoice_context_t cosyvoice_load_from_file_with_params_v3(const char* filename, const cosyvoice_context_params_v3_t* params)
{
    return cosyvoice_load_from_file_ext(filename, params, nullptr, 0);
}

cosyvoice_context_t cosyvoice_duplicate_context(cosyvoice_context_t ctx)
{
    if (auto cv3_ctx = dynamic_cast<cosyvoice_context_3*>(ctx); cv3_ctx)
        return new cosyvoice_context_3(*cv3_ctx);
    return nullptr;
}

void cosyvoice_free(cosyvoice_context_t ctx)
{
    delete static_cast<cosyvoice_internal_context*>(ctx);
}

cosyvoice_tokenizer_context_t cosyvoice_tokenizer_load_from_file(const char* filename)
{
    file_mmap mmap(filename);
    if (!mmap)
        return nullptr;

    gguf_parser parser;
    if (!parser.parse(
        static_cast<const uint8_t*>(mmap.data()), mmap.size(), nullptr))
        return nullptr;

    gguf_metadata_loader loader(&parser);
    auto ctx = new cosyvoice_tokenizer();
    ctx->load(loader);
    return ctx;
}

cosyvoice_tokenizer_context_t cosyvoice_tokenizer_load(const void* data, size_t size)
{
    gguf_parser parser;
    if (!parser.parse(
        static_cast<const uint8_t*>(data), size, nullptr))
        return nullptr;

    gguf_metadata_loader loader(&parser);
    auto ctx = new cosyvoice_tokenizer();
    ctx->load(loader);
    return ctx;
}

bool cosyvoice_llm_prefill(
    cosyvoice_context_t ctx,
    ggml_type type,
    const void* data,
    uint32_t n_tokens
)
{
    return ctx->llm_prefill(type, data, n_tokens);
}

bool cosyvoice_llm_decode(cosyvoice_context_t ctx, ggml_type type, const void* data)
{
    return ctx->llm_decode(type, data);
}

void cosyvoice_llm_prepare_probs(cosyvoice_context_t ctx, bool allow_stop_tokens)
{
    ctx->llm_prepare_probs(allow_stop_tokens);
}

uint32_t cosyvoice_llm_get_kv_cache_len(cosyvoice_context_t ctx)
{
    return ctx->llm_get_kv_cache_len();
}

bool cosyvoice_llm_set_kv_cache_len(cosyvoice_context_t ctx, uint32_t len)
{
    return ctx->llm_set_kv_cache_len(len);
}

void cosyvoice_llm_offload_kv_cache(cosyvoice_context_t ctx)
{
    ctx->llm_offload_kv_cache();
}

void cosyvoice_llm_load_kv_cache(cosyvoice_context_t ctx)
{
    ctx->llm_load_kv_cache();
}

int cosyvoice_llm_sample_token(cosyvoice_context_t ctx)
{
    return ctx->llm_sample_token();
}

bool cosyvoice_llm_is_stop_token(cosyvoice_context_t ctx, int token_id)
{
    return ctx->llm_is_stop_token(token_id);
}

void cosyvoice_llm_accept_token(cosyvoice_context_t ctx, int token_id)
{
    ctx->llm_accept_token(token_id);
}

void cosyvoice_llm_clear_accepted_tokens(cosyvoice_context_t ctx)
{
    ctx->llm_clear_accepted_tokens();
}

uint32_t cosyvoice_llm_get_n_accepted_tokens(cosyvoice_context_t ctx)
{
    return ctx->llm_get_n_accepted_tokens();
}

const int* cosyvoice_llm_get_accepted_tokens(cosyvoice_context_t ctx)
{
    return ctx->llm_get_accepted_tokens();
}

bool cosyvoice_llm_job(cosyvoice_context_t ctx, const int* text, uint32_t text_len, cosyvoice_prompt_t prompt)
{
    return ctx->llm_job(text, text_len, prompt);
}

bool cosyvoice_llm_job_ext(cosyvoice_context_t ctx, const int* text, uint32_t text_len, cosyvoice_prompt_t prompt, uint32_t max_new_tokens, bool* final)
{
    return ctx->llm_job_ext(text, text_len, prompt, max_new_tokens, final);
}

bool cosyvoice_token2wav(cosyvoice_context_t ctx, const int* token_ids, uint32_t n_tokens, float speed, cosyvoice_prompt_t prompt, cosyvoice_generated_speech_ptr generated_speech)
{
    return ctx->token2wav(token_ids, n_tokens, speed, prompt, generated_speech);
}

bool cosyvoice_token2wav_ext(cosyvoice_context_t ctx, const int* token_ids, uint32_t n_tokens, float speed, cosyvoice_prompt_t prompt, uint32_t* offset, bool streaming, bool finalize, cosyvoice_generated_speech_ptr result)
{
    return ctx->token2wav_ext(token_ids, n_tokens, speed, prompt, offset, streaming, finalize, result);
}

static bool check_length(cosyvoice_prompt_t prompt, uint32_t text_len, uint32_t n_max_seq)
{
    // Defensive guard: bail out before llm_prefill / llm_decode would refuse the request because
    // the prefill layout exceeds n_max_seq. Without this, llm_job throws and emits an error log
    // for every doomed call; rejecting up-front lets the caller know the input is too long.
    //
    // The guard only looks at the predicted *fresh* prefill length, not the current KV-cache
    // length. llm_job internally truncates the cache back to either SOS or SOS+prompt_text
    // before re-prefilling, so a large residual cache from a previous call (e.g. when callers
    // synthesize chunked text via tts_job) is not a real budget consumer.
    const uint32_t prompt_text_len = static_cast<uint32_t>(prompt->prompt_text.size());
    const uint32_t prompt_speech_len = prompt->llm_prompt_speech_tokens.second;
    // Worst-case prefill: SOS(1) + prompt_text + text + (task_token(1) + prompt_speech_tokens) when present.
    uint64_t prefill_len = 1ull + prompt_text_len + text_len;
    if (prompt_speech_len != 0)
        prefill_len += 1ull + prompt_speech_len;
    if (prefill_len + 1 > n_max_seq)
        return false;
    return true;
}

bool cosyvoice_tts(cosyvoice_context_t ctx, const int* text, uint32_t text_len, float speed, cosyvoice_prompt_t prompt, cosyvoice_generated_speech_ptr result)
{
    use_count_guard guard(ctx);

    {
        cosyvoice_context_params_t params;
        ctx->get_context_params(&params);
        if (!check_length(prompt, text_len, params.n_max_seq))
        {
            result->data = nullptr;
            result->length = 0;
            return false;
        }
    }

    if (ctx->llm_job(text, text_len, prompt)
        && ctx->token2wav(ctx->llm_get_accepted_tokens(), ctx->llm_get_n_accepted_tokens(), speed, prompt, result))
    {
        // Check if stop was requested during token2wav
        if (ctx->stop_requested())
        {
            result->data = nullptr;
            result->length = 0;
            return false;
        }
        return true;
    }
    // Clear stop flag if generation failed due to stop request
    ctx->stop_requested();
    result->data = nullptr;
    result->length = 0;
    return false;
}

bool cosyvoice_tts_stream(cosyvoice_context_t ctx, const int* text, uint32_t text_len, float speed, cosyvoice_prompt_t prompt, cosyvoice_tts_audio_callback_t callback, void* user_data)
{
    use_count_guard guard(ctx);

    cosyvoice_context_params_t params;
    ctx->get_context_params(&params);
    if (!check_length(prompt, text_len, params.n_max_seq))
        return false;

    const auto chunk_tokens = ctx->get_chunk_tokens();

    ctx->llm_clear_accepted_tokens();
    uint32_t offset = 0;
    uint32_t n_tokens = 0;
    uint32_t last_length = 0;
    bool final = false;
    do
    {
        if (ctx->stop_requested())
            return false;

        if (!final)
            if (!ctx->llm_job_ext(text, text_len, prompt, chunk_tokens + 1, &final))
                return false;

        n_tokens = std::min(n_tokens + chunk_tokens, ctx->llm_get_n_accepted_tokens());
        text = nullptr;
        if (params.inference_buffer_policy == COSYVOICE_INFERENCE_BUFFER_POLICY_BALANCED)
            ctx->llm_offload_kv_cache();

        cosyvoice_generated_speech result = {};
        if (!ctx->token2wav_ext(ctx->llm_get_accepted_tokens(), n_tokens, speed, prompt, &offset, true, ctx->llm_get_n_accepted_tokens() == n_tokens, &result))
            return false;

        if (result.data && result.length > last_length
            && !callback(result.data + last_length, result.length - last_length, user_data))
            return false;
        last_length = result.length;
    } while (n_tokens != ctx->llm_get_n_accepted_tokens());

    return true;
}

uint32_t cosyvoice_get_chunk_tokens(cosyvoice_context_t ctx)
{
    return ctx->get_chunk_tokens();
}

void cosyvoice_set_chunk_tokens(cosyvoice_context_t ctx, uint32_t n_tokens)
{
    return ctx->set_chunk_tokens(n_tokens);
}

ggml_status cosyvoice_get_last_status(cosyvoice_context_t ctx)
{
    return ctx->get_last_status();
}

void cosyvoice_get_memory_usage(cosyvoice_context_t ctx, cosyvoice_memory_usage_t* usage)
{
    ctx->get_memory_usage(usage);
}

void cosyvoice_get_total_memory_usage(cosyvoice_context_t ctx, cosyvoice_memory_usage_t* usage)
{
    return ctx->get_total_memory_usage(usage);
}

void cosyvoice_empty_buffer_cache(cosyvoice_context_t ctx)
{
    ctx->empty_buffer_cache();
}

void cosyvoice_request_stop(cosyvoice_context_t ctx)
{
    ctx->request_stop();
}

bool cosyvoice_stop_requested(cosyvoice_context_t ctx)
{
    return ctx->stop_requested();
}

void cosyvoice_set_noise_callback(cosyvoice_context_t ctx, cosyvoice_noise_callback_t callback, void* callback_ctx)
{
    ctx->set_noise_callback(callback, callback_ctx);
}

void cosyvoice_get_noise_callback(cosyvoice_context_t ctx, cosyvoice_noise_callback_t* callback, void** callback_ctx)
{
    ctx->get_noise_callback(callback, callback_ctx);
}

uint32_t cosyvoice_get_hift_rand_ini_len(cosyvoice_context_t ctx)
{
    return ctx->get_hift_rand_ini_len();
}

void cosyvoice_set_hift_rand_ini(cosyvoice_context_t ctx, const float* data)
{
    ctx->set_hift_rand_ini(data);
}

const ggml_tensor* cosyvoice_get_word_token_embed_weight(cosyvoice_context_t ctx)
{
    return ctx->get_word_token_embed_weight();
}

const ggml_tensor* cosyvoice_get_speech_token_embed_weight(cosyvoice_context_t ctx)
{
    return ctx->get_speech_token_embed_weight();
}

uint32_t cosyvoice_tokenize(cosyvoice_tokenizer_context_t ctx, const char* text, cosyvoice_tokenization_result_t result, bool parse_special)
{
    return ctx->tokenize(text, result, parse_special);
}

uint32_t cosyvoice_tokenize_ext(cosyvoice_tokenizer_context_t ctx, const char* text, uint32_t text_len, cosyvoice_tokenization_result_t result, bool parse_special)
{
    return ctx->tokenize(text, text_len, result, parse_special);
}

void cosyvoice_tokenizer_free(cosyvoice_tokenizer_context_t ctx)
{
    delete dynamic_cast<cosyvoice_tokenizer*>(ctx);
}

cosyvoice_tokenizer_context_t cosyvoice_get_tokenizer(cosyvoice_context_t ctx)
{
    return ctx;
}

void cosyvoice_get_default_generation_config(cosyvoice_context_t ctx, cosyvoice_generation_config_t* config)
{
    return ctx->get_default_generation_config(config);
}

void cosyvoice_get_generation_config(cosyvoice_context_t ctx, cosyvoice_generation_config_t* config)
{
    ctx->get_generation_config(config);
}

bool cosyvoice_set_generation_config(cosyvoice_context_t ctx, const cosyvoice_generation_config_t* config)
{
    return ctx->set_generation_config(config);
}

uint32_t cosyvoice_get_sample_rate(cosyvoice_context_t ctx)
{
    return ctx->get_sample_rate();
}

void cosyvoice_get_context_params(cosyvoice_context_t ctx, cosyvoice_context_params_t* params)
{
    ctx->get_context_params(params);
}

uint32_t cosyvoice_get_n_workers(cosyvoice_context_t ctx)
{
    return ctx->get_n_workers();
}

uint32_t cosyvoice_get_worker_no(cosyvoice_context_t ctx)
{
    return ctx->get_worker_no();
}

bool cosyvoice_set_worker_no(cosyvoice_context_t ctx, uint32_t worker_no)
{
    return ctx->set_worker_no(worker_no);
}

const char* cosyvoice_get_architecture(cosyvoice_context_t ctx)
{
    return ctx->get_architecture();
}

bool cosyvoice_is_backend_uma(cosyvoice_context_t ctx)
{
    return ctx->is_backend_uma();
}

void cosyvoice_set_sampler(cosyvoice_context_t ctx, cosyvoice_sampler_t sampler, void* sampler_ctx)
{
    return ctx->set_sampler(sampler, sampler_ctx);
}

void cosyvoice_get_sampler(cosyvoice_context_t ctx, cosyvoice_sampler_t* sampler, void** sampler_ctx)
{
    return ctx->get_sampler(sampler, sampler_ctx);
}

cosyvoice_builtin_sampler_rng_policy_t cosyvoice_get_builtin_sampler_rng_policy(cosyvoice_context_t ctx)
{
    return ctx->get_builtin_sampler_rng_policy();
}

bool cosyvoice_set_sampler_seed(cosyvoice_context_t ctx, uint32_t seed)
{
    return ctx->set_sampler_seed(seed);
}

uint32_t cosyvoice_get_sampler_seed(cosyvoice_context_t ctx)
{
    return ctx->get_sampler_seed();
}

bool cosyvoice_set_builtin_sampler_rng_policy(cosyvoice_context_t ctx, cosyvoice_builtin_sampler_rng_policy_t policy)
{
    return ctx->set_builtin_sampler_rng_policy(policy);
}

cosyvoice_tokenization_result_t cosyvoice_tokenization_result_create()
{
    return cosyvoice_tokenization_result_t(new cosyvoice_tokenization_result_impl());
}

void cosyvoice_tokenization_result_free(cosyvoice_tokenization_result_t result)
{
    delete static_cast<cosyvoice_tokenization_result_impl*>(result);
}

int* cosyvoice_tokenization_result_get_tokens(cosyvoice_tokenization_result_t result)
{
    return result->get_tokens();
}

uint32_t cosyvoice_tokenization_result_get_n_tokens(cosyvoice_tokenization_result_t result)
{
    return result->get_n_tokens();
}

const char* cosyvoice_get_instruction_prefix(cosyvoice_context_t ctx)
{
    return ctx->get_instruction_prefix();
}

void cosyvoice_call_ggml_log_callback(ggml_log_level level, const char* message)
{
    ggml_log_callback callback;
    void* user_data;
    ggml_log_get(&callback, &user_data);
    callback(level, message, user_data);
}

bool cosyvoice_save_wav(const char* filename, const float* data, uint32_t data_len, uint32_t sample_rate)
{
    auto f = open_ofstream_utf8(filename);
    if (!f) return false;

    struct {
        char     riff[4] = { 'R', 'I', 'F', 'F' };
        uint32_t riff_size;
        char     wave[4] = { 'W', 'A', 'V', 'E' };
        char     fmt[4] = { 'f', 'm', 't', ' ' };
        uint32_t fmt_size = 16;
        uint16_t audio_format = 3; // float
        uint16_t num_channels = 1;
        uint32_t sample_rate;
        uint32_t byte_rate;
        uint16_t block_align;
        uint16_t bits_per_sample = 32;
        char     data[4] = { 'd', 'a', 't', 'a' };
        uint32_t data_size;
    } header = { .sample_rate = sample_rate };
    header.byte_rate = sample_rate * header.num_channels * header.bits_per_sample / 8;
    header.block_align = header.num_channels * header.bits_per_sample / 8;
    header.data_size = static_cast<uint32_t>(data_len * header.num_channels * header.bits_per_sample / 8);
    header.riff_size = 4 + (8 + header.fmt_size) + (8 + header.data_size);

    f.write(reinterpret_cast<const char*>(&header), static_cast<std::streamsize>(sizeof(header)));
    f.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(sizeof(float) * data_len));
    return f.good();
}
