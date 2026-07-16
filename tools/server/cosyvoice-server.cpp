#include "cosyvoice-server.h"
#include "tool_common_cosyvoice.h"

#include <ggml-backend.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <utility>

struct voice_prompt_option
{
    std::string voice;
    std::string prompt_speech;
};

struct server_options
{
    std::string model;
    std::string host = "127.0.0.1";
    std::string api_key;
    std::string served_model_name;
    std::string backend_path;
    std::string backend = "auto";
    uint16_t port = 8080;
    bool has_served_model_name = false;
    std::vector<voice_prompt_option> voice_prompts;

    std::string single_voice = "alloy";
    std::string single_prompt_speech;
    bool has_single_voice_arg = false;

    // Mode selection
    enum class run_mode { unspecified, webui, api };
    run_mode mode = run_mode::unspecified;

    std::string frontend_model;
    std::string speech_tokenizer;
    std::string campplus;

    uint32_t max_llm_len = COSYVOICE_DEFAULT_LLM_MAX_SEQ_LEN;
    uint32_t n_threads = 0;
    uint32_t concurrency = 1;
    bool has_inference_buffer_policy = false;
    cosyvoice_inference_buffer_policy_t inference_buffer_policy = COSYVOICE_INFERENCE_BUFFER_POLICY_BALANCED;

    bool has_seed = false;
    uint32_t seed = 0;

    bool has_llm_kv_cache_type = false;
    cosyvoice_kv_cache_type_t llm_kv_cache_type = COSYVOICE_MAKE_SEPARATE_KV_CACHE(
        COSYVOICE_KV_CACHE_TYPE_Q8_0,
        COSYVOICE_KV_CACHE_TYPE_Q4_0,
        COSYVOICE_KV_CACHE_TYPE_Q8_0);

    bool has_dit_kv_cache_type = false;
    cosyvoice_kv_cache_type_t dit_kv_cache_type = COSYVOICE_MAKE_SEPARATE_KV_CACHE(
        COSYVOICE_KV_CACHE_TYPE_Q8_0,
        COSYVOICE_KV_CACHE_TYPE_Q4_0,
        COSYVOICE_KV_CACHE_TYPE_Q8_0);
    uint32_t dit_kv_fixed_slots = 0;
    uint32_t dit_kv_offloadable_slots = 0;
    uint32_t dit_kv_cache_length = 0;
    bool stream = false;
    bool has_chunk_tokens = false;
    uint32_t chunk_tokens = 0;

    bool has_temperature = false;
    float temperature = 0.0f;
    bool has_top_k = false;
    int top_k = 0;
    bool has_top_p = false;
    float top_p = 0.0f;
    bool has_win_size = false;
    int win_size = 0;
    bool has_tau_r = false;
    float tau_r = 0.0f;
    bool has_min_token_text_ratio = false;
    float min_token_text_ratio = 0.0f;
    bool has_max_token_text_ratio = false;
    float max_token_text_ratio = 0.0f;

#ifndef COSYVOICE_NO_ICU
    bool text_normalization_enabled = true;
#endif
    bool split_text_enabled = true;
    bool fast_split_text_enabled = true;
    bool fade_in_enabled = true;
    bool verbose = false;
    bool quiet = false;
    bool has_llm_flash_attn = false;
    bool llm_flash_attn = true;
    bool has_flow_flash_attn = false;
    bool flow_flash_attn = true;
};

static void print_usage(const char* argv0)
{
    printf("cosyvoice-server - OpenAI Speech compatible API server\n\n");
    printf("Usage:\n");
    printf("  %s --model <file.gguf> --voice-prompt <voice=prompt_speech.gguf> [--voice-prompt ...] [options]\n", argv0);
    printf("  %s --model <file.gguf> --voice <voice> --prompt-speech <prompt_speech.gguf> [options]\n", argv0);
    printf("  %s                                       (default: WebUI mode)\n", argv0);

    printf("\nCore options:\n");
    printf("  --help, -h                                  Show this help message and exit.\n");
    printf("  --model, -m <file>                          CosyVoice model file (.gguf).\n");
    printf("  --backend-path <dir>                        GGML backend directory. Default: load from the executable's directory.\n");
    printf("  --backend <name>                            GGML backend name. Default: auto (best available).\n");
    printf("  --cpu                                       Use CPU backend (equivalent to --backend cpu).\n");
    printf("  --cuda                                      Use CUDA backend (equivalent to --backend cuda0).\n");
    printf("  --served-model-name <name>                  Exposed model name for API requests.\n");
    printf("  --host <host>                               Listen host. Default: 127.0.0.1.\n");
    printf("  --port <port>                               Listen port. Default: 8080.\n");
    printf("  --api-key <key>                             Require Bearer auth when provided.\n");

    printf("\nVoice mapping:\n");
    printf("  --voice-prompt <voice=prompt_speech.gguf>   Map one voice to one prompt_speech file. Repeatable.\n");
    printf("  --voice <voice>                             Voice name for single mapping mode. Default: alloy.\n");
    printf("  --prompt-speech <file>                      Prompt speech file for single mapping mode.\n");

    printf("\nRuntime options:\n");
    printf("  --max-llm-len <value>                       Maximum LLM sequence length. Default: " COSYVOICE_DEFAULT_LLM_MAX_SEQ_LEN_STR ".\n");
    printf("  --threads, -j <value>                       CPU thread count. Default: 0 (hardware concurrency).\n");
    printf("  --concurrency, -c <value>                   Concurrent request slots. Default: 1.\n");
    printf("  --inference-buffer-policy <shared|balanced|dedicated>\n");
    printf("                                              Inference buffer policy. Default: balanced.\n");
    printf("  --llm-kv-cache-type <f32|f16|q8_0|q5_1|q5_0|q4_1|q4_0|k=<type>,v=<type>[,fallback=<type>]>\n");
    printf("                                              KV cache type. Single type (e.g. q8_0) uses the same format for K and V.\n");
    printf("                                              Default: k=q8_0,v=q4_0,fallback=q8_0.\n");
    printf("  --seed <value>                              Default random seed for built-in sampler.\n");
    printf("  --dit-kv-cache-type <f32|f16|q8_0|q5_1|q5_0|q4_1|q4_0|k=<type>,v=<type>[,fallback=<type>]>\n");
    printf("                                              DiT KV cache type. Default: k=q8_0,v=q4_0,fallback=q8_0.\n");
    printf("  --dit-kv-fixed-slots <value>                DiT KV fixed slots (0 = auto).\n");
    printf("  --dit-kv-offloadable-slots <value>          DiT KV offloadable slots (0 = auto).\n");
    printf("  --dit-kv-cache-length <value>               DiT KV cache length (0 = auto).\n");
    printf("  --stream                                    Enable streaming for TTS requests.\n");
    printf("  --chunk-tokens <value>                      Tokens per streaming chunk. Default: model-defined.\n");

    printf("\nSampling defaults (server-level):\n");
    printf("  --temperature <value>                       Sampling temperature (> 0).\n");
    printf("  --top-k <value>                             Sampling top-k (>= 0).\n");
    printf("  --top-p <value>                             Sampling top-p in [0, 1].\n");
    printf("  --win-size <value>                          Repetition window size (> 0).\n");
    printf("  --tau-r <value>                             Repetition penalty coefficient (>= 0).\n");
    printf("  --min-token-text-ratio <value>              Minimum token/text ratio (>= 0).\n");
    printf("  --max-token-text-ratio <value>              Maximum token/text ratio (>= 0).\n");
    printf("  --verbose, -v                               Verbose logs.\n");
    printf("  --quiet, -q                               Suppress non-error logs.\n");
    printf("  --llm-flash-attn <0|1>                    Enable/disable LLM flash attention. Default: 1.\n");
    printf("  --flow-flash-attn <0|1>                   Enable/disable Flow/DiT flash attention. Default: 1.\n");

#ifndef COSYVOICE_NO_ICU
    printf("\nText normalization:\n");
    printf("  --disable-text-normalization                Disable ICU text normalization in TTS context.\n");
#endif

    printf("\nText splitting:\n");
    printf("  --disable-text-splitting                    Disable fragment splitting in TTS context.\n");
    printf("  --disable-fast-split                        Disable fast token-based splitting in TTS context.\n");

    printf("\nOutput postprocess:\n");
    printf("  --disable-fade-in                           Disable the default 20ms fade-in on output audio.\n");

    printf("\nOpenAI-compatible endpoints:\n");
    printf("  GET  /healthz\n");
    printf("  GET  /v1/models\n");
    printf("  POST /v1/audio/speech\n");

    printf("\nMode selection:\n");
    printf("  --api                                       Enable OpenAI-compatible API server.\n");
#ifndef COSYVOICE_SERVER_NO_WEBUI
    printf("  --webui                                     Launch WebUI (default, --model and voices optional).\n");
#endif
    printf("  --api requires --model and --voice-prompt.\n");
#ifndef COSYVOICE_SERVER_NO_WEBUI
    printf("\nWebUI options:\n");
    printf("  --frontend-model <dir>                      Frontend ONNX model directory (for speaker extraction).\n");
    printf("  --speech-tokenizer <file>                   Speech tokenizer ONNX model file.\n");
    printf("  --campplus <file>                           Campplus ONNX model file.\n");
#endif
}

static bool parse_voice_prompt_mapping(const std::string& value, voice_prompt_option* mapping)
{
    const auto eq = value.find('=');
    if (eq == std::string::npos)
        return false;

    mapping->voice = trim_copy(value.substr(0, eq));
    mapping->prompt_speech = trim_copy(value.substr(eq + 1));
    return !mapping->voice.empty() && !mapping->prompt_speech.empty();
}

static bool validate_options(server_options& options)
{
    if (options.quiet && options.verbose)
    {
        fprintf(stderr, "Error: --quiet and --verbose cannot be used together.\n");
        return false;
    }

    if (options.model.empty())
    {
        fprintf(stderr, "Error: --model is required.\n");
        return false;
    }

    if (options.single_prompt_speech.empty() && options.has_single_voice_arg)
    {
        fprintf(stderr, "Error: --voice requires --prompt-speech in single mapping mode.\n");
        return false;
    }

    if (!options.single_prompt_speech.empty())
        options.voice_prompts.push_back({ options.single_voice, options.single_prompt_speech });

    if (options.voice_prompts.empty())
    {
        fprintf(stderr, "Error: at least one voice mapping is required. Use --voice-prompt or --prompt-speech.\n");
        return false;
    }

    return true;
}

static server_log_level get_log_level(const server_options& options)
{
    if (options.quiet)
        return server_log_level::quiet;
    if (options.verbose)
        return server_log_level::verbose;
    return server_log_level::concise;
}

static std::string derive_served_model_name(const std::string& model_path)
{
    const auto last_sep = model_path.find_last_of("/\\");
    const size_t start = last_sep == std::string::npos ? 0 : last_sep + 1;

    const auto last_dot = model_path.find_last_of('.');
    const size_t end = (last_dot == std::string::npos || last_dot < start) ? model_path.size() : last_dot;
    if (end <= start)
        return "cosyvoice";
    return model_path.substr(start, end - start);
}

static bool init_model_context(const server_options& options, ggml_backend_t backend, server_runtime* runtime)
{
    cosyvoice_context_params_v3_cpp context_params;
    cosyvoice_init_default_context_params(&context_params);
    context_params.inference_buffer_policy = options.inference_buffer_policy;
    context_params.n_max_seq = options.max_llm_len;
    context_params.llm_kv_cache_type = options.llm_kv_cache_type;
    if (options.has_llm_flash_attn)
        context_params.llm_use_flash_attn = options.llm_flash_attn;
    if (options.has_flow_flash_attn)
        context_params.flow_use_flash_attn = options.flow_flash_attn;
    context_params.dit_kv_cache_type = options.dit_kv_cache_type;
    context_params.dit_kv_fixed_slots = options.dit_kv_fixed_slots;
    context_params.dit_kv_offloadable_slots = options.dit_kv_offloadable_slots;
    context_params.dit_kv_cache_length = options.dit_kv_cache_length;
    if (options.has_seed)
        context_params.seed = options.seed;
    context_params.n_workers = options.concurrency;

    runtime->model_slots.reserve(options.concurrency);
    runtime->model_slots.emplace_back(cosyvoice_load_from_file_ext(
        options.model.c_str(),
        &context_params,
        backend,
        options.n_threads));

    if (!runtime->model_slots.back())
    {
        fprintf(stderr, "Error: failed to load model file \"%s\".\n", options.model.c_str());
        return false;
    }

    // Save effective DiT KV cache params from the context used to load
    runtime->dit_kv_fixed_slots = context_params.dit_kv_fixed_slots;
    runtime->dit_kv_offloadable_slots = context_params.dit_kv_offloadable_slots;
    runtime->dit_kv_cache_length = context_params.dit_kv_cache_length;

    return true;
}

static bool init_model_slots(const server_options& options, server_runtime* runtime)
{
    runtime->voice_sessions.reserve(options.concurrency);
    for (uint32_t i = 0; i != options.concurrency; ++i)
        runtime->voice_sessions.emplace_back();

    auto build_sessions = [&](cosyvoice_context_t ctx, uint32_t slot_index) -> bool
    {
        auto& sessions = runtime->voice_sessions[slot_index];
        for (const auto& [name, voice] : runtime->voices)
        {
            cosyvoice_tts_context_handle session(cosyvoice_tts_context_new(ctx, voice.prompt.get()));
            if (!session)
                return false;
#ifndef COSYVOICE_NO_ICU
            cosyvoice_tts_context_set_text_normalization_enabled(session.get(), options.text_normalization_enabled);
#endif
            cosyvoice_tts_context_set_split_text_enabled(session.get(), options.split_text_enabled);
            cosyvoice_tts_context_set_fast_split_text_enabled(session.get(), options.fast_split_text_enabled);
            cosyvoice_tts_context_set_fade_in_enabled(session.get(), options.fade_in_enabled);
            sessions.emplace_back(name, std::move(session));
        }
        return true;
    };

    if (!build_sessions(runtime->model_slots.front().get(), 0))
        return false;

    for (uint32_t i = 1; i != options.concurrency; ++i)
    {
        auto slot = cosyvoice_duplicate_context(runtime->model_slots.front().get());
        if (!slot)
            return false;
        if (!cosyvoice_set_worker_no(slot, i))
            return false;
        runtime->model_slots.emplace_back(slot);
        if (!build_sessions(slot, i))
            return false;
    }

    return true;
}

static bool apply_generation_defaults(const server_options& options, server_runtime* runtime)
{
    cosyvoice_get_default_generation_config(runtime->model_slots.front().get(), &runtime->default_generation_config);
    if (options.has_temperature)
        runtime->default_generation_config.temperature = options.temperature;
    if (options.has_top_k)
        runtime->default_generation_config.sampling.top_k = options.top_k;
    if (options.has_top_p)
        runtime->default_generation_config.sampling.top_p = options.top_p;
    if (options.has_win_size)
        runtime->default_generation_config.sampling.win_size = options.win_size;
    if (options.has_tau_r)
        runtime->default_generation_config.sampling.tau_r = options.tau_r;
    if (options.has_min_token_text_ratio)
        runtime->default_generation_config.min_token_text_ratio = options.min_token_text_ratio;
    if (options.has_max_token_text_ratio)
        runtime->default_generation_config.max_token_text_ratio = options.max_token_text_ratio;

    for (auto& slot : runtime->model_slots)
        if (!cosyvoice_set_generation_config(slot.get(), &runtime->default_generation_config))
        {
            fprintf(stderr, "Error: invalid server-level sampling defaults.\n");
            return false;
        }
    return true;
}

static bool load_voice_runtime(const voice_prompt_option& mapping, server_runtime* runtime)
{
    voice_runtime voice;
    voice.name = mapping.voice;

    voice.prompt_speech.reset(cosyvoice_prompt_speech_load_from_file(mapping.prompt_speech.c_str()));
    if (!voice.prompt_speech)
    {
        fprintf(stderr, "Error: failed to load prompt_speech file \"%s\" for voice \"%s\".\n",
            mapping.prompt_speech.c_str(),
            mapping.voice.c_str());
        if (errno != 0)
            fprintf(stderr, "Reason: %s\n", strerror(errno));
        return false;
    }

    voice.prompt.reset(cosyvoice_prompt_init_from_prompt_speech(runtime->model_slots.front().get(), voice.prompt_speech.get()));
    if (!voice.prompt)
    {
        fprintf(stderr, "Error: failed to initialize prompt from prompt_speech for voice \"%s\".\n", mapping.voice.c_str());
        return false;
    }

    runtime->voice_names.push_back(voice.name);
    runtime->voices.emplace(voice.name, std::move(voice));
    return true;
}

static bool build_runtime(const server_options& options, server_runtime* runtime)
{
    runtime->model = options.model;
    runtime->host = options.host;
    runtime->api_key = options.api_key;
    runtime->port = options.port;
    runtime->has_seed = options.has_seed;
    runtime->seed = options.seed;
    runtime->concurrency = options.concurrency;
    runtime->inference_buffer_policy = options.inference_buffer_policy;
#ifndef COSYVOICE_NO_ICU
    runtime->text_normalization_enabled = options.text_normalization_enabled;
#endif
    runtime->split_text_enabled = options.split_text_enabled;
    runtime->fast_split_text_enabled = options.fast_split_text_enabled;
    runtime->fade_in_enabled = options.fade_in_enabled;
    runtime->stream = options.stream;
    runtime->has_chunk_tokens = options.has_chunk_tokens;
    runtime->chunk_tokens = options.chunk_tokens;
    runtime->log_level = get_log_level(options);

    runtime->seed_rng.seed(cosyvoice_generate_random_seed());
    runtime->slot_in_use.resize(runtime->concurrency, false);

    cosyvoice_init_backend_from_path(options.backend_path.empty() ? nullptr : options.backend_path.c_str());

    ggml_backend_t backend = nullptr;
    if (options.backend == "auto")
        backend = ggml_backend_init_best();
    else
    {
        backend = ggml_backend_init_by_name(options.backend.c_str(), nullptr);
        if (!backend)
        {
            fprintf(stderr, "Error: failed to initialize backend \"%s\".\n", options.backend.c_str());
            return false;
        }
    }

    if (!init_model_context(options, backend, runtime))
        return false;

    {
        cosyvoice_context_params_t actual_params;
        cosyvoice_get_context_params(runtime->model_slots.front().get(), &actual_params);
        runtime->has_llm_kv_cache_override  = options.has_llm_kv_cache_type;
        runtime->requested_llm_kv_cache_type = options.llm_kv_cache_type;
        runtime->actual_llm_kv_cache_type    = actual_params.llm_kv_cache_type;
        if (!options.quiet)
            fprintf(stderr, "llm_kv_cache_type: requested: %s, actual: %s\n",
                kv_cache_type_to_string(options.llm_kv_cache_type).c_str(),
                kv_cache_type_to_string(actual_params.llm_kv_cache_type).c_str());
    }

    if (!apply_generation_defaults(options, runtime))
        return false;

    if (options.has_served_model_name)
        runtime->served_model_name = options.served_model_name;
    else
    {
        auto arch = cosyvoice_get_architecture(runtime->model_slots.front().get());
        if (!arch || !*arch)
            runtime->served_model_name = derive_served_model_name(options.model);
        else
            runtime->served_model_name = arch;
    }
    
    runtime->sample_rate = cosyvoice_get_sample_rate(runtime->model_slots.front().get());

#ifndef COSYVOICE_NO_AUDIO
    runtime->audio_encoder.reset(cosyvoice_audio_encoder_create(runtime->sample_rate));
#endif

    for (const auto& mapping : options.voice_prompts)
        if (!load_voice_runtime(mapping, runtime))
            return false;

    if (!init_model_slots(options, runtime))
        return false;

    return true;
}

#ifndef COSYVOICE_SERVER_NO_WEBUI
// Simplified runtime builder for WebUI mode. Loads the model and optional voices,
// but does NOT require voice mappings (speakers are registered via the WebUI).
static bool webui_build_runtime(const server_options& options, server_runtime* runtime)
{
    runtime->model = options.model;
    runtime->host = options.host;
    runtime->api_key = options.api_key;
    runtime->port = options.port;
    runtime->has_seed = options.has_seed;
    runtime->seed = options.seed;
    runtime->concurrency = 1;
    runtime->inference_buffer_policy = options.inference_buffer_policy;
#ifndef COSYVOICE_NO_ICU
    runtime->text_normalization_enabled = options.text_normalization_enabled;
#endif
    runtime->split_text_enabled = options.split_text_enabled;
    runtime->fast_split_text_enabled = options.fast_split_text_enabled;
    runtime->fade_in_enabled = options.fade_in_enabled;
    runtime->stream = options.stream;
    runtime->has_chunk_tokens = options.has_chunk_tokens;
    runtime->chunk_tokens = options.chunk_tokens;
    runtime->log_level = get_log_level(options);

    runtime->seed_rng.seed(cosyvoice_generate_random_seed());
    runtime->slot_in_use.resize(runtime->concurrency, false);

    cosyvoice_init_backend_from_path(options.backend_path.empty() ? nullptr : options.backend_path.c_str());

    ggml_backend_t backend = nullptr;
    if (options.backend == "auto")
        backend = ggml_backend_init_best();
    else
    {
        backend = ggml_backend_init_by_name(options.backend.c_str(), nullptr);
        if (!backend)
        {
            fprintf(stderr, "Error: failed to initialize backend \"%s\".\n", options.backend.c_str());
            return false;
        }
    }

    if (!init_model_context(options, backend, runtime))
        return false;

    if (!apply_generation_defaults(options, runtime))
        return false;

    if (options.has_served_model_name)
        runtime->served_model_name = options.served_model_name;
    else
    {
        auto arch = cosyvoice_get_architecture(runtime->model_slots.front().get());
        if (!arch || !*arch)
            runtime->served_model_name = derive_served_model_name(options.model);
        else
            runtime->served_model_name = arch;
    }

    runtime->sample_rate = cosyvoice_get_sample_rate(runtime->model_slots.front().get());

#ifndef COSYVOICE_NO_AUDIO
    runtime->audio_encoder.reset(cosyvoice_audio_encoder_create(runtime->sample_rate));
#endif

    // Load optional voice mappings (if provided via --voice-prompt etc.)
    // Unlike API mode, these are NOT required in WebUI mode.
    for (const auto& mapping : options.voice_prompts)
        if (!load_voice_runtime(mapping, runtime))
            return false;

    if (!options.single_prompt_speech.empty())
    {
        voice_prompt_option fallback = { options.single_voice, options.single_prompt_speech };
        if (!load_voice_runtime(fallback, runtime))
            return false;
    }

    // Initialize a single model slot (WebUI mode does not support concurrency)
    runtime->voice_sessions.reserve(1);
    runtime->voice_sessions.emplace_back();

    if (!runtime->voices.empty())
    {
        auto ctx = runtime->model_slots.front().get();
        auto& sessions = runtime->voice_sessions[0];
        for (const auto& [name, voice] : runtime->voices)
        {
            cosyvoice_tts_context_handle session(cosyvoice_tts_context_new(ctx, voice.prompt.get()));
            if (!session)
                return false;
#ifndef COSYVOICE_NO_ICU
            cosyvoice_tts_context_set_text_normalization_enabled(session.get(), options.text_normalization_enabled);
#endif
            cosyvoice_tts_context_set_split_text_enabled(session.get(), options.split_text_enabled);
            cosyvoice_tts_context_set_fast_split_text_enabled(session.get(), options.fast_split_text_enabled);
            cosyvoice_tts_context_set_fade_in_enabled(session.get(), options.fade_in_enabled);
            sessions.emplace_back(name, std::move(session));
        }
    }

    return true;
}
#endif // !COSYVOICE_SERVER_NO_WEBUI

int tool_entry(int argc, char** argv)
{
#ifndef COSYVOICE_SERVER_NO_WEBUI
    if (argc == 1)
    {
        // No args → enter WebUI mode directly (no help, no model)
        server_runtime runtime;
        server_options  empty_opts;

        cosyvoice_init_backend_from_path(nullptr);

        runtime.host       = empty_opts.host;
        runtime.port       = empty_opts.port;
        runtime.api_key    = empty_opts.api_key;
        runtime.concurrency = 1;
        runtime.log_level  = get_log_level(empty_opts);

        return cosyvoice_server_webui_run(runtime);
    }
#endif

    server_runtime runtime;
    {
        server_options options;
        for (int i = 1; i != argc; ++i)
        {
            auto arg = argv[i];
            auto get_arg_value = [&]() -> char*
            {
                if (++i == argc)
                {
                    const auto option = arg;
                    fprintf(stderr, "Error: missing value for the command-line option \"%s\".\n", option);
                    exit(1);
                }
                return argv[i];
            };

            if (str_casecmp(arg, "--help") == 0 || str_casecmp(arg, "-h") == 0)
            {
                print_usage(argv[0]);
                return 0;
            }
            else if (str_casecmp(arg, "--model") == 0 || str_casecmp(arg, "-m") == 0)
                options.model = get_arg_value();
            else if (str_casecmp(arg, "--backend-path") == 0)
                options.backend_path = get_arg_value();
            else if (str_casecmp(arg, "--backend") == 0)
            {
                if (options.backend != "auto")
                {
                    fprintf(stderr, "Error: --backend, --cpu, and --cuda are mutually exclusive.\n");
                    return 1;
                }
                options.backend = get_arg_value();
            }
            else if (str_casecmp(arg, "--cpu") == 0)
            {
                if (options.backend != "auto")
                {
                    fprintf(stderr, "Error: --backend, --cpu, and --cuda are mutually exclusive.\n");
                    return 1;
                }
                options.backend = "cpu";
            }
            else if (str_casecmp(arg, "--cuda") == 0)
            {
                if (options.backend != "auto")
                {
                    fprintf(stderr, "Error: --backend, --cpu, and --cuda are mutually exclusive.\n");
                    return 1;
                }
                options.backend = "cuda0";
            }
            else if (str_casecmp(arg, "--served-model-name") == 0)
            {
                options.served_model_name = get_arg_value();
                options.has_served_model_name = true;
            }
            else if (str_casecmp(arg, "--host") == 0)
                options.host = get_arg_value();
            else if (str_casecmp(arg, "--port") == 0)
            {
                const auto value = get_arg_value();
                uint16_t port;
                if (!parse_uint16_port(value, &port))
                {
                    fprintf(stderr, "Error: invalid --port value \"%s\".\n", value);
                    return 1;
                }
                options.port = port;
            }
            else if (str_casecmp(arg, "--api-key") == 0)
                options.api_key = get_arg_value();
            else if (str_casecmp(arg, "--voice-prompt") == 0)
            {
                const auto value = get_arg_value();
                voice_prompt_option mapping;
                if (!parse_voice_prompt_mapping(value, &mapping))
                {
                    fprintf(stderr, "Error: invalid --voice-prompt value \"%s\". Expected <voice=prompt_speech.gguf>.\n", value);
                    return 1;
                }
                options.voice_prompts.push_back(std::move(mapping));
            }
            else if (str_casecmp(arg, "--voice") == 0)
            {
                options.single_voice = get_arg_value();
                options.has_single_voice_arg = true;
            }
            else if (str_casecmp(arg, "--prompt-speech") == 0)
                options.single_prompt_speech = get_arg_value();
            else if (str_casecmp(arg, "--api") == 0)
            {
                if (options.mode != server_options::run_mode::unspecified)
                {
                    fprintf(stderr, "Error: --api and --webui are mutually exclusive.\n");
                    return 1;
                }
                options.mode = server_options::run_mode::api;
            }
#ifndef COSYVOICE_SERVER_NO_WEBUI
            else if (str_casecmp(arg, "--webui") == 0)
            {
                if (options.mode != server_options::run_mode::unspecified)
                {
                    fprintf(stderr, "Error: --api and --webui are mutually exclusive.\n");
                    return 1;
                }
                options.mode = server_options::run_mode::webui;
            }
            else if (str_casecmp(arg, "--frontend-model") == 0)
                options.frontend_model = get_arg_value();
            else if (str_casecmp(arg, "--speech-tokenizer") == 0)
                options.speech_tokenizer = get_arg_value();
            else if (str_casecmp(arg, "--campplus") == 0)
                options.campplus = get_arg_value();
#endif
            else if (str_casecmp(arg, "--max-llm-len") == 0)
            {
                const auto value = get_arg_value();
                uint32_t max_llm_len;
                if (!parse_uint32_arg(value, &max_llm_len) || max_llm_len == 0)
                {
                    fprintf(stderr, "Error: invalid --max-llm-len value \"%s\".\n", value);
                    return 1;
                }
                options.max_llm_len = max_llm_len;
            }
            else if (str_casecmp(arg, "--inference-buffer-policy") == 0)
            {
                const auto value = get_arg_value();
                cosyvoice_inference_buffer_policy_t policy;
                if (!parse_inference_buffer_policy_arg(value, &policy))
                {
                    fprintf(stderr, "Error: invalid --inference-buffer-policy value \"%s\".\n", value);
                    return 1;
                }
                options.inference_buffer_policy = policy;
                options.has_inference_buffer_policy = true;
            }
            else if (str_casecmp(arg, "--threads") == 0 || str_casecmp(arg, "-j") == 0)
            {
                const auto value = get_arg_value();
                uint32_t n_threads;
                if (!parse_uint32_arg(value, &n_threads))
                {
                    fprintf(stderr, "Error: invalid --threads value \"%s\".\n", value);
                    return 1;
                }
                options.n_threads = n_threads;
            }
            else if (str_casecmp(arg, "--concurrency") == 0 || str_casecmp(arg, "-c") == 0)
            {
                const auto value = get_arg_value();
                uint32_t concurrency;
                if (!parse_uint32_arg(value, &concurrency) || concurrency == 0)
                {
                    fprintf(stderr, "Error: invalid --concurrency value \"%s\".\n", value);
                    return 1;
                }
                options.concurrency = concurrency;
            }
            else if (str_casecmp(arg, "--seed") == 0)
            {
                const auto value = get_arg_value();
                uint32_t seed;
                if (!parse_uint32_arg(value, &seed))
                {
                    fprintf(stderr, "Error: invalid --seed value \"%s\".\n", value);
                    return 1;
                }
                options.seed = seed;
                options.has_seed = true;
            }
            else if (str_casecmp(arg, "--temperature") == 0)
            {
                const auto value = get_arg_value();
                float v;
                if (!parse_float_arg(value, &v))
                {
                    fprintf(stderr, "Error: invalid --temperature value \"%s\".\n", value);
                    return 1;
                }
                options.temperature = v;
                options.has_temperature = true;
            }
            else if (str_casecmp(arg, "--top-k") == 0)
            {
                const auto value = get_arg_value();
                int v;
                if (!parse_int_arg(value, &v))
                {
                    fprintf(stderr, "Error: invalid --top-k value \"%s\".\n", value);
                    return 1;
                }
                options.top_k = v;
                options.has_top_k = true;
            }
            else if (str_casecmp(arg, "--top-p") == 0)
            {
                const auto value = get_arg_value();
                float v;
                if (!parse_float_arg(value, &v))
                {
                    fprintf(stderr, "Error: invalid --top-p value \"%s\".\n", value);
                    return 1;
                }
                options.top_p = v;
                options.has_top_p = true;
            }
            else if (str_casecmp(arg, "--win-size") == 0)
            {
                const auto value = get_arg_value();
                int v;
                if (!parse_int_arg(value, &v))
                {
                    fprintf(stderr, "Error: invalid --win-size value \"%s\".\n", value);
                    return 1;
                }
                options.win_size = v;
                options.has_win_size = true;
            }
            else if (str_casecmp(arg, "--tau-r") == 0)
            {
                const auto value = get_arg_value();
                float v;
                if (!parse_float_arg(value, &v))
                {
                    fprintf(stderr, "Error: invalid --tau-r value \"%s\".\n", value);
                    return 1;
                }
                options.tau_r = v;
                options.has_tau_r = true;
            }
            else if (str_casecmp(arg, "--min-token-text-ratio") == 0)
            {
                const auto value = get_arg_value();
                float v;
                if (!parse_float_arg(value, &v))
                {
                    fprintf(stderr, "Error: invalid --min-token-text-ratio value \"%s\".\n", value);
                    return 1;
                }
                options.min_token_text_ratio = v;
                options.has_min_token_text_ratio = true;
            }
            else if (str_casecmp(arg, "--max-token-text-ratio") == 0)
            {
                const auto value = get_arg_value();
                float v;
                if (!parse_float_arg(value, &v))
                {
                    fprintf(stderr, "Error: invalid --max-token-text-ratio value \"%s\".\n", value);
                    return 1;
                }
                options.max_token_text_ratio = v;
                options.has_max_token_text_ratio = true;
            }
            else if (str_casecmp(arg, "--llm-kv-cache-type") == 0)
            {
                const auto value = get_arg_value();
                cosyvoice_kv_cache_type_t type;
                if (!parse_kv_cache_type_arg(value, &type))
                {
                    fprintf(stderr, "Error: invalid --llm-kv-cache-type value \"%s\".\n", value);
                    return 1;
                }
                options.llm_kv_cache_type = type;
                options.has_llm_kv_cache_type = true;
            }
            else if (str_casecmp(arg, "--dit-kv-cache-type") == 0)
            {
                const auto value = get_arg_value();
                cosyvoice_kv_cache_type_t type;
                if (!parse_kv_cache_type_arg(value, &type))
                {
                    fprintf(stderr, "Error: invalid --dit-kv-cache-type value \"%s\".\n", value);
                    return 1;
                }
                options.dit_kv_cache_type = type;
                options.has_dit_kv_cache_type = true;
            }
            else if (str_casecmp(arg, "--dit-kv-fixed-slots") == 0)
            {
                const auto value = get_arg_value();
                uint32_t v;
                if (!parse_uint32_arg(value, &v))
                {
                    fprintf(stderr, "Error: invalid --dit-kv-fixed-slots value \"%s\".\n", value);
                    return 1;
                }
                options.dit_kv_fixed_slots = v;
            }
            else if (str_casecmp(arg, "--dit-kv-offloadable-slots") == 0)
            {
                const auto value = get_arg_value();
                uint32_t v;
                if (!parse_uint32_arg(value, &v))
                {
                    fprintf(stderr, "Error: invalid --dit-kv-offloadable-slots value \"%s\".\n", value);
                    return 1;
                }
                options.dit_kv_offloadable_slots = v;
            }
            else if (str_casecmp(arg, "--dit-kv-cache-length") == 0)
            {
                const auto value = get_arg_value();
                uint32_t v;
                if (!parse_uint32_arg(value, &v))
                {
                    fprintf(stderr, "Error: invalid --dit-kv-cache-length value \"%s\".\n", value);
                    return 1;
                }
                options.dit_kv_cache_length = v;
            }
            else if (str_casecmp(arg, "--stream") == 0)
                options.stream = true;
            else if (str_casecmp(arg, "--chunk-tokens") == 0)
            {
                const auto value = get_arg_value();
                uint32_t v;
                if (!parse_uint32_arg(value, &v))
                {
                    fprintf(stderr, "Error: invalid --chunk-tokens value \"%s\".\n", value);
                    return 1;
                }
                options.chunk_tokens = v;
                options.has_chunk_tokens = true;
            }
            else if (str_casecmp(arg, "--verbose") == 0 || str_casecmp(arg, "-v") == 0)
                options.verbose = true;
            else if (str_casecmp(arg, "--quiet") == 0 || str_casecmp(arg, "-q") == 0)
                options.quiet = true;
#ifndef COSYVOICE_NO_ICU
            else if (str_casecmp(arg, "--disable-text-normalization") == 0)
                options.text_normalization_enabled = false;
#endif
            else if (str_casecmp(arg, "--disable-text-splitting") == 0)
                options.split_text_enabled = false;
            else if (str_casecmp(arg, "--disable-fast-split") == 0)
                options.fast_split_text_enabled = false;
            else if (str_casecmp(arg, "--disable-fade-in") == 0)
                options.fade_in_enabled = false;
            else if (str_casecmp(arg, "--llm-flash-attn") == 0)
            {
                const auto v = to_lower(get_arg_value());
                if (v == "1" || v == "yes" || v == "true" || v == "on")
                    { options.llm_flash_attn = true; options.has_llm_flash_attn = true; }
                else if (v == "0" || v == "no" || v == "false" || v == "off")
                    { options.llm_flash_attn = false; options.has_llm_flash_attn = true; }
                else
                    { fprintf(stderr, "Error: invalid --llm-flash-attn value \"%s\". Use 0/1, yes/no, true/false, on/off.\n", v.c_str()); return 1; }
            }
            else if (str_casecmp(arg, "--flow-flash-attn") == 0)
            {
                const auto v = to_lower(get_arg_value());
                if (v == "1" || v == "yes" || v == "true" || v == "on")
                    { options.flow_flash_attn = true; options.has_flow_flash_attn = true; }
                else if (v == "0" || v == "no" || v == "false" || v == "off")
                    { options.flow_flash_attn = false; options.has_flow_flash_attn = true; }
                else
                    { fprintf(stderr, "Error: invalid --flow-flash-attn value \"%s\". Use 0/1, yes/no, true/false, on/off.\n", v.c_str()); return 1; }
            }
            else
            {
                const auto option = arg;
                fprintf(stderr, "Error: the program doesn't recognize the command-line option \"%s\".\n", option);
                return 1;
            }
        }

        // ---- Determine effective mode ----
#ifdef COSYVOICE_SERVER_NO_WEBUI
        // WebUI not available — force API mode regardless of --api/--webui flags
        options.mode = server_options::run_mode::api;
#else
        if (options.mode == server_options::run_mode::unspecified)
        {
#ifdef COSYVOICE_SERVER_DEFAULT_MODE_API
            // Default mode = API: enter API if sufficient args, otherwise fall back to WebUI
            bool cannot_api = options.model.empty() || options.single_prompt_speech.empty() && options.has_single_voice_arg;
            options.mode = !cannot_api ? server_options::run_mode::api : server_options::run_mode::webui;
#else
            options.mode = server_options::run_mode::webui;
#endif
        }
#endif

        // ---- Dispatch based on resolved mode ----
        if (options.mode == server_options::run_mode::api)
        {
            if (!validate_options(options))
            {
                print_usage(argv[0]);
#ifdef COSYVOICE_SERVER_NO_WEBUI
                return 0;   // silent help, not an error
#else
                return 1;   // explicit --api failure, error exit
#endif
            }
            if (!build_runtime(options, &runtime))
                return 1;
            return cosyvoice_server_backend_run(runtime);
        }

        // WebUI mode (only reachable when COSYVOICE_SERVER_NO_WEBUI is NOT set)
#ifndef COSYVOICE_SERVER_NO_WEBUI
        {
            runtime.webui_enabled = true;
            runtime.frontend_model  = options.frontend_model;
            runtime.speech_tokenizer = options.speech_tokenizer;
            runtime.campplus         = options.campplus;
            runtime.host = options.host;
            runtime.port = options.port;
            runtime.api_key = options.api_key;
            runtime.concurrency = 1;
            runtime.log_level = get_log_level(options);

            // Init backend early — needed even without a model so the user can load one later via the WebUI
            cosyvoice_init_backend_from_path(options.backend_path.empty() ? nullptr : options.backend_path.c_str());

            // Load model if given (voices are optional in WebUI mode)
            if (!options.model.empty())
            {
                if (!webui_build_runtime(options, &runtime))
                {
                    fprintf(stderr, "Warning: failed to load model, starting WebUI without model.\n");
                    runtime.model_slots.clear();
                    runtime.sample_rate = 0;
                }
            }

            return cosyvoice_server_webui_run(runtime);
        }
#else
        // Not reachable — COSYVOICE_SERVER_NO_WEBUI forces API mode above
        return 1;
#endif
    }
}
