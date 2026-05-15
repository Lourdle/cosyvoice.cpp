#include "cosyvoice-server.h"
#include "tool_common_cosyvoice.h"

#include <chrono>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
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
    uint16_t port = 8080;
    bool has_served_model_name = false;
    std::vector<voice_prompt_option> voice_prompts;

    std::string single_voice = "alloy";
    std::string single_prompt_speech;
    bool has_single_voice_arg = false;

    uint32_t max_llm_len = 8192;
    uint32_t n_threads = 0;
    bool has_inference_buffer_policy = false;
    cosyvoice_inference_buffer_policy_t inference_buffer_policy = COSYVOICE_INFERENCE_BUFFER_POLICY_BALANCED;

    bool has_seed = false;
    uint32_t seed = 0;

    bool has_llm_kv_cache_type = false;
    cosyvoice_llm_kv_cache_type_t llm_kv_cache_type = COSYVOICE_LLM_KV_CACHE_TYPE_Q8_0;

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
    bool verbose = false;
    bool quiet = false;
};

static void print_usage(const char* argv0)
{
    printf("cosyvoice-server - OpenAI Speech compatible API server\n\n");
    printf("Usage:\n");
    printf("  %s --model <file.gguf> --voice-prompt <voice=prompt_speech.gguf> [--voice-prompt ...] [options]\n", argv0);
    printf("  %s --model <file.gguf> --voice <voice> --prompt-speech <prompt_speech.gguf> [options]\n", argv0);

    printf("\nCore options:\n");
    printf("  --help, -h                                  Show this help message and exit.\n");
    printf("  --model, -m <file>                          CosyVoice model file (.gguf).\n");
    printf("  --backend-path <dir>                        GGML backend directory. Default: load from the executable's directory.\n");
    printf("  --served-model-name <name>                  Exposed model name for API requests.\n");
    printf("  --host <host>                               Listen host. Default: 127.0.0.1.\n");
    printf("  --port <port>                               Listen port. Default: 8080.\n");
    printf("  --api-key <key>                             Require Bearer auth when provided.\n");

    printf("\nVoice mapping:\n");
    printf("  --voice-prompt <voice=prompt_speech.gguf>   Map one voice to one prompt_speech file. Repeatable.\n");
    printf("  --voice <voice>                             Voice name for single mapping mode. Default: alloy.\n");
    printf("  --prompt-speech <file>                      Prompt speech file for single mapping mode.\n");

    printf("\nRuntime options:\n");
    printf("  --max-llm-len <value>                       Maximum LLM sequence length. Default: 8192.\n");
    printf("  --threads, -j <value>                       CPU thread count. Default: 0 (hardware concurrency).\n");
    printf("  --inference-buffer-policy <shared|balanced|dedicated>\n");
    printf("                                              Inference buffer policy. Default: balanced.\n");
    printf("  --llm-kv-cache-type <f32|f16|q8_0|q5_1|q5_0|q4_1|q4_0>\n");
    printf("                                              KV cache type. Default: q8_0.\n");
    printf("  --seed <value>                              Default random seed for built-in sampler.\n");

    printf("\nSampling defaults (server-level):\n");
    printf("  --temperature <value>                       Sampling temperature (> 0).\n");
    printf("  --top-k <value>                             Sampling top-k (>= 0).\n");
    printf("  --top-p <value>                             Sampling top-p in [0, 1].\n");
    printf("  --win-size <value>                          Repetition window size (> 0).\n");
    printf("  --tau-r <value>                             Repetition penalty coefficient (>= 0).\n");
    printf("  --min-token-text-ratio <value>              Minimum token/text ratio (>= 0).\n");
    printf("  --max-token-text-ratio <value>              Maximum token/text ratio (>= 0).\n");
    printf("  --verbose, -v                               Verbose logs.\n");
    printf("  --quiet, -q                                 Suppress non-error logs.\n");

#ifndef COSYVOICE_NO_ICU
    printf("\nText normalization:\n");
    printf("  --disable-text-normalization                Disable ICU text normalization in TTS context.\n");
#endif

    printf("\nOpenAI-compatible endpoints:\n");
    printf("  GET  /healthz\n");
    printf("  GET  /v1/models\n");
    printf("  POST /v1/audio/speech\n");
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

static void seed_runtime_rng(server_runtime* runtime)
{
    uint32_t seed_rng_seed = static_cast<uint32_t>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count());
    std::random_device rd;
    if (rd.entropy() > 0.0)
        seed_rng_seed ^= static_cast<uint32_t>(rd());
    runtime->seed_rng.seed(seed_rng_seed);
}

static bool init_model_context(const server_options& options, server_runtime* runtime)
{
    cosyvoice_context_params_t context_params;
    cosyvoice_init_default_context_params(&context_params);
    context_params.inference_buffer_policy = options.inference_buffer_policy;
    context_params.n_max_seq = options.max_llm_len;
    if (options.has_llm_kv_cache_type)
        context_params.llm_kv_cache_type = options.llm_kv_cache_type;
    if (options.has_seed)
        context_params.seed = options.seed;

    runtime->model_context.reset(cosyvoice_load_from_file_ext(
        options.model.c_str(),
        &context_params,
        nullptr,
        options.n_threads,
        0));

    if (!runtime->model_context)
    {
        fprintf(stderr, "Error: failed to load model file \"%s\".\n", options.model.c_str());
        return false;
    }

    return true;
}

static bool apply_generation_defaults(const server_options& options, server_runtime* runtime)
{
    cosyvoice_get_generation_config(runtime->model_context.get(), &runtime->default_generation_config);
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

    if (!cosyvoice_set_generation_config(runtime->model_context.get(), &runtime->default_generation_config))
    {
        fprintf(stderr, "Error: invalid server-level sampling defaults.\n");
        return false;
    }

    cosyvoice_get_generation_config(runtime->model_context.get(), &runtime->default_generation_config);
    return true;
}

static bool load_voice_runtime(const server_options& options, const voice_prompt_option& mapping, server_runtime* runtime)
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

    voice.prompt.reset(cosyvoice_prompt_init_from_prompt_speech(runtime->model_context.get(), voice.prompt_speech.get()));
    if (!voice.prompt)
    {
        fprintf(stderr, "Error: failed to initialize prompt from prompt_speech for voice \"%s\".\n", mapping.voice.c_str());
        return false;
    }

    voice.tts_context.reset(cosyvoice_tts_context_new(runtime->model_context.get(), voice.prompt.get()));
    if (!voice.tts_context)
    {
        fprintf(stderr, "Error: failed to create TTS context for voice \"%s\".\n", mapping.voice.c_str());
        return false;
    }

#ifndef COSYVOICE_NO_ICU
    cosyvoice_tts_context_set_text_normalization_enabled(voice.tts_context.get(), options.text_normalization_enabled);
#endif

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
    runtime->inference_buffer_policy = options.inference_buffer_policy;
#ifndef COSYVOICE_NO_ICU
    runtime->text_normalization_enabled = options.text_normalization_enabled;
#endif
    runtime->log_level = get_log_level(options);

    seed_runtime_rng(runtime);

    cosyvoice_init_backend_from_path(options.backend_path.empty() ? nullptr : options.backend_path.c_str());

    if (!init_model_context(options, runtime))
        return false;

    if (!apply_generation_defaults(options, runtime))
        return false;

    if (options.has_served_model_name)
        runtime->served_model_name = options.served_model_name;
    else
    {
        auto arch = cosyvoice_get_architecture(runtime->model_context.get());
        if (!arch || !*arch)
            runtime->served_model_name = derive_served_model_name(options.model);
        else
            runtime->served_model_name = arch;
    }
    
    runtime->sample_rate = cosyvoice_get_sample_rate(runtime->model_context.get());

#ifndef COSYVOICE_NO_AUDIO
    runtime->audio_encoder.reset(cosyvoice_audio_encoder_create(runtime->sample_rate));
#endif

    for (const auto& mapping : options.voice_prompts)
        if (!load_voice_runtime(options, mapping, runtime))
            return false;

    return true;
}

int tool_entry(int argc, char** argv)
{
    if (argc == 1)
    {
        print_usage(argv[0]);
        return 0;
    }

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
                cosyvoice_llm_kv_cache_type_t type;
                if (!parse_llm_kv_cache_type_arg(value, &type))
                {
                    fprintf(stderr, "Error: invalid --llm-kv-cache-type value \"%s\".\n", value);
                    return 1;
                }
                options.llm_kv_cache_type = type;
                options.has_llm_kv_cache_type = true;
            }
            else if (str_casecmp(arg, "--verbose") == 0 || str_casecmp(arg, "-v") == 0)
                options.verbose = true;
            else if (str_casecmp(arg, "--quiet") == 0 || str_casecmp(arg, "-q") == 0)
                options.quiet = true;
#ifndef COSYVOICE_NO_ICU
            else if (str_casecmp(arg, "--disable-text-normalization") == 0)
                options.text_normalization_enabled = false;
#endif
            else
            {
                const auto option = arg;
                fprintf(stderr, "Error: the program doesn't recognize the command-line option \"%s\".\n", option);
                return 1;
            }
        }

        if (!validate_options(options))
            return 1;

        if (!build_runtime(options, &runtime))
            return 1;
    }
    return cosyvoice_server_backend_run(runtime);
}
