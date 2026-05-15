#ifdef _MSC_VER
    #define _CRT_SECURE_NO_WARNINGS
#endif

#include "tool_common_cosyvoice.h"

#ifdef COSYVOICE_NO_AUDIO
    #define cosyvoice_audio_save_to_file cosyvoice_save_wav
    #define cosyvoice_audio_supported_encoding_formats() "wav"
#endif

#include <ggml-backend.h>

#include <cstdarg>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <format>

struct cli_options
{
    uint32_t max_llm_len = 2048;
    float speed = 1.0f;
#ifndef COSYVOICE_NO_ICU
    bool text_normalization_enabled = true;
#endif
    std::string model;
    std::string backend_path;
#ifndef COSYVOICE_NO_FRONTEND
    bool frontend_only = false;
    std::string speech_tokenizer;
    std::string campplus;
    #ifdef COSYVOICE_NO_AUDIO
    std::string prompt_audio_16k;
    std::string prompt_audio_24k;
    #else
    std::string prompt_audio;
    #endif
    std::string prompt_text;
    std::string prompt_speech_output;
#endif
    std::string prompt_speech;
    std::string text;
    std::string instruction;
    std::string output;
    std::string mode = "auto";
    std::string seed;
    uint32_t n_threads = 0;
    bool has_llm_kv_cache_type = false;
    cosyvoice_llm_kv_cache_type_t llm_kv_cache_type = COSYVOICE_LLM_KV_CACHE_TYPE_Q8_0;
    bool verbose = false;
    bool quiet = false;
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
};

enum class cli_log_level
{
    quiet,
    concise,
    verbose
};

static bool g_quiet_logs = false;
static constexpr const char* ANSI_RESET = "\033[0m";
static constexpr const char* ANSI_RED = "\033[31m";
static constexpr const char* ANSI_YELLOW = "\033[93m";
static constexpr const char* ANSI_BOLD_CYAN = "\033[1;36m";
static constexpr const char* ANSI_BOLD_BLUE = "\033[1;34m";
static constexpr const char* ANSI_DIM = "\033[2m";

static bool has_generation_overrides(const cli_options& options)
{
    return options.has_temperature
        || options.has_top_k
        || options.has_top_p
        || options.has_win_size
        || options.has_tau_r
        || options.has_min_token_text_ratio
        || options.has_max_token_text_ratio;
}

static cli_log_level get_log_level(const cli_options& options)
{
    if (options.quiet)
        return cli_log_level::quiet;
    if (options.verbose)
        return cli_log_level::verbose;
    return cli_log_level::concise;
}

#ifdef COSYVOICE_NO_AUDIO
    #define PROMPT_AUDIO_CMD "--prompt-audio-16k <16k_pcm_file> --prompt-audio-24k <24k_pcm_file>"
#else
    #define PROMPT_AUDIO_CMD "--prompt-audio <file>"
#endif

static void print_usage(const char* argv0)
{
    const char* supported_formats = cosyvoice_audio_supported_encoding_formats();
    printf("cosyvoice-cli - command line TTS tool\n\n");
    printf("Usage:\n");
    printf("  %s --model <file.gguf> --prompt-speech <file> --text <text> --output <file>\n", argv0);
#ifndef COSYVOICE_NO_FRONTEND
    printf("  %s --frontend-only --speech-tokenizer <file.onnx> --campplus <file.onnx> %s --prompt-text <text> --prompt-speech-output <file>\n", argv0, PROMPT_AUDIO_CMD);
    printf("  %s --model <file.gguf> --speech-tokenizer <file.onnx> --campplus <file.onnx> %s --prompt-text <text> --text <text> --output <file>\n", argv0, PROMPT_AUDIO_CMD);
#endif

    printf("\nCore options:\n");
    printf("  --help, -h                                  Show this help message and exit.\n");
    printf("  --model, -m <file>                          CosyVoice model file.\n");
    printf("  --backend-path <dir>                        GGML backend directory. Default: load from the executable's directory.\n");
    printf("  --text, -t <text>                           Text to synthesize.\n");
    printf("  --instruction, -i <text>                    Only used in instruct mode.\n");
#ifdef COSYVOICE_NO_AUDIO
    printf("  --output, -o <file>                         Output audio file path (WAV).\n");
#else
    printf("  --output, -o <file>                         Output audio file path (%s).\n", supported_formats);
#endif
    printf("  --mode <zero-shot|instruct|cross-lingual>   TTS mode. Default: auto-detect by --instruction.\n");
    printf("  --speed, -s <value>                         Speech speed. Default: 1.0.\n");
    printf("  --max-llm-len <value>                       Maximum LLM sequence length. Default: 2048.\n");
    printf("  --threads, -j <value>                       CPU thread count. Default: 0 (hardware concurrency).\n");
    printf("  --llm-kv-cache-type <f32|f16|q8_0|q5_1|q5_0|q4_1|q4_0>\n");
    printf("                                              KV cache type. Default: q8_0.\n");
    printf("  --seed <value>                              Random seed for sampling. Default: random.\n");

    printf("\nSampling overrides:\n");
    printf("  --temperature <value>                       Sampling temperature (> 0).\n");
    printf("  --top-k <value>                             Sampling top-k (>= 0).\n");
    printf("  --top-p <value>                             Sampling top-p in [0, 1].\n");
    printf("  --win-size <value>                          Repetition window size (> 0).\n");
    printf("  --tau-r <value>                             Repetition penalty coefficient (>= 0).\n");
    printf("  --min-token-text-ratio <value>              Minimum token/text ratio (>= 0).\n");
    printf("  --max-token-text-ratio <value>              Maximum token/text ratio (>= 0).\n");

    printf("\nLog options:\n");
    printf("  --verbose, -v                               Show detailed runtime info.\n");
    printf("  --quiet, -q                                 Suppress non-error logs.\n");

#ifndef COSYVOICE_NO_ICU
    printf("\nText normalization:\n");
    printf("  --disable-text-normalization                Disable ICU text normalization before tokenization.\n");
#endif

#ifndef COSYVOICE_NO_FRONTEND
    printf("\nFrontend options:\n");
    printf("  --frontend-only                             Only run frontend and save prompt_speech.\n");
    printf("  --speech-tokenizer <file>                   Frontend speech tokenizer ONNX file.\n");
    printf("  --campplus <file>                           Frontend campplus ONNX file.\n");
#ifdef COSYVOICE_NO_AUDIO
    printf("  --prompt-audio-16k <16k_pcm_file>           Reference audio in 16kHz PCM float format.\n");
    printf("  --prompt-audio-24k <24k_pcm_file>           Reference audio in 24kHz PCM float format.\n");
#else
    printf("  --prompt-audio <file>                       Reference audio file for frontend.\n");
#endif
    printf("  --prompt-text <text>                        Transcript of the reference audio.\n");
    printf("  --prompt-speech-output <file>               Save generated prompt_speech to file.\n");
#endif

    printf("\nPrompt source:\n");
    printf("  --prompt-speech <file>                      Saved prompt_speech file.\n");

    printf("\nRequired combinations:\n");
    printf("  TTS: --model --text --output + one prompt source.\n");
    printf("  If --prompt-speech is not provided, frontend inputs are required.\n");
    printf("  Frontend-only: --frontend-only --speech-tokenizer --campplus + audio input + --prompt-speech-output.\n");

    printf("\nDefaults and sources:\n");
    printf("  CLI defaults: speed=1.0, max-llm-len=2048, threads=0 (hardware concurrency), llm-kv-cache-type=q8_0, mode=auto.\n");
    printf("  Sampling defaults (temperature/top-k/top-p/win-size/tau-r/token-text ratios) come from model metadata unless overridden.\n");
}

static void print_error_log(const char* format, ...)
{
    fprintf(stderr, ANSI_RED); // Set text color to red
    va_list args;
    va_start(args, format);
    vfprintf(stderr, format, args);
    va_end(args);
    fprintf(stderr, ANSI_RESET); // Reset text color
}

static void print_warning_log(const char* format, ...)
{
    if (g_quiet_logs)
        return;

    fprintf(stderr, ANSI_YELLOW); // Set text color to yellow
    va_list args;
    va_start(args, format);
    vfprintf(stderr, format, args);
    va_end(args);
    fprintf(stderr, ANSI_RESET); // Reset text color
}

struct cli_timing_info
{
    double backend_init_ms = 0.0;
    double prompt_prepare_ms = 0.0;
    double model_load_ms = 0.0;
    double tts_generate_ms = 0.0;
    double save_output_ms = 0.0;
    double total_ms = 0.0;
};

class loading_spinner
{
public:
    loading_spinner(cli_log_level log_level, const char* label)
        : enabled(log_level != cli_log_level::quiet), label(label)
    {
    }

    void start()
    {
        if (!enabled)
            return;

        running = true;
        worker = std::thread([this]()
            {
                constexpr char frames[] = { '|', '/', '-', '\\' };
                size_t i = 0;
                while (running)
                {
                    printf("\r%s%s%s %c", ANSI_BOLD_CYAN, label.c_str(), ANSI_RESET, frames[i++ % (sizeof(frames) / sizeof(frames[0]))]);
                    fflush(stdout);
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                }
            });
    }

    void stop(bool success)
    {
        if (!enabled)
            return;

        running = false;
        if (worker.joinable())
            worker.join();

        printf("\r%s%s%s %s\n", ANSI_BOLD_CYAN, label.c_str(), ANSI_RESET, success ? "done" : "failed");
        fflush(stdout);
    }

private:
    bool enabled = false;
    std::string label;
    volatile bool running = false;
    std::thread worker;
};

static void print_section_title(const char* title)
{
    printf("\n%s== %s ==%s\n", ANSI_BOLD_BLUE, title, ANSI_RESET);
}

static void print_kv_line_string(const char* key, const char* value)
{
    printf("  %s%-24s%s : %s\n", ANSI_DIM, key, ANSI_RESET, value);
}

static void print_kv_line_u32(const char* key, uint32_t value)
{
    printf("  %s%-24s%s : %u\n", ANSI_DIM, key, ANSI_RESET, value);
}

static void print_kv_line_size(const char* key, size_t value)
{
    printf("  %s%-24s%s : %zu\n", ANSI_DIM, key, ANSI_RESET, value);
}

static void print_kv_line_int_with_source(const char* key, int value, bool overridden)
{
    printf("  %s%-24s%s : %d (%s)\n", ANSI_DIM, key, ANSI_RESET, value, overridden ? "cli override" : "model default");
}

static void print_kv_line_float3(const char* key, float value)
{
    printf("  %s%-24s%s : %.3f\n", ANSI_DIM, key, ANSI_RESET, value);
}

static void print_kv_line_float4_with_source(const char* key, float value, bool overridden)
{
    printf("  %s%-24s%s : %.4f (%s)\n", ANSI_DIM, key, ANSI_RESET, value, overridden ? "cli override" : "model default");
}

static void print_kv_line_ms(const char* key, double value)
{
    printf("  %s%-24s%s : %.2f ms\n", ANSI_DIM, key, ANSI_RESET, value);
}

static void print_kv_line_mib(const char* key, size_t bytes)
{
    printf("  %s%-24s%s : %zu bytes (%.2f MiB)\n", ANSI_DIM, key, ANSI_RESET, bytes, bytes_to_mib(bytes));
}

static void print_kv_line_mib_delta(const char* key, size_t before, size_t after)
{
    const auto delta = static_cast<long long>(after) - static_cast<long long>(before);
    printf("  %s%-24s%s : %zu bytes (%.2f MiB), delta %+lld bytes\n", ANSI_DIM, key, ANSI_RESET, after, bytes_to_mib(after), delta);
}

static const char* enabled_to_string(bool enabled)
{
    return enabled ? "enabled" : "disabled";
}

static const char* get_prompt_source(const cli_options& options)
{
#ifndef COSYVOICE_NO_FRONTEND
    return options.prompt_speech.empty() ? "frontend" : "prompt-speech";
#else
    (void)options;
    return "prompt-speech";
#endif
}

static void print_preload_run_info(const cli_options& options, cli_log_level log_level)
{
    if (log_level == cli_log_level::quiet)
        return;

    printf("\n================ cosyvoice-cli ================\n");
    print_section_title("Run");
    print_kv_line_string("model", options.model.c_str());
    print_kv_line_string("mode", options.mode.c_str());
    print_kv_line_string("prompt_source", get_prompt_source(options));
    print_kv_line_string("output", options.output.c_str());
    print_kv_line_float3("speed", options.speed);
    print_kv_line_u32("n_threads", options.n_threads ? options.n_threads : std::thread::hardware_concurrency());
#ifndef COSYVOICE_NO_ICU
    print_kv_line_string("text_normalization", enabled_to_string(options.text_normalization_enabled));
#else
    print_kv_line_string("text_normalization", "unavailable (COSYVOICE_NO_ICU)");
#endif
    print_kv_line_string("llm_kv_cache_type", llm_kv_cache_type_to_string(options.llm_kv_cache_type));
    if (log_level == cli_log_level::verbose)
    {
        print_kv_line_size("text_length", options.text.size());
        if (options.mode == "instruct")
            print_kv_line_size("instruction_length", options.instruction.size());
    }
}

static void print_tts_runtime_info(
    const cli_options& options,
    ggml_backend_t backend,
    const cosyvoice_context_params_t& context_params,
    const cosyvoice_generation_config_t& generation_config,
    uint32_t sample_rate,
    cli_log_level log_level)
{
    if (log_level == cli_log_level::quiet)
        return;

    ggml_backend_dev_props props;
    ggml_backend_dev_get_props(ggml_backend_get_device(backend), &props);
    print_section_title("Backend");
    print_kv_line_string("device", std::format("{} ({})", props.name, props.description).c_str());
    print_kv_line_string("device_id", props.device_id);
    print_kv_line_mib("total_memory", props.memory_total);
    print_kv_line_mib("free_memory", props.memory_free);
#ifndef GGML_SHARED
    ggml_backend_free(backend);
#endif

    print_section_title("Model");
    print_kv_line_u32("sample_rate", sample_rate);
    if (!options.seed.empty())
    {
        char buf[64];
        snprintf(buf, sizeof(buf), "%u (--seed)", context_params.seed);
        print_kv_line_string("seed", buf);
    }
    else
    {
        char buf[64];
        snprintf(buf, sizeof(buf), "%u (random)", context_params.seed);
        print_kv_line_string("seed", buf);
    }
    {
        char buf[64];
        snprintf(buf, sizeof(buf), "%s (%s)", llm_kv_cache_type_to_string(context_params.llm_kv_cache_type),
            options.has_llm_kv_cache_type ? "cli override" : "default");
        print_kv_line_string("llm_kv_cache_type", buf);
    }
    if (log_level == cli_log_level::verbose)
    {
        print_kv_line_u32("n_batch", context_params.n_batch);
        print_kv_line_u32("n_max_seq", context_params.n_max_seq);
        print_kv_line_string("llm_use_flash_attn", enabled_to_string(context_params.llm_use_flash_attn));
        print_kv_line_string("flow_use_flash_attn", enabled_to_string(context_params.flow_use_flash_attn));
        print_kv_line_string("llm_kv_fallback", enabled_to_string(context_params.llm_allow_kv_cache_fallback));
    }

    print_section_title("Sampling");
    print_kv_line_float4_with_source("temperature", generation_config.temperature, options.has_temperature);
    print_kv_line_int_with_source("top_k", generation_config.sampling.top_k, options.has_top_k);
    print_kv_line_float4_with_source("top_p", generation_config.sampling.top_p, options.has_top_p);
    print_kv_line_int_with_source("win_size", generation_config.sampling.win_size, options.has_win_size);
    print_kv_line_float4_with_source("tau_r", generation_config.sampling.tau_r, options.has_tau_r);
    print_kv_line_float4_with_source("min_token_text_ratio", generation_config.min_token_text_ratio, options.has_min_token_text_ratio);
    print_kv_line_float4_with_source("max_token_text_ratio", generation_config.max_token_text_ratio, options.has_max_token_text_ratio);
}

static void print_memory_runtime_info(
    const cosyvoice_memory_usage_t& memory_usage_before,
    const cosyvoice_memory_usage_t& memory_usage_after,
    cli_log_level log_level)
{
    if (log_level != cli_log_level::verbose)
        return;

    print_section_title("Memory");
    print_kv_line_mib_delta("parameters", memory_usage_before.parameters, memory_usage_after.parameters);
    print_kv_line_mib_delta("kv_cache", memory_usage_before.kv_cache, memory_usage_after.kv_cache);
    print_kv_line_mib_delta("token2wav", memory_usage_before.token2wav, memory_usage_after.token2wav);
    print_kv_line_mib_delta("buffers", memory_usage_before.buffers, memory_usage_after.buffers);
    print_kv_line_mib_delta("cpu_buffers", memory_usage_before.cpu_buffers, memory_usage_after.cpu_buffers);
    print_kv_line_mib_delta("offloaded_kv", memory_usage_before.offloaded_kv_cache, memory_usage_after.offloaded_kv_cache);
    print_kv_line_mib_delta("random_noise", memory_usage_before.random_noise, memory_usage_after.random_noise);
}

#ifndef COSYVOICE_NO_FRONTEND
static void print_frontend_runtime_info(const cli_options& options, cli_log_level log_level)
{
    if (log_level == cli_log_level::quiet)
        return;

    printf("\n================ cosyvoice-cli ================\n");
    print_section_title("Frontend");
    print_kv_line_string("mode", "frontend-only");
    print_kv_line_string("prompt_speech_output", options.prompt_speech_output.c_str());
    if (log_level != cli_log_level::verbose)
        return;

    print_kv_line_string("speech_tokenizer", options.speech_tokenizer.c_str());
    print_kv_line_string("campplus", options.campplus.c_str());
#ifdef COSYVOICE_NO_AUDIO
    print_kv_line_string("prompt_audio_16k", options.prompt_audio_16k.c_str());
    print_kv_line_string("prompt_audio_24k", options.prompt_audio_24k.c_str());
#else
    print_kv_line_string("prompt_audio", options.prompt_audio.c_str());
#endif
    print_kv_line_size("prompt_text_length", options.prompt_text.size());
}
#endif

static void print_timing_info(const cli_timing_info& timing, bool include_tts_stages, cli_log_level log_level)
{
    if (log_level == cli_log_level::quiet)
        return;

    print_section_title("Timing");
    if (log_level == cli_log_level::verbose)
        print_kv_line_ms("backend_init", timing.backend_init_ms);
    print_kv_line_ms("prompt_prepare", timing.prompt_prepare_ms);
    if (include_tts_stages)
    {
        print_kv_line_ms("model_load", timing.model_load_ms);
        print_kv_line_ms("tts_generate", timing.tts_generate_ms);
        if (log_level == cli_log_level::verbose)
            print_kv_line_ms("save_output", timing.save_output_ms);
    }
    print_kv_line_ms("total", timing.total_ms);
}

static bool validate_options(cli_options& options)
{
    if (options.quiet && options.verbose)
    {
        print_error_log("Error: --quiet and --verbose cannot be used together.\n");
        return false;
    }

    bool mode_auto_detected = false;

#ifndef COSYVOICE_NO_FRONTEND
    if (options.frontend_only)
    {
        bool ok = true;
        if (options.speech_tokenizer.empty())
        {
            print_error_log("Error: --speech-tokenizer is required in frontend-only mode.\n");
            ok = false;
        }
        if (options.campplus.empty())
        {
            print_error_log("Error: --campplus is required in frontend-only mode.\n");
            ok = false;
        }
        if (!options.seed.empty())
            print_warning_log("Warning: --seed is ignored in frontend-only mode.\n");
        if (options.n_threads != 0)
            print_warning_log("Warning: --threads is ignored in frontend-only mode.\n");
        if (has_generation_overrides(options))
            print_warning_log("Warning: sampling overrides are ignored in frontend-only mode.\n");
#ifdef COSYVOICE_NO_AUDIO
        if (options.prompt_audio_16k.empty())
        {
            print_error_log("Error: --prompt-audio-16k is required in frontend-only mode when audio is not available.\n");
            ok = false;
        }
        if (options.prompt_audio_24k.empty())
        {
            print_error_log("Error: --prompt-audio-24k is required in frontend-only mode when audio is not available.\n");
            ok = false;
        }
#else
        if (options.prompt_audio.empty())
        {
            print_error_log("Error: --prompt-audio is required in frontend-only mode.\n");
            ok = false;
        }
#endif
        if (options.prompt_speech_output.empty())
        {
            print_error_log("Error: --prompt-speech-output is required in frontend-only mode.\n");
            ok = false;
        }
        return ok;
    }
#endif

    bool ok = true;
    if (options.model.empty())
    {
        print_error_log("Error: --model is required for TTS.\n");
        ok = false;
    }
    if (options.text.empty())
    {
        print_error_log("Error: --text is required for TTS.\n");
        ok = false;
    }
    if (options.output.empty())
    {
        print_error_log("Error: --output is required for TTS.\n");
        ok = false;
    }

    const bool has_prompt_speech = !options.prompt_speech.empty();
#ifndef COSYVOICE_NO_FRONTEND
    const bool has_any_frontend_input = !options.speech_tokenizer.empty()
        || !options.campplus.empty()
#ifdef COSYVOICE_NO_AUDIO
        || !options.prompt_audio_16k.empty()
        || !options.prompt_audio_24k.empty()
#else
        || !options.prompt_audio.empty()
#endif
        || !options.prompt_text.empty();

    if (has_prompt_speech && has_any_frontend_input)
    {
        print_error_log("Error: use either --prompt-speech or frontend inputs, not both.\n");
        ok = false;
    }
#endif

    if (options.mode == "auto")
    {
    auto_detect:
        if (!options.instruction.empty())
            options.mode = "instruct";
        else
            options.mode = "zero-shot";
        mode_auto_detected = true;
    }
    else if (options.mode == "instruct")
    {
        if (options.instruction.empty())
        {
            print_warning_log("Warning: --instruction is not provided, falling back to zero-shot mode.\n");
            options.mode = "zero-shot";
        }
    }
    else if (options.mode == "zero-shot")
    {
        if (!options.instruction.empty())
            print_warning_log("Warning: --instruction is provided but will be ignored in zero-shot mode.\n");
    }
    else if (options.mode != "cross-lingual")
    {
        print_warning_log("Warning: unrecognized mode \"%s\", auto-detecting mode.\n", options.mode.c_str());
        goto auto_detect;
    }

    if (!has_prompt_speech)
    {
#ifndef COSYVOICE_NO_FRONTEND
        if (options.speech_tokenizer.empty())
        {
            print_error_log("Error: --speech-tokenizer is required when --prompt-speech is not provided.\n");
            ok = false;
        }
        if (options.campplus.empty())
        {
            print_error_log("Error: --campplus is required when --prompt-speech is not provided.\n");
            ok = false;
        }
#ifdef COSYVOICE_NO_AUDIO
        if (options.prompt_audio_16k.empty())
        {
            print_error_log("Error: --prompt-audio-16k is required when --prompt-speech is not provided.\n");
            ok = false;
        }
        if (options.prompt_audio_24k.empty())
        {
            print_error_log("Error: --prompt-audio-24k is required when --prompt-speech is not provided.\n");
            ok = false;
        }
#else
        if (options.prompt_audio.empty())
        {
            print_error_log("Error: --prompt-audio is required when --prompt-speech is not provided.\n");
            ok = false;
        }
#endif
        if (options.prompt_text.empty())
        {
            if (options.mode == "zero-shot")
            {
                print_error_log("Error: --prompt-text is required when mode is zero-shot.\n");
                ok = false;
            }
        }
        else if (options.mode != "zero-shot")
            print_warning_log("Warning: --prompt-text is ignored when mode is not zero-shot.\n");
#else
        print_error_log("Error: --prompt-speech is required when the frontend is not available.\n");
#endif
    }

    if (!options.seed.empty())
    {
        uint32_t seed_value;
        if (!parse_uint32_arg(options.seed, &seed_value))
        {
            print_error_log("Error: invalid --seed value \"%s\". It should be a non-negative integer between 0 and %u.\n", options.seed.c_str(), UINT32_MAX);
            ok = false;
        }
    }

    if (options.has_temperature && options.temperature <= 0.0f)
    {
        print_error_log("Error: --temperature must be > 0.\n");
        ok = false;
    }

    if (options.has_top_k && options.top_k < 0)
    {
        print_error_log("Error: --top-k must be >= 0.\n");
        ok = false;
    }

    if (options.has_top_p && (options.top_p < 0.0f || options.top_p > 1.0f))
    {
        print_error_log("Error: --top-p must be in [0, 1].\n");
        ok = false;
    }

    if (options.has_llm_kv_cache_type
        && (options.llm_kv_cache_type < 0 || options.llm_kv_cache_type >= COSYVOICE_LLM_KV_CACHE_TYPE_COUNT))
    {
        print_error_log("Error: invalid --llm-kv-cache-type. Allowed values: f32, f16, q8_0, q5_1, q5_0, q4_1, q4_0.\n");
        ok = false;
    }

    if (options.has_win_size && options.win_size <= 0)
    {
        print_error_log("Error: --win-size must be > 0.\n");
        ok = false;
    }

    if (options.has_tau_r && options.tau_r < 0.0f)
    {
        print_error_log("Error: --tau-r must be >= 0.\n");
        ok = false;
    }

    if (options.has_min_token_text_ratio && options.min_token_text_ratio < 0.0f)
    {
        print_error_log("Error: --min-token-text-ratio must be >= 0.\n");
        ok = false;
    }

    if (options.has_max_token_text_ratio && options.max_token_text_ratio < 0.0f)
    {
        print_error_log("Error: --max-token-text-ratio must be >= 0.\n");
        ok = false;
    }

    if (options.has_min_token_text_ratio && options.has_max_token_text_ratio
        && options.max_token_text_ratio < options.min_token_text_ratio)
    {
        print_error_log("Error: --max-token-text-ratio must be >= --min-token-text-ratio.\n");
        ok = false;
    }

    if (mode_auto_detected && ok && !options.quiet)
        printf("Info: auto-detected mode \"%s\".\n", options.mode.c_str());

    return ok;
}

int tool_entry(int argc, char** argv)
{
    if (argc == 1)
    {
        print_usage(argv[0]);
        return 0;
    }

    cli_options options;
    for (int i = 1; i != argc; ++i)
    {
        auto arg = argv[i];
        auto get_arg_value = [&]() -> char*
        {
            if (++i == argc)
            {
                auto option = arg;
                print_error_log("Error: missing value for the command-line option \"%s\".\n", option);
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
        else if (str_casecmp(arg, "--max-llm-len") == 0)
        {
            auto value = get_arg_value();
            uint32_t max_llm_len;
            if (!parse_uint32_arg(value, &max_llm_len) || max_llm_len <= 0)
            {
                print_error_log("Error: invalid --max-llm-len value \"%s\".\n", value);
                return 1;
            }
            options.max_llm_len = max_llm_len;
        }
        else if (str_casecmp(arg, "--threads") == 0 || str_casecmp(arg, "-j") == 0)
        {
            auto value = get_arg_value();
            uint32_t n_threads;
            if (!parse_uint32_arg(value, &n_threads))
            {
                print_error_log("Error: invalid --threads value \"%s\". It should be a non-negative integer between 0 and %u.\n", value, UINT32_MAX);
                return 1;
            }
            options.n_threads = n_threads;
        }
        else if (str_casecmp(arg, "--llm-kv-cache-type") == 0)
        {
            auto value = get_arg_value();
            cosyvoice_llm_kv_cache_type_t type;
            if (!parse_llm_kv_cache_type_arg(value, &type))
            {
                print_error_log("Error: invalid --llm-kv-cache-type value \"%s\".\n", value);
                return 1;
            }
            options.llm_kv_cache_type = type;
            options.has_llm_kv_cache_type = true;
        }
#ifndef COSYVOICE_NO_FRONTEND
        else if (str_casecmp(arg, "--frontend-only") == 0)
            options.frontend_only = true;
        else if (str_casecmp(arg, "--speech-tokenizer") == 0)
            options.speech_tokenizer = get_arg_value();
        else if (str_casecmp(arg, "--campplus") == 0)
            options.campplus = get_arg_value();
    #ifdef COSYVOICE_NO_AUDIO
        else if (str_casecmp(arg, "--prompt-audio-16k") == 0)
            options.prompt_audio_16k = get_arg_value();
        else if (str_casecmp(arg, "--prompt-audio-24k") == 0)
            options.prompt_audio_24k = get_arg_value();
    #else
        else if (str_casecmp(arg, "--prompt-audio") == 0)
            options.prompt_audio = get_arg_value();
    #endif
        else if (str_casecmp(arg, "--prompt-text") == 0)
            options.prompt_text = get_arg_value();
        else if (str_casecmp(arg, "--prompt-speech-output") == 0)
            options.prompt_speech_output = get_arg_value();
#endif
        else if (str_casecmp(arg, "--prompt-speech") == 0)
            options.prompt_speech = get_arg_value();
        else if (str_casecmp(arg, "--text") == 0 || str_casecmp(arg, "-t") == 0)
            options.text = get_arg_value();
        else if (str_casecmp(arg, "--instruction") == 0 || str_casecmp(arg, "-i") == 0)
            options.instruction = get_arg_value();
        else if (str_casecmp(arg, "--output") == 0 || str_casecmp(arg, "-o") == 0)
            options.output = get_arg_value();
        else if (str_casecmp(arg, "--mode") == 0)
            options.mode = to_lower(get_arg_value());
        else if (str_casecmp(arg, "--verbose") == 0 || str_casecmp(arg, "-v") == 0)
            options.verbose = true;
        else if (str_casecmp(arg, "--quiet") == 0 || str_casecmp(arg, "-q") == 0)
            options.quiet = true;
#ifndef COSYVOICE_NO_ICU
        else if (str_casecmp(arg, "--disable-text-normalization") == 0)
            options.text_normalization_enabled = false;
#endif
        else if (str_casecmp(arg, "--speed") == 0 || str_casecmp(arg, "-s") == 0)
        {
            auto value = get_arg_value();
            float speed;
            if (!parse_float_arg(value, &speed) || speed <= 0.0f)
            {
                print_error_log("Error: invalid --speed value \"%s\".\n", value);
                return 1;
            }
            options.speed = speed;
        }
        else if (str_casecmp(arg, "--seed") == 0)
            options.seed = get_arg_value();
        else if (str_casecmp(arg, "--temperature") == 0)
        {
            auto value = get_arg_value();
            float v;
            if (!parse_float_arg(value, &v))
            {
                print_error_log("Error: invalid --temperature value \"%s\".\n", value);
                return 1;
            }
            options.temperature = v;
            options.has_temperature = true;
        }
        else if (str_casecmp(arg, "--top-k") == 0)
        {
            auto value = get_arg_value();
            int v;
            if (!parse_int_arg(value, &v))
            {
                print_error_log("Error: invalid --top-k value \"%s\".\n", value);
                return 1;
            }
            options.top_k = v;
            options.has_top_k = true;
        }
        else if (str_casecmp(arg, "--top-p") == 0)
        {
            auto value = get_arg_value();
            float v;
            if (!parse_float_arg(value, &v))
            {
                print_error_log("Error: invalid --top-p value \"%s\".\n", value);
                return 1;
            }
            options.top_p = v;
            options.has_top_p = true;
        }
        else if (str_casecmp(arg, "--win-size") == 0)
        {
            auto value = get_arg_value();
            int v;
            if (!parse_int_arg(value, &v))
            {
                print_error_log("Error: invalid --win-size value \"%s\".\n", value);
                return 1;
            }
            options.win_size = v;
            options.has_win_size = true;
        }
        else if (str_casecmp(arg, "--tau-r") == 0)
        {
            auto value = get_arg_value();
            float v;
            if (!parse_float_arg(value, &v))
            {
                print_error_log("Error: invalid --tau-r value \"%s\".\n", value);
                return 1;
            }
            options.tau_r = v;
            options.has_tau_r = true;
        }
        else if (str_casecmp(arg, "--min-token-text-ratio") == 0)
        {
            auto value = get_arg_value();
            float v;
            if (!parse_float_arg(value, &v))
            {
                print_error_log("Error: invalid --min-token-text-ratio value \"%s\".\n", value);
                return 1;
            }
            options.min_token_text_ratio = v;
            options.has_min_token_text_ratio = true;
        }
        else if (str_casecmp(arg, "--max-token-text-ratio") == 0)
        {
            auto value = get_arg_value();
            float v;
            if (!parse_float_arg(value, &v))
            {
                print_error_log("Error: invalid --max-token-text-ratio value \"%s\".\n", value);
                return 1;
            }
            options.max_token_text_ratio = v;
            options.has_max_token_text_ratio = true;
        }
        else
        {
            auto option = arg;
            print_error_log("Error: the program doesn't recognize the command-line option \"%s\".\n", option);
            return 1;
        }
    }

    g_quiet_logs = options.quiet;
    if (!validate_options(options))
        return 1;

    const auto log_level = get_log_level(options);
    cli_timing_info timing;
    const auto total_start = std::chrono::steady_clock::now();

#ifndef COSYVOICE_NO_FRONTEND
    if (!options.frontend_only)
        print_preload_run_info(options, log_level);
#endif

    auto stage_start = std::chrono::steady_clock::now();
    cosyvoice_init_backend_from_path(options.backend_path.empty() ? nullptr : options.backend_path.c_str());
    timing.backend_init_ms = elapsed_ms(stage_start, std::chrono::steady_clock::now());

    cosyvoice_prompt_speech_handle prompt_speech;
    stage_start = std::chrono::steady_clock::now();
    if (!options.prompt_speech.empty())
    {
        prompt_speech.reset(cosyvoice_prompt_speech_load_from_file(options.prompt_speech.c_str()));
        if (!prompt_speech)
        {
            print_error_log("Error: failed to load prompt_speech file \"%s\".\n", options.prompt_speech.c_str());
            if (errno != 0)
                print_error_log("Reason: %s\n", strerror(errno));
            return 1;
        }
    }
#ifndef COSYVOICE_NO_FRONTEND
    else
    {
        cosyvoice_frontend_handle frontend(cosyvoice_frontend_load_from_files(options.speech_tokenizer.c_str(), options.campplus.c_str()));
        if (!frontend)
        {
            print_error_log("Error: failed to load frontend models.\n");
            return 1;
        }
    #ifdef COSYVOICE_NO_AUDIO
        auto f = open_ifstream_utf8(options.prompt_audio_16k.c_str());
        if (!f)
        {
            print_error_log("Error: failed to open prompt audio file \"%s\".\n", options.prompt_audio_16k.c_str());
            return 1;
        }
        f.seekg(0, std::ios::end);
        const auto prompt_audio_16k_end = f.tellg();
        if (prompt_audio_16k_end == std::streampos(-1))
        {
            print_error_log("Error: failed to query prompt audio file \"%s\".\n", options.prompt_audio_16k.c_str());
            return 1;
        }
        const auto prompt_audio_16k_size = static_cast<std::streamoff>(prompt_audio_16k_end);
        if (prompt_audio_16k_size <= 0)
        {
            print_error_log("Error: prompt audio file \"%s\" is empty.\n", options.prompt_audio_16k.c_str());
            return 1;
        }
        if (prompt_audio_16k_size % static_cast<std::streamoff>(sizeof(float)) != 0)
        {
            print_error_log("Error: prompt audio file \"%s\" size is not aligned to float samples.\n", options.prompt_audio_16k.c_str());
            return 1;
        }
        if (prompt_audio_16k_size / static_cast<std::streamoff>(sizeof(float)) > static_cast<std::streamoff>(std::numeric_limits<uint32_t>::max()))
        {
            print_error_log("Error: prompt audio file \"%s\" is too large.\n", options.prompt_audio_16k.c_str());
            return 1;
        }
        uint32_t prompt_audio_16k_length = static_cast<uint32_t>(prompt_audio_16k_size / static_cast<std::streamoff>(sizeof(float)));
        auto prompt_audio_16k_data = std::make_unique<float[]>(prompt_audio_16k_length);
        f.seekg(0, std::ios::beg);
        f.read(reinterpret_cast<char*>(prompt_audio_16k_data.get()), static_cast<std::streamsize>(prompt_audio_16k_length) * static_cast<std::streamsize>(sizeof(float)));
        if (!f)
        {
            print_error_log("Error: failed to read prompt audio file \"%s\".\n", options.prompt_audio_16k.c_str());
            return 1;
        }

        f = open_ifstream_utf8(options.prompt_audio_24k.c_str());
        if (!f)
        {
            print_error_log("Error: failed to open prompt audio file \"%s\".\n", options.prompt_audio_24k.c_str());
            return 1;
        }
        f.seekg(0, std::ios::end);
        const auto prompt_audio_24k_end = f.tellg();
        if (prompt_audio_24k_end == std::streampos(-1))
        {
            print_error_log("Error: failed to query prompt audio file \"%s\".\n", options.prompt_audio_24k.c_str());
            return 1;
        }
        const auto prompt_audio_24k_size = static_cast<std::streamoff>(prompt_audio_24k_end);
        if (prompt_audio_24k_size <= 0)
        {
            print_error_log("Error: prompt audio file \"%s\" is empty.\n", options.prompt_audio_24k.c_str());
            return 1;
        }
        if (prompt_audio_24k_size % static_cast<std::streamoff>(sizeof(float)) != 0)
        {
            print_error_log("Error: prompt audio file \"%s\" size is not aligned to float samples.\n", options.prompt_audio_24k.c_str());
            return 1;
        }
        if (prompt_audio_24k_size / static_cast<std::streamoff>(sizeof(float)) > static_cast<std::streamoff>(std::numeric_limits<uint32_t>::max()))
        {
            print_error_log("Error: prompt audio file \"%s\" is too large.\n", options.prompt_audio_24k.c_str());
            return 1;
        }
        uint32_t prompt_audio_24k_length = static_cast<uint32_t>(prompt_audio_24k_size / static_cast<std::streamoff>(sizeof(float)));
        auto prompt_audio_24k_data = std::make_unique<float[]>(prompt_audio_24k_length);
        f.seekg(0, std::ios::beg);
        f.read(reinterpret_cast<char*>(prompt_audio_24k_data.get()), static_cast<std::streamsize>(prompt_audio_24k_length) * static_cast<std::streamsize>(sizeof(float)));
        if (!f)
        {
            print_error_log("Error: failed to read prompt audio file \"%s\".\n", options.prompt_audio_24k.c_str());
            return 1;
        }

        prompt_speech.reset(cosyvoice_frontend_prompt_speech_direct(frontend.get(), prompt_audio_16k_data.get(), prompt_audio_16k_length, prompt_audio_24k_data.get(), prompt_audio_24k_length, options.prompt_text.c_str()));
        if (!prompt_speech)
        {
            print_error_log("Error: failed to generate prompt_speech from frontend inputs.\n");
            return 1;
        }
    #else
        float* prompt_audio_data = nullptr;
        uint32_t prompt_audio_length = 0;
        uint32_t prompt_audio_sample_rate = 0;
        if (!cosyvoice_audio_load_from_file(options.prompt_audio.c_str(), &prompt_audio_data, &prompt_audio_length, &prompt_audio_sample_rate))
        {
            print_error_log("Error: failed to load prompt audio file \"%s\".\n", options.prompt_audio.c_str());
            return 1;
        }

        audio_buffer_handle audio(prompt_audio_data);
        prompt_speech.reset(cosyvoice_frontend_prompt_speech(frontend.get(), audio.get(), prompt_audio_length, prompt_audio_sample_rate, options.prompt_text.c_str()));
        if (!prompt_speech)
        {
            print_error_log("Error: failed to generate prompt_speech from frontend inputs.\n");
            return 1;
        }
    #endif
    }

    if (!options.prompt_speech_output.empty() && !cosyvoice_prompt_speech_save_to_file(prompt_speech.get(), options.prompt_speech_output.c_str()))
    {
        print_error_log("Error: failed to save prompt_speech file \"%s\".\n", options.prompt_speech_output.c_str());
        return 1;
    }
    timing.prompt_prepare_ms = elapsed_ms(stage_start, std::chrono::steady_clock::now());

    if (options.frontend_only)
    {
        timing.total_ms = elapsed_ms(total_start, std::chrono::steady_clock::now());
        print_frontend_runtime_info(options, log_level);
        print_timing_info(timing, false, log_level);
        if (log_level != cli_log_level::quiet)
            printf("\n-----------------------------------------------\n");
        return 0;
    }
#endif

    cosyvoice_context_params_t params;
    cosyvoice_init_default_context_params(&params);
    params.inference_buffer_policy = COSYVOICE_INFERENCE_BUFFER_POLICY_SHARED;
    params.n_max_seq = options.max_llm_len;
    if (options.has_llm_kv_cache_type)
        params.llm_kv_cache_type = options.llm_kv_cache_type;
    if (!options.seed.empty())
        params.seed = stoul(options.seed);
    stage_start = std::chrono::steady_clock::now();
    loading_spinner model_loading_spinner(log_level, "Loading model");
    model_loading_spinner.start();

    ggml_backend_t backend = ggml_backend_init_best();
    cosyvoice_context_handle ctx(cosyvoice_load_from_file_ext(options.model.c_str(), &params, backend, options.n_threads, 0));
    model_loading_spinner.stop(ctx != nullptr);
    timing.model_load_ms = elapsed_ms(stage_start, std::chrono::steady_clock::now());
    if (!ctx)
    {
        print_error_log("Error: failed to load model file \"%s\".\n", options.model.c_str());
        return 1;
    }

    cosyvoice_prompt_handle prompt(cosyvoice_prompt_init_from_prompt_speech(ctx.get(), prompt_speech.get()));
    if (!prompt)
    {
        print_error_log("Error: failed to initialize prompt from prompt_speech.\n");
        return 1;
    }

    cosyvoice_tts_context_handle tts_ctx(cosyvoice_tts_context_new(ctx.get(), prompt.get()));
    if (!tts_ctx)
    {
        print_error_log("Error: failed to create TTS context.\n");
        return 1;
    }
#ifndef COSYVOICE_NO_ICU
    cosyvoice_tts_context_set_text_normalization_enabled(tts_ctx.get(), options.text_normalization_enabled);
#endif
    cosyvoice_context_params_t effective_params;
    cosyvoice_get_context_params(ctx.get(), &effective_params);
    cosyvoice_generation_config_t generation_config;
    cosyvoice_get_generation_config(ctx.get(), &generation_config);
    if (has_generation_overrides(options))
    {
        if (options.has_temperature)
            generation_config.temperature = options.temperature;
        if (options.has_top_k)
            generation_config.sampling.top_k = options.top_k;
        if (options.has_top_p)
            generation_config.sampling.top_p = options.top_p;
        if (options.has_win_size)
            generation_config.sampling.win_size = options.win_size;
        if (options.has_tau_r)
            generation_config.sampling.tau_r = options.tau_r;
        if (options.has_min_token_text_ratio)
            generation_config.min_token_text_ratio = options.min_token_text_ratio;
        if (options.has_max_token_text_ratio)
            generation_config.max_token_text_ratio = options.max_token_text_ratio;

        if (!cosyvoice_set_generation_config(ctx.get(), &generation_config))
        {
            print_error_log("Error: invalid sampling override values.\n");
            return 1;
        }
        cosyvoice_get_generation_config(ctx.get(), &generation_config);
    }
    const uint32_t sample_rate = cosyvoice_get_sample_rate(ctx.get());
    cosyvoice_memory_usage_t memory_usage_before;
    cosyvoice_get_memory_usage(ctx.get(), &memory_usage_before);
    print_tts_runtime_info(options, backend, effective_params, generation_config, sample_rate, log_level);

    cosyvoice_generated_speech result = {};
    stage_start = std::chrono::steady_clock::now();
    bool ok = false;
    if (options.mode == "cross-lingual")
        ok = cosyvoice_tts_cross_lingual(tts_ctx.get(), options.text.c_str(), options.speed, &result);
    else if (options.mode == "zero-shot")
        ok = cosyvoice_tts_zero_shot(tts_ctx.get(), options.text.c_str(), options.speed, &result);
    else
        ok = cosyvoice_tts_instruct(tts_ctx.get(), options.text.c_str(), options.instruction.c_str(), options.speed, &result);
    timing.tts_generate_ms = elapsed_ms(stage_start, std::chrono::steady_clock::now());

    if (!ok && (!result.data || result.length == 0))
    {
        print_error_log("Error: TTS generation failed.\n");
        return 1;
    }
    cosyvoice_memory_usage_t memory_usage_after;
    cosyvoice_get_memory_usage(ctx.get(), &memory_usage_after);
    print_memory_runtime_info(memory_usage_before, memory_usage_after, log_level);

    stage_start = std::chrono::steady_clock::now();
    if (!cosyvoice_audio_save_to_file(options.output.c_str(), result.data, result.length, sample_rate))
    {
        print_error_log("Error: failed to save output audio file \"%s\".\n", options.output.c_str());
        return 1;
    }
    timing.save_output_ms = elapsed_ms(stage_start, std::chrono::steady_clock::now());
    timing.total_ms = elapsed_ms(total_start, std::chrono::steady_clock::now());
    print_timing_info(timing, true, log_level);
    if (log_level != cli_log_level::quiet)
        printf("\n-----------------------------------------------\n");

    return 0;
}
