#ifdef _MSC_VER
    #define _CRT_SECURE_NO_WARNINGS
#endif

#include "tool_common_cosyvoice.h"

#ifndef COSYVOICE_NO_AUDIO
    #include "cosyvoice-audio.h"
#endif

#include "httplib.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

using json = nlohmann::json;

enum class server_log_level
{
    quiet,
    concise,
    verbose
};

enum class request_log_status
{
    ok,
    bad_request,
    unauthorized,
    not_found,
    failed
};

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
    uint16_t port = 8080;
    bool has_served_model_name = false;
    std::vector<voice_prompt_option> voice_prompts;

    std::string single_voice = "alloy";
    std::string single_prompt_speech;
    bool has_single_voice_arg = false;

    uint32_t max_llm_len = 2048;
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

struct voice_runtime
{
    std::string name;
    std::string prompt_speech_path;
    cosyvoice_prompt_speech_handle prompt_speech;
    cosyvoice_prompt_handle prompt;
    cosyvoice_tts_context_handle tts_context;
};

struct server_runtime
{
    server_options options;
    server_log_level log_level = server_log_level::concise;
    cosyvoice_context_handle model_context;
    uint32_t sample_rate = 0;
    cosyvoice_generation_config_t default_generation_config = {};

#ifndef COSYVOICE_NO_AUDIO
    cosyvoice_audio_encoder_handle audio_encoder;
#endif

    std::unordered_map<std::string, voice_runtime> voices;
    std::vector<std::string> voice_names;

    std::mt19937 seed_rng;
    std::mutex infer_mutex;
};

enum class response_audio_format
{
    mp3,
    wav,
    flac,
    pcm,
    aac,
    opus,
    unknown
};

struct speech_request
{
    std::string model;
    std::string input;
    std::string voice;
    std::string response_format = "mp3";
    float speed = 1.0f;
    std::string instructions;
    bool has_instructions = false;

    bool has_seed = false;
    uint32_t seed = 0;

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

struct request_log_context
{
    uint64_t id = 0;
    std::string endpoint;
    std::string remote_addr;
    std::string model;
    std::string voice;
    std::string response_format;
    bool has_instructions = false;
    bool has_seed = false;
    uint32_t seed = 0;
    std::chrono::steady_clock::time_point start;
};

static std::atomic<uint64_t> g_request_id = 0;

static bool log_level_enabled(server_log_level current, server_log_level required)
{
    return static_cast<int>(current) >= static_cast<int>(required);
}

static void log_message(server_log_level current, server_log_level required, const char* tag, const char* format, ...)
{
    if (!log_level_enabled(current, required))
        return;

    const auto ts = get_local_timestamp_ms();
    fprintf(stdout, "[%s] [%s] ", ts.c_str(), tag);

    va_list args;
    va_start(args, format);
    vfprintf(stdout, format, args);
    va_end(args);

    fprintf(stdout, "\n");
    fflush(stdout);
}

static void print_error_log(const char* format, ...)
{
    const auto ts = get_local_timestamp_ms();
    fprintf(stderr, "[%s] [ERROR] ", ts.c_str());

    va_list args;
    va_start(args, format);
    vfprintf(stderr, format, args);
    va_end(args);

    fflush(stderr);
}

static void print_info_log(server_log_level level, const char* format, ...)
{
    if (level == server_log_level::quiet)
        return;

    va_list args;
    va_start(args, format);
    vprintf(format, args);
    va_end(args);
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

static std::string join_voice_names(const std::vector<std::string>& voices)
{
    std::string result;
    for (size_t i = 0; i < voices.size(); ++i)
    {
        if (i != 0)
            result += ", ";
        result += voices[i];
    }
    return result;
}

static server_log_level get_log_level(const server_options& options)
{
    if (options.quiet)
        return server_log_level::quiet;
    if (options.verbose)
        return server_log_level::verbose;
    return server_log_level::concise;
}

static void print_usage(const tchar* argv0)
{
    const auto exe = tchar_to_utf8(argv0);
    printf("cosyvoice-server - OpenAI Speech compatible API server\n\n");
    printf("Usage:\n");
    printf("  %s --model <file.gguf> --voice-prompt <voice=prompt_speech.gguf> [--voice-prompt ...] [options]\n", exe.c_str());
    printf("  %s --model <file.gguf> --voice <voice> --prompt-speech <prompt_speech.gguf> [options]\n", exe.c_str());

    printf("\nCore options:\n");
    printf("  --help, -h                                  Show this help message and exit.\n");
    printf("  --model, -m <file>                          CosyVoice model file (.gguf).\n");
    printf("  --served-model-name <name>                  Exposed model name for API requests.\n");
    printf("  --host <host>                               Listen host. Default: 127.0.0.1.\n");
    printf("  --port <port>                               Listen port. Default: 8080.\n");
    printf("  --api-key <key>                             Require Bearer auth when provided.\n");

    printf("\nVoice mapping:\n");
    printf("  --voice-prompt <voice=prompt_speech.gguf>   Map one voice to one prompt_speech file. Repeatable.\n");
    printf("  --voice <voice>                             Voice name for single mapping mode. Default: alloy.\n");
    printf("  --prompt-speech <file>                      Prompt speech file for single mapping mode.\n");

    printf("\nRuntime options:\n");
    printf("  --max-llm-len <value>                       Maximum LLM sequence length. Default: 2048.\n");
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

    printf("\nPOST /v1/audio/speech request fields:\n");
    printf("  Required: model, input, voice\n");
    printf("  Optional standard: instructions, speed, response_format\n");
    printf("  Optional extensions: seed, temperature, top_k, top_p, win_size, tau_r,\n");
    printf("                      min_token_text_ratio, max_token_text_ratio\n");
    printf("  response_format standard values: mp3, opus, aac, flac, wav, pcm\n");
    printf("  Unsupported formats return OpenAI-style 400 error in this build/runtime.\n");
}

static bool validate_options(server_options& options)
{
    if (options.quiet && options.verbose)
    {
        print_error_log("Error: --quiet and --verbose cannot be used together.\n");
        return false;
    }

    if (options.model.empty())
    {
        print_error_log("Error: --model is required.\n");
        return false;
    }

    if (options.single_prompt_speech.empty() && options.has_single_voice_arg)
    {
        print_error_log("Error: --voice requires --prompt-speech in single mapping mode.\n");
        return false;
    }

    if (!options.single_prompt_speech.empty())
        options.voice_prompts.push_back({ options.single_voice, options.single_prompt_speech });

    if (options.voice_prompts.empty())
    {
        print_error_log("Error: at least one voice mapping is required. Use --voice-prompt or --prompt-speech.\n");
        return false;
    }

    std::unordered_map<std::string, bool> seen_voices;
    for (const auto& mapping : options.voice_prompts)
    {
        if (mapping.voice.empty() || mapping.prompt_speech.empty())
        {
            print_error_log("Error: invalid voice mapping. Voice name and prompt_speech path must be non-empty.\n");
            return false;
        }

        if (seen_voices.find(mapping.voice) != seen_voices.end())
        {
            print_error_log("Error: duplicate voice name \"%s\".\n", mapping.voice.c_str());
            return false;
        }
        seen_voices.emplace(mapping.voice, true);
    }

    if (options.max_llm_len == 0)
    {
        print_error_log("Error: --max-llm-len must be > 0.\n");
        return false;
    }

    if (options.has_temperature && options.temperature <= 0.0f)
    {
        print_error_log("Error: --temperature must be > 0.\n");
        return false;
    }

    if (options.has_top_k && options.top_k < 0)
    {
        print_error_log("Error: --top-k must be >= 0.\n");
        return false;
    }

    if (options.has_top_p && (options.top_p < 0.0f || options.top_p > 1.0f))
    {
        print_error_log("Error: --top-p must be in [0, 1].\n");
        return false;
    }

    if (options.has_win_size && options.win_size <= 0)
    {
        print_error_log("Error: --win-size must be > 0.\n");
        return false;
    }

    if (options.has_tau_r && options.tau_r < 0.0f)
    {
        print_error_log("Error: --tau-r must be >= 0.\n");
        return false;
    }

    if (options.has_min_token_text_ratio && options.min_token_text_ratio < 0.0f)
    {
        print_error_log("Error: --min-token-text-ratio must be >= 0.\n");
        return false;
    }

    if (options.has_max_token_text_ratio && options.max_token_text_ratio < 0.0f)
    {
        print_error_log("Error: --max-token-text-ratio must be >= 0.\n");
        return false;
    }

    if (options.has_min_token_text_ratio && options.has_max_token_text_ratio
        && options.max_token_text_ratio < options.min_token_text_ratio)
    {
        print_error_log("Error: --max-token-text-ratio must be >= --min-token-text-ratio.\n");
        return false;
    }

    return true;
}

static inline int16_t float_to_pcm16(float value)
{
    if (value > 1.0f)
        value = 1.0f;
    else if (value < -1.0f)
        value = -1.0f;

    const float scaled = value * 32767.0f;
    long rounded = lroundf(scaled);
    if (rounded < std::numeric_limits<int16_t>::min())
        rounded = std::numeric_limits<int16_t>::min();
    else if (rounded > std::numeric_limits<int16_t>::max())
        rounded = std::numeric_limits<int16_t>::max();
    return static_cast<int16_t>(rounded);
}

static bool build_pcm16_bytes(const float* data, uint32_t length, std::string* output, std::string* error)
{
    if (!data || length == 0)
    {
        *error = "Generated audio buffer is empty.";
        return false;
    }

    constexpr size_t kBytesPerSample = sizeof(int16_t);
    if (length > std::numeric_limits<size_t>::max() / kBytesPerSample)
    {
        *error = "Generated audio buffer is too large.";
        return false;
    }

    const size_t total_bytes = static_cast<size_t>(length) * kBytesPerSample;
    output->resize(total_bytes);
    for (uint32_t i = 0; i < length; ++i)
    {
        const auto sample = float_to_pcm16(data[i]);
        (*output)[2 * i] = static_cast<char>(sample & 0xFF);
        (*output)[2 * i + 1] = static_cast<char>((sample >> 8) & 0xFF);
    }
    return true;
}

static void append_u16_le(std::string* output, uint16_t value)
{
    output->push_back(static_cast<char>(value & 0xFF));
    output->push_back(static_cast<char>((value >> 8) & 0xFF));
}

static void append_u32_le(std::string* output, uint32_t value)
{
    output->push_back(static_cast<char>(value & 0xFF));
    output->push_back(static_cast<char>((value >> 8) & 0xFF));
    output->push_back(static_cast<char>((value >> 16) & 0xFF));
    output->push_back(static_cast<char>((value >> 24) & 0xFF));
}

static bool build_wav_bytes(const float* data, uint32_t length, uint32_t sample_rate, std::string* output, std::string* error)
{
    std::string pcm16;
    if (!build_pcm16_bytes(data, length, &pcm16, error))
        return false;

    const uint64_t data_size_64 = static_cast<uint64_t>(pcm16.size());
    if (data_size_64 > UINT32_MAX)
    {
        *error = "WAV payload is too large.";
        return false;
    }

    const auto data_size = static_cast<uint32_t>(data_size_64);
    const uint64_t riff_size_64 = 36ull + data_size;
    if (riff_size_64 > UINT32_MAX)
    {
        *error = "WAV RIFF size exceeds format limit.";
        return false;
    }

    const auto riff_size = static_cast<uint32_t>(riff_size_64);
    output->clear();
    output->reserve(44 + pcm16.size());

    output->append("RIFF", 4);
    append_u32_le(output, riff_size);
    output->append("WAVE", 4);

    output->append("fmt ", 4);
    append_u32_le(output, 16);
    append_u16_le(output, 1);
    append_u16_le(output, 1);
    append_u32_le(output, sample_rate);
    append_u32_le(output, sample_rate * 2);
    append_u16_le(output, 2);
    append_u16_le(output, 16);

    output->append("data", 4);
    append_u32_le(output, data_size);
    output->append(pcm16);
    return true;
}

static response_audio_format parse_response_format(const std::string& value)
{
    const auto lowered = to_lower(value);
    if (lowered == "mp3") return response_audio_format::mp3;
    if (lowered == "opus") return response_audio_format::opus;
    if (lowered == "aac") return response_audio_format::aac;
    if (lowered == "flac") return response_audio_format::flac;
    if (lowered == "wav") return response_audio_format::wav;
    if (lowered == "pcm") return response_audio_format::pcm;
    return response_audio_format::unknown;
}

static const char* response_format_to_string(response_audio_format format)
{
    switch (format)
    {
    case response_audio_format::mp3:
        return "mp3";
    case response_audio_format::opus:
        return "opus";
    case response_audio_format::aac:
        return "aac";
    case response_audio_format::flac:
        return "flac";
    case response_audio_format::wav:
        return "wav";
    case response_audio_format::pcm:
        return "pcm";
    default:
        return "unknown";
    }
}

static bool is_response_format_supported(const server_runtime& runtime, response_audio_format format)
{
    switch (format)
    {
    case response_audio_format::pcm:
        return true;
    case response_audio_format::wav:
#ifndef COSYVOICE_NO_AUDIO
        return static_cast<bool>(runtime.audio_encoder);
#else
        return true;
#endif
    case response_audio_format::mp3:
    case response_audio_format::opus:
    case response_audio_format::aac:
    case response_audio_format::flac:
    default:
        return false;
    }
}

static std::string supported_response_formats_to_string(const server_runtime& runtime)
{
    constexpr response_audio_format kStandardFormats[] = {
        response_audio_format::mp3,
        response_audio_format::opus,
        response_audio_format::aac,
        response_audio_format::flac,
        response_audio_format::wav,
        response_audio_format::pcm
    };

    std::string result;
    for (auto format : kStandardFormats)
    {
        if (!is_response_format_supported(runtime, format))
            continue;
        if (!result.empty())
            result += ", ";
        result += response_format_to_string(format);
    }

    if (result.empty())
        return "(none)";
    return result;
}

static const char* response_format_to_content_type(response_audio_format format)
{
    switch (format)
    {
    case response_audio_format::mp3:
        return "audio/mpeg";
    case response_audio_format::opus:
        return "audio/opus";
    case response_audio_format::aac:
        return "audio/aac";
    case response_audio_format::wav:
        return "audio/wav";
    case response_audio_format::flac:
        return "audio/flac";
    case response_audio_format::pcm:
        return "audio/pcm";
    default:
        return "application/octet-stream";
    }
}

#ifndef COSYVOICE_NO_AUDIO
static bool build_wav_bytes_with_audio_encoder(cosyvoice_audio_encoder_t encoder, const float* data, uint32_t length, std::string* output, std::string* error)
{
    if (!encoder)
    {
        *error = "WAV encoder is unavailable in this runtime environment.";
        return false;
    }

    if (!data || length == 0)
    {
        *error = "Generated audio buffer is empty.";
        return false;
    }

    if (!cosyvoice_audio_encoder_encode(encoder, data, length, COSYVOICE_AUDIO_ENCODING_FORMAT_WAV))
    {
        *error = "Failed to encode WAV payload with cosyvoice_audio_encoder.";
        return false;
    }

    const uint8_t* encoded = nullptr;
    uint32_t encoded_length = 0;
    cosyvoice_audio_encoder_get_encoded_data(encoder, &encoded, &encoded_length);

    if (!encoded || encoded_length == 0)
    {
        *error = "Encoded WAV payload is empty.";
        return false;
    }

    output->assign(reinterpret_cast<const char*>(encoded), static_cast<size_t>(encoded_length));
    return true;
}
#endif

static bool build_audio_payload(response_audio_format format, const cosyvoice_generated_speech& generated, const server_runtime& runtime, std::string* payload, std::string* error)
{
    switch (format)
    {
    case response_audio_format::wav:
#ifndef COSYVOICE_NO_AUDIO
        return build_wav_bytes_with_audio_encoder(runtime.audio_encoder.get(), generated.data, generated.length, payload, error);
#else
        return build_wav_bytes(generated.data, generated.length, runtime.sample_rate, payload, error);
#endif
    case response_audio_format::pcm:
        return build_pcm16_bytes(generated.data, generated.length, payload, error);
    default:
        *error = "Unsupported response format.";
        return false;
    }
}

static void set_openai_error(httplib::Response& res, int status, const std::string& message, const std::string& type, const char* param, const char* code)
{
    json payload;
    payload["error"] = {
        { "message", message },
        { "type", type },
        { "param", param ? json(param) : json(nullptr) },
        { "code", code ? json(code) : json(nullptr) }
    };

    res.status = status;
    res.set_content(payload.dump(), "application/json");
}

static bool require_auth(const httplib::Request& req, const server_options& options, httplib::Response& res)
{
    if (options.api_key.empty())
        return true;

    const std::string auth = req.get_header_value("Authorization");
    constexpr const char* kPrefix = "Bearer ";
    constexpr size_t kPrefixLen = 7;

    if (auth.size() <= kPrefixLen || auth.compare(0, kPrefixLen, kPrefix) != 0)
    {
        res.set_header("WWW-Authenticate", "Bearer");
        set_openai_error(res, 401, "Missing or invalid Authorization header.", "invalid_request_error", "Authorization", "invalid_api_key");
        return false;
    }

    const auto token = trim_copy(auth.substr(kPrefixLen));
    if (token != options.api_key)
    {
        res.set_header("WWW-Authenticate", "Bearer");
        set_openai_error(res, 401, "Invalid API key.", "invalid_request_error", "Authorization", "invalid_api_key");
        return false;
    }

    return true;
}

static bool parse_required_string_field(const json& body, const char* key, std::string* value, httplib::Response& res)
{
    const auto it = body.find(key);
    if (it == body.end())
    {
        set_openai_error(res, 400, std::string("Missing required field: ") + key, "invalid_request_error", key, "missing_parameter");
        return false;
    }

    if (!it->is_string())
    {
        set_openai_error(res, 400, std::string("Field must be a string: ") + key, "invalid_request_error", key, "invalid_type");
        return false;
    }

    *value = it->get<std::string>();
    if (value->empty())
    {
        set_openai_error(res, 400, std::string("Field must be non-empty: ") + key, "invalid_request_error", key, "invalid_value");
        return false;
    }

    return true;
}

static bool parse_speech_request_json(const json& body, speech_request* request, httplib::Response& res)
{
    if (!parse_required_string_field(body, "model", &request->model, res))
        return false;
    if (!parse_required_string_field(body, "input", &request->input, res))
        return false;
    if (!parse_required_string_field(body, "voice", &request->voice, res))
        return false;

    if (const auto it = body.find("response_format"); it != body.end())
    {
        if (!it->is_string())
        {
            set_openai_error(res, 400, "Field must be a string: response_format", "invalid_request_error", "response_format", "invalid_type");
            return false;
        }
        request->response_format = to_lower(it->get<std::string>());
    }

    if (const auto it = body.find("speed"); it != body.end())
    {
        if (!it->is_number())
        {
            set_openai_error(res, 400, "Field must be a number: speed", "invalid_request_error", "speed", "invalid_type");
            return false;
        }
        request->speed = it->get<float>();
        if (!std::isfinite(request->speed) || request->speed <= 0.0f)
        {
            set_openai_error(res, 400, "Field must be > 0: speed", "invalid_request_error", "speed", "invalid_value");
            return false;
        }
    }

    if (const auto it = body.find("seed"); it != body.end())
    {
        if (!it->is_number_integer())
        {
            set_openai_error(res, 400, "Field must be an integer: seed", "invalid_request_error", "seed", "invalid_type");
            return false;
        }
        const auto value = it->get<long long>();
        if (value < 0 || static_cast<unsigned long long>(value) > UINT32_MAX)
        {
            set_openai_error(res, 400, "Field must be in [0, 4294967295]: seed", "invalid_request_error", "seed", "invalid_value");
            return false;
        }
        request->seed = static_cast<uint32_t>(value);
        request->has_seed = true;
    }

    if (const auto it = body.find("temperature"); it != body.end())
    {
        if (!it->is_number())
        {
            set_openai_error(res, 400, "Field must be a number: temperature", "invalid_request_error", "temperature", "invalid_type");
            return false;
        }
        request->temperature = it->get<float>();
        if (!std::isfinite(request->temperature) || request->temperature <= 0.0f)
        {
            set_openai_error(res, 400, "Field must be > 0: temperature", "invalid_request_error", "temperature", "invalid_value");
            return false;
        }
        request->has_temperature = true;
    }

    if (const auto it = body.find("top_k"); it != body.end())
    {
        if (!it->is_number_integer())
        {
            set_openai_error(res, 400, "Field must be an integer: top_k", "invalid_request_error", "top_k", "invalid_type");
            return false;
        }
        const auto value = it->get<long long>();
        if (value < static_cast<long long>(std::numeric_limits<int>::min())
            || value > static_cast<long long>(std::numeric_limits<int>::max()))
        {
            set_openai_error(res, 400, "Field is out of range for int: top_k", "invalid_request_error", "top_k", "invalid_value");
            return false;
        }
        request->top_k = static_cast<int>(value);
        if (request->top_k < 0)
        {
            set_openai_error(res, 400, "Field must be >= 0: top_k", "invalid_request_error", "top_k", "invalid_value");
            return false;
        }
        request->has_top_k = true;
    }

    if (const auto it = body.find("top_p"); it != body.end())
    {
        if (!it->is_number())
        {
            set_openai_error(res, 400, "Field must be a number: top_p", "invalid_request_error", "top_p", "invalid_type");
            return false;
        }
        request->top_p = it->get<float>();
        if (!std::isfinite(request->top_p) || request->top_p < 0.0f || request->top_p > 1.0f)
        {
            set_openai_error(res, 400, "Field must be in [0, 1]: top_p", "invalid_request_error", "top_p", "invalid_value");
            return false;
        }
        request->has_top_p = true;
    }

    if (const auto it = body.find("win_size"); it != body.end())
    {
        if (!it->is_number_integer())
        {
            set_openai_error(res, 400, "Field must be an integer: win_size", "invalid_request_error", "win_size", "invalid_type");
            return false;
        }
        const auto value = it->get<long long>();
        if (value < static_cast<long long>(std::numeric_limits<int>::min())
            || value > static_cast<long long>(std::numeric_limits<int>::max()))
        {
            set_openai_error(res, 400, "Field is out of range for int: win_size", "invalid_request_error", "win_size", "invalid_value");
            return false;
        }
        request->win_size = static_cast<int>(value);
        if (request->win_size <= 0)
        {
            set_openai_error(res, 400, "Field must be > 0: win_size", "invalid_request_error", "win_size", "invalid_value");
            return false;
        }
        request->has_win_size = true;
    }

    if (const auto it = body.find("tau_r"); it != body.end())
    {
        if (!it->is_number())
        {
            set_openai_error(res, 400, "Field must be a number: tau_r", "invalid_request_error", "tau_r", "invalid_type");
            return false;
        }
        request->tau_r = it->get<float>();
        if (!std::isfinite(request->tau_r) || request->tau_r < 0.0f)
        {
            set_openai_error(res, 400, "Field must be >= 0: tau_r", "invalid_request_error", "tau_r", "invalid_value");
            return false;
        }
        request->has_tau_r = true;
    }

    if (const auto it = body.find("min_token_text_ratio"); it != body.end())
    {
        if (!it->is_number())
        {
            set_openai_error(res, 400, "Field must be a number: min_token_text_ratio", "invalid_request_error", "min_token_text_ratio", "invalid_type");
            return false;
        }
        request->min_token_text_ratio = it->get<float>();
        if (!std::isfinite(request->min_token_text_ratio) || request->min_token_text_ratio < 0.0f)
        {
            set_openai_error(res, 400, "Field must be >= 0: min_token_text_ratio", "invalid_request_error", "min_token_text_ratio", "invalid_value");
            return false;
        }
        request->has_min_token_text_ratio = true;
    }

    if (const auto it = body.find("max_token_text_ratio"); it != body.end())
    {
        if (!it->is_number())
        {
            set_openai_error(res, 400, "Field must be a number: max_token_text_ratio", "invalid_request_error", "max_token_text_ratio", "invalid_type");
            return false;
        }
        request->max_token_text_ratio = it->get<float>();
        if (!std::isfinite(request->max_token_text_ratio) || request->max_token_text_ratio < 0.0f)
        {
            set_openai_error(res, 400, "Field must be >= 0: max_token_text_ratio", "invalid_request_error", "max_token_text_ratio", "invalid_value");
            return false;
        }
        request->has_max_token_text_ratio = true;
    }

    if (request->has_min_token_text_ratio && request->has_max_token_text_ratio
        && request->max_token_text_ratio < request->min_token_text_ratio)
    {
        set_openai_error(res, 400, "max_token_text_ratio must be >= min_token_text_ratio", "invalid_request_error", "max_token_text_ratio", "invalid_value");
        return false;
    }

    const auto it_instructions = body.find("instructions");
    const auto it_instruction = body.find("instruction");
    if (it_instructions != body.end())
    {
        if (it_instructions->is_null())
        {
            request->instructions.clear();
            request->has_instructions = false;
        }
        else if (!it_instructions->is_string())
        {
            set_openai_error(res, 400, "Field must be a string: instructions", "invalid_request_error", "instructions", "invalid_type");
            return false;
        }
        else
        {
            request->instructions = it_instructions->get<std::string>();
            request->has_instructions = !request->instructions.empty();
        }
    }
    else if (it_instruction != body.end())
    {
        if (it_instruction->is_null())
        {
            request->instructions.clear();
            request->has_instructions = false;
        }
        else if (!it_instruction->is_string())
        {
            set_openai_error(res, 400, "Field must be a string: instruction", "invalid_request_error", "instruction", "invalid_type");
            return false;
        }
        else
        {
            request->instructions = it_instruction->get<std::string>();
            request->has_instructions = !request->instructions.empty();
        }
    }

    return true;
}

static bool apply_generation_overrides(
    const speech_request& request,
    server_runtime* runtime,
    cosyvoice_context_t model_ctx,
    uint32_t* applied_seed,
    std::string* error_message)
{
    bool has_override =
        request.has_temperature
        || request.has_top_k
        || request.has_top_p
        || request.has_win_size
        || request.has_tau_r
        || request.has_min_token_text_ratio
        || request.has_max_token_text_ratio;

    cosyvoice_generation_config_t cfg = runtime->default_generation_config;
    if (has_override)
    {
        if (request.has_temperature)
            cfg.temperature = request.temperature;
        if (request.has_top_k)
            cfg.sampling.top_k = request.top_k;
        if (request.has_top_p)
            cfg.sampling.top_p = request.top_p;
        if (request.has_win_size)
            cfg.sampling.win_size = request.win_size;
        if (request.has_tau_r)
            cfg.sampling.tau_r = request.tau_r;
        if (request.has_min_token_text_ratio)
            cfg.min_token_text_ratio = request.min_token_text_ratio;
        if (request.has_max_token_text_ratio)
            cfg.max_token_text_ratio = request.max_token_text_ratio;
    }

    if (!cosyvoice_set_generation_config(model_ctx, &cfg))
    {
        *error_message = "Invalid generation parameter values.";
        return false;
    }

    uint32_t effective_seed = 0;
    if (request.has_seed)
        effective_seed = request.seed;
    else if (runtime->options.has_seed)
        effective_seed = runtime->options.seed;
    else
        effective_seed = (*runtime).seed_rng();

    if (!cosyvoice_set_sampler_seed(model_ctx, effective_seed))
    {
        *error_message = "Failed to apply seed with current sampler configuration.";
        return false;
    }

    if (applied_seed)
        *applied_seed = effective_seed;

    return true;
}

static request_log_context make_request_log_context(const httplib::Request& req, const char* endpoint)
{
    request_log_context ctx;
    ctx.id = ++g_request_id;
    ctx.endpoint = endpoint;
    ctx.remote_addr = req.remote_addr;
    ctx.start = std::chrono::steady_clock::now();
    return ctx;
}

static void fill_request_log_context_from_request(request_log_context* ctx, const speech_request& request)
{
    ctx->model = request.model;
    ctx->voice = request.voice;
    ctx->response_format = request.response_format;
    ctx->has_instructions = request.has_instructions;
    ctx->has_seed = request.has_seed;
    ctx->seed = request.seed;
}

static const char* request_log_status_to_string(request_log_status status)
{
    switch (status)
    {
    case request_log_status::ok:
        return "ok";
    case request_log_status::bad_request:
        return "bad_request";
    case request_log_status::unauthorized:
        return "unauthorized";
    case request_log_status::not_found:
        return "not_found";
    case request_log_status::failed:
        return "failed";
    default:
        return "unknown";
    }
}

static void log_request_start(server_log_level level, const request_log_context& ctx)
{
    log_message(
        level,
        server_log_level::concise,
        "REQ",
        "id=%llu start endpoint=%s remote=%s",
        static_cast<unsigned long long>(ctx.id),
        ctx.endpoint.c_str(),
        ctx.remote_addr.c_str());
}

static void log_request_details(server_log_level level, const request_log_context& ctx)
{
    log_message(
        level,
        server_log_level::verbose,
        "REQ",
        "id=%llu details model=%s voice=%s format=%s instructions=%s seed_in_request=%s",
        static_cast<unsigned long long>(ctx.id),
        ctx.model.empty() ? "-" : ctx.model.c_str(),
        ctx.voice.empty() ? "-" : ctx.voice.c_str(),
        ctx.response_format.empty() ? "-" : ctx.response_format.c_str(),
        ctx.has_instructions ? "yes" : "no",
        ctx.has_seed ? "yes" : "no");
}

static void log_request_done(
    server_log_level level,
    const request_log_context& ctx,
    request_log_status status,
    int http_status,
    uint32_t effective_seed,
    size_t response_bytes,
    const char* note)
{
    const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - ctx.start).count();

    if (effective_seed != 0)
    {
        log_message(
            level,
            server_log_level::concise,
            "REQ",
            "id=%llu done status=%s http=%d elapsed_ms=%lld bytes=%zu seed=%u note=%s",
            static_cast<unsigned long long>(ctx.id),
            request_log_status_to_string(status),
            http_status,
            static_cast<long long>(elapsed_ms),
            response_bytes,
            effective_seed,
            note ? note : "-");
    }
    else
    {
        log_message(
            level,
            server_log_level::concise,
            "REQ",
            "id=%llu done status=%s http=%d elapsed_ms=%lld bytes=%zu note=%s",
            static_cast<unsigned long long>(ctx.id),
            request_log_status_to_string(status),
            http_status,
            static_cast<long long>(elapsed_ms),
            response_bytes,
            note ? note : "-");
    }
}

static bool initialize_runtime(const server_options& options, server_runtime* runtime)
{
    runtime->options = options;
    runtime->log_level = get_log_level(options);

    uint32_t seed_rng_seed = static_cast<uint32_t>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count());
    std::random_device rd;
    if (rd.entropy() > 0.0)
        seed_rng_seed ^= static_cast<uint32_t>(rd());
    runtime->seed_rng.seed(seed_rng_seed);

    cosyvoice_init_backend();

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
        print_error_log("Error: failed to load model file \"%s\".\n", options.model.c_str());
        return false;
    }

    // If the user didn't explicitly provide a served_model_name, try to use
    // the model's architecture reported by cosyvoice_get_architecture(). If
    // that isn't available, fall back to deriving the name from the path.
    if (!runtime->options.has_served_model_name)
    {
        const char* arch = cosyvoice_get_architecture(runtime->model_context.get());
        if (arch && arch[0] != '\0')
            runtime->options.served_model_name = arch;
        else
            runtime->options.served_model_name = derive_served_model_name(runtime->options.model);
    }

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
        print_error_log("Error: invalid server-level sampling defaults.\n");
        return false;
    }
    cosyvoice_get_generation_config(runtime->model_context.get(), &runtime->default_generation_config);

    runtime->sample_rate = cosyvoice_get_sample_rate(runtime->model_context.get());

#ifndef COSYVOICE_NO_AUDIO
    runtime->audio_encoder.reset(cosyvoice_audio_encoder_create(runtime->sample_rate));
#endif

    for (const auto& mapping : options.voice_prompts)
    {
        voice_runtime voice;
        voice.name = mapping.voice;
        voice.prompt_speech_path = mapping.prompt_speech;

        voice.prompt_speech.reset(cosyvoice_prompt_speech_load_from_file(mapping.prompt_speech.c_str()));
        if (!voice.prompt_speech)
        {
            print_error_log("Error: failed to load prompt_speech file \"%s\" for voice \"%s\".\n",
                mapping.prompt_speech.c_str(),
                mapping.voice.c_str());
            if (errno != 0)
                print_error_log("Reason: %s\n", strerror(errno));
            return false;
        }

        voice.prompt.reset(cosyvoice_prompt_init_from_prompt_speech(runtime->model_context.get(), voice.prompt_speech.get()));
        if (!voice.prompt)
        {
            print_error_log("Error: failed to initialize prompt from prompt_speech for voice \"%s\".\n", mapping.voice.c_str());
            return false;
        }

        voice.tts_context.reset(cosyvoice_tts_context_new(runtime->model_context.get(), voice.prompt.get()));
        if (!voice.tts_context)
        {
            print_error_log("Error: failed to create TTS context for voice \"%s\".\n", mapping.voice.c_str());
            return false;
        }

#ifndef COSYVOICE_NO_ICU
        cosyvoice_tts_context_set_text_normalization_enabled(voice.tts_context.get(), options.text_normalization_enabled);
#endif

        runtime->voice_names.push_back(voice.name);
        runtime->voices.emplace(voice.name, std::move(voice));
    }

    return true;
}

#ifdef _WIN32
int wmain(int argc, wchar_t** argv)
{
    setup_console_utf8();
#else
int main(int argc, char** argv)
{
#endif
    if (argc == 1)
    {
        print_usage(argv[0]);
        return 0;
    }

    server_options options;
    for (int i = 1; i != argc; ++i)
    {
        auto arg = argv[i];
        auto get_arg_value = [&]() -> tchar*
        {
            if (++i == argc)
            {
                const auto option = tchar_to_utf8(arg);
                print_error_log("Error: missing value for the command-line option \"%s\".\n", option.c_str());
                exit(1);
            }
            return argv[i];
        };

        if (tchar_casecmp(arg, COSYVOICE_TEXT("--help")) == 0 || tchar_casecmp(arg, COSYVOICE_TEXT("-h")) == 0)
        {
            print_usage(argv[0]);
            return 0;
        }
        else if (tchar_casecmp(arg, COSYVOICE_TEXT("--model")) == 0 || tchar_casecmp(arg, COSYVOICE_TEXT("-m")) == 0)
            options.model = tchar_to_utf8(get_arg_value());
        else if (tchar_casecmp(arg, COSYVOICE_TEXT("--served-model-name")) == 0)
        {
            options.served_model_name = tchar_to_utf8(get_arg_value());
            options.has_served_model_name = true;
        }
        else if (tchar_casecmp(arg, COSYVOICE_TEXT("--host")) == 0)
            options.host = tchar_to_utf8(get_arg_value());
        else if (tchar_casecmp(arg, COSYVOICE_TEXT("--port")) == 0)
        {
            const auto value = tchar_to_utf8(get_arg_value());
            uint16_t port;
            if (!parse_uint16_port(value, &port))
            {
                print_error_log("Error: invalid --port value \"%s\".\n", value.c_str());
                return 1;
            }
            options.port = port;
        }
        else if (tchar_casecmp(arg, COSYVOICE_TEXT("--api-key")) == 0)
            options.api_key = tchar_to_utf8(get_arg_value());
        else if (tchar_casecmp(arg, COSYVOICE_TEXT("--voice-prompt")) == 0)
        {
            const auto value = tchar_to_utf8(get_arg_value());
            voice_prompt_option mapping;
            if (!parse_voice_prompt_mapping(value, &mapping))
            {
                print_error_log("Error: invalid --voice-prompt value \"%s\". Expected <voice=prompt_speech.gguf>.\n", value.c_str());
                return 1;
            }
            options.voice_prompts.push_back(std::move(mapping));
        }
        else if (tchar_casecmp(arg, COSYVOICE_TEXT("--voice")) == 0)
        {
            options.single_voice = tchar_to_utf8(get_arg_value());
            options.has_single_voice_arg = true;
        }
        else if (tchar_casecmp(arg, COSYVOICE_TEXT("--prompt-speech")) == 0)
            options.single_prompt_speech = tchar_to_utf8(get_arg_value());
        else if (tchar_casecmp(arg, COSYVOICE_TEXT("--max-llm-len")) == 0)
        {
            const auto value = tchar_to_utf8(get_arg_value());
            uint32_t max_llm_len;
            if (!parse_uint32_arg(value, &max_llm_len) || max_llm_len == 0)
            {
                print_error_log("Error: invalid --max-llm-len value \"%s\".\n", value.c_str());
                return 1;
            }
            options.max_llm_len = max_llm_len;
        }
        else if (tchar_casecmp(arg, COSYVOICE_TEXT("--inference-buffer-policy")) == 0)
        {
            const auto value = tchar_to_utf8(get_arg_value());
            cosyvoice_inference_buffer_policy_t policy;
            if (!parse_inference_buffer_policy_arg(value, &policy))
            {
                print_error_log("Error: invalid --inference-buffer-policy value \"%s\".\n", value.c_str());
                return 1;
            }
            options.inference_buffer_policy = policy;
            options.has_inference_buffer_policy = true;
        }
        else if (tchar_casecmp(arg, COSYVOICE_TEXT("--threads")) == 0 || tchar_casecmp(arg, COSYVOICE_TEXT("-j")) == 0)
        {
            const auto value = tchar_to_utf8(get_arg_value());
            uint32_t n_threads;
            if (!parse_uint32_arg(value, &n_threads))
            {
                print_error_log("Error: invalid --threads value \"%s\".\n", value.c_str());
                return 1;
            }
            options.n_threads = n_threads;
        }
        else if (tchar_casecmp(arg, COSYVOICE_TEXT("--seed")) == 0)
        {
            const auto value = tchar_to_utf8(get_arg_value());
            uint32_t seed;
            if (!parse_uint32_arg(value, &seed))
            {
                print_error_log("Error: invalid --seed value \"%s\".\n", value.c_str());
                return 1;
            }
            options.seed = seed;
            options.has_seed = true;
        }
        else if (tchar_casecmp(arg, COSYVOICE_TEXT("--temperature")) == 0)
        {
            const auto value = tchar_to_utf8(get_arg_value());
            float v;
            if (!parse_float_arg(value, &v))
            {
                print_error_log("Error: invalid --temperature value \"%s\".\n", value.c_str());
                return 1;
            }
            options.temperature = v;
            options.has_temperature = true;
        }
        else if (tchar_casecmp(arg, COSYVOICE_TEXT("--top-k")) == 0)
        {
            const auto value = tchar_to_utf8(get_arg_value());
            int v;
            if (!parse_int_arg(value, &v))
            {
                print_error_log("Error: invalid --top-k value \"%s\".\n", value.c_str());
                return 1;
            }
            options.top_k = v;
            options.has_top_k = true;
        }
        else if (tchar_casecmp(arg, COSYVOICE_TEXT("--top-p")) == 0)
        {
            const auto value = tchar_to_utf8(get_arg_value());
            float v;
            if (!parse_float_arg(value, &v))
            {
                print_error_log("Error: invalid --top-p value \"%s\".\n", value.c_str());
                return 1;
            }
            options.top_p = v;
            options.has_top_p = true;
        }
        else if (tchar_casecmp(arg, COSYVOICE_TEXT("--win-size")) == 0)
        {
            const auto value = tchar_to_utf8(get_arg_value());
            int v;
            if (!parse_int_arg(value, &v))
            {
                print_error_log("Error: invalid --win-size value \"%s\".\n", value.c_str());
                return 1;
            }
            options.win_size = v;
            options.has_win_size = true;
        }
        else if (tchar_casecmp(arg, COSYVOICE_TEXT("--tau-r")) == 0)
        {
            const auto value = tchar_to_utf8(get_arg_value());
            float v;
            if (!parse_float_arg(value, &v))
            {
                print_error_log("Error: invalid --tau-r value \"%s\".\n", value.c_str());
                return 1;
            }
            options.tau_r = v;
            options.has_tau_r = true;
        }
        else if (tchar_casecmp(arg, COSYVOICE_TEXT("--min-token-text-ratio")) == 0)
        {
            const auto value = tchar_to_utf8(get_arg_value());
            float v;
            if (!parse_float_arg(value, &v))
            {
                print_error_log("Error: invalid --min-token-text-ratio value \"%s\".\n", value.c_str());
                return 1;
            }
            options.min_token_text_ratio = v;
            options.has_min_token_text_ratio = true;
        }
        else if (tchar_casecmp(arg, COSYVOICE_TEXT("--max-token-text-ratio")) == 0)
        {
            const auto value = tchar_to_utf8(get_arg_value());
            float v;
            if (!parse_float_arg(value, &v))
            {
                print_error_log("Error: invalid --max-token-text-ratio value \"%s\".\n", value.c_str());
                return 1;
            }
            options.max_token_text_ratio = v;
            options.has_max_token_text_ratio = true;
        }
        else if (tchar_casecmp(arg, COSYVOICE_TEXT("--llm-kv-cache-type")) == 0)
        {
            const auto value = tchar_to_utf8(get_arg_value());
            cosyvoice_llm_kv_cache_type_t type;
            if (!parse_llm_kv_cache_type_arg(value, &type))
            {
                print_error_log("Error: invalid --llm-kv-cache-type value \"%s\".\n", value.c_str());
                return 1;
            }
            options.llm_kv_cache_type = type;
            options.has_llm_kv_cache_type = true;
        }
        else if (tchar_casecmp(arg, COSYVOICE_TEXT("--verbose")) == 0 || tchar_casecmp(arg, COSYVOICE_TEXT("-v")) == 0)
            options.verbose = true;
        else if (tchar_casecmp(arg, COSYVOICE_TEXT("--quiet")) == 0 || tchar_casecmp(arg, COSYVOICE_TEXT("-q")) == 0)
            options.quiet = true;
#ifndef COSYVOICE_NO_ICU
        else if (tchar_casecmp(arg, COSYVOICE_TEXT("--disable-text-normalization")) == 0)
            options.text_normalization_enabled = false;
#endif
        else
        {
            const auto option = tchar_to_utf8(arg);
            print_error_log("Error: the program doesn't recognize the command-line option \"%s\".\n", option.c_str());
            return 1;
        }
    }

    if (!validate_options(options))
        return 1;

    server_runtime runtime;
    if (!initialize_runtime(options, &runtime))
        return 1;

    httplib::Server server;

    server.set_exception_handler([&](const httplib::Request& req, httplib::Response& res, std::exception_ptr ep)
    {
        (void)ep;
        if (req.path.rfind("/v1/", 0) == 0)
            set_openai_error(res, 500, "Internal server error.", "server_error", nullptr, "internal_error");
        else
            res.status = 500;
    });

    server.Get("/", [](const httplib::Request&, httplib::Response& res)
    {
        res.status = 200;
        res.set_content("CosyVoice OpenAI Speech API Server is running.\n", "text/plain");
    });

    server.Get("/healthz", [&](const httplib::Request&, httplib::Response& res)
    {
        json payload = {
            { "status", "ok" },
            { "model", runtime.options.served_model_name },
            { "voices", runtime.voice_names }
        };
        res.status = 200;
        res.set_content(payload.dump(), "application/json");
    });

    server.Get("/v1/models", [&](const httplib::Request& req, httplib::Response& res)
    {
        request_log_context log_ctx = make_request_log_context(req, "/v1/models");
        log_request_start(runtime.log_level, log_ctx);

        if (!require_auth(req, runtime.options, res))
        {
            log_request_done(runtime.log_level, log_ctx, request_log_status::unauthorized, res.status, 0, res.body.size(), "auth");
            return;
        }

        const auto created = static_cast<int64_t>(time(nullptr));
        json payload = {
            { "object", "list" },
            {
                "data",
                json::array({
                    {
                        { "id", runtime.options.served_model_name },
                        { "object", "model" },
                        { "created", created },
                        { "owned_by", "cosyvoice-server" }
                    }
                })
            }
        };

        res.status = 200;
        res.set_content(payload.dump(), "application/json");

        log_request_done(runtime.log_level, log_ctx, request_log_status::ok, res.status, 0, res.body.size(), "models");
    });

    server.Post("/v1/audio/speech", [&](const httplib::Request& req, httplib::Response& res)
    {
        request_log_context log_ctx = make_request_log_context(req, "/v1/audio/speech");
        log_request_start(runtime.log_level, log_ctx);

        if (!require_auth(req, runtime.options, res))
        {
            log_request_done(runtime.log_level, log_ctx, request_log_status::unauthorized, res.status, 0, res.body.size(), "auth");
            return;
        }

        json body;
        try
        {
            body = json::parse(req.body);
        }
        catch (const std::exception&)
        {
            set_openai_error(res, 400, "Malformed JSON body.", "invalid_request_error", nullptr, "invalid_json");
            log_request_done(runtime.log_level, log_ctx, request_log_status::bad_request, res.status, 0, res.body.size(), "invalid_json");
            return;
        }

        if (!body.is_object())
        {
            set_openai_error(res, 400, "JSON body must be an object.", "invalid_request_error", nullptr, "invalid_json");
            log_request_done(runtime.log_level, log_ctx, request_log_status::bad_request, res.status, 0, res.body.size(), "invalid_json");
            return;
        }

        speech_request request;
        if (!parse_speech_request_json(body, &request, res))
        {
            log_request_done(runtime.log_level, log_ctx, request_log_status::bad_request, res.status, 0, res.body.size(), "request_validation");
            return;
        }

        fill_request_log_context_from_request(&log_ctx, request);
        log_request_details(runtime.log_level, log_ctx);

        if (request.model != runtime.options.served_model_name)
        {
            set_openai_error(
                res,
                400,
                std::string("Model not found: ") + request.model,
                "invalid_request_error",
                "model",
                "model_not_found");
            log_request_done(runtime.log_level, log_ctx, request_log_status::bad_request, res.status, 0, res.body.size(), "model_mismatch");
            return;
        }

        const auto voice_it = runtime.voices.find(request.voice);
        if (voice_it == runtime.voices.end())
        {
            std::string message = "Unknown voice \"" + request.voice + "\". Available voices: " + join_voice_names(runtime.voice_names);
            set_openai_error(res, 400, message, "invalid_request_error", "voice", "invalid_voice");
            log_request_done(runtime.log_level, log_ctx, request_log_status::bad_request, res.status, 0, res.body.size(), "voice_mismatch");
            return;
        }

        const auto format = parse_response_format(request.response_format);
        if (format == response_audio_format::unknown)
        {
            set_openai_error(res, 400,
                "Unsupported response_format. Allowed standard values: mp3, opus, aac, flac, wav, pcm.",
                "invalid_request_error",
                "response_format",
                "invalid_value");
            log_request_done(runtime.log_level, log_ctx, request_log_status::bad_request, res.status, 0, res.body.size(), "response_format");
            return;
        }

        if (!is_response_format_supported(runtime, format))
        {
            const auto supported_formats = supported_response_formats_to_string(runtime);
            const auto requested_format = response_format_to_string(format);
            set_openai_error(
                res,
                400,
                std::string("response_format \"") + requested_format + "\" is not supported in this build/runtime. "
                + "Supported formats: " + supported_formats + ".",
                "invalid_request_error",
                "response_format",
                "unsupported_format");
            log_request_done(runtime.log_level, log_ctx, request_log_status::bad_request, res.status, 0, res.body.size(), "response_format");
            return;
        }

        cosyvoice_generated_speech generated = {};
        std::string payload;
        std::string encoding_error;
        std::string generation_error;
        uint32_t applied_seed = 0;

        {
            std::lock_guard<std::mutex> lock(runtime.infer_mutex);
            if (!apply_generation_overrides(request, &runtime, runtime.model_context.get(), &applied_seed, &generation_error))
            {
                set_openai_error(res, 400, generation_error, "invalid_request_error", nullptr, "invalid_generation_params");
                log_request_done(runtime.log_level, log_ctx, request_log_status::bad_request, res.status, 0, res.body.size(), "generation_override");
                return;
            }

            bool ok = false;
            if (request.has_instructions)
                ok = cosyvoice_tts_instruct(voice_it->second.tts_context.get(), request.input.c_str(), request.instructions.c_str(), request.speed, &generated);
            else
                ok = cosyvoice_tts_zero_shot(voice_it->second.tts_context.get(), request.input.c_str(), request.speed, &generated);

            if (!ok || !generated.data || generated.length == 0)
            {
                set_openai_error(res, 500, "TTS generation failed.", "server_error", nullptr, "generation_failed");
                log_request_done(runtime.log_level, log_ctx, request_log_status::failed, res.status, applied_seed, res.body.size(), "generation_failed");
                return;
            }

            if (!build_audio_payload(format, generated, runtime, &payload, &encoding_error))
            {
                set_openai_error(res, 500, encoding_error, "server_error", nullptr, "audio_encode_failed");
                log_request_done(runtime.log_level, log_ctx, request_log_status::failed, res.status, applied_seed, res.body.size(), "audio_encode_failed");
                return;
            }
        }

        res.status = 200;
        res.set_content(std::move(payload), response_format_to_content_type(format));
        log_request_done(runtime.log_level, log_ctx, request_log_status::ok, res.status, applied_seed, res.body.size(), "speech");
    });

    server.set_error_handler([&](const httplib::Request& req, httplib::Response& res)
    {
        if (!res.body.empty())
            return;

        if (req.path.rfind("/v1/", 0) == 0)
        {
            set_openai_error(res, 404, "Endpoint not found.", "invalid_request_error", nullptr, "not_found");
            request_log_context log_ctx = make_request_log_context(req, req.path.c_str());
            log_request_done(runtime.log_level, log_ctx, request_log_status::not_found, res.status, 0, res.body.size(), "endpoint");
        }
    });

    print_info_log(runtime.log_level, "\n================ cosyvoice-server ================\n");
    print_info_log(runtime.log_level, "  model              : %s\n", runtime.options.model.c_str());
    print_info_log(runtime.log_level, "  served_model_name  : %s\n", runtime.options.served_model_name.c_str());
    print_info_log(runtime.log_level, "  bind               : %s:%u\n", runtime.options.host.c_str(), static_cast<unsigned>(runtime.options.port));
    print_info_log(runtime.log_level, "  sample_rate        : %u\n", runtime.sample_rate);
    print_info_log(runtime.log_level, "  api_key_required   : %s\n", runtime.options.api_key.empty() ? "no" : "yes");
    print_info_log(runtime.log_level, "  voices             : %s\n", join_voice_names(runtime.voice_names).c_str());
    print_info_log(runtime.log_level, "  buffer_policy      : %s\n", inference_buffer_policy_to_string(runtime.options.inference_buffer_policy));
    if (runtime.options.has_seed)
        print_info_log(runtime.log_level, "  seed_strategy      : server default=%u, request override allowed\n", runtime.options.seed);
    else
        print_info_log(runtime.log_level, "  seed_strategy      : random per request, request override allowed\n");
#ifndef COSYVOICE_NO_ICU
    print_info_log(runtime.log_level, "  text_normalization : %s\n", runtime.options.text_normalization_enabled ? "enabled" : "disabled");
#else
    print_info_log(runtime.log_level, "  text_normalization : unavailable (COSYVOICE_NO_ICU)\n");
#endif
#ifndef COSYVOICE_NO_AUDIO
    print_info_log(runtime.log_level, "  audio_encoder      : %s\n", runtime.audio_encoder ? "available" : "unavailable");
#else
    print_info_log(runtime.log_level, "  audio_encoders     : unavailable (COSYVOICE_NO_AUDIO)\n");
#endif
    print_info_log(runtime.log_level, "-----------------------------------------------\n\n");

    if (!server.listen(runtime.options.host.c_str(), runtime.options.port))
    {
        print_error_log("Error: failed to start server at %s:%u.\n",
            runtime.options.host.c_str(),
            static_cast<unsigned>(runtime.options.port));
        return 1;
    }

    return 0;
}
