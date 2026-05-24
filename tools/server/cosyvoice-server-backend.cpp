#include "cosyvoice-server.h"
#include "tool_common_cosyvoice.h"

#ifndef COSYVOICE_NO_AUDIO
    #include "cosyvoice-audio.h"
#endif

#include "httplib.h"
#include "nlohmann/json.hpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <limits>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using json = nlohmann::json;

enum class request_log_status
{
    ok,
    bad_request,
    unauthorized,
    not_found,
    failed
};

enum class response_audio_format
{
    mp3,
    wav,
    flac,
    pcm,
    aac,
    m4a,
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

static response_audio_format parse_response_format(const std::string& value)
{
    const auto lowered = to_lower(value);
    if (lowered == "mp3") return response_audio_format::mp3;
    if (lowered == "aac") return response_audio_format::aac;
    if (lowered == "m4a") return response_audio_format::m4a;
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
    case response_audio_format::aac:
        return "aac";
    case response_audio_format::m4a:
        return "m4a";
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

static std::string join_voice_names(const std::vector<std::string>& voice_names)
{
    std::string joined;
    for (const auto& voice_name : voice_names)
    {
        if (!joined.empty())
            joined += ", ";
        joined += voice_name;
    }
    return joined.empty() ? std::string("-") : joined;
}

static bool is_response_format_supported(const server_runtime& runtime, response_audio_format format)
{
    switch (format)
    {
    case response_audio_format::pcm:
        return true;
    case response_audio_format::wav:
#ifdef COSYVOICE_NO_AUDIO
        return true;
#else
        return cosyvoice_audio_encoding_format_supported(COSYVOICE_AUDIO_ENCODING_FORMAT_WAV);
#endif
#ifdef COSYVOICE_AUDIO_BACKEND_FFMPEG
    case response_audio_format::mp3:
        return cosyvoice_audio_encoding_format_supported(COSYVOICE_AUDIO_ENCODING_FORMAT_MP3);
    case response_audio_format::m4a:
        return cosyvoice_audio_encoding_format_supported(COSYVOICE_AUDIO_ENCODING_FORMAT_AAC);
    case response_audio_format::aac:
        return cosyvoice_audio_encoding_format_supported(COSYVOICE_AUDIO_ENCODING_FORMAT_AAC);
    case response_audio_format::flac:
        return cosyvoice_audio_encoding_format_supported(COSYVOICE_AUDIO_ENCODING_FORMAT_FLAC);
    case response_audio_format::opus:
        return cosyvoice_audio_encoding_format_supported(COSYVOICE_AUDIO_ENCODING_FORMAT_OPUS);
#endif
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
        response_audio_format::m4a,
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
    case response_audio_format::m4a:
        return "audio/mp4";
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
static bool build_wav_bytes_with_audio_encoder(cosyvoice_audio_encoder_t encoder, const float* data, uint32_t length, cosyvoice_audio_encoding_format_t format, std::string* output, std::string* error)
{
    if (!encoder)
    {
        *error = "Encoder is unavailable in this runtime environment.";
        return false;
    }

    if (!data || length == 0)
    {
        *error = "Generated audio buffer is empty.";
        return false;
    }

    if (!cosyvoice_audio_encoder_encode(encoder, data, length, format))
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
#else
static bool build_wav_bytes(const float* data, uint32_t length, uint32_t sample_rate, std::string* output, std::string* error)
{
    if (!data || length == 0)
    {
        *error = "Generated audio buffer is empty.";
        return false;
    }
    constexpr uint16_t num_channels = 1;
    constexpr uint16_t bits_per_sample = 16;
    const uint32_t byte_rate = sample_rate * num_channels * bits_per_sample / 8;
    const uint16_t block_align = num_channels * bits_per_sample / 8;
    const uint32_t subchunk2_size = length * num_channels * bits_per_sample / 8;
    const uint32_t chunk_size = 4 + (8 + 16) + (8 + subchunk2_size);
    output->clear();
    output->reserve(44 + subchunk2_size);
    // RIFF header
    output->append("RIFF", 4);
    output->append(reinterpret_cast<const char*>(&chunk_size), 4);
    output->append("WAVE", 4);
    // fmt subchunk
    output->append("fmt ", 4);
    uint32_t fmt_subchunk_size = 16;
    output->append(reinterpret_cast<const char*>(&fmt_subchunk_size), 4);
    uint16_t audio_format = 1; // PCM
    output->append(reinterpret_cast<const char*>(&audio_format), 2);
    output->append(reinterpret_cast<const char*>(&num_channels), 2);
    output->append(reinterpret_cast<const char*>(&sample_rate), 4);
    output->append(reinterpret_cast<const char*>(&byte_rate), 4);
    output->append(reinterpret_cast<const char*>(&block_align), 2);
    output->append(reinterpret_cast<const char*>(&bits_per_sample), 2);
    // data subchunk
    output->append("data", 4);
    output->append(reinterpret_cast<const char*>(&subchunk2_size), 4);
    for (uint32_t i = 0; i < length; ++i)
    {
        const auto sample = float_to_pcm16(data[i]);
        output->push_back(static_cast<char>(sample & 0xFF));
        output->push_back(static_cast<char>((sample >> 8) & 0xFF));
    }
    return true;
}
#endif

static bool build_audio_payload(response_audio_format format, const cosyvoice_generated_speech& generated, const server_runtime& runtime, std::string* payload, std::string* error)
{
    switch (format)
    {
    case response_audio_format::pcm:
        return build_pcm16_bytes(generated.data, generated.length, payload, error);
    case response_audio_format::wav:
#ifdef COSYVOICE_NO_AUDIO
        return build_wav_bytes(generated.data, generated.length, runtime.sample_rate, payload, error);
#else
        return build_wav_bytes_with_audio_encoder(runtime.audio_encoder.get(), generated.data, generated.length, COSYVOICE_AUDIO_ENCODING_FORMAT_WAV, payload, error);
    case response_audio_format::mp3:
        return build_wav_bytes_with_audio_encoder(runtime.audio_encoder.get(), generated.data, generated.length, COSYVOICE_AUDIO_ENCODING_FORMAT_MP3, payload, error);
    case response_audio_format::m4a:
        return build_wav_bytes_with_audio_encoder(runtime.audio_encoder.get(), generated.data, generated.length, COSYVOICE_AUDIO_ENCODING_FORMAT_AAC, payload, error);
    case response_audio_format::aac:
        return build_wav_bytes_with_audio_encoder(runtime.audio_encoder.get(), generated.data, generated.length, COSYVOICE_AUDIO_ENCODING_FORMAT_AAC, payload, error);
    case response_audio_format::flac:
        return build_wav_bytes_with_audio_encoder(runtime.audio_encoder.get(), generated.data, generated.length, COSYVOICE_AUDIO_ENCODING_FORMAT_FLAC, payload, error);
    case response_audio_format::opus:
        return build_wav_bytes_with_audio_encoder(runtime.audio_encoder.get(), generated.data, generated.length, COSYVOICE_AUDIO_ENCODING_FORMAT_OPUS, payload, error);
#endif
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

static bool require_auth(const httplib::Request& req, const server_runtime& runtime, httplib::Response& res)
{
    if (runtime.api_key.empty())
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
    if (token != runtime.api_key)
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
    server_runtime& runtime,
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

    // Only touch the generation config when the caller actually overrides
    // something. cosyvoice_model::set_generation_config reallocates the
    // nucleus_probs buffer and re-installs sampling state on every call;
    // doing that between identical requests perturbs the sampler enough to
    // tank throughput (measured ~3x slower than a fresh CLI invocation on
    // the same input). The startup default was already installed once when
    // the runtime was set up, so the no-override path needs no further
    // configuration here.
    if (has_override)
    {
        cosyvoice_generation_config_t cfg = runtime.default_generation_config;
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

        if (!cosyvoice_set_generation_config(model_ctx, &cfg))
        {
            *error_message = "Invalid generation parameter values.";
            return false;
        }
    }

    // The sampler seed must vary across requests when the caller didn't pin
    // one — pinning a single seed across the lifetime of a shared context
    // makes the sampler emit the same stop-token pattern for every text and
    // truncates the audio (e.g. a 55-char Japanese sentence collapses to
    // ~4.8s instead of the natural ~12s). Re-seeding here is the cheap part;
    // it was the set_generation_config call above that did the harm.
    uint32_t effective_seed = 0;
    if (request.has_seed)
        effective_seed = request.seed;
    else if (runtime.has_seed)
        effective_seed = runtime.seed;
    else
        effective_seed = runtime.seed_rng();

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

int cosyvoice_server_backend_run(server_runtime& runtime)
{
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
            { "model", runtime.served_model_name },
            { "voices", runtime.voice_names }
        };
        res.status = 200;
        res.set_content(payload.dump(), "application/json");
    });

    server.Get("/v1/models", [&](const httplib::Request& req, httplib::Response& res)
    {
        request_log_context log_ctx = make_request_log_context(req, "/v1/models");
        log_request_start(runtime.log_level, log_ctx);

        if (!require_auth(req, runtime, res))
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
                        { "id", runtime.served_model_name },
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

        if (!require_auth(req, runtime, res))
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

        if (request.model != runtime.served_model_name)
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
                "Unsupported response_format. Allowed standard values: mp3, opus, aac, flac, wav, pcm, m4a.",
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
            if (!apply_generation_overrides(request, runtime, runtime.model_context.get(), &applied_seed, &generation_error))
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
    print_info_log(runtime.log_level, "  model              : %s\n", runtime.model.c_str());
    print_info_log(runtime.log_level, "  served_model_name  : %s\n", runtime.served_model_name.c_str());
    print_info_log(runtime.log_level, "  bind               : %s:%u\n", runtime.host.c_str(), static_cast<unsigned>(runtime.port));
    print_info_log(runtime.log_level, "  sample_rate        : %u\n", runtime.sample_rate);
    print_info_log(runtime.log_level, "  api_key_required   : %s\n", runtime.api_key.empty() ? "no" : "yes");
    print_info_log(runtime.log_level, "  voices             : %s\n", join_voice_names(runtime.voice_names).c_str());
    print_info_log(runtime.log_level, "  buffer_policy      : %s\n", inference_buffer_policy_to_string(runtime.inference_buffer_policy));
    if (runtime.has_seed)
        print_info_log(runtime.log_level, "  seed_strategy      : server default=%u, request override allowed\n", runtime.seed);
    else
        print_info_log(runtime.log_level, "  seed_strategy      : random per request, request override allowed\n");
#ifndef COSYVOICE_NO_ICU
    print_info_log(runtime.log_level, "  text_normalization : %s\n", runtime.text_normalization_enabled ? "enabled" : "disabled");
#else
    print_info_log(runtime.log_level, "  text_normalization : unavailable (COSYVOICE_NO_ICU)\n");
#endif
    print_info_log(runtime.log_level, "  text_splitting     : %s\n", runtime.split_text_enabled ? "enabled" : "disabled");
    print_info_log(runtime.log_level, "  fast_split         : %s\n", runtime.fast_split_text_enabled ? "enabled" : "disabled");
    print_info_log(runtime.log_level, "  fade_in            : %s\n", runtime.fade_in_enabled ? "enabled" : "disabled");
#ifndef COSYVOICE_NO_AUDIO
    print_info_log(runtime.log_level, "  audio_encoder      : %s\n", runtime.audio_encoder ? "available" : "unavailable");
#else
    print_info_log(runtime.log_level, "  audio_encoders     : unavailable (COSYVOICE_NO_AUDIO)\n");
#endif
    print_info_log(runtime.log_level, "-----------------------------------------------\n\n");

    if (!server.listen(runtime.host.c_str(), runtime.port))
    {
        print_error_log("Error: failed to start server at %s:%u.\n",
            runtime.host.c_str(),
            static_cast<unsigned>(runtime.port));
        return 1;
    }

    return 0;
}

