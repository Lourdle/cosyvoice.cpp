#include "pch.h"
#include "tool_common.h"
#include "server_common.h"

#include "httplib.h"

#include <atomic>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <limits>

// ---------------------------------------------------------------------------
// Level-gated logging
// ---------------------------------------------------------------------------

bool log_level_enabled(server_log_level current, server_log_level required)
{
    return static_cast<int>(current) >= static_cast<int>(required);
}

void log_message(server_log_level current, server_log_level required, const char* tag, const char* format, ...)
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

void print_error_log(const char* format, ...)
{
    const auto ts = get_local_timestamp_ms();
    fprintf(stderr, "[%s] [ERROR] ", ts.c_str());

    va_list args;
    va_start(args, format);
    vfprintf(stderr, format, args);
    va_end(args);

    fflush(stderr);
}

void print_info_log(server_log_level level, const char* format, ...)
{
    if (level == server_log_level::quiet)
        return;

    va_list args;
    va_start(args, format);
    vprintf(format, args);
    va_end(args);
}

// ---------------------------------------------------------------------------
// Request ID generation
// ---------------------------------------------------------------------------

static std::atomic<uint64_t> g_request_id = 0;

// ---------------------------------------------------------------------------
// Generation config overrides
// ---------------------------------------------------------------------------

bool apply_generation_overrides(
    const generation_config_overrides& overrides,
    server_runtime& runtime,
    cosyvoice_context_t model_ctx,
    uint32_t* applied_seed,
    std::string* error_message)
{
    cosyvoice_generation_config_t cfg = runtime.default_generation_config;
    if (overrides.has_temperature)
        cfg.temperature = overrides.temperature;
    if (overrides.has_top_k)
        cfg.sampling.top_k = overrides.top_k;
    if (overrides.has_top_p)
        cfg.sampling.top_p = overrides.top_p;
    if (overrides.has_win_size)
        cfg.sampling.win_size = overrides.win_size;
    if (overrides.has_tau_r)
        cfg.sampling.tau_r = overrides.tau_r;
    if (overrides.has_min_token_text_ratio)
        cfg.min_token_text_ratio = overrides.min_token_text_ratio;
    if (overrides.has_max_token_text_ratio)
        cfg.max_token_text_ratio = overrides.max_token_text_ratio;

    if (!cosyvoice_set_generation_config(model_ctx, &cfg))
    {
        *error_message = "Invalid generation parameter values.";
        return false;
    }

    // The sampler seed must vary across requests when the caller didn't pin
    // one — pinning a single seed across the lifetime of a shared context
    // makes the sampler emit the same stop-token pattern for every text and
    // truncates the audio (e.g. a 55-char Japanese sentence collapses to
    // ~4.8s instead of the natural ~12s). Re-seeding here is the cheap part;
    // it was the set_generation_config call above that did the harm.
    uint32_t effective_seed = 0;
    if (overrides.has_seed)
        effective_seed = overrides.seed;
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

// ---------------------------------------------------------------------------
// Request logging
// ---------------------------------------------------------------------------

request_log_context make_request_log_context(const httplib::Request& req, const char* endpoint)
{
    request_log_context ctx;
    ctx.id = ++g_request_id;
    ctx.endpoint = endpoint;
    ctx.remote_addr = req.remote_addr;
    ctx.start = std::chrono::steady_clock::now();
    return ctx;
}

void log_request_start(server_log_level level, const request_log_context& ctx)
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

void log_request_details(server_log_level level, const request_log_context& ctx)
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

void log_request_done(
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

const char* request_log_status_to_string(request_log_status status)
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

// ---------------------------------------------------------------------------
// PCM16 bytes
// ---------------------------------------------------------------------------

bool build_pcm16_bytes(const float* data, uint32_t length, std::string* output, std::string* error)
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

// ---------------------------------------------------------------------------
// WAV building helpers
// ---------------------------------------------------------------------------

#ifdef COSYVOICE_NO_AUDIO

bool build_wav_bytes(const float* data, uint32_t length, uint32_t sample_rate, std::string* output, std::string* error)
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

#else // !COSYVOICE_NO_AUDIO

bool build_wav_bytes_with_audio_encoder(cosyvoice_audio_encoder_t encoder, const float* data, uint32_t length, cosyvoice_audio_encoding_format_t format, std::string* output, std::string* error)
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

#endif // COSYVOICE_NO_AUDIO

// ---------------------------------------------------------------------------
// Main audio payload builder
// ---------------------------------------------------------------------------

bool build_audio_payload(response_audio_format format, const cosyvoice_generated_speech& generated, const server_runtime& runtime, std::string* payload, std::string* error)
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

// ---------------------------------------------------------------------------
// Response format helpers
// ---------------------------------------------------------------------------

response_audio_format parse_response_format(const std::string& value)
{
    const auto lowered = to_lower(value);
    if (lowered == "mp3")  return response_audio_format::mp3;
    if (lowered == "aac")  return response_audio_format::aac;
    if (lowered == "m4a")  return response_audio_format::m4a;
    if (lowered == "flac") return response_audio_format::flac;
    if (lowered == "wav")  return response_audio_format::wav;
    if (lowered == "pcm")  return response_audio_format::pcm;
    return response_audio_format::unknown;
}

const char* response_format_to_string(response_audio_format format)
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

const char* response_format_to_content_type(response_audio_format format)
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

bool is_response_format_supported(const server_runtime& runtime, response_audio_format format)
{
    (void)runtime;
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

std::string supported_response_formats_to_string(const server_runtime& runtime)
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

// ---------------------------------------------------------------------------
// String join
// ---------------------------------------------------------------------------

std::string join_strings(const std::vector<std::string>& items, const char* sep)
{
    std::string r;
    for (size_t i = 0; i < items.size(); ++i)
    {
        if (i > 0) r += sep;
        r += items[i];
    }
    return r;
}
