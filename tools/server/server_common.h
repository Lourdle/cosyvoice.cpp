#pragma once

#include "cosyvoice-server.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

namespace httplib { class Request; class Response; }

#ifdef COSYVOICE_SERVER_USE_PCH
namespace httplib { class Server; }
using Server = httplib::Server;
#else
class Server;
#endif

// ---------------------------------------------------------------------------
// Audio output format
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Level-gated logging (timestamped, tag-prefixed)
// ---------------------------------------------------------------------------

bool log_level_enabled(server_log_level current, server_log_level required);

// Timestamped, level-gated message to stdout with a [tag] prefix.
void log_message(server_log_level current, server_log_level required,
                 const char* tag, const char* format, ...);

// Timestamped error message to stderr.
void print_error_log(const char* format, ...);

// Level-gated info - no timestamp, no trailing newline (for startup banners).
void print_info_log(server_log_level level, const char* format, ...);

// ---------------------------------------------------------------------------
// Float-to-PCM16 conversion helper
// ---------------------------------------------------------------------------

inline int16_t float_to_pcm16(float value)
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

// ---------------------------------------------------------------------------
// Generation config overrides (applied per-request)
// ---------------------------------------------------------------------------

struct generation_config_overrides
{
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
    bool has_seed = false;
    uint32_t seed = 0;
};

// Apply generation config overrides to a model context, handling seed selection.
// Returns false and sets error_message on failure.
bool apply_generation_overrides(
    const generation_config_overrides& overrides,
    server_runtime& runtime,
    cosyvoice_context_t model_ctx,
    uint32_t* applied_seed,
    std::string* error_message);

// ---------------------------------------------------------------------------
// Request logging (request ID, timing, status) - shared between API and WebUI
// ---------------------------------------------------------------------------

enum class request_log_status
{
    ok,
    bad_request,
    unauthorized,
    not_found,
    failed
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
    bool stream = false;
    std::chrono::steady_clock::time_point start;
};

request_log_context make_request_log_context(const httplib::Request& req, const char* endpoint);

void log_request_start(server_log_level level, const request_log_context& ctx);

void log_request_details(server_log_level level, const request_log_context& ctx);

void log_request_done(
    server_log_level level,
    const request_log_context& ctx,
    request_log_status status,
    int http_status,
    uint32_t effective_seed,
    size_t response_bytes,
    const char* note);

const char* request_log_status_to_string(request_log_status status);

// ---------------------------------------------------------------------------
// PCM / WAV audio building
// ---------------------------------------------------------------------------

// Convert float PCM array to raw PCM16 bytes (little-endian).
bool build_pcm16_bytes(const float* data, uint32_t length,
                       std::string* output, std::string* error);

// Native WAV builder (no audio encoder) - only compiled when COSYVOICE_NO_AUDIO.
#ifdef COSYVOICE_NO_AUDIO
bool build_wav_bytes(const float* data, uint32_t length, uint32_t sample_rate,
                     std::string* output, std::string* error);
#endif

// Audio-encoder-based builder - only compiled when !COSYVOICE_NO_AUDIO.
#ifndef COSYVOICE_NO_AUDIO
bool build_wav_bytes_with_audio_encoder(cosyvoice_audio_encoder_t encoder,
                                        const float* data, uint32_t length,
                                        cosyvoice_audio_encoding_format_t format,
                                        std::string* output, std::string* error);
#endif

// Main audio payload builder - dispatches to the correct encoder based on format.
bool build_audio_payload(response_audio_format format,
                         const cosyvoice_generated_speech& generated,
                         const server_runtime& runtime,
                         std::string* payload, std::string* error);

// ---------------------------------------------------------------------------
// Response format helpers
// ---------------------------------------------------------------------------

response_audio_format parse_response_format(const std::string& value);
const char*           response_format_to_string(response_audio_format format);
const char*           response_format_to_content_type(response_audio_format format);
bool                  is_response_format_supported(const server_runtime& runtime,
                                                   response_audio_format format);
std::string           supported_response_formats_to_string(const server_runtime& runtime);

// ---------------------------------------------------------------------------
// Streaming audio helpers
// ---------------------------------------------------------------------------

// Forward declarations from httplib
namespace httplib { class DataSink; }

// Context passed through the cosyvoice streaming TTS callback.
// encoder is a cosyvoice_audio_encoder_t stored as void* to avoid
// conditional-compilation complexity in this header.
struct streaming_callback_context
{
    httplib::DataSink*          sink;
    void*                       encoder;     // cosyvoice_audio_encoder_t or NULL
    response_audio_format       format;
    uint32_t                    sample_rate;
    bool                        wav_header_written;
    std::string*                error_out;
    std::atomic<bool>*          aborted;
};

// Build a standard 44-byte RIFF/WAV header with a placeholder data size
// (0x7FFFFFFF, meaning "until EOF" — accepted by most WAV players).
bool build_wav_header(std::string* output, uint32_t sample_rate);

// Encode a single PCM chunk and write it to the DataSink.
// Called from within a cosyvoice streaming TTS callback.
// Returns false to abort TTS if the sink write fails (sets *aborted).
bool stream_audio_chunk(streaming_callback_context* ctx, const float* pcm, uint32_t n_samples);

// ---------------------------------------------------------------------------
// Generic string join
// ---------------------------------------------------------------------------

std::string join_strings(const std::vector<std::string>& items, const char* sep);

// OpenAI-compatible error response helper (shared between API and WebUI modes)
void set_openai_error(httplib::Response& res, int status, const std::string& message,
                      const std::string& type, const char* param, const char* code);

// Bearer token authentication (shared between API and WebUI modes)
bool require_auth(const httplib::Request& req, const server_runtime& runtime, httplib::Response& res);

// Register OpenAI-compatible API routes on an existing server instance.
// Called by both API mode and WebUI mode (when a model is loaded).
void cosyvoice_server_register_api_routes(Server& server, server_runtime& runtime);
