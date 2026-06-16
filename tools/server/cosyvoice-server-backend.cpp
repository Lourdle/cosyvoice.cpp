#include "pch.h"
#include "cosyvoice-server.h"
#include "server_common.h"

#ifndef COSYVOICE_NO_AUDIO
    #include "cosyvoice-audio.h"
#endif

#include "nlohmann/json.hpp"
#include "httplib.h"

#include <cmath>
#include <cstdint>
#include <ctime>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>

using json = nlohmann::json;

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

static void fill_request_log_context_from_request(request_log_context* ctx, const speech_request& request)
{
    ctx->model = request.model;
    ctx->voice = request.voice;
    ctx->response_format = request.response_format;
    ctx->has_instructions = request.has_instructions;
    ctx->has_seed = request.has_seed;
    ctx->seed = request.seed;
}

int cosyvoice_server_backend_run(server_runtime& runtime)
{
    httplib::Server server;
    server.new_task_queue = [&runtime]() {
        return new httplib::ThreadPool(runtime.concurrency, runtime.concurrency);
    };

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
            const auto voice_str = join_strings(runtime.voice_names, ", ");
            std::string message = "Unknown voice \"" + request.voice + "\". Available voices: " + (voice_str.empty() ? "-" : voice_str);
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

        const uint32_t slot = get_or_assign_thread_slot(runtime);
        if (slot >= runtime.concurrency)
        {
            set_openai_error(res, 503, "Server is overloaded.", "server_error", nullptr, "server_overloaded");
            log_request_done(runtime.log_level, log_ctx, request_log_status::failed, res.status, 0, res.body.size(), "server_overloaded");
            return;
        }

        auto model_ctx = get_slot_model_context(runtime, slot);
        auto voice_ctx = get_slot_voice_session(runtime, slot, request.voice);
        if (!voice_ctx)
        {
            set_openai_error(res, 400, "Unknown voice for current slot.", "invalid_request_error", "voice", "invalid_voice");
            log_request_done(runtime.log_level, log_ctx, request_log_status::bad_request, res.status, 0, res.body.size(), "voice_mismatch");
            return;
        }

        cosyvoice_generated_speech generated = {};
        std::string payload;
        std::string encoding_error;
        std::string generation_error;
        uint32_t applied_seed = 0;

        generation_config_overrides go;
        go.has_temperature = request.has_temperature; go.temperature = request.temperature;
        go.has_top_k = request.has_top_k; go.top_k = request.top_k;
        go.has_top_p = request.has_top_p; go.top_p = request.top_p;
        go.has_win_size = request.has_win_size; go.win_size = request.win_size;
        go.has_tau_r = request.has_tau_r; go.tau_r = request.tau_r;
        go.has_min_token_text_ratio = request.has_min_token_text_ratio; go.min_token_text_ratio = request.min_token_text_ratio;
        go.has_max_token_text_ratio = request.has_max_token_text_ratio; go.max_token_text_ratio = request.max_token_text_ratio;
        go.has_seed = request.has_seed; go.seed = request.seed;

        if (!apply_generation_overrides(go, runtime, model_ctx, &applied_seed, &generation_error))
        {
            set_openai_error(res, 400, generation_error, "invalid_request_error", nullptr, "invalid_generation_params");
            log_request_done(runtime.log_level, log_ctx, request_log_status::bad_request, res.status, 0, res.body.size(), "generation_override");
            return;
        }

        bool ok = false;
        if (request.has_instructions)
            ok = cosyvoice_tts_instruct(voice_ctx, request.input.c_str(), request.instructions.c_str(), request.speed, &generated);
        else
            ok = cosyvoice_tts_zero_shot(voice_ctx, request.input.c_str(), request.speed, &generated);

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
    print_info_log(runtime.log_level, "  concurrency        : %u\n", runtime.concurrency);
    print_info_log(runtime.log_level, "  api_key_required   : %s\n", runtime.api_key.empty() ? "no" : "yes");
    const auto voices_str = join_strings(runtime.voice_names, ", ");
    print_info_log(runtime.log_level, "  voices             : %s\n", voices_str.empty() ? "-" : voices_str.c_str());
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

