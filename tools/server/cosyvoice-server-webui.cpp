#include "cosyvoice-server.h"
#include "server_common.h"
#include "resource.h"
#include "tool_common_cosyvoice.h"
#include "common.h"

#include <ggml-backend.h>

#include <cstring>
#include <memory>
#include <string>
#include <vector>

#ifndef COSYVOICE_NO_AUDIO
    #include "cosyvoice-audio.h"
#endif

#ifndef COSYVOICE_NO_FRONTEND
    #include "cosyvoice-frontend.h"
#endif

#ifndef COSYVOICE_USE_PCH
import nlohmann_json;
import httplib;
#endif

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Dynamic speaker registration (helpers called from route handlers)
// ---------------------------------------------------------------------------

static bool register_speaker_from_gguf(
    server_runtime& runtime,
    const std::string& name,
    const std::string& path)
{
    if (runtime.model_slots.empty())
        return false;

    auto model_ctx = runtime.model_slots[0].get();

    cosyvoice_prompt_speech_handle ps(cosyvoice_prompt_speech_load_from_file(path.c_str()));
    if (!ps)
        return false;

    cosyvoice_prompt_handle prompt(cosyvoice_prompt_init_from_prompt_speech(model_ctx, ps.get()));
    if (!prompt)
        return false;

    cosyvoice_tts_context_handle session(cosyvoice_tts_context_new(model_ctx, prompt.get()));
    if (!session)
        return false;

    // Apply current TTS context settings from the first existing session if any
    if (!runtime.voice_sessions.empty() && !runtime.voice_sessions[0].empty())
    {
        auto& first = runtime.voice_sessions[0].front().second;
        // Copy settings from existing session (text_norm, splitting, fade-in)
        // We can read the first session's config — but for simplicity, just use defaults.
        // cosyvoice_tts_context_set_text_normalization_enabled(session.get(), ...);
        // cosyvoice_tts_context_set_split_text_enabled(session.get(), ...);
        // cosyvoice_tts_context_set_fast_split_text_enabled(session.get(), ...);
        // cosyvoice_tts_context_set_fade_in_enabled(session.get(), ...);
        (void)first;
    }

    voice_runtime voice;
    voice.name         = name;
    voice.prompt_speech = std::move(ps);
    voice.prompt        = std::move(prompt);

    runtime.voice_names.push_back(name);
    runtime.voices.emplace(name, std::move(voice));
    runtime.voice_sessions[0].emplace_back(name, std::move(session));
    return true;
}

#ifndef COSYVOICE_NO_AUDIO
#ifndef COSYVOICE_NO_FRONTEND
static bool register_speaker_from_audio(
    server_runtime& runtime,
    const std::string& name,
    const std::string& prompt_text,
    const void* audio_data,
    size_t audio_size)
{
    if (runtime.model_slots.empty())
        return false;

    // Decode audio from memory
    using cosyvoice_audio_decoder_handle = std::unique_ptr<cosyvoice_audio_decoder, tool_deleter<cosyvoice_audio_decoder, &cosyvoice_audio_decoder_destroy>>;
    cosyvoice_audio_decoder_handle decoder(cosyvoice_audio_decoder_create());
    if (!decoder)
        return false;

    if (!cosyvoice_audio_decoder_decode(decoder.get(), audio_data, static_cast<uint32_t>(audio_size)))
        return false;

    float* audio_float = nullptr;
    uint32_t audio_length = 0;
    uint32_t audio_sample_rate = 0;
    cosyvoice_audio_decoder_get_decoded_data(decoder.get(), &audio_float, &audio_length, &audio_sample_rate);
    if (!audio_float || audio_length == 0 || audio_sample_rate == 0)
        return false;

    // Use pre-loaded frontend if available, otherwise load on-the-fly
    cosyvoice_frontend_context_t frontend = runtime.frontend_ctx.get();
    bool own_frontend = false;
    if (!frontend)
    {
        frontend = cosyvoice_frontend_load_from_files(
            runtime.speech_tokenizer.c_str(),
            runtime.campplus.c_str());
        if (!frontend)
            return false;
        own_frontend = true;
    }

    // Extract prompt speech
    cosyvoice_prompt_speech_handle ps(
        cosyvoice_frontend_prompt_speech(
            frontend,
            audio_float,
            audio_length,
            audio_sample_rate,
            prompt_text.c_str()));

    if (own_frontend)
        cosyvoice_frontend_free(frontend);

    if (!ps)
        return false;

    // Create prompt from extracted features
    auto model_ctx = runtime.model_slots[0].get();
    cosyvoice_prompt_handle prompt(cosyvoice_prompt_init_from_prompt_speech(model_ctx, ps.get()));
    if (!prompt)
        return false;

    cosyvoice_tts_context_handle session(cosyvoice_tts_context_new(model_ctx, prompt.get()));
    if (!session)
        return false;

    voice_runtime voice;
    voice.name          = name;
    voice.prompt_speech = std::move(ps);
    voice.prompt        = std::move(prompt);

    runtime.voice_names.push_back(name);
    runtime.voices.emplace(name, std::move(voice));
    runtime.voice_sessions[0].emplace_back(name, std::move(session));
    return true;
}
#endif // !COSYVOICE_NO_FRONTEND
#endif // !COSYVOICE_NO_AUDIO

// ---------------------------------------------------------------------------
// Route handlers — WebUI mode
// ---------------------------------------------------------------------------

int cosyvoice_server_webui_run(server_runtime& runtime)
{
    Server server;
    // Use default task queue (no explicit override)

    server.set_exception_handler([](const Request&, Response& res, std::exception_ptr ep)
    {
        (void)ep;
        res.status = 500;

        static const char json[] = R"({"error":"Internal server error"})";
        res.set_content(json, "application/json");
    });

    // ---- GET / — serve the WebUI HTML ----
    server.Get("/", [&runtime](const Request&, Response& res)
    {
        res.status = 200;

        size_t size = 0;
        const void* data = server_resource_load(IDR_WEBUI_HTML, &size);
        if (!data)
        {
            res.set_content("CosyVoice WebUI\nEmbedded resource not found.\n", "text/plain");
            return;
        }

        std::string html(static_cast<const char*>(data), size);

        // Build injected config
        nlohmann::json cfg;
        cfg["frontendModel"]   = runtime.frontend_model;
        cfg["speechTokenizer"] = runtime.speech_tokenizer;
        cfg["campplus"]        = runtime.campplus;
        cfg["apiKeyRequired"]  = !runtime.api_key.empty();
        cfg["modelLoaded"]     = !runtime.model_slots.empty();
        cfg["defaultMaxLlmLen"] = static_cast<uint32_t>(COSYVOICE_DEFAULT_LLM_MAX_SEQ_LEN);

        std::string script = "\n<script>window.__COSYVOICE_CONFIG__ = " + cfg.dump() + ";</script>\n";

        // Insert config before the external JS reference
        const std::string js_marker = "<script src=\"/cosyvoice-webui.js\">";
        const auto pos = html.find(js_marker);
        if (pos != std::string::npos)
            html.insert(pos, script);

        res.set_content(std::move(html), "text/html");
    });

    // ---- GET /cosyvoice-webui.css — serve embedded CSS ----
    server.Get("/cosyvoice-webui.css", [](const Request&, Response& res)
    {
        size_t size = 0;
        const void* data = server_resource_load(IDR_WEBUI_CSS, &size);
        if (!data)
        {
            res.status = 404;
            res.set_content("/* Not found */", "text/css");
            return;
        }
        res.status = 200;
        res.set_content(std::string(static_cast<const char*>(data), size), "text/css");
    });

    // ---- GET /cosyvoice-webui.js — serve embedded JavaScript ----
    server.Get("/cosyvoice-webui.js", [](const Request&, Response& res)
    {
        size_t size = 0;
        const void* data = server_resource_load(IDR_WEBUI_JS, &size);
        if (!data)
        {
            res.status = 404;
            res.set_content("// Not found", "application/javascript");
            return;
        }
        res.status = 200;
        res.set_content(std::string(static_cast<const char*>(data), size), "application/javascript");
    });

    // ---- Simple ping endpoint (always works) ----
    server.Get("/ping", [](const Request&, Response& res)
    {
        res.set_content("pong", "text/plain");
    });

    // ---- GET /backends — list available GGML backends ----
    server.Get("/backends", [](const Request&, Response& res)
    {
        nlohmann::json list = nlohmann::json::array();

        // "auto" pseudo-entry
        nlohmann::json auto_entry;
        auto_entry["name"]        = "auto";
        auto_entry["description"] = "Choose the best available backend automatically";
        list.push_back(auto_entry);

        size_t nd = ggml_backend_dev_count();
        for (size_t i = 0; i < nd; i++)
        {
            auto dev = ggml_backend_dev_get(i);
            struct ggml_backend_dev_props props;
            ggml_backend_dev_get_props(dev, &props);

            nlohmann::json entry;
            entry["name"]        = props.name ? props.name : "unknown";
            entry["description"] = props.description ? props.description : "";
            entry["has_device_id"] = props.device_id ? true : false;
            list.push_back(entry);
        }

        res.status = 200;
        res.set_content(list.dump(), "application/json");
    });

    // ---- GET /formats — list supported audio output formats ----
    server.Get("/formats", [](const Request&, Response& res)
    {
        nlohmann::json list = nlohmann::json::array();
        list.push_back("wav");
        list.push_back("pcm");
#ifndef COSYVOICE_NO_AUDIO
        // Query each format via the API (does NOT require an audio encoder)
        static const struct { const char* name; cosyvoice_audio_encoding_format_t fmt; } fmts[] = {
            {"mp3",  COSYVOICE_AUDIO_ENCODING_FORMAT_MP3},
            {"opus", COSYVOICE_AUDIO_ENCODING_FORMAT_OPUS},
            {"flac", COSYVOICE_AUDIO_ENCODING_FORMAT_FLAC},
            {"aac",  COSYVOICE_AUDIO_ENCODING_FORMAT_AAC},
            {"m4a",  COSYVOICE_AUDIO_ENCODING_FORMAT_M4A},
        };
        for (auto& f : fmts)
            if (cosyvoice_audio_encoding_format_supported(f.fmt))
                list.push_back(f.name);
#endif
        res.status = 200;
        res.set_content(list.dump(), "application/json");
    });

    // ---- GET /status — server status ----
    server.Get("/status", [&runtime](const Request&, Response& res)
    {
        // Use manual JSON building — avoids an issue with nlohmann::json in this handler
        std::string json = "{\"status\":\"ok\"";
        json += ",\"model_loaded\":";
        json += runtime.model_slots.empty() ? "false" : "true";
        json += ",\"model\":";
        if (runtime.model_slots.empty())
            json += "null";
        else
            json += "\"" + runtime.served_model_name + "\"";
        json += ",\"sample_rate\":" + std::to_string(runtime.sample_rate);
        if (!runtime.model_slots.empty())
        {
            cosyvoice_context_params_t actual_params;
            cosyvoice_get_context_params(runtime.model_slots.front().get(), &actual_params);
            json += ",\"max_llm_len\":" + std::to_string(actual_params.n_max_seq);
            json += ",\"k_cache_type\":\"" + llm_kv_cache_type_to_string(actual_params.llm_k_cache_type) + "\"";
            json += ",\"v_cache_type\":\"" + llm_kv_cache_type_to_string(actual_params.llm_v_cache_type) + "\"";
            json += ",\"buffer_policy\":\"" + std::string(inference_buffer_policy_to_string(actual_params.inference_buffer_policy)) + "\"";
            auto arch = cosyvoice_get_architecture(runtime.model_slots.front().get());
            if (arch && *arch)
                json += ",\"model_arch\":\"" + std::string(arch) + "\"";
        }
#if !defined(COSYVOICE_NO_FRONTEND)
        bool fe_avail = runtime.frontend_ctx ? true : false;
#else
        bool fe_avail = false;
#endif
        json += ",\"frontend_available\":" + std::string(fe_avail ? "true" : "false");
        json += ",\"speakers\":[";
        for (size_t i = 0; i < runtime.voice_names.size(); ++i)
        {
            if (i > 0) json += ",";
            json += "\"" + runtime.voice_names[i] + "\"";
        }
        json += "]}";

        res.status = 200;
        res.set_content(std::move(json), "application/json");
    });

    // ---- GET /speaker — list registered speakers ----
    server.Get("/speaker", [&runtime](const Request&, Response& res)
    {
        nlohmann::json payload = runtime.voice_names;
        res.status = 200;
        res.set_content(payload.dump(), "application/json");
    });

    // ---- POST /speaker — register a new speaker ----
    server.Post("/speaker", [&runtime](const Request& req, Response& res)
    {
        // Detect multipart vs JSON
        bool is_multipart = req.has_header("Content-Type") &&
            req.get_header_value("Content-Type").find("multipart/form-data") != std::string::npos;

        std::string type, name, path, text;

        if (is_multipart)
        {
            // ?type=extract (always extract when multipart)
            type = "extract";

            // Get form fields
            name = req.form.get_field("name");
            text = req.form.get_field("text");
        }
        else
        {
            // JSON body
            nlohmann::json body;
            try { body = nlohmann::json::parse(req.body); }
            catch (...)
            {
                res.status = 400;
                nlohmann::json err = {{"error", "Invalid JSON body"}};
                res.set_content(err.dump(), "application/json");
                return;
            }

            type = body.value("type", "gguf");
            name = body.value("name", "");
            path = body.value("path", "");
            text = body.value("text", "");
        }

        if (name.empty())
        {
            res.status = 400;
            nlohmann::json err = {{"error", "Missing 'name' field"}};
            res.set_content(err.dump(), "application/json");
            return;
        }

        // Check if name already exists
        if (runtime.voices.find(name) != runtime.voices.end())
        {
            res.status = 409;
            nlohmann::json err = {{"error", "Speaker '" + name + "' already exists"}};
            res.set_content(err.dump(), "application/json");
            return;
        }

        // Model must be loaded to register speakers
        if (runtime.model_slots.empty())
        {
            res.status = 503;
            nlohmann::json err = {{"error", "No model loaded. Start with --model to enable speaker registration."}};
            res.set_content(err.dump(), "application/json");
            return;
        }

        if (type == "gguf")
        {
            if (path.empty())
            {
                res.status = 400;
                nlohmann::json err = {{"error", "Missing 'path' field for gguf type"}};
                res.set_content(err.dump(), "application/json");
                return;
            }
            if (!register_speaker_from_gguf(runtime, name, path))
            {
                res.status = 500;
                nlohmann::json err = {{"error", "Failed to load prompt speech from: " + path}};
                res.set_content(err.dump(), "application/json");
                return;
            }
        }
        else if (type == "extract")
        {
#if !defined(COSYVOICE_NO_AUDIO) && !defined(COSYVOICE_NO_FRONTEND)
            if (!req.form.has_file("audio"))
            {
                res.status = 400;
                nlohmann::json err = {{"error", "Missing 'audio' file in multipart upload"}};
                res.set_content(err.dump(), "application/json");
                return;
            }

            // Get the uploaded file data
            const auto& audio_entry = *req.form.files.find("audio");
            const auto& audio_formdata = audio_entry.second;
            if (!register_speaker_from_audio(
                    runtime, name, text,
                    audio_formdata.content.data(), audio_formdata.content.size()))
            {
                res.status = 500;
                nlohmann::json err = {{"error", "Failed to extract speaker features from audio"}};
                res.set_content(err.dump(), "application/json");
                return;
            }
#else
            (void)text;
            res.status = 501;
            nlohmann::json err = {{"error",
                "Audio extraction is not available. "
                "This build has COSYVOICE_NO_AUDIO or COSYVOICE_NO_FRONTEND defined."}};
            res.set_content(err.dump(), "application/json");
            return;
#endif
        }
        else
        {
            res.status = 400;
            nlohmann::json err = {{"error", "Unknown type '" + type + "'. Use 'gguf' or 'extract'."}};
            res.set_content(err.dump(), "application/json");
            return;
        }

        nlohmann::json ok = {{"success", true}, {"name", name}};
        res.status = 200;
        res.set_content(ok.dump(), "application/json");
        log_message(runtime.log_level, server_log_level::concise, "WEBUI",
            "Speaker registered: %s (type=%s, total=%zu)", name.c_str(), type.c_str(), runtime.voice_names.size());
    });

    // ---- DELETE /speaker/<name> — remove a speaker ----
    server.Delete(R"(/speaker/(.*))", [&runtime](const Request& req, Response& res)
    {
        std::string name = req.matches[1];

        auto it = runtime.voices.find(name);
        if (it == runtime.voices.end())
        {
            res.status = 404;
            nlohmann::json err = {{"error", "Speaker '" + name + "' not found"}};
            res.set_content(err.dump(), "application/json");
            return;
        }

        // Remove from voices map
        runtime.voices.erase(it);

        // Remove from voice_names
        for (auto nit = runtime.voice_names.begin(); nit != runtime.voice_names.end(); ++nit)
        {
            if (*nit == name)
            {
                runtime.voice_names.erase(nit);
                break;
            }
        }

        // Remove from sessions
        for (auto& sessions : runtime.voice_sessions)
        {
            for (auto sit = sessions.begin(); sit != sessions.end(); ++sit)
            {
                if (sit->first == name)
                {
                    sessions.erase(sit);
                    break;
                }
            }
        }

        nlohmann::json ok = {{"success", true}};
        res.status = 200;
        res.set_content(ok.dump(), "application/json");
        log_message(runtime.log_level, server_log_level::concise, "WEBUI",
            "Speaker removed: %s (remaining=%zu)", name.c_str(), runtime.voice_names.size());
    });

    // ---- POST /speaker/save — save speaker prompt speech to a server-side path ----
    server.Post("/speaker/save", [&runtime](const Request& req, Response& res)
    {
        nlohmann::json body;
        try { body = nlohmann::json::parse(req.body); }
        catch (...)
        {
            res.status = 400;
            nlohmann::json err = {{"error", "Invalid JSON body"}};
            res.set_content(err.dump(), "application/json");
            return;
        }

        std::string name = body.value("name", "");
        std::string path = body.value("path", "");
        if (name.empty())
        {
            res.status = 400;
            nlohmann::json err = {{"error", "Missing 'name' field"}};
            res.set_content(err.dump(), "application/json");
            return;
        }
        if (path.empty())
        {
            res.status = 400;
            nlohmann::json err = {{"error", "Missing 'path' field"}};
            res.set_content(err.dump(), "application/json");
            return;
        }

        auto it = runtime.voices.find(name);
        if (it == runtime.voices.end())
        {
            res.status = 404;
            nlohmann::json err = {{"error", "Speaker '" + name + "' not found"}};
            res.set_content(err.dump(), "application/json");
            return;
        }

        if (!cosyvoice_prompt_speech_save_to_file(it->second.prompt_speech.get(), path.c_str()))
        {
            res.status = 500;
            nlohmann::json err = {{"error", "Failed to save prompt speech to: " + path}};
            res.set_content(err.dump(), "application/json");
            return;
        }

        nlohmann::json ok = {{"success", true}, {"path", path}};
        res.status = 200;
        res.set_content(ok.dump(), "application/json");
        log_message(runtime.log_level, server_log_level::concise, "WEBUI",
            "Speaker prompt speech saved: %s -> %s", name.c_str(), path.c_str());
    });

    server.Post("/tts", [&runtime](const Request& req, Response& res)
    {
        request_log_context log_ctx = make_request_log_context(req, "/tts");
        log_request_start(runtime.log_level, log_ctx);

        // Parse JSON body
        nlohmann::json body;
        try { body = nlohmann::json::parse(req.body); }
        catch (...)
        {
            res.status = 400;
            nlohmann::json err = {{"error", "Invalid JSON body"}};
            res.set_content(err.dump(), "application/json");
            log_request_done(runtime.log_level, log_ctx, request_log_status::bad_request, res.status, 0, res.body.size(), "invalid_json");
            return;
        }

        // Validate
        if (runtime.model_slots.empty())
        {
            res.status = 503;
            nlohmann::json err = {{"error", "No model loaded"}};
            res.set_content(err.dump(), "application/json");
            log_request_done(runtime.log_level, log_ctx, request_log_status::failed, res.status, 0, res.body.size(), "no_model");
            return;
        }

        const std::string text  = body.value("input", body.value("text", ""));
        if (text.empty())
        {
            res.status = 400;
            nlohmann::json err = {{"error", "Missing 'text' or 'input' field"}};
            res.set_content(err.dump(), "application/json");
            log_request_done(runtime.log_level, log_ctx, request_log_status::bad_request, res.status, 0, res.body.size(), "missing_text");
            return;
        }

        const std::string voice = body.value("voice", "");
        if (voice.empty())
        {
            res.status = 400;
            nlohmann::json err = {{"error", "Missing 'voice' field"}};
            res.set_content(err.dump(), "application/json");
            log_request_done(runtime.log_level, log_ctx, request_log_status::bad_request, res.status, 0, res.body.size(), "missing_voice");
            return;
        }

        auto voice_it = runtime.voices.find(voice);
        if (voice_it == runtime.voices.end())
        {
            res.status = 400;
            nlohmann::json err = {{"error", "Unknown voice '" + voice + "'. Available: " + join_strings(runtime.voice_names, ", ")}};
            res.set_content(err.dump(), "application/json");
            log_request_done(runtime.log_level, log_ctx, request_log_status::bad_request, res.status, 0, res.body.size(), "unknown_voice");
            return;
        }

        // Get TTS context using shared inline helper (WebUI: slot 0)
        cosyvoice_tts_context_t tts_ctx = get_slot_voice_session(runtime, 0, voice);
        if (!tts_ctx)
        {
            res.status = 500;
            nlohmann::json err = {{"error", "TTS context not initialized for voice '" + voice + "'"}};
            res.set_content(err.dump(), "application/json");
            log_request_done(runtime.log_level, log_ctx, request_log_status::failed, res.status, 0, res.body.size(), "no_tts_ctx");
            return;
        }

        // Fill request context fields for detailed logging
        log_ctx.voice = voice;
        log_ctx.response_format = body.value("format", "wav");
        log_ctx.has_instructions = body.contains("instructions") || body.contains("instruction");
        log_ctx.has_seed = body.contains("seed");
        if (log_ctx.has_seed)
            log_ctx.seed = body["seed"].get<uint32_t>();
        log_request_details(runtime.log_level, log_ctx);

        // Speed
        float speed = body.value("speed", 1.0f);
        if (speed <= 0.0f) speed = 1.0f;

        // Apply TTS context flags (optional overrides per request)
        if (body.contains("fade_in"))
            cosyvoice_tts_context_set_fade_in_enabled(tts_ctx, body["fade_in"].get<bool>());
        if (body.contains("text_normalization"))
            cosyvoice_tts_context_set_text_normalization_enabled(tts_ctx, body["text_normalization"].get<bool>());
        if (body.contains("split_text"))
            cosyvoice_tts_context_set_split_text_enabled(tts_ctx, body["split_text"].get<bool>());
        if (body.contains("fast_split"))
            cosyvoice_tts_context_set_fast_split_text_enabled(tts_ctx, body["fast_split"].get<bool>());

        // Apply generation overrides (shared function)
        uint32_t applied_seed = 0;
        {
            auto model_ctx = runtime.model_slots[0].get();

            generation_config_overrides go;
            if (body.contains("temperature")) { go.has_temperature = true; go.temperature = body["temperature"].get<float>(); }
            if (body.contains("top_k"))       { go.has_top_k = true; go.top_k = body["top_k"].get<int>(); }
            if (body.contains("top_p"))       { go.has_top_p = true; go.top_p = body["top_p"].get<float>(); }
            if (body.contains("win_size"))    { go.has_win_size = true; go.win_size = body["win_size"].get<int>(); }
            if (body.contains("tau_r"))       { go.has_tau_r = true; go.tau_r = body["tau_r"].get<float>(); }
            if (body.contains("seed"))        { go.has_seed = true; go.seed = body["seed"].get<uint32_t>(); }

            std::string gen_error;
            if (!apply_generation_overrides(go, runtime, model_ctx, &applied_seed, &gen_error))
            {
                res.status = 400;
                nlohmann::json err = {{"error", gen_error}};
                res.set_content(err.dump(), "application/json");
                log_request_done(runtime.log_level, log_ctx, request_log_status::bad_request, res.status, applied_seed, res.body.size(), "generation_override");
                return;
            }
        }

        // Generate speech
        cosyvoice_generated_speech generated = {};
        bool ok = false;

        const std::string mode         = body.value("mode", "auto");
        const std::string instructions = body.value("instructions", body.value("instruction", ""));

        if (mode == "cross_lingual")
        {
            ok = cosyvoice_tts_cross_lingual(tts_ctx, text.c_str(), speed, &generated);
        }
        else if (mode == "instruct" || (mode == "auto" && !instructions.empty()))
        {
            ok = cosyvoice_tts_instruct(tts_ctx, text.c_str(), instructions.c_str(), speed, &generated);
        }
        else // "zero_shot" or "auto"
        {
            ok = cosyvoice_tts_zero_shot(tts_ctx, text.c_str(), speed, &generated);
        }

        if (!ok || !generated.data || generated.length == 0)
        {
            res.status = 500;
            nlohmann::json err = {{"error", "TTS generation failed"}};
            res.set_content(err.dump(), "application/json");
            log_request_done(runtime.log_level, log_ctx, request_log_status::failed, res.status, applied_seed, res.body.size(), "generation_failed");
            return;
        }

        // Encode output using shared utilities
        std::string audio_payload;
        const std::string format_str = body.value("format", "wav");
        auto fmt = parse_response_format(format_str);
        if (fmt == response_audio_format::unknown)
            fmt = response_audio_format::wav;

        std::string encoding_error;
        if (!build_audio_payload(fmt, generated, runtime, &audio_payload, &encoding_error))
        {
            res.status = 500;
            nlohmann::json err = {{"error", "Audio encoding failed: " + encoding_error}};
            res.set_content(err.dump(), "application/json");
            log_request_done(runtime.log_level, log_ctx, request_log_status::failed, res.status, applied_seed, res.body.size(), "audio_encode_failed");
            return;
        }

        log_message(runtime.log_level, server_log_level::concise, "WEBUI",
            "TTS generated: voice=%s, text_len=%zu, format=%s, samples=%u",
            voice.c_str(), text.size(), format_str.c_str(), generated.length);

        res.status = 200;
        res.set_content(std::move(audio_payload), response_format_to_content_type(fmt));
        log_request_done(runtime.log_level, log_ctx, request_log_status::ok, res.status, applied_seed, audio_payload.size(), "tts");
    });

    // ---- GET /frontend/model — return frontend model paths ----
    server.Get("/frontend/model", [&runtime](const Request&, Response& res)
    {
        nlohmann::json payload;
        payload["speech_tokenizer"] = runtime.speech_tokenizer;
        payload["campplus"]         = runtime.campplus;
        res.status = 200;
        res.set_content(payload.dump(), "application/json");
    });

    // ---- PUT /frontend/model — update frontend model paths ----
    server.Put("/frontend/model", [&runtime](const Request& req, Response& res)
    {
        nlohmann::json body;
        try { body = nlohmann::json::parse(req.body); }
        catch (...)
        {
            res.status = 400;
            nlohmann::json err = {{"error", "Invalid JSON body"}};
            res.set_content(err.dump(), "application/json");
            return;
        }

        if (body.contains("speech_tokenizer"))
            runtime.speech_tokenizer = body["speech_tokenizer"].get<std::string>();
        if (body.contains("campplus"))
            runtime.campplus = body["campplus"].get<std::string>();

        nlohmann::json ok = {{"success", true}};
        res.status = 200;
        res.set_content(ok.dump(), "application/json");
    });

    // ---- POST /frontend/model/load — load ONNX frontend models into memory ----
#if !defined(COSYVOICE_NO_FRONTEND)
    server.Post("/frontend/model/load", [&runtime](const Request& req, Response& res)
    {
        nlohmann::json body;
        try { body = nlohmann::json::parse(req.body); }
        catch (...)
        {
            res.status = 400;
            nlohmann::json err = {{"error", "Invalid JSON body"}};
            res.set_content(err.dump(), "application/json");
            return;
        }

        std::string tokenizer_path = body.value("speech_tokenizer", runtime.speech_tokenizer);
        std::string campplus_path  = body.value("campplus", runtime.campplus);

        if (tokenizer_path.empty() || campplus_path.empty())
        {
            res.status = 400;
            nlohmann::json err = {{"error", "Both speech_tokenizer and campplus paths are required"}};
            res.set_content(err.dump(), "application/json");
            return;
        }

        auto frontend = cosyvoice_frontend_load_from_files(tokenizer_path.c_str(), campplus_path.c_str());
        if (!frontend)
        {
            res.status = 500;
            nlohmann::json err = {{"error", "Failed to load frontend ONNX models"}};
            res.set_content(err.dump(), "application/json");
            return;
        }

        runtime.frontend_ctx.reset(frontend);
        runtime.speech_tokenizer = tokenizer_path;
        runtime.campplus         = campplus_path;

        log_message(runtime.log_level, server_log_level::concise, "WEBUI",
            "Frontend ONNX models loaded: speech_tokenizer=%s, campplus=%s",
            runtime.speech_tokenizer.c_str(), runtime.campplus.c_str());

        nlohmann::json ok2 = {{"success", true}};
        res.status = 200;
        res.set_content(ok2.dump(), "application/json");
    });
#else
    server.Post("/frontend/model/load", [](const Request&, Response& res)
    {
        res.status = 501;
        nlohmann::json err = {{"error", "Frontend support is not available in this build (COSYVOICE_NO_FRONTEND)."}};
        res.set_content(err.dump(), "application/json");
    });
#endif

    // ---- POST /frontend/model/unload — unload ONNX frontend models ----
#if !defined(COSYVOICE_NO_FRONTEND)
    server.Post("/frontend/model/unload", [&runtime](const Request&, Response& res)
    {
        if (!runtime.frontend_ctx)
        {
            res.status = 409;
            nlohmann::json err2 = {{"error", "No frontend model is currently loaded."}};
            res.set_content(err2.dump(), "application/json");
            return;
        }

        runtime.frontend_ctx.reset();
        log_message(runtime.log_level, server_log_level::concise, "WEBUI",
            "Frontend ONNX models unloaded");

        nlohmann::json ok2 = {{"success", true}};
        res.status = 200;
        res.set_content(ok2.dump(), "application/json");
    });
#else
    server.Post("/frontend/model/unload", [](const Request&, Response& res)
    {
        res.status = 501;
        nlohmann::json err = {{"error", "Frontend support is not available in this build (COSYVOICE_NO_FRONTEND)."}};
        res.set_content(err.dump(), "application/json");
    });
#endif

    // ---- POST /model/load — dynamically load a GGUF model ----
    server.Post("/model/load", [&runtime](const Request& req, Response& res)
    {
        request_log_context log_ctx = make_request_log_context(req, "/model/load");
        log_request_start(runtime.log_level, log_ctx);

        // Must not already have a model loaded
        if (!runtime.model_slots.empty())
        {
            res.status = 409;
            nlohmann::json err = {{"error", "A model is already loaded. Unload it first via POST /model/unload."}};
            res.set_content(err.dump(), "application/json");
            log_request_done(runtime.log_level, log_ctx, request_log_status::bad_request, res.status, 0, res.body.size(), "model_loaded");
            return;
        }

        nlohmann::json body;
        try { body = nlohmann::json::parse(req.body); }
        catch (...)
        {
            res.status = 400;
            nlohmann::json err = {{"error", "Invalid JSON body"}};
            res.set_content(err.dump(), "application/json");
            log_request_done(runtime.log_level, log_ctx, request_log_status::bad_request, res.status, 0, res.body.size(), "invalid_json");
            return;
        }

        std::string model_path = body.value("model", "");
        if (model_path.empty())
        {
            res.status = 400;
            nlohmann::json err = {{"error", "Missing 'model' field"}};
            res.set_content(err.dump(), "application/json");
            log_request_done(runtime.log_level, log_ctx, request_log_status::bad_request, res.status, 0, res.body.size(), "missing_model");
            return;
        }

        std::string backend_type = body.value("backend", "auto");
        uint32_t    n_threads    = body.value("n_threads", 0u);

        // Select backend
        ggml_backend_t backend = nullptr;
        if (backend_type == "auto" || backend_type.empty())
            backend = ggml_backend_init_best();
        else
        {
            // Try matching by the full device name or backend name
            backend = ggml_backend_init_by_name(backend_type.c_str(), nullptr);

            // Fallback: iterate devices and try matching by name/description prefix
            if (!backend)
            {
                size_t nd = ggml_backend_dev_count();
                for (size_t i = 0; i < nd; i++)
                {
                    auto dev = ggml_backend_dev_get(i);
                    struct ggml_backend_dev_props props;
                    ggml_backend_dev_get_props(dev, &props);
                    if (props.name && (backend_type == props.name ||
                        (props.description && backend_type == props.description)))
                    {
                        backend = ggml_backend_dev_init(dev, nullptr);
                        break;
                    }
                }
            }
        }

        if (!backend)
        {
            res.status = 500;
            nlohmann::json err = {{"error", "Failed to initialize backend: " + backend_type}};
            res.set_content(err.dump(), "application/json");
            log_request_done(runtime.log_level, log_ctx, request_log_status::failed, res.status, 0, res.body.size(), "backend_init");
            return;
        }

        // Build context params with defaults
        cosyvoice_context_params_v2_cpp context_params;
        cosyvoice_init_default_context_params(&context_params);

        if (runtime.has_seed)
            context_params.seed = runtime.seed;
        context_params.n_workers = 1;

        // Apply optional advanced config from request
        if (body.contains("llm_kv_cache_type"))
        {
            cosyvoice_llm_kv_cache_type_t kv_type;
            if (parse_llm_kv_cache_type_arg(body["llm_kv_cache_type"].get<std::string>(), &kv_type))
                context_params.llm_kv_cache_type = kv_type;
        }
        if (body.contains("inference_buffer_policy"))
        {
            cosyvoice_inference_buffer_policy_t policy;
            if (parse_inference_buffer_policy_arg(body["inference_buffer_policy"].get<std::string>(), &policy))
                context_params.inference_buffer_policy = policy;
        }
        if (body.contains("max_llm_len"))
        {
            uint32_t v = body["max_llm_len"].get<uint32_t>();
            if (v > 0) context_params.n_max_seq = v;
        }

        // Load the model — check result FIRST, then store
        auto loaded_ctx = cosyvoice_load_from_file_ext(
            model_path.c_str(),
            &context_params,
            backend,
            n_threads);

        if (!loaded_ctx)
        {
            res.status = 500;
            nlohmann::json err = {{"error", "Failed to load model: " + model_path}};
            res.set_content(err.dump(), "application/json");
            log_request_done(runtime.log_level, log_ctx, request_log_status::failed, res.status, 0, res.body.size(), "model_load_failed");
            return;
        }

        runtime.model_slots.emplace_back(loaded_ctx);

        // Log model details
        {
            cosyvoice_context_params_t p;
            cosyvoice_get_context_params(loaded_ctx, &p);
            log_message(runtime.log_level, server_log_level::concise, "WEBUI",
                "Model loaded: %s (arch=%s, backend=%s, threads=%u, kv_cache_type=%s, buffer=%s, max_llm_len=%u)",
                model_path.c_str(),
                cosyvoice_get_architecture(loaded_ctx) ? cosyvoice_get_architecture(loaded_ctx) : "?",
                backend_type.c_str(),
                n_threads ? n_threads : (uint32_t)0,
                llm_kv_cache_type_to_string(p.llm_kv_cache_type).c_str(),
                inference_buffer_policy_to_string(p.inference_buffer_policy),
                p.n_max_seq);
        }

        // Set served model name
        {
            auto arch = cosyvoice_get_architecture(runtime.model_slots.front().get());
            if (arch && *arch)
                runtime.served_model_name = arch;
            else
            {
                const auto last_sep = model_path.find_last_of("/\\");
                const size_t start = last_sep == std::string::npos ? 0 : last_sep + 1;
                const auto last_dot = model_path.find_last_of('.');
                const size_t end = (last_dot == std::string::npos || last_dot < start) ? model_path.size() : last_dot;
                runtime.served_model_name = (end <= start) ? "cosyvoice" : model_path.substr(start, end - start);
            }
        }

        // Sample rate
        runtime.sample_rate = cosyvoice_get_sample_rate(runtime.model_slots.front().get());

        // Get default generation config
        cosyvoice_get_default_generation_config(runtime.model_slots.front().get(), &runtime.default_generation_config);
        for (auto& slot : runtime.model_slots)
            cosyvoice_set_generation_config(slot.get(), &runtime.default_generation_config);

    #ifndef COSYVOICE_NO_AUDIO
        runtime.audio_encoder.reset(cosyvoice_audio_encoder_create(runtime.sample_rate));
    #endif

        // Initialize empty voice sessions (WebUI: concurrency = 1)
        runtime.voice_sessions.clear();
        runtime.voice_sessions.reserve(1);
        runtime.voice_sessions.emplace_back();

        nlohmann::json ok = {
            {"success", true},
            {"model",    runtime.served_model_name},
            {"sample_rate", runtime.sample_rate}
        };
        res.status = 200;
        res.set_content(ok.dump(), "application/json");

        log_request_done(runtime.log_level, log_ctx, request_log_status::ok, res.status, 0, res.body.size(), "model_loaded");
    });

    // ---- POST /model/unload — unload the current model ----
    server.Post("/model/unload", [&runtime](const Request& req, Response& res)
    {
        request_log_context log_ctx = make_request_log_context(req, "/model/unload");
        log_request_start(runtime.log_level, log_ctx);

        if (runtime.model_slots.empty())
        {
            res.status = 409;
            nlohmann::json err = {{"error", "No model is currently loaded."}};
            res.set_content(err.dump(), "application/json");
            log_request_done(runtime.log_level, log_ctx, request_log_status::bad_request, res.status, 0, res.body.size(), "no_model");
            return;
        }

        // Order matters: TTS sessions → voices → model context
        runtime.voice_sessions.clear();
        runtime.voices.clear();
        runtime.voice_names.clear();
        runtime.model_slots.clear();

        runtime.sample_rate = 0;
        runtime.served_model_name.clear();
        memset(&runtime.default_generation_config, 0, sizeof(runtime.default_generation_config));

    #ifndef COSYVOICE_NO_AUDIO
        runtime.audio_encoder.reset();
    #endif

        log_message(runtime.log_level, server_log_level::concise, "WEBUI",
            "Model unloaded");

        // Re-initialize empty voice sessions vector
        runtime.voice_sessions.reserve(1);
        runtime.voice_sessions.emplace_back();

        nlohmann::json ok = {{"success", true}};
        res.status = 200;
        res.set_content(ok.dump(), "application/json");
        log_request_done(runtime.log_level, log_ctx, request_log_status::ok, res.status, 0, res.body.size(), "model_unloaded");
    });

    // ---- GET /model/defaults — return defaults for all configurable parameters ----
    // Works even without a loaded model (sensible fallbacks for generation params).
    server.Get("/model/defaults", [&runtime](const Request&, Response& res)
    {
        nlohmann::json d;

        // Model loading defaults (always available, from compiled constants)
        d["default_max_llm_len"]   = static_cast<uint32_t>(COSYVOICE_DEFAULT_LLM_MAX_SEQ_LEN);
        d["default_k_cache_type"]  = "q8_0";
        d["default_v_cache_type"]  = "q4_0";
        d["default_buffer_policy"] = "balanced";
        d["default_backend"]       = "auto";
        d["default_n_threads"]     = 0;

        // Generation defaults (model-dependent or sensible fallbacks)
        if (!runtime.model_slots.empty())
        {
            d["temperature"]    = runtime.default_generation_config.temperature;
            d["top_k"]          = runtime.default_generation_config.sampling.top_k;
            d["top_p"]          = runtime.default_generation_config.sampling.top_p;
            d["win_size"]       = runtime.default_generation_config.sampling.win_size;
            d["tau_r"]          = runtime.default_generation_config.sampling.tau_r;
        }
        else
        {
            d["temperature"]    = 0.8;
            d["top_k"]          = 10;
            d["top_p"]          = 0.8;
            d["win_size"]       = 100;
            d["tau_r"]          = 0.0;
        }
    #ifndef COSYVOICE_NO_ICU
        d["text_normalization"] = runtime.text_normalization_enabled;
    #else
        d["text_normalization"] = false;
    #endif
        d["split_text"]         = runtime.split_text_enabled;
        d["fast_split"]         = runtime.fast_split_text_enabled;
        d["fade_in"]            = runtime.fade_in_enabled;
        res.status = 200;
        res.set_content(d.dump(), "application/json");
    });

    // ---- Error handler (404) ----
    server.set_error_handler([&runtime](const Request& req, Response& res)
    {
        (void)runtime;
        if (!res.body.empty())
            return;

        nlohmann::json err = {
            {"error", "Not found"},
            {"path", req.path}
        };
        res.status = 404;
        res.set_content(err.dump(), "application/json");
    });

    // ---- Startup banner ----
    log_message(runtime.log_level, server_log_level::concise, "WEBUI",
        "CosyVoice WebUI starting: host=%s, port=%u",
        runtime.host.c_str(), static_cast<unsigned>(runtime.port));

    print_info_log(runtime.log_level, "  host               : %s\n", runtime.host.c_str());
    print_info_log(runtime.log_level, "  port               : %u\n", static_cast<unsigned>(runtime.port));
    print_info_log(runtime.log_level, "  model loaded       : %s\n",
        runtime.model_slots.empty() ? "no" : runtime.served_model_name.c_str());
    print_info_log(runtime.log_level, "  api_key_required   : %s\n",
        runtime.api_key.empty() ? "no" : "yes");
    if (runtime.sample_rate > 0)
        print_info_log(runtime.log_level, "  sample_rate        : %u\n", runtime.sample_rate);
    print_info_log(runtime.log_level, "  speakers           : %zu\n", runtime.voice_names.size());
#if !defined(COSYVOICE_NO_FRONTEND)
    print_info_log(runtime.log_level, "  frontend_available : %s\n",
        runtime.frontend_ctx ? "yes" : "no");
#endif

    if (!server.listen(runtime.host.c_str(), runtime.port))
    {
        print_error_log("Error: failed to start WebUI server at %s:%u.\n",
            runtime.host.c_str(),
            static_cast<unsigned>(runtime.port));
        return 1;
    }

    return 0;
}
