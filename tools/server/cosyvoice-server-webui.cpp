#include "cosyvoice-server-webui.h"
#include "resource.h"
#include "tool_common_cosyvoice.h"
#include "common.h"

#include "httplib.h"
#include "nlohmann/json.hpp"

#include <ggml-backend.h>

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
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

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

// ---------------------------------------------------------------------------
// Logging helpers
// ---------------------------------------------------------------------------

static void print_info_log(server_log_level level, const char* format, ...)
{
    if (level == server_log_level::quiet)
        return;
    va_list args;
    va_start(args, format);
    vprintf(format, args);
    va_end(args);
}

static void print_error_log(const char* format, ...)
{
    va_list args;
    va_start(args, format);
    vfprintf(stderr, format, args);
    va_end(args);
}

// Activity log: always printed (unless quiet), timestamped, prefixed
static void log_activity(const server_runtime&, const char* format, ...)
{
    va_list args;
    va_start(args, format);
    printf("[cosyvoice] ");
    vprintf(format, args);
    printf("\n");
    va_end(args);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static std::string join_strings(const std::vector<std::string>& items, const char* sep)
{
    std::string r;
    for (size_t i = 0; i < items.size(); ++i)
    {
        if (i > 0) r += sep;
        r += items[i];
    }
    return r;
}

// Build a minimal WAV payload from float PCM by converting to 16-bit.
static bool build_wav_payload(const cosyvoice_generated_speech& speech, uint32_t sample_rate, std::string* out)
{
    if (!speech.data || speech.length == 0)
        return false;

    const uint32_t num_samples = speech.length;
    const uint32_t data_bytes  = num_samples * sizeof(int16_t);
    const uint32_t file_size   = 44 + data_bytes;

    out->resize(file_size);
    auto* buf = out->data();

    // Convert float PCM to 16-bit signed PCM inline in the buffer
    auto* pcm16 = reinterpret_cast<int16_t*>(buf + 44);
    for (uint32_t i = 0; i < num_samples; ++i)
    {
        float s = speech.data[i];
        if (s < -1.0f) s = -1.0f;
        if (s >  1.0f) s =  1.0f;
        pcm16[i] = static_cast<int16_t>(s * 32767.0f);
    }

    // RIFF header
    memcpy(buf, "RIFF", 4);
    const uint32_t riff_size = file_size - 8;
    memcpy(buf + 4, &riff_size, 4);
    memcpy(buf + 8, "WAVE", 4);

    // fmt chunk (PCM, mono, 16-bit)
    memcpy(buf + 12, "fmt ", 4);
    const uint32_t fmt_chunk_size = 16;
    memcpy(buf + 16, &fmt_chunk_size, 4);
    const uint16_t audio_format = 1;   // PCM
    memcpy(buf + 20, &audio_format, 2);
    const uint16_t num_channels = 1;
    memcpy(buf + 22, &num_channels, 2);
    memcpy(buf + 24, &sample_rate, 4);
    const uint32_t byte_rate = sample_rate * 2;
    memcpy(buf + 28, &byte_rate, 4);
    const uint16_t block_align = 2;
    memcpy(buf + 32, &block_align, 2);
    const uint16_t bits_per_sample = 16;
    memcpy(buf + 34, &bits_per_sample, 2);

    // data chunk
    memcpy(buf + 36, "data", 4);
    memcpy(buf + 40, &data_bytes, 4);

    return true;
}

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
    size_t audio_size,
    const char* temp_ext)
{
    if (runtime.model_slots.empty())
        return false;

    // Write audio data to a temp file so the audio loader can read it.
    // Generate a unique temp file path (cross-platform).
#ifdef _WIN32
    char temp_dir[MAX_PATH];
    DWORD ret = GetTempPathA(MAX_PATH, temp_dir);
    if (ret == 0 || ret > MAX_PATH) return false;

    char temp_file[MAX_PATH];
    if (!GetTempFileNameA(temp_dir, "csw", 0, temp_file)) return false;

    char temp_path[MAX_PATH];
    strncpy(temp_path, temp_file, MAX_PATH - 1);
    temp_path[MAX_PATH - 1] = '\0';
#else
    char temp_path[] = "/tmp/cosyvoice_audio_XXXXXX.wav";
    int fd = mkstemp(temp_path);
    if (fd == -1) return false;
    close(fd);
    // mkstemp modifies in-place; strip the .wav suffix we'll re-add below
    temp_path[strlen(temp_path) - 4] = '\0';
#endif

    // Append the correct extension
    std::string final_path = std::string(temp_path) + "." + (temp_ext && *temp_ext ? temp_ext : "wav");

    // Write file
    {
        auto f = fopen(final_path.c_str(), "wb");
        if (!f) { (void)remove(final_path.c_str()); return false; }
        (void)fwrite(audio_data, 1, audio_size, f);
        fclose(f);
    }

    // Remove the original temp file (without extension) — Windows creates it as empty
    (void)remove(temp_path);

    // Load audio
    float* audio_float = nullptr;
    uint32_t audio_length = 0;
    uint32_t audio_sample_rate = 0;
    bool ok = cosyvoice_audio_load_from_file(
        final_path.c_str(),
        &audio_float,
        &audio_length,
        &audio_sample_rate);
    if (!ok || !audio_float)
    {
        (void)remove(final_path.c_str());
        return false;
    }

    audio_buffer_handle audio_buf(audio_float);

    // Use pre-loaded frontend if available, otherwise load on-the-fly
    cosyvoice_frontend_context_t frontend = runtime.frontend_ctx.get();
    bool own_frontend = false;
    if (!frontend)
    {
        frontend = cosyvoice_frontend_load_from_files(
            runtime.speech_tokenizer.c_str(),
            runtime.campplus.c_str());
        if (!frontend)
        {
            (void)remove(final_path.c_str());
            return false;
        }
        own_frontend = true;
    }

    // Extract prompt speech
    cosyvoice_prompt_speech_handle ps(
        cosyvoice_frontend_prompt_speech(
            frontend,
            audio_buf.get(),
            audio_length,
            audio_sample_rate,
            prompt_text.c_str()));

    if (own_frontend)
        cosyvoice_frontend_free(frontend);
    (void)remove(final_path.c_str());

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

// Build raw 16-bit PCM from float PCM (no WAV header).
static bool build_pcm_raw(const cosyvoice_generated_speech& speech, std::string* out)
{
    if (!speech.data || speech.length == 0)
        return false;

    const uint32_t data_bytes = speech.length * sizeof(int16_t);
    out->resize(data_bytes);
    auto* pcm16 = reinterpret_cast<int16_t*>(out->data());
    for (uint32_t i = 0; i < speech.length; ++i)
    {
        float s = speech.data[i];
        if (s < -1.0f) s = -1.0f;
        if (s >  1.0f) s =  1.0f;
        pcm16[i] = static_cast<int16_t>(s * 32767.0f);
    }
    return true;
}

// ---------------------------------------------------------------------------
// Route handlers — WebUI mode
// ---------------------------------------------------------------------------

int cosyvoice_server_webui_run(server_runtime& runtime)
{
    httplib::Server server;
    // Use default task queue (no explicit override)

    server.set_exception_handler([](const httplib::Request&, httplib::Response& res, std::exception_ptr ep)
    {
        (void)ep;
        res.status = 500;

        static const char json[] = R"({"error":"Internal server error"})";
        res.set_content(json, "application/json");
    });

    // ---- GET / — serve the WebUI HTML ----
    server.Get("/", [&runtime](const httplib::Request&, httplib::Response& res)
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

        const auto pos = html.find("</body>");
        if (pos != std::string::npos)
            html.insert(pos, script);
        else
            html.append(script);

        res.set_content(std::move(html), "text/html");
    });

    // ---- Simple ping endpoint (always works) ----
    server.Get("/ping", [](const httplib::Request&, httplib::Response& res)
    {
        res.set_content("pong", "text/plain");
    });

    // ---- GET /backends — list available GGML backends ----
    server.Get("/backends", [](const httplib::Request&, httplib::Response& res)
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
    server.Get("/formats", [](const httplib::Request&, httplib::Response& res)
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
    server.Get("/status", [&runtime](const httplib::Request&, httplib::Response& res)
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
    server.Get("/speaker", [&runtime](const httplib::Request&, httplib::Response& res)
    {
        nlohmann::json payload = runtime.voice_names;
        res.status = 200;
        res.set_content(payload.dump(), "application/json");
    });

    // ---- POST /speaker — register a new speaker ----
    server.Post("/speaker", [&runtime](const httplib::Request& req, httplib::Response& res)
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
                    audio_formdata.content.data(), audio_formdata.content.size(),
                    audio_formdata.filename.c_str()))
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
        log_activity(runtime, "Speaker registered: %s (type=%s)", name.c_str(), type.c_str());
    });

    // ---- DELETE /speaker/<name> — remove a speaker ----
    server.Delete(R"(/speaker/(.*))", [&runtime](const httplib::Request& req, httplib::Response& res)
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
        log_activity(runtime, "Speaker removed: %s", name.c_str());
    });

    // ---- GET /speaker/<name>/download.gguf — download prompt_speech as GGUF ----
    server.Get(R"(/speaker/(.*)/download\.gguf)", [&runtime](const httplib::Request& req, httplib::Response& res)
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

        // Save prompt_speech to a temp file, read it back, send as download
#ifdef _WIN32
        char temp_dir[MAX_PATH];
        DWORD ret = GetTempPathA(MAX_PATH, temp_dir);
        if (ret == 0 || ret > MAX_PATH) { res.status = 500; return; }

        char temp_path[MAX_PATH];
        if (!GetTempFileNameA(temp_dir, "csd", 0, temp_path)) { res.status = 500; return; }
        std::string gguf_path = temp_path;
#else
        char temp_path[] = "/tmp/cosyvoice_dl_XXXXXX.gguf";
        int fd = mkstemp(temp_path);
        if (fd == -1) { res.status = 500; return; }
        close(fd);
        std::string gguf_path = temp_path;
#endif

        if (!cosyvoice_prompt_speech_save_to_file(it->second.prompt_speech.get(), gguf_path.c_str()))
        {
            (void)remove(gguf_path.c_str());
            res.status = 500;
            nlohmann::json err = {{"error", "Failed to serialize prompt speech"}};
            res.set_content(err.dump(), "application/json");
            return;
        }

        // Read file
        FILE* f = fopen(gguf_path.c_str(), "rb");
        if (!f)
        {
            (void)remove(gguf_path.c_str());
            res.status = 500;
            return;
        }
        fseek(f, 0, SEEK_END);
        long fsize = ftell(f);
        rewind(f);
        std::string payload(static_cast<size_t>(fsize), '\0');
        (void)fread(&payload[0], 1, static_cast<size_t>(fsize), f);
        fclose(f);
        (void)remove(gguf_path.c_str());

        res.status = 200;
        res.set_header("Content-Disposition", "attachment; filename=\"" + name + ".gguf\"");
        res.set_content(std::move(payload), "application/octet-stream");
    });
    server.Post("/tts", [&runtime](const httplib::Request& req, httplib::Response& res)
    {
        // Parse JSON body
        nlohmann::json body;
        try { body = nlohmann::json::parse(req.body); }
        catch (...)
        {
            res.status = 400;
            nlohmann::json err = {{"error", "Invalid JSON body"}};
            res.set_content(err.dump(), "application/json");
            return;
        }

        // Validate
        if (runtime.model_slots.empty())
        {
            res.status = 503;
            nlohmann::json err = {{"error", "No model loaded"}};
            res.set_content(err.dump(), "application/json");
            return;
        }

        const std::string text  = body.value("input", body.value("text", ""));
        if (text.empty())
        {
            res.status = 400;
            nlohmann::json err = {{"error", "Missing 'text' or 'input' field"}};
            res.set_content(err.dump(), "application/json");
            return;
        }

        const std::string voice = body.value("voice", "");
        if (voice.empty())
        {
            res.status = 400;
            nlohmann::json err = {{"error", "Missing 'voice' field"}};
            res.set_content(err.dump(), "application/json");
            return;
        }

        auto voice_it = runtime.voices.find(voice);
        if (voice_it == runtime.voices.end())
        {
            res.status = 400;
            nlohmann::json err = {{"error", "Unknown voice '" + voice + "'. Available: " + join_strings(runtime.voice_names, ", ")}};
            res.set_content(err.dump(), "application/json");
            return;
        }

        // Get TTS context (WebUI mode: only slot 0)
        auto& sessions = runtime.voice_sessions[0];
        cosyvoice_tts_context_t tts_ctx = nullptr;
        for (auto& s : sessions)
        {
            if (s.first == voice)
            {
                tts_ctx = s.second.get();
                break;
            }
        }
        if (!tts_ctx)
        {
            res.status = 500;
            nlohmann::json err = {{"error", "TTS context not initialized for voice '" + voice + "'"}};
            res.set_content(err.dump(), "application/json");
            return;
        }

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

        // Apply generation overrides (temperature, sampling, seed)
        {
            auto model_ctx = runtime.model_slots[0].get();

            cosyvoice_generation_config_t cfg = runtime.default_generation_config;
            if (body.contains("temperature"))
                cfg.temperature = body["temperature"].get<float>();
            if (body.contains("top_k"))
                cfg.sampling.top_k = body["top_k"].get<int>();
            if (body.contains("top_p"))
                cfg.sampling.top_p = body["top_p"].get<float>();
            if (body.contains("win_size"))
                cfg.sampling.win_size = body["win_size"].get<int>();
            if (body.contains("tau_r"))
                cfg.sampling.tau_r = body["tau_r"].get<float>();

            if (!cosyvoice_set_generation_config(model_ctx, &cfg))
            {
                res.status = 400;
                nlohmann::json err = {{"error", "Invalid generation parameter values."}};
                res.set_content(err.dump(), "application/json");
                return;
            }

            // Seed
            uint32_t effective_seed;
            if (body.contains("seed"))
                effective_seed = body["seed"].get<uint32_t>();
            else if (runtime.has_seed)
                effective_seed = runtime.seed;
            else
                effective_seed = cosyvoice_generate_random_seed() ^ static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&req));

            if (!cosyvoice_set_sampler_seed(model_ctx, effective_seed))
            {
                res.status = 400;
                nlohmann::json err = {{"error", "Failed to apply seed."}};
                res.set_content(err.dump(), "application/json");
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
            return;
        }

        // Encode output
        std::string audio_payload;
        const std::string format = body.value("format", "wav");
        bool encode_ok = false;

#ifndef COSYVOICE_NO_AUDIO
        if (format != "wav" && runtime.audio_encoder)
        {
            // Use audio encoder for compressed formats
            cosyvoice_audio_encoding_format_t af = COSYVOICE_AUDIO_ENCODING_FORMAT_WAV;
            if (format == "mp3")       af = COSYVOICE_AUDIO_ENCODING_FORMAT_MP3;
            else if (format == "opus") af = COSYVOICE_AUDIO_ENCODING_FORMAT_OPUS;
            else if (format == "flac") af = COSYVOICE_AUDIO_ENCODING_FORMAT_FLAC;
            else if (format == "aac")  af = COSYVOICE_AUDIO_ENCODING_FORMAT_AAC;
            else if (format == "m4a")  af = COSYVOICE_AUDIO_ENCODING_FORMAT_M4A;

            if (cosyvoice_audio_encoder_encode(runtime.audio_encoder.get(), generated.data, generated.length, af))
            {
                const uint8_t* enc_data = nullptr;
                uint32_t enc_length = 0;
                cosyvoice_audio_encoder_get_encoded_data(runtime.audio_encoder.get(), &enc_data, &enc_length);
                if (enc_data && enc_length > 0)
                {
                    audio_payload.assign(reinterpret_cast<const char*>(enc_data), enc_length);
                    encode_ok = true;
                }
            }
        }
        else
#endif
        {
            // Fall back to WAV or raw PCM
            if (format == "pcm")
                encode_ok = build_pcm_raw(generated, &audio_payload);
            else
                encode_ok = build_wav_payload(generated, runtime.sample_rate, &audio_payload);
        }

        if (!encode_ok)
        {
            res.status = 500;
            nlohmann::json err = {{"error", "Audio encoding failed"}};
            res.set_content(err.dump(), "application/json");
            return;
        }

        log_activity(runtime, "TTS generated: voice=%s, text_len=%zu, format=%s, samples=%u",
            voice.c_str(), text.size(), format.c_str(), generated.length);

        // Content-Type mapping
        std::string content_type;
        if (format == "mp3")       content_type = "audio/mpeg";
        else if (format == "opus") content_type = "audio/opus";
        else if (format == "flac") content_type = "audio/flac";
        else if (format == "pcm")  content_type = "audio/pcm";
        else                       content_type = "audio/wav";

        res.status = 200;
        res.set_content(std::move(audio_payload), content_type);
    });

    // ---- GET /frontend/model — return frontend model paths ----
    server.Get("/frontend/model", [&runtime](const httplib::Request&, httplib::Response& res)
    {
        nlohmann::json payload;
        payload["speech_tokenizer"] = runtime.speech_tokenizer;
        payload["campplus"]         = runtime.campplus;
        res.status = 200;
        res.set_content(payload.dump(), "application/json");
    });

    // ---- PUT /frontend/model — update frontend model paths ----
    server.Put("/frontend/model", [&runtime](const httplib::Request& req, httplib::Response& res)
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
    server.Post("/frontend/model/load", [&runtime](const httplib::Request& req, httplib::Response& res)
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

        log_activity(runtime, "Frontend ONNX models loaded");

        nlohmann::json ok2 = {{"success", true}};
        res.status = 200;
        res.set_content(ok2.dump(), "application/json");
    });
#else
    server.Post("/frontend/model/load", [](const httplib::Request&, httplib::Response& res)
    {
        res.status = 501;
        nlohmann::json err = {{"error", "Frontend support is not available in this build (COSYVOICE_NO_FRONTEND)."}};
        res.set_content(err.dump(), "application/json");
    });
#endif

    // ---- POST /frontend/model/unload — unload ONNX frontend models ----
#if !defined(COSYVOICE_NO_FRONTEND)
    server.Post("/frontend/model/unload", [&runtime](const httplib::Request&, httplib::Response& res)
    {
        if (!runtime.frontend_ctx)
        {
            res.status = 409;
            nlohmann::json err2 = {{"error", "No frontend model is currently loaded."}};
            res.set_content(err2.dump(), "application/json");
            return;
        }

        runtime.frontend_ctx.reset();
        log_activity(runtime, "Frontend ONNX models unloaded");

        nlohmann::json ok2 = {{"success", true}};
        res.status = 200;
        res.set_content(ok2.dump(), "application/json");
    });
#else
    server.Post("/frontend/model/unload", [](const httplib::Request&, httplib::Response& res)
    {
        res.status = 501;
        nlohmann::json err = {{"error", "Frontend support is not available in this build (COSYVOICE_NO_FRONTEND)."}};
        res.set_content(err.dump(), "application/json");
    });
#endif

    // ---- POST /model/load — dynamically load a GGUF model ----
    server.Post("/model/load", [&runtime](const httplib::Request& req, httplib::Response& res)
    {
        // Must not already have a model loaded
        if (!runtime.model_slots.empty())
        {
            res.status = 409;
            nlohmann::json err = {{"error", "A model is already loaded. Unload it first via POST /model/unload."}};
            res.set_content(err.dump(), "application/json");
            return;
        }

        nlohmann::json body;
        try { body = nlohmann::json::parse(req.body); }
        catch (...)
        {
            res.status = 400;
            nlohmann::json err = {{"error", "Invalid JSON body"}};
            res.set_content(err.dump(), "application/json");
            return;
        }

        std::string model_path = body.value("model", "");
        if (model_path.empty())
        {
            res.status = 400;
            nlohmann::json err = {{"error", "Missing 'model' field"}};
            res.set_content(err.dump(), "application/json");
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
            return;
        }

        runtime.model_slots.emplace_back(loaded_ctx);

        // Log model details
        {
            cosyvoice_context_params_t p;
            cosyvoice_get_context_params(loaded_ctx, &p);
            log_activity(runtime, "Model loaded: %s (arch=%s, backend=%s, threads=%u, kv_cache_type=%s, buffer=%s, max_llm_len=%u)",
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
    });

    // ---- POST /model/unload — unload the current model ----
    server.Post("/model/unload", [&runtime](const httplib::Request&, httplib::Response& res)
    {
        if (runtime.model_slots.empty())
        {
            res.status = 409;
            nlohmann::json err = {{"error", "No model is currently loaded."}};
            res.set_content(err.dump(), "application/json");
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

        log_activity(runtime, "Model unloaded");

        // Re-initialize empty voice sessions vector
        runtime.voice_sessions.reserve(1);
        runtime.voice_sessions.emplace_back();

        nlohmann::json ok = {{"success", true}};
        res.status = 200;
        res.set_content(ok.dump(), "application/json");
    });

    // ---- GET /model/defaults — return defaults for all configurable parameters ----
    // Works even without a loaded model (sensible fallbacks for generation params).
    server.Get("/model/defaults", [&runtime](const httplib::Request&, httplib::Response& res)
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
    server.set_error_handler([&runtime](const httplib::Request& req, httplib::Response& res)
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

    // ---- Startup banner (minimal) ----
    print_info_log(runtime.log_level, "cosyvoice-server WebUI listening on %s:%u\n",
        runtime.host.c_str(), static_cast<unsigned>(runtime.port));

    if (!server.listen(runtime.host.c_str(), runtime.port))
    {
        print_error_log("Error: failed to start WebUI server at %s:%u.\n",
            runtime.host.c_str(),
            static_cast<unsigned>(runtime.port));
        return 1;
    }

    return 0;
}
