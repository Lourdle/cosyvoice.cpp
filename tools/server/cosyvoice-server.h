#pragma once

#include "tool_common_cosyvoice.h"

#include <cstdint>
#include <mutex>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

enum class server_log_level
{
    quiet,
    concise,
    verbose
};

struct voice_runtime
{
    std::string name;
    cosyvoice_prompt_speech_handle prompt_speech;
    cosyvoice_prompt_handle prompt;
};

struct server_runtime
{
    std::string model;
    std::string served_model_name;
    std::string host = "127.0.0.1";
    std::string api_key;
    uint16_t port = 8080;
    bool has_seed = false;
    bool webui_enabled = false;
    uint32_t seed = 0;
    uint32_t concurrency = 1;
    cosyvoice_inference_buffer_policy_t inference_buffer_policy = COSYVOICE_INFERENCE_BUFFER_POLICY_BALANCED;
    bool has_llm_kv_cache_override = false;
    cosyvoice_kv_cache_type_t requested_llm_kv_cache_type = static_cast<cosyvoice_kv_cache_type_t>(0);
    cosyvoice_kv_cache_type_t actual_llm_kv_cache_type = static_cast<cosyvoice_kv_cache_type_t>(0);
#ifndef COSYVOICE_NO_ICU
    bool text_normalization_enabled = true;
#endif
    bool split_text_enabled = true;
    bool fast_split_text_enabled = true;
    bool fade_in_enabled = true;
    server_log_level log_level = server_log_level::concise;

    bool stream = false;
    bool has_chunk_tokens = false;
    uint32_t chunk_tokens = 0;

    // Effective DiT KV cache params (populated after model load)
    uint32_t dit_kv_fixed_slots          = 0;
    uint32_t dit_kv_offloadable_slots    = 0;
    uint32_t dit_kv_cache_length         = 0;

    // Frontend model paths (ONNX, for feature extraction)
    std::string frontend_model;
    std::string speech_tokenizer;
    std::string campplus;

#ifndef COSYVOICE_NO_FRONTEND
    cosyvoice_frontend_handle frontend_ctx; ///< Pre-loaded ONNX frontend context (kept in memory across extractions)
#endif
    uint32_t sample_rate = 0;
    cosyvoice_generation_config_t default_generation_config = {};

#ifndef COSYVOICE_NO_AUDIO
    cosyvoice_audio_encoder_handle audio_encoder;
#endif

    std::vector<std::string> voice_names;
    std::unordered_map<std::string, voice_runtime> voices;
    std::vector<cosyvoice_context_handle> model_slots;
    std::vector<std::vector<std::pair<std::string, cosyvoice_tts_context_handle>>> voice_sessions;

    std::mt19937 seed_rng;

    // Concurrency slot management (request-scoped, not thread-local)
    std::mutex              slot_mutex;
    std::vector<bool>       slot_in_use;
};

inline cosyvoice_context_t get_slot_model_context(server_runtime& runtime, uint32_t slot)
{
    return runtime.model_slots[slot].get();
}

inline cosyvoice_tts_context_t get_slot_voice_session(server_runtime& runtime, uint32_t slot, const std::string& voice)
{
    auto& sessions = runtime.voice_sessions[slot];
    for (auto& item : sessions)
        if (item.first == voice)
            return item.second.get();
    return nullptr;
}

// Acquire a concurrency slot for this request. Returns UINT32_MAX if all
// slots are in use (caller should return 503 / overloaded).
inline uint32_t acquire_thread_slot(server_runtime& runtime)
{
    std::lock_guard<std::mutex> lock(runtime.slot_mutex);
    for (uint32_t i = 0; i < runtime.concurrency; ++i)
        if (!runtime.slot_in_use[i])
        {
            runtime.slot_in_use[i] = true;
            return i;
        }
    return UINT32_MAX;
}

// Release a concurrency slot previously acquired via acquire_thread_slot.
inline void release_thread_slot(server_runtime& runtime, uint32_t slot)
{
    if (slot >= runtime.concurrency)
        return;
    std::lock_guard<std::mutex> lock(runtime.slot_mutex);
    runtime.slot_in_use[slot] = false;
}

inline uint32_t get_or_assign_thread_slot(server_runtime& runtime)
{
    return acquire_thread_slot(runtime);
}

int cosyvoice_server_backend_run(server_runtime& runtime);
#ifndef COSYVOICE_SERVER_NO_WEBUI
int cosyvoice_server_webui_run(server_runtime& runtime);
#endif
