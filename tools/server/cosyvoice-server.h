#pragma once

#include "tool_common_cosyvoice.h"

#include <cstdint>
#include <atomic>
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
    cosyvoice_llm_kv_cache_type_t requested_llm_kv_cache_type = static_cast<cosyvoice_llm_kv_cache_type_t>(0);
    cosyvoice_llm_kv_cache_type_t actual_llm_kv_cache_type = static_cast<cosyvoice_llm_kv_cache_type_t>(0);
#ifndef COSYVOICE_NO_ICU
    bool text_normalization_enabled = true;
#endif
    bool split_text_enabled = true;
    bool fast_split_text_enabled = true;
    bool fade_in_enabled = true;
    server_log_level log_level = server_log_level::concise;

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
    std::atomic_uint32_t thread_slot_counter{0};
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

inline uint32_t get_or_assign_thread_slot(server_runtime& runtime)
{
    thread_local uint32_t slot = UINT32_MAX;
    if (slot == UINT32_MAX)
        slot = runtime.thread_slot_counter.fetch_add(1, std::memory_order_relaxed);
    return slot;
}

int cosyvoice_server_backend_run(server_runtime& runtime);
int cosyvoice_server_webui_run(server_runtime& runtime);
