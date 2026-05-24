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
    cosyvoice_tts_context_handle tts_context;
};

struct server_runtime
{
    std::string model;
    std::string served_model_name;
    std::string host = "127.0.0.1";
    std::string api_key;
    uint16_t port = 8080;
    bool has_seed = false;
    uint32_t seed = 0;
    cosyvoice_inference_buffer_policy_t inference_buffer_policy = COSYVOICE_INFERENCE_BUFFER_POLICY_BALANCED;
#ifndef COSYVOICE_NO_ICU
    bool text_normalization_enabled = true;
#endif
    bool split_text_enabled = true;
    bool fast_split_text_enabled = true;
    bool fade_in_enabled = true;
    server_log_level log_level = server_log_level::concise;
    cosyvoice_context_handle model_context;
    uint32_t sample_rate = 0;
    cosyvoice_generation_config_t default_generation_config = {};

#ifndef COSYVOICE_NO_AUDIO
    cosyvoice_audio_encoder_handle audio_encoder;
#endif

    std::vector<std::string> voice_names;
    std::unordered_map<std::string, voice_runtime> voices;

    std::mt19937 seed_rng;
    std::mutex infer_mutex;
};

int cosyvoice_server_backend_run(server_runtime& runtime);
