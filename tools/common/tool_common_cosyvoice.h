#pragma once

#include "cosyvoice.h"
#include "cosyvoice-lowlevel.h"
#include "tool_common.h"

#ifndef COSYVOICE_NO_FRONTEND
    #include "cosyvoice-frontend.h"
#endif
#ifndef COSYVOICE_NO_AUDIO
    #include "cosyvoice-audio.h"
#endif

#include <memory>

template <typename T, auto deleter>
struct tool_deleter { void operator()(T* ptr) const noexcept { deleter(ptr); } };

using cosyvoice_context_handle = std::unique_ptr<cosyvoice_context, tool_deleter<cosyvoice_context, &cosyvoice_free>>;
#ifndef COSYVOICE_NO_FRONTEND
using cosyvoice_frontend_handle = std::unique_ptr<cosyvoice_frontend_context, tool_deleter<cosyvoice_frontend_context, &cosyvoice_frontend_free>>;
#endif
using cosyvoice_prompt_speech_handle = std::unique_ptr<cosyvoice_prompt_speech, tool_deleter<cosyvoice_prompt_speech, &cosyvoice_prompt_speech_free>>;
using cosyvoice_prompt_handle = std::unique_ptr<cosyvoice_prompt, tool_deleter<cosyvoice_prompt, &cosyvoice_prompt_free>>;
using cosyvoice_tts_context_handle = std::unique_ptr<cosyvoice_tts_context, tool_deleter<cosyvoice_tts_context, &cosyvoice_tts_context_free>>;
#ifndef COSYVOICE_NO_AUDIO
using audio_buffer_handle = std::unique_ptr<float, tool_deleter<float, &cosyvoice_audio_free>>;
using cosyvoice_audio_encoder_handle = std::unique_ptr<cosyvoice_audio_encoder, tool_deleter<cosyvoice_audio_encoder, &cosyvoice_audio_encoder_destroy>>;
#endif

inline bool parse_llm_kv_cache_type_arg(const std::string& value, cosyvoice_llm_kv_cache_type_t* result)
{
    // Check for separate K/V format: "k=<type>,v=<type>" or "k=<type>,v=<type>,fallback=<type>"
    auto k_pos = value.find("k=");
    auto v_pos = value.find("v=");
    if (k_pos != std::string::npos && v_pos != std::string::npos)
    {
        auto k_start = k_pos + 2;
        auto k_end = value.find_first_of(",", k_start);
        auto k_str = value.substr(k_start, k_end - k_start);
        auto v_start = v_pos + 2;
        auto v_end = value.find_first_of(",", v_start);
        auto v_str = value.substr(v_start, v_end - v_start);

        cosyvoice_llm_kv_cache_type_t k_type, v_type;

        if (!parse_llm_kv_cache_type_arg(k_str, &k_type) ||
            !parse_llm_kv_cache_type_arg(v_str, &v_type))
            return false;

        // Parse optional fallback
        auto f_pos = value.find("fallback=");
        cosyvoice_llm_kv_cache_type_t fallback_type;
        if (f_pos != std::string::npos)
        {
            auto f_start = f_pos + 9;
            auto f_end = value.find_first_of(",", f_start);
            auto f_str = value.substr(f_start, f_end - f_start);
            if (!parse_llm_kv_cache_type_arg(f_str, &fallback_type))
                return false;
        }
        else
        {
            fallback_type = v_type; // default fallback = V type
        }

        *result = COSYVOICE_MAKE_SEPARATE_KV_CACHE(k_type, v_type, fallback_type);
        return true;
    }

    // Unified type.
    const auto lowered = to_lower(value);
    if (lowered == "f32")
        *result = COSYVOICE_LLM_KV_CACHE_TYPE_F32;
    else if (lowered == "f16")
        *result = COSYVOICE_LLM_KV_CACHE_TYPE_F16;
    else if (lowered == "q8_0")
        *result = COSYVOICE_LLM_KV_CACHE_TYPE_Q8_0;
    else if (lowered == "q5_1")
        *result = COSYVOICE_LLM_KV_CACHE_TYPE_Q5_1;
    else if (lowered == "q5_0")
        *result = COSYVOICE_LLM_KV_CACHE_TYPE_Q5_0;
    else if (lowered == "q4_1")
        *result = COSYVOICE_LLM_KV_CACHE_TYPE_Q4_1;
    else if (lowered == "q4_0")
        *result = COSYVOICE_LLM_KV_CACHE_TYPE_Q4_0;
    else
        return false;
    return true;
}

inline bool parse_inference_buffer_policy_arg(const std::string& value, cosyvoice_inference_buffer_policy_t* result)
{
    const auto lowered = to_lower(value);
    if (lowered == "shared")
        *result = COSYVOICE_INFERENCE_BUFFER_POLICY_SHARED;
    else if (lowered == "balanced")
        *result = COSYVOICE_INFERENCE_BUFFER_POLICY_BALANCED;
    else if (lowered == "dedicated")
        *result = COSYVOICE_INFERENCE_BUFFER_POLICY_DEDICATED;
    else
        return false;
    return true;
}

inline const char* inference_buffer_policy_to_string(cosyvoice_inference_buffer_policy_t policy)
{
    switch (policy)
    {
    case COSYVOICE_INFERENCE_BUFFER_POLICY_SHARED:
        return "shared";
    case COSYVOICE_INFERENCE_BUFFER_POLICY_BALANCED:
        return "balanced";
    case COSYVOICE_INFERENCE_BUFFER_POLICY_DEDICATED:
        return "dedicated";
    default:
        return "unknown";
    }
}

inline std::string llm_kv_cache_type_to_string(cosyvoice_llm_kv_cache_type_t type)
{
    if (COSYVOICE_IS_SEPARATE_KV_CACHE(type))
    {
        auto fallback = COSYVOICE_KV_CACHE_FALLBACK(type);
        auto result = "k=" + std::string(llm_kv_cache_type_to_string(COSYVOICE_K_CACHE_TYPE(type)))
            + ",v=" + llm_kv_cache_type_to_string(COSYVOICE_V_CACHE_TYPE(type));
        if (fallback != COSYVOICE_V_CACHE_TYPE(type))
            result += ",fallback=" + llm_kv_cache_type_to_string(fallback);
        return result;
    }

    switch (type)
    {
    case COSYVOICE_LLM_KV_CACHE_TYPE_F32:
        return "f32";
    case COSYVOICE_LLM_KV_CACHE_TYPE_F16:
        return "f16";
    case COSYVOICE_LLM_KV_CACHE_TYPE_Q8_0:
        return "q8_0";
    case COSYVOICE_LLM_KV_CACHE_TYPE_Q5_1:
        return "q5_1";
    case COSYVOICE_LLM_KV_CACHE_TYPE_Q5_0:
        return "q5_0";
    case COSYVOICE_LLM_KV_CACHE_TYPE_Q4_1:
        return "q4_1";
    case COSYVOICE_LLM_KV_CACHE_TYPE_Q4_0:
        return "q4_0";
    default:
        return "unknown";
    }
}
