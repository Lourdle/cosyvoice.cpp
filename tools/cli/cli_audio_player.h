#pragma once

#include <cstdint>
#include <string>
#include <atomic>
#include <functional>

inline std::atomic_bool g_playback_interrupted;

inline void interrupt_playback()
{
    g_playback_interrupted.store(true, std::memory_order_relaxed);
}

inline bool is_playback_interrupted()
{
    return g_playback_interrupted.load(std::memory_order_relaxed);
}

using streaming_callback_t = std::function<bool(float*, uint32_t)>;
bool cli_audio_play_pcm_streaming(uint32_t sample_rate, std::string* error, streaming_callback_t callback);
bool cli_audio_play_pcm_blocking(const float* data, uint32_t length, uint32_t sample_rate, std::string* error);
