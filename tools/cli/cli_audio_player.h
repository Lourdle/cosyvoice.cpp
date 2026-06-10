#pragma once

#include <cstdint>
#include <string>
#include <atomic>

inline std::atomic_bool g_playback_interrupted;

inline void interrupt_playback()
{
    g_playback_interrupted.store(true, std::memory_order_relaxed);
}

inline bool is_playback_interrupted()
{
    return g_playback_interrupted.load(std::memory_order_relaxed);
}

bool cli_audio_play_pcm_blocking(const float* data, uint32_t length, uint32_t sample_rate, std::string* error);
