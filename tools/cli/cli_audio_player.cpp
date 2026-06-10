#include "cli_audio_player.h"

#ifndef COSYVOICE_STATIC
    #define MINIAUDIO_IMPLEMENTATION
    #define MA_API static
#endif

#define MA_NO_RESOURCE_MANAGER
#define MA_NO_NODE_GRAPH
#define MA_NO_ENGINE
#define MA_NO_GENERATION
#define MA_NO_DECODING
#define MA_NO_ENCODING
#define MA_NO_VORBIS
#define MA_NO_MP3
#define MA_NO_FLAC
#define MA_NO_WAV
#include "miniaudio.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <thread>

struct playback_state
{
    const float* data = nullptr;
    uint32_t length = 0;
    std::atomic<uint32_t> cursor{0};
};

static void data_callback(ma_device* device, void* output, const void* input, ma_uint32 frame_count)
{
    auto* state = reinterpret_cast<playback_state*>(device->pUserData);
    float* out = reinterpret_cast<float*>(output);
    if (!state || !out)
        return;

    const uint32_t cursor = state->cursor.load(std::memory_order_relaxed);
    const uint32_t remaining = cursor < state->length ? (state->length - cursor) : 0;
    const uint32_t frames_to_copy = std::min<uint32_t>(frame_count, remaining);
    if (frames_to_copy > 0)
    {
        std::memcpy(out, state->data + cursor, frames_to_copy * sizeof(float));
        state->cursor.store(cursor + frames_to_copy, std::memory_order_relaxed);
    }
    if (frames_to_copy < frame_count)
        std::fill(out + frames_to_copy, out + frame_count, 0.0f);
}

bool cli_audio_play_pcm_blocking(const float* data, uint32_t length, uint32_t sample_rate, std::string* error)
{
    if (!data || length == 0 || sample_rate == 0)
    {
        if (error)
            *error = "Audio playback input is empty.";
        return false;
    }

    playback_state state;
    state.data = data;
    state.length = length;

    ma_device_config config = ma_device_config_init(ma_device_type_playback);
    config.playback.format = ma_format_f32;
    config.playback.channels = 1;
    config.sampleRate = sample_rate;
    config.dataCallback = data_callback;
    config.pUserData = &state;

    ma_device device;
    if (ma_device_init(nullptr, &config, &device) != MA_SUCCESS)
    {
        if (error)
            *error = "Failed to initialize audio device.";
        return false;
    }

    if (ma_device_start(&device) != MA_SUCCESS)
    {
        ma_device_uninit(&device);
        if (error)
            *error = "Failed to start audio playback.";
        return false;
    }

    const auto sleep_chunk = std::chrono::milliseconds(20);
    bool interrupted = false;
    while (state.cursor.load(std::memory_order_relaxed) < state.length)
    {
        interrupted = is_playback_interrupted();
        if (interrupted)
            break;
        std::this_thread::sleep_for(sleep_chunk);
    }

    if (!interrupted && !is_playback_interrupted())
        std::this_thread::sleep_for(std::chrono::milliseconds(50));

    ma_device_stop(&device);
    ma_device_uninit(&device);
    return !interrupted;
}
