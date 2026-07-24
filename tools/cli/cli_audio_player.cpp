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
#define NOMINMAX
#include "miniaudio.h"

#include <algorithm>
#include <condition_variable>
#include <mutex>

#include <cstring>

struct playback_context
{
    const streaming_callback_t& procedure;
    std::mutex mutex;
    std::condition_variable cv;
    bool done = false;

    void call_procedure(float* output, uint32_t frame_count)
    {
        if (!done && !procedure(output, frame_count)
            || is_playback_interrupted())
        {
            {
                std::unique_lock<std::mutex> lock(mutex);
                done = true;
            }
            cv.notify_one();
        }
    }

    void wait_for_completion()
    {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [this] { return done; });
    }
};

static void streaming_callback(ma_device* device, void* output, const void* input, ma_uint32 frame_count)
{
    reinterpret_cast<playback_context*>(device->pUserData)->call_procedure(reinterpret_cast<float*>(output), frame_count);
}

bool cli_audio_play_pcm_streaming(uint32_t sample_rate, std::string* error, streaming_callback_t callback)
{
    if (sample_rate == 0)
    {
        if (error)
            *error = "Invalid sample rate.";
        return false;
    }

    playback_context context{ callback };
    ma_device_config config = ma_device_config_init(ma_device_type_playback);
    config.playback.format = ma_format_f32;
    config.playback.channels = 1;
    config.sampleRate = sample_rate;
    config.dataCallback = streaming_callback;
    config.pUserData = &context;

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

    context.wait_for_completion();
    ma_device_stop(&device);
    ma_device_uninit(&device);
    return !is_playback_interrupted();
}

bool cli_audio_play_pcm_blocking(const float* data, uint32_t length, uint32_t sample_rate, std::string* error)
{
    if (!data || length == 0)
    {
        if (error)
            *error = "Audio playback input is empty.";
        return false;
    }

    uint32_t offset = 0;
    return cli_audio_play_pcm_streaming(sample_rate, error, [data, length, &offset](float* output, uint32_t frame_count)
        {
            uint32_t remaining_frames = length - offset;
            if (remaining_frames == 0) return false;
            uint32_t frames_to_copy = std::min(frame_count, remaining_frames);
            std::memcpy(output, data + offset, frames_to_copy * sizeof(float));
            if (frame_count > frames_to_copy)
                std::memset(output + frames_to_copy, 0, (frame_count - frames_to_copy) * sizeof(float));
            offset += frames_to_copy;
            return true;
        });
}
