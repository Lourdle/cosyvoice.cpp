#include "cosyvoice-internal.h"
#include "cosyvoice-audio.h"
#include "common.h"

#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

#include <ggml.h>

#define MINIAUDIO_IMPLEMENTATION
#define MA_NO_RESOURCE_MANAGER
#define MA_NO_NODE_GRAPH
#define MA_NO_ENGINE
#define MA_NO_GENERATION
#include "miniaudio.h"

#ifdef _WIN32
    #define strcasecmp _stricmp
#else
    #include <strings.h>
#endif

struct cosyvoice_audio_encoder
{
    cosyvoice_audio_encoder(uint32_t sample_rate) :
        config(ma_encoder_config_init(ma_encoding_format_wav, ma_format_f32, 1, sample_rate)) {}

    bool encode(const float* input, uint32_t length, ma_encoding_format format)
    {
        config.encodingFormat = format;
        offset = 0;
        buffer.clear();

        ma_result result = ma_encoder_init(
            [](ma_encoder* pEncoder, const void* pBufferIn, size_t bytesToWrite, size_t* pBytesWritten)
            {
                auto ctx = reinterpret_cast<cosyvoice_audio_encoder*>(pEncoder->pUserData);
                const uint8_t* bufferIn = reinterpret_cast<const uint8_t*>(pBufferIn);

                *pBytesWritten = bytesToWrite;
                if (ctx->offset + bytesToWrite > ctx->buffer.size())
                    ctx->buffer.resize(ctx->offset + bytesToWrite);
                memcpy(ctx->buffer.data() + ctx->offset, bufferIn, bytesToWrite);
                ctx->offset += bytesToWrite;

                return MA_SUCCESS;
            },
            [](ma_encoder* pEncoder, ma_int64 offset, ma_seek_origin origin)
            {
                auto ctx = reinterpret_cast<cosyvoice_audio_encoder*>(pEncoder->pUserData);
                switch (origin)
                {
                case ma_seek_origin_start:
                    ctx->offset = offset;
                    break;
                case ma_seek_origin_current:
                    ctx->offset += offset;
                    break;
                case ma_seek_origin_end:
                    ctx->offset = ctx->buffer.size() + offset;
                    break;
                default:
                    return MA_INVALID_ARGS;
                }
                return MA_SUCCESS;
            }, this, &config, &encoder);

        if (result != MA_SUCCESS) return false;

        result = ma_encoder_write_pcm_frames(&encoder, input, length, nullptr);
        ma_encoder_uninit(&encoder);
        return result == MA_SUCCESS;
    }

    ma_encoder_config config;
    ma_encoder encoder;
    std::vector<uint8_t> buffer;
    ma_int64 offset = 0;
};


bool cosyvoice_audio_encoding_format_supported(cosyvoice_audio_encoding_format_t format)
{
    switch (format)
    {
    case COSYVOICE_AUDIO_ENCODING_FORMAT_WAV:
        return true;
    default:
        return false;
    }
}

const char* cosyvoice_audio_supported_encoding_formats(void)
{
    return "wav";
}

cosyvoice_audio_encoder_t cosyvoice_audio_encoder_create(uint32_t sample_rate)
{
    return new cosyvoice_audio_encoder(sample_rate);
}

void cosyvoice_audio_encoder_destroy(cosyvoice_audio_encoder_t encoder)
{
    delete encoder;
}

bool cosyvoice_audio_encoder_encode(cosyvoice_audio_encoder_t encoder, const float* input, uint32_t length, cosyvoice_audio_encoding_format_t format)
{
    if (!encoder || !input) return false;

    ma_encoding_format miniaudio_format;
    switch (format)
    {
    case COSYVOICE_AUDIO_ENCODING_FORMAT_WAV:
        miniaudio_format = ma_encoding_format_wav;
        break;
    default:
        return false;
    }
    return encoder->encode(input, length, miniaudio_format);
}

void cosyvoice_audio_encoder_get_encoded_data(cosyvoice_audio_encoder_t encoder, const uint8_t** data, uint32_t* length)
{
    if (!encoder || !data || !length)
    {
        if (data) *data = nullptr;
        if (length) *length = 0;
        return;
    }

    *data = encoder->buffer.data();
    *length = static_cast<uint32_t>(encoder->buffer.size());
}

bool cosyvoice_audio_load_from_file(const char* filename, float** data, uint32_t* length, uint32_t* sample_rate)
{
    if (!filename || !data || !length || !sample_rate) return false;

    try
    {
        struct audio_decoder : ma_decoder
        {
            audio_decoder(const char* filename)
            {
                ma_decoder_config config = ma_decoder_config_init(ma_format_f32, 1, 0);

#ifdef _WIN32
                ma_result result = ma_decoder_init_file_w(utf8_to_wstr(filename).c_str(), &config, this);
#else
                ma_result result = ma_decoder_init_file(filename, &config, this);
#endif

                if (result != MA_SUCCESS) throw std::runtime_error("Failed to open audio file\n");
            }

            ~audio_decoder() { ma_decoder_uninit(this); }
        } decoder(filename);

        *sample_rate = decoder.outputSampleRate;

        ma_uint64 length64;
        ma_result result = ma_decoder_get_length_in_pcm_frames(&decoder, &length64);
        if (result != MA_SUCCESS) throw std::runtime_error("Failed to get audio length\n");
        *length = static_cast<uint32_t>(length64);

        std::unique_ptr<float[]> buffer(new float[length64]);
        result = ma_decoder_read_pcm_frames(&decoder, buffer.get(), length64, nullptr);
        if (result != MA_SUCCESS) throw std::runtime_error("Failed to read audio data\n");
        *data = buffer.release();
        return true;
    }
    catch (const std::exception& e)
    {
        cosyvoice_call_ggml_log_callback(GGML_LOG_LEVEL_ERROR, e.what());
        return false;
    }
}

bool cosyvoice_audio_resample(const float* input, uint32_t input_length, uint32_t input_sample_rate, float** output, uint32_t* output_length, uint32_t output_sample_rate)
{
    if (!input || !output || !output_length) return false;

    try
    {
        struct audio_resampler : ma_resampler
        {
            audio_resampler(uint32_t input_sample_rate, uint32_t output_sample_rate)
            {
                ma_resampler_config config = ma_resampler_config_init(ma_format_f32, 1, input_sample_rate, output_sample_rate, ma_resample_algorithm_linear);
                ma_result result = ma_resampler_init(&config, nullptr, this);
                if (result != MA_SUCCESS) throw std::runtime_error("Failed to initialize resampler\n");
            }
            ~audio_resampler() { ma_resampler_uninit(this, nullptr); }
        } resampler(input_sample_rate, output_sample_rate);

        ma_uint64 input_frames = input_length;
        ma_uint64 output_frames = ma_calculate_frame_count_after_resampling(output_sample_rate, input_sample_rate, input_frames);

        std::unique_ptr<float[]> buffer(new float[output_frames]);
        ma_result result = ma_resampler_process_pcm_frames(&resampler, input, &input_frames, buffer.get(), &output_frames);
        if (result != MA_SUCCESS) throw std::runtime_error("Failed to process audio data\n");
        *output = buffer.release();
        *output_length = static_cast<uint32_t>(output_frames);
    }
    catch (const std::exception& e)
    {
        cosyvoice_call_ggml_log_callback(GGML_LOG_LEVEL_ERROR, e.what());
        return false;
    }
    return true;
}

bool cosyvoice_audio_save_to_file(const char* filename, const float* data, uint32_t length, uint32_t sample_rate)
{
    if (!filename || !data || length == 0 || sample_rate == 0) return false;

    try
    {
        struct audio_encoder : ma_encoder
        {
            audio_encoder(const char* filename, uint32_t sample_rate)
            {
                ma_encoding_format encoding_format = ma_encoding_format_wav;
                if (const char* ext = strrchr(filename, '.'); ext && strcasecmp(ext, ".wav") != 0)
                {
                encoding_fallback:
                    ggml_log_callback callback;
                    void* user_data;
                    ggml_log_get(&callback, &user_data);
                    callback(GGML_LOG_LEVEL_WARN, "Unknown audio file extension, defaulting to WAV format\n", user_data);
                }
                ma_encoder_config config = ma_encoder_config_init(encoding_format, ma_format_f32, 1, sample_rate);
#ifdef _WIN32
                ma_result result = ma_encoder_init_file_w(utf8_to_wstr(filename).c_str(), &config, this);
#else
                ma_result result = ma_encoder_init_file(filename, &config, this);
#endif
                if (result != MA_SUCCESS) throw std::runtime_error("Failed to open audio file\n");
            }

            ~audio_encoder() { ma_encoder_uninit(this); }
        } encoder(filename, sample_rate);

        ma_result result = ma_encoder_write_pcm_frames(&encoder, data, length, nullptr);
        if (result != MA_SUCCESS) throw std::runtime_error("Failed to write audio data\n");
        return true;
    }
    catch (const std::exception& e)
    {
        cosyvoice_call_ggml_log_callback(GGML_LOG_LEVEL_ERROR, e.what());
        return false;
    }
}

void cosyvoice_audio_free(float* data)
{
    delete[] data;
}

// ---------------------------------------------------------------------------
// In-Memory Audio Decoder (miniaudio backend)
// ---------------------------------------------------------------------------

struct cosyvoice_audio_decoder
{
    std::vector<float> buffer;
    ma_decoder_config config = ma_decoder_config_init(ma_format_f32, 1, 0);
    uint32_t sample_rate = 0;
};

bool cosyvoice_audio_decoding_format_supported(cosyvoice_audio_encoding_format_t format)
{
    switch (format)
    {
    case COSYVOICE_AUDIO_ENCODING_FORMAT_WAV:
    case COSYVOICE_AUDIO_ENCODING_FORMAT_FLAC:
    case COSYVOICE_AUDIO_ENCODING_FORMAT_MP3:
        return true;
    default:
        return false;
    }
}

cosyvoice_audio_decoder_t cosyvoice_audio_decoder_create(void)
{
    return new cosyvoice_audio_decoder();
}

void cosyvoice_audio_decoder_destroy(cosyvoice_audio_decoder_t decoder)
{
    delete decoder;
}

bool cosyvoice_audio_decoder_decode(
    cosyvoice_audio_decoder_t decoder,
    const void*               input,
    uint32_t                  input_length)
{
    if (!decoder || !input || input_length == 0)
        return false;

    decoder->buffer.clear();
    decoder->sample_rate = 0;

    ma_decoder ma_dec;
    ma_result result = ma_decoder_init_memory(input, input_length, &decoder->config, &ma_dec);
    if (result != MA_SUCCESS)
        return false;

    decoder->sample_rate = ma_dec.outputSampleRate;

    ma_uint64 total = 0;
    result = ma_decoder_get_length_in_pcm_frames(&ma_dec, &total);
    if (result != MA_SUCCESS)
    {
        ma_decoder_uninit(&ma_dec);
        return false;
    }

    decoder->buffer.resize(static_cast<size_t>(total));
    ma_uint64 frames_read = 0;
    result = ma_decoder_read_pcm_frames(&ma_dec, decoder->buffer.data(), total, &frames_read);
    ma_decoder_uninit(&ma_dec);

    if (result != MA_SUCCESS)
    {
        decoder->buffer.clear();
        decoder->sample_rate = 0;
        return false;
    }

    decoder->buffer.resize(static_cast<size_t>(frames_read));
    return true;
}

void cosyvoice_audio_decoder_get_decoded_data(
    cosyvoice_audio_decoder_t decoder,
    float**                   data,
    uint32_t*                 length,
    uint32_t*                 sample_rate)
{
    if (!decoder || !data || !length || !sample_rate)
    {
        if (data)        *data = nullptr;
        if (length)      *length = 0;
        if (sample_rate) *sample_rate = 0;
        return;
    }

    *data        = decoder->buffer.data();
    *length      = static_cast<uint32_t>(decoder->buffer.size());
    *sample_rate = decoder->sample_rate;
}
