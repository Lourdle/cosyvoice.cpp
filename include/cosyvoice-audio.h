#pragma once

#ifndef COSYVOICE_API
    #ifdef COSYVOICE_STATIC
        #define COSYVOICE_API
    #else
        #ifdef _WIN32
            #define COSYVOICE_API __declspec(dllimport)
        #else
            #define COSYVOICE_API __attribute__((visibility("default")))
        #endif
    #endif
#endif

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#else
    #include <stdbool.h>
#endif

// ----------------------------------------------------------------------------
// Audio File Management & Utility API
// ----------------------------------------------------------------------------

/**
 * @brief Opaque in-memory audio encoder handle.
 *
 * The encoder converts mono float PCM (`[-1, 1]`) to encoded audio bytes
 * without touching the filesystem. Use one of the encode functions, then read
 * the encoded payload via `cosyvoice_audio_encoder_get_encoded_data`.
 */
typedef struct cosyvoice_audio_encoder* cosyvoice_audio_encoder_t;

/**
 * @brief Supported in-memory audio encoding formats.
 *
 * Note: Available formats depend on the audio backend:
 * - MINIAUDIO backend: only WAV
 * - FFMPEG backend: WAV, MP3, AAC, FLAC, M4A, OPUS
 */
typedef enum cosyvoice_audio_encoding_format
{
    COSYVOICE_AUDIO_ENCODING_FORMAT_WAV = 0,
    COSYVOICE_AUDIO_ENCODING_FORMAT_MP3,
    COSYVOICE_AUDIO_ENCODING_FORMAT_AAC,
    COSYVOICE_AUDIO_ENCODING_FORMAT_FLAC,
    COSYVOICE_AUDIO_ENCODING_FORMAT_M4A,
    COSYVOICE_AUDIO_ENCODING_FORMAT_OPUS,
    COSYVOICE_AUDIO_ENCODING_FORMAT_COUNT
} cosyvoice_audio_encoding_format_t;

/**
 * @brief Check whether an audio encoding format is supported.
 * @param format Format to test.
 * @return True if the format is supported, otherwise false.
 */
COSYVOICE_API bool cosyvoice_audio_encoding_format_supported(cosyvoice_audio_encoding_format_t format);

/**
 * @brief Get the supported audio encoding formats as a comma-separated string.
 *
 * The returned pointer is owned by the library and valid until process exit.
 */
COSYVOICE_API const char* cosyvoice_audio_supported_encoding_formats(void);

/**
 * @brief Create an in-memory audio encoder.
 * @param sample_rate Input sample rate in Hz.
 * @return Encoder handle on success; otherwise `NULL`.
 */
COSYVOICE_API cosyvoice_audio_encoder_t cosyvoice_audio_encoder_create(uint32_t sample_rate);

/**
 * @brief Destroy an encoder created by `cosyvoice_audio_encoder_create`.
 * @param encoder Encoder handle. `NULL` is allowed.
 */
COSYVOICE_API void cosyvoice_audio_encoder_destroy(cosyvoice_audio_encoder_t encoder);

/**
 * @brief Encode mono float PCM to a specific audio format.
 * @param encoder Encoder handle.
 * @param input Input mono float PCM buffer.
 * @param length Number of input samples.
 * @param format Target audio format.
 * @return True on success, otherwise false.
 */
COSYVOICE_API bool cosyvoice_audio_encoder_encode(
    cosyvoice_audio_encoder_t encoder,
    const float* input,
    uint32_t length,
    cosyvoice_audio_encoding_format_t format
);

/**
 * @brief Get encoded payload from the last successful encode call.
 * @param encoder Encoder handle.
 * @param data Receives pointer to encoded bytes owned by the encoder.
 * @param length Receives encoded byte length.
 *
 * The returned `data` pointer remains valid until the next encode call on the
 * same encoder, or until the encoder is destroyed.
 */
COSYVOICE_API void cosyvoice_audio_encoder_get_encoded_data(cosyvoice_audio_encoder_t encoder,
    const uint8_t** data,
    uint32_t* length
);

// ----------------------------------------------------------------------------
// In-Memory Audio Decoder API
// ----------------------------------------------------------------------------

/**
 * @brief Opaque in-memory audio decoder handle.
 *
 * Decodes encoded audio from a memory buffer to mono float PCM
 * without touching the filesystem.
 */
typedef struct cosyvoice_audio_decoder* cosyvoice_audio_decoder_t;

/**
 * @brief Check whether an audio encoding format can be decoded.
 * @param format Format to test.
 * @return True if the format is supported for decoding, otherwise false.
 */
COSYVOICE_API bool cosyvoice_audio_decoding_format_supported(cosyvoice_audio_encoding_format_t format);

/**
 * @brief Create an in-memory audio decoder.
 * @return Decoder handle on success; otherwise NULL.
 */
COSYVOICE_API cosyvoice_audio_decoder_t cosyvoice_audio_decoder_create(void);

/**
 * @brief Destroy a decoder created by cosyvoice_audio_decoder_create.
 * @param decoder Decoder handle. NULL is allowed.
 */
COSYVOICE_API void cosyvoice_audio_decoder_destroy(cosyvoice_audio_decoder_t decoder);

/**
 * @brief Decode audio from a memory buffer to mono float PCM.
 *
 * The decoder auto-detects the audio format from the buffer content.
 * After a successful decode, call cosyvoice_audio_decoder_get_decoded_data
 * to retrieve the PCM data.
 *
 * @param decoder      Decoder handle.
 * @param input        Pointer to encoded audio data.
 * @param input_length Length of encoded data in bytes.
 * @return True on success, otherwise false.
 */
COSYVOICE_API bool cosyvoice_audio_decoder_decode(
    cosyvoice_audio_decoder_t decoder,
    const void*               input,
    uint32_t                  input_length
);

/**
 * @brief Get decoded PCM data from the last successful decode call.
 *
 * The returned `data` pointer remains valid until the next decode call on
 * the same decoder, or until the decoder is destroyed.  Caller must NOT
 * free the data.
 *
 * @param decoder     Decoder handle.
 * @param data        Receives pointer to mono float PCM buffer ([-1, 1]).
 * @param length      Receives number of samples.
 * @param sample_rate Receives sample rate in Hz.
 */
COSYVOICE_API void cosyvoice_audio_decoder_get_decoded_data(
    cosyvoice_audio_decoder_t decoder,
    float**                   data,
    uint32_t*                 length,
    uint32_t*                 sample_rate
);

/**
 * @brief Load an audio file from disk and decode it to mono float PCM.
 * @param filename Path to the input audio file.
 * @param data Receives the allocated PCM buffer.
 * @param length Receives the number of samples.
 * @param sample_rate Receives the sample rate in Hz.
 * @return True on success, otherwise false.
 */
COSYVOICE_API bool cosyvoice_audio_load_from_file(
    const char* filename,
    float**     data,
    uint32_t*   length,
    uint32_t*   sample_rate
);

/**
 * @brief Resample mono float PCM to the requested sample rate.
 * @param input Input PCM buffer.
 * @param input_length Number of input samples.
 * @param input_sample_rate Input sample rate in Hz.
 * @param output Receives the allocated resampled data.
 * @param output_length Receives the number of output samples.
 * @param output_sample_rate Target sample rate in Hz.
 * @return True on success, otherwise false.
 */
COSYVOICE_API bool cosyvoice_audio_resample(
    const float* input,
    uint32_t     input_length,
    uint32_t     input_sample_rate,
    float**      output,
    uint32_t*    output_length,
    uint32_t     output_sample_rate
);

/**
 * @brief Save mono float PCM data to an audio file.
 * @param filename Path to the output file.
 * @param data Input PCM buffer.
 * @param length Number of samples.
 * @param sample_rate Output sample rate in Hz.
 * @return True on success, otherwise false.
 */
COSYVOICE_API bool cosyvoice_audio_save_to_file(
    const char*  filename,
    const float* data,
    uint32_t     length,
    uint32_t     sample_rate
);

/**
 * @brief Free audio data allocated by the audio helper APIs.
 * @param data Buffer returned by an audio helper.
 */
COSYVOICE_API void cosyvoice_audio_free(float* data);

#ifdef __cplusplus
}
#endif

