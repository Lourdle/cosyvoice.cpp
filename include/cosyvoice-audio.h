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
 * @brief Load an audio file from disk into memory.
 * @param filename Path to the input audio file.
 * @param data Receives the allocated mono PCM data.
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
 * @brief Resample audio to the requested sample rate.
 * @param input Input mono PCM data.
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
 * @brief Save mono PCM data to an audio file.
 * @param filename Path to the output file.
 * @param data Input mono PCM data.
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
 * @brief Free audio data allocated by the audio helpers.
 * @param data Audio buffer returned by the audio API.
 */
COSYVOICE_API void cosyvoice_audio_free(float* data);

#ifdef __cplusplus
}
#endif

