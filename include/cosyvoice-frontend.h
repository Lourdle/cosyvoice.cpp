#ifndef COSYVOICE_FRONTEND_H
#define COSYVOICE_FRONTEND_H

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

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Normalized text returned by frontend utilities.
 */
typedef struct cosyvoice_text
{
	uint32_t    length; ///< Length of `text` in bytes, excluding the null terminator.
	const char* text;   ///< Pointer to a null-terminated UTF-8 string.
}* cosyvoice_text_ptr;

// ----------------------------------------------------------------------------
// Normalized Text Utilities
// ----------------------------------------------------------------------------

/**
* @brief Opaque struct for normalized text storage used by frontend utilities.
*/
COSYVOICE_API cosyvoice_text_ptr cosyvoice_text_create();

/**
 * @brief Free a normalized text object created by `cosyvoice_text_create`.
 */
COSYVOICE_API void cosyvoice_text_free(cosyvoice_text_ptr text);

/**
 * @brief Normalize input text for prompt processing.
 * @param locale Optional locale hint. Pass null to auto-detect.
 */
COSYVOICE_API void cosyvoice_frontend_util_text_normalize(
	const char* text,
	uint32_t    text_len,
	const char* locale,
	cosyvoice_text_ptr normalized_text
);

typedef struct cosyvoice_frontend_context* cosyvoice_frontend_context_t; // Handle to a frontend context.

// ----------------------------------------------------------------------------
// Frontend Context Loading
// ----------------------------------------------------------------------------

/**
 * @brief Load a frontend context from model files on disk.
 */
COSYVOICE_API cosyvoice_frontend_context_t cosyvoice_frontend_load_from_files(
	const char* speech_tokenizer,
	const char* campplus
);

struct OrtEnv;
struct OrtSessionOptions;

/**
 * @brief Load a frontend context from in-memory model blobs.
 */
COSYVOICE_API cosyvoice_frontend_context_t cosyvoice_frontend_load(
	const void* speech_tokenizer_data,
	size_t speech_tokenizer_size,
	const void* campplus_data,
	size_t campplus_size,
	const struct OrtEnv* env,
	const struct OrtSessionOptions* session_options
);

// ----------------------------------------------------------------------------
// Prompt-Speech Extraction
// ----------------------------------------------------------------------------

/**
 * @brief Extract prompt speech features from a single input waveform. This needs audio API support.
 * @param speech Input mono PCM waveform.
 * @param speech_len Number of samples in `speech`.
 * @param sample_rate Sample rate of `speech` in Hz.
 * @param prompt_text Optional prompt text associated with the speech.
 */
COSYVOICE_API cosyvoice_prompt_speech_t cosyvoice_frontend_prompt_speech(
	cosyvoice_frontend_context_t ctx,
	float* speech,
	uint32_t speech_len,
	uint32_t sample_rate,
	const char* prompt_text
);

/**
 * @brief Extract prompt speech features from pre-resampled 16 kHz and 24 kHz waveforms.
 * @param speech_16k Input mono PCM waveform at 16 kHz.
 * @param speech_16k_len Number of samples in `speech_16k`.
 * @param speech_24k Input mono PCM waveform at 24 kHz.
 * @param speech_24k_len Number of samples in `speech_24k`.
 * @param prompt_text Optional prompt text associated with the speech.
 */
COSYVOICE_API cosyvoice_prompt_speech_t cosyvoice_frontend_prompt_speech_direct(
	cosyvoice_frontend_context_t ctx,
	float* speech_16k,
	uint32_t speech_16k_len,
	float* speech_24k,
	uint32_t speech_24k_len,
	const char* prompt_text
);

// ----------------------------------------------------------------------------
// Prompt-Speech Cleanup
// ----------------------------------------------------------------------------

/**
 * @brief Free a frontend context.
 */
COSYVOICE_API void cosyvoice_frontend_free(cosyvoice_frontend_context_t ctx);

#ifdef __cplusplus
}
#endif

#endif // COSYVOICE_FRONTEND_H
