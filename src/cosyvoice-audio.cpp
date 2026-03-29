#include "cosyvoice-internal.h"
#include "cosyvoice-audio.h"

#include <stdexcept>
#include <memory>

#include <ggml.h>

#define MINIAUDIO_IMPLEMENTATION
#define MA_NO_DEVICE_IO
#define MA_NO_RESOURCE_MANAGER
#define MA_NO_NODE_GRAPH
#define MA_NO_ENGINE
#define MA_NO_GENERATION
#define MA_NO_THREADING
#include "miniaudio.h"

#ifdef _WIN32
	#include <Windows.h>
	#define strcasecmp _stricmp

static std::unique_ptr<WCHAR[]> utf8_to_utf16(const char* str)
{
	int cch = MultiByteToWideChar(CP_UTF8, 0, str, -1, nullptr, 0);
	std::unique_ptr<WCHAR[]> wstr(new WCHAR[cch]);
	MultiByteToWideChar(CP_UTF8, 0, str, -1, wstr.get(), cch);
	return wstr;
}
#else
	#include <strings.h>
#endif

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
				ma_result result = ma_decoder_init_file_w(utf8_to_utf16(filename).get(), &config, this);
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
				if (const char* ext = strrchr(filename, '.'); ext)
				{
					if (strcasecmp(ext, ".flac") == 0) encoding_format = ma_encoding_format_flac;
					else if (strcasecmp(ext, ".mp3") == 0) encoding_format = ma_encoding_format_mp3;
					else if (strcasecmp(ext, ".ogg") == 0) encoding_format = ma_encoding_format_vorbis;
					else if (strcasecmp(ext, ".wav") != 0) goto encoding_fallback;
				}
				else
				{
				encoding_fallback:
					ggml_log_callback callback;
					void* user_data;
					ggml_log_get(&callback, &user_data);
					callback(GGML_LOG_LEVEL_WARN, "Unknown audio file extension, defaulting to WAV format\n", user_data);
				}
				ma_encoder_config config = ma_encoder_config_init(encoding_format, ma_format_f32, 1, sample_rate);
#ifdef _WIN32
				ma_result result = ma_encoder_init_file_w(utf8_to_utf16(filename).get(), &config, this);
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
