#ifdef _MSC_VER
	#define _CRT_SECURE_NO_WARNINGS
#endif

#include "cosyvoice.h"

#ifndef COSYVOICE_NO_AUDIO
	#include "cosyvoice-audio.h"
#else
	#include <ggml.h>
	#define cosyvoice_audio_save_to_file cosyvoice_save_wav
#endif

#ifndef COSYVOICE_NO_FRONTEND
	#include "cosyvoice-frontend.h"
#endif

#include "common.h"

#include <cstdarg>
#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <string>

struct cli_options
{
	uint32_t max_llm_len = 2048;
	float speed = 1.0f;
#ifndef COSYVOICE_NO_ICU
	bool text_normalization_enabled = true;
#endif
	std::string model;
#ifndef COSYVOICE_NO_FRONTEND
	bool frontend_only = false;
	std::string speech_tokenizer;
	std::string campplus;
	#ifdef COSYVOICE_NO_AUDIO
	std::string prompt_audio_16k;
	std::string prompt_audio_24k;
	#else
	std::string prompt_audio;
	#endif
	std::string prompt_text;
	std::string prompt_speech_output;
#endif
	std::string prompt_speech;
	std::string text;
	std::string instruction;
	std::string output;
	std::string mode = "auto";
};

struct cosyvoice_context_deleter { void operator()(cosyvoice_context_t ctx) const noexcept { cosyvoice_free(ctx); } };
#ifndef COSYVOICE_NO_FRONTEND
struct cosyvoice_frontend_deleter { void operator()(cosyvoice_frontend_context_t ctx) const noexcept { cosyvoice_frontend_free(ctx); } };
#endif
struct cosyvoice_prompt_speech_deleter { void operator()(cosyvoice_prompt_speech_t prompt_speech) const noexcept { cosyvoice_prompt_speech_free(prompt_speech); } };
struct cosyvoice_prompt_deleter { void operator()(cosyvoice_prompt_t prompt) const noexcept { cosyvoice_prompt_free(prompt); } };
struct cosyvoice_tts_context_deleter { void operator()(cosyvoice_tts_context_t ctx) const noexcept { cosyvoice_tts_context_free(ctx); } };
#ifndef COSYVOICE_NO_AUDIO
struct audio_buffer_deleter { void operator()(float* data) const noexcept { cosyvoice_audio_free(data); } };
#endif

using cosyvoice_context_handle = std::unique_ptr<cosyvoice_context, cosyvoice_context_deleter>;
#ifndef COSYVOICE_NO_FRONTEND
using cosyvoice_frontend_handle = std::unique_ptr<cosyvoice_frontend_context, cosyvoice_frontend_deleter>;
#endif
using cosyvoice_prompt_speech_handle = std::unique_ptr<cosyvoice_prompt_speech, cosyvoice_prompt_speech_deleter>;
using cosyvoice_prompt_handle = std::unique_ptr<cosyvoice_prompt, cosyvoice_prompt_deleter>;
using cosyvoice_tts_context_handle = std::unique_ptr<cosyvoice_tts_context, cosyvoice_tts_context_deleter>;
#ifndef COSYVOICE_NO_AUDIO
using audio_buffer_handle = std::unique_ptr<float, audio_buffer_deleter>;
#endif

static std::string to_lower(std::string&& value)
{
	std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
	return value;
}

static bool parse_float_arg(const std::string& value, float* result)
{
	char* end = nullptr;
	*result = strtof(value.c_str(), &end);
	return end != value.c_str() && end && *end == '\0';
}

static bool parse_uint32_arg(const std::string& value, uint32_t* result)
{
	char* end = nullptr;
	unsigned long long val = strtoull(value.c_str(), &end, 10);
	if (end == value.c_str() || !end || *end != '\0' || val > UINT32_MAX)
		return false;
	*result = static_cast<uint32_t>(val);
	return true;
}

#ifdef COSYVOICE_NO_AUDIO
	#define PROMPT_AUDIO_CMD "--prompt-audio-16k <16k_pcm_file> --prompt-audio-24k <24k_pcm_file>"
#else
	#define PROMPT_AUDIO_CMD "--prompt-audio <file>"
#endif

static void print_usage(const tchar* argv0)
{
	auto exe = tchar_to_utf8(argv0);
	printf(
		R"(cosyvoice-cli - command line TTS tool
Usage:
  %s --model <file.gguf> --prompt-speech <file> --text <text> --output <file>)"
#ifndef COSYVOICE_NO_FRONTEND
		R"("  %s --frontend-only --speech-tokenizer <file.onnx> --campplus <file.onnx> )" PROMPT_AUDIO_CMD R"( --prompt-text <text> --prompt-speech-output <file>
  %s --model <file.gguf> --speech-tokenizer <file.onnx> --campplus <file.onnx> )" PROMPT_AUDIO_CMD R"( --prompt-text <text> --text <text> --output <file>)"
#endif
		R"(
Options:
  --help, -h                                  Show this help message and exit.
  --model, -m <file>                          CosyVoice model file.
  --max-llm-len <value>                       Maximum number of tokens to feed into the LLM. Default: 2048.)"
#ifndef COSYVOICE_NO_FRONTEND
		R"(
  --frontend-only                             Only run frontend and save prompt_speech.
  --speech-tokenizer <file>                   Frontend speech tokenizer ONNX file.
  --campplus <file>                           Frontend campplus ONNX file.)"
#ifdef COSYVOICE_NO_AUDIO
		R"(
  --prompt-audio-16k <16k_pcm_file>           Reference audio file in 16kHz PCM float format.
  --prompt-audio-24k <24k_pcm_file>           Reference audio file in 24kHz PCM float format.)"
#else
		R"(
  --prompt-audio <file>                       Reference audio file for frontend.)"
#endif
		R"(
  --prompt-text <text>                        Transcript of the reference audio.
  --prompt-speech-output <file>               Save generated prompt_speech to file.)"
#endif
		R"(
  --prompt-speech <file>                      Saved prompt_speech file.
  --text, -t <text>                           Text to synthesize.
  --instruction, -i <text>                    Only used in instruct mode.
)"
#ifdef COSYVOICE_NO_AUDIO
"  --output, -o <file>                         Output audio file path. The output format is WAV."
#else
"  --output, -o <file>                         Output audio file path. The output format is determined by the file extension."
#endif
R"(
  --mode <zero-shot|instruct|cross-lingual>   TTS mode. Default: auto-detect based on the presence of --instruction.
)"
#ifndef COSYVOICE_NO_ICU
"  --disable-text-normalization                Disable ICU text normalization before tokenization.\n"
#endif
R"(  --speed, -s <value>                         Speech speed. Default: 1.0.
)", exe.c_str()
#ifndef COSYVOICE_NO_FRONTEND
, exe.c_str(), exe.c_str()
#endif
);
}

static void print_error_log(const char* format, ...)
{
	fprintf(stderr, "\033[31m"); // Set text color to red
	va_list args;
	va_start(args, format);
	vfprintf(stderr, format, args);
	va_end(args);
	fprintf(stderr, "\033[0m"); // Reset text color
}

static void print_warning_log(const char* format, ...)
{
	fprintf(stderr, "\033[93m"); // Set text color to yellow
	va_list args;
	va_start(args, format);
	vfprintf(stderr, format, args);
	va_end(args);
	fprintf(stderr, "\033[0m"); // Reset text color
}

static bool validate_options(cli_options& options)
{
#ifndef COSYVOICE_NO_FRONTEND
	if (options.frontend_only)
	{
		bool ok = true;
		if (options.speech_tokenizer.empty())
		{
			print_error_log("Error: --speech-tokenizer is required in frontend-only mode.\n");
			ok = false;
		}
		if (options.campplus.empty())
		{
			print_error_log("Error: --campplus is required in frontend-only mode.\n");
			ok = false;
		}
#ifdef COSYVOICE_NO_AUDIO
		if (options.prompt_audio_16k.empty())
		{
			print_error_log("Error: --prompt-audio-16k is required in frontend-only mode when audio is not available.\n");
			ok = false;
		}
		if (options.prompt_audio_24k.empty())
		{
			print_error_log("Error: --prompt-audio-24k is required in frontend-only mode when audio is not available.\n");
			ok = false;
		}
#else
		if (options.prompt_audio.empty())
		{
			print_error_log("Error: --prompt-audio is required in frontend-only mode.\n");
			ok = false;
		}
#endif
		if (options.prompt_speech_output.empty())
		{
			print_error_log("Error: --prompt-speech-output is required in frontend-only mode.\n");
			ok = false;
		}
		return ok;
	}
#endif

	bool ok = true;
	if (options.model.empty())
	{
		print_error_log("Error: --model is required for TTS.\n");
		ok = false;
	}
	if (options.text.empty())
	{
		print_error_log("Error: --text is required for TTS.\n");
		ok = false;
	}
	if (options.output.empty())
	{
		print_error_log("Error: --output is required for TTS.\n");
		ok = false;
	}

	const bool has_prompt_speech = !options.prompt_speech.empty();
#ifndef COSYVOICE_NO_FRONTEND
	const bool has_any_frontend_input = !options.speech_tokenizer.empty()
		|| !options.campplus.empty()
	#ifdef COSYVOICE_NO_AUDIO
		|| !options.prompt_audio_16k.empty()
		|| !options.prompt_audio_24k.empty()
	#else
		|| !options.prompt_audio.empty()
	#endif
		|| !options.prompt_text.empty();

	if (has_prompt_speech && has_any_frontend_input)
	{
		print_error_log("Error: use either --prompt-speech or frontend inputs, not both.\n");
		ok = false;
	}
#endif

	if (options.mode == "auto")
	{
	auto_detect:
		if (!options.instruction.empty())
			options.mode = "instruct";
		else
			options.mode = "zero-shot";
		printf("Info: auto-detected mode \"%s\".\n", options.mode.c_str());
	}
	else if (options.mode == "instruct")
	{
		if (options.instruction.empty())
		{
			print_warning_log("Warning: --instruction is not provided, falling back to zero-shot mode.\n");
			options.mode = "zero-shot";
		}
	}
	else if (options.mode == "zero-shot")
	{
		if (!options.instruction.empty())
			print_warning_log("Warning: --instruction is provided but will be ignored in zero-shot mode.\n");
	}
	else if (options.mode != "cross-lingual")
	{
		print_warning_log("Warning: unrecognized mode \"%s\", auto-detecting mode.\n", options.mode.c_str());
		goto auto_detect;
	}
	
	if (!has_prompt_speech)
	{
#ifndef COSYVOICE_NO_FRONTEND
		if (options.speech_tokenizer.empty())
		{
			print_error_log("Error: --speech-tokenizer is required when --prompt-speech is not provided.\n");
			ok = false;
		}
		if (options.campplus.empty())
		{
			print_error_log("Error: --campplus is required when --prompt-speech is not provided.\n");
			ok = false;
		}
	#ifdef COSYVOICE_NO_AUDIO
		if (options.prompt_audio_16k.empty())
		{
			print_error_log("Error: --prompt-audio-16k is required when --prompt-speech is not provided.\n");
			ok = false;
		}
		if (options.prompt_audio_24k.empty())
		{
			print_error_log("Error: --prompt-audio-24k is required when --prompt-speech is not provided.\n");
			ok = false;
		}
	#else
		if (options.prompt_audio.empty())
		{
			print_error_log("Error: --prompt-audio is required when --prompt-speech is not provided.\n");
			ok = false;
		}
	#endif
		if (options.prompt_text.empty())
		{
			if (options.mode == "zero-shot")
			{
				print_error_log("Error: --prompt-text is required when mode is zero-shot.\n");
				ok = false;
			}
		}
		else if (options.mode != "zero-shot")
			print_warning_log("Warning: --prompt-text is ignored when mode is not zero-shot.\n");
#else
		print_error_log("Error: --prompt-speech is required when the frontend is not available.\n");
#endif
	}

	return ok;
}

#ifdef _WIN32
int wmain(int argc, wchar_t** argv)
{
	setup_console_utf8();
#else
int main(int argc, char** argv)
{
#endif
	if (argc == 1)
	{
		print_usage(argv[0]);
		return 0;
	}

	cli_options options;
	for (int i = 1; i != argc; ++i)
	{
		auto arg = argv[i];
		auto get_arg_value = [&]() -> tchar*
		{
			if (++i == argc)
			{
				auto option = tchar_to_utf8(arg);
				print_error_log("Error: missing value for the command-line option \"%s\".\n", option.c_str());
				exit(1);
			}

			return argv[i];
		};

		if (tchar_casecmp(arg, COSYVOICE_TEXT("--help")) == 0 || tchar_casecmp(arg, COSYVOICE_TEXT("-h")) == 0)
		{
			print_usage(argv[0]);
			return 0;
		}
		else if (tchar_casecmp(arg, COSYVOICE_TEXT("--model")) == 0 || tchar_casecmp(arg, COSYVOICE_TEXT("-m")) == 0)
			options.model = tchar_to_utf8(get_arg_value());
		else if (tchar_casecmp(arg, COSYVOICE_TEXT("--max-llm-len")) == 0)
		{
			auto value = tchar_to_utf8(get_arg_value());
			uint32_t max_llm_len;
			if (!parse_uint32_arg(value, &max_llm_len) || max_llm_len <= 0)
			{
				print_error_log("Error: invalid --max-llm-len value \"%s\".\n", value.c_str());
				return 1;
			}
			options.max_llm_len = max_llm_len;
		}
#ifndef COSYVOICE_NO_FRONTEND
		else if (tchar_casecmp(arg, COSYVOICE_TEXT("--frontend-only")) == 0)
			options.frontend_only = true;
		else if (tchar_casecmp(arg, COSYVOICE_TEXT("--speech-tokenizer")) == 0)
			options.speech_tokenizer = tchar_to_utf8(get_arg_value());
		else if (tchar_casecmp(arg, COSYVOICE_TEXT("--campplus")) == 0)
			options.campplus = tchar_to_utf8(get_arg_value());
	#ifdef COSYVOICE_NO_AUDIO
		else if (tchar_casecmp(arg, COSYVOICE_TEXT("--prompt-audio-16k")) == 0)
			options.prompt_audio_16k = tchar_to_utf8(get_arg_value());
		else if (tchar_casecmp(arg, COSYVOICE_TEXT("--prompt-audio-24k")) == 0)
			options.prompt_audio_24k = tchar_to_utf8(get_arg_value());
	#else
		else if (tchar_casecmp(arg, COSYVOICE_TEXT("--prompt-audio")) == 0)
			options.prompt_audio = tchar_to_utf8(get_arg_value());
	#endif
		else if (tchar_casecmp(arg, COSYVOICE_TEXT("--prompt-text")) == 0)
			options.prompt_text = tchar_to_utf8(get_arg_value());
		else if (tchar_casecmp(arg, COSYVOICE_TEXT("--prompt-speech-output")) == 0)
			options.prompt_speech_output = tchar_to_utf8(get_arg_value());
#endif
		else if (tchar_casecmp(arg, COSYVOICE_TEXT("--prompt-speech")) == 0)
			options.prompt_speech = tchar_to_utf8(get_arg_value());
		else if (tchar_casecmp(arg, COSYVOICE_TEXT("--text")) == 0 || tchar_casecmp(arg, COSYVOICE_TEXT("-t")) == 0)
			options.text = tchar_to_utf8(get_arg_value());
		else if (tchar_casecmp(arg, COSYVOICE_TEXT("--instruction")) == 0 || tchar_casecmp(arg, COSYVOICE_TEXT("-i")) == 0)
			options.instruction = tchar_to_utf8(get_arg_value());
		else if (tchar_casecmp(arg, COSYVOICE_TEXT("--output")) == 0 || tchar_casecmp(arg, COSYVOICE_TEXT("-o")) == 0)
			options.output = tchar_to_utf8(get_arg_value());
		else if (tchar_casecmp(arg, COSYVOICE_TEXT("--mode")) == 0)
			options.mode = to_lower(tchar_to_utf8(get_arg_value()));
#ifndef COSYVOICE_NO_ICU
		else if (tchar_casecmp(arg, COSYVOICE_TEXT("--disable-text-normalization")) == 0)
			options.text_normalization_enabled = false;
#endif
		else if (tchar_casecmp(arg, COSYVOICE_TEXT("--speed")) == 0 || tchar_casecmp(arg, COSYVOICE_TEXT("-s")) == 0)
		{
			auto value = tchar_to_utf8(get_arg_value());
			float speed;
			if (!parse_float_arg(value, &speed) || speed <= 0.0f)
			{
				print_error_log("Error: invalid --speed value \"%s\".\n", value.c_str());
				return 1;
			}
			options.speed = speed;
		}
		else
		{
			auto option = tchar_to_utf8(arg);
			print_error_log("Error: the program doesn't recognize the command-line option \"%s\".\n", option.c_str());
			return 1;
		}
	}

	if (!validate_options(options))
		return 1;

	cosyvoice_init_backend();
	cosyvoice_prompt_speech_handle prompt_speech;
	if (!options.prompt_speech.empty())
	{
		prompt_speech.reset(cosyvoice_prompt_speech_load_from_file(options.prompt_speech.c_str()));
		if (!prompt_speech)
		{
			print_error_log("Error: failed to load prompt_speech file \"%s\".\n", options.prompt_speech.c_str());
			if (errno != 0)
				print_error_log("Reason: %s\n", strerror(errno));
			return 1;
		}
	}
#ifndef COSYVOICE_NO_FRONTEND
	else
	{
		cosyvoice_frontend_handle frontend(cosyvoice_frontend_load_from_files(options.speech_tokenizer.c_str(), options.campplus.c_str()));
		if (!frontend)
		{
			print_error_log("Error: failed to load frontend models.\n");
			return 1;
		}
	#ifdef COSYVOICE_NO_AUDIO
		auto f = open_ifstream_utf8(options.prompt_audio_16k.c_str());
		if (!f)
		{
			print_error_log("Error: failed to open prompt audio file \"%s\".\n", options.prompt_audio_16k.c_str());
			return 1;
		}
		f.seekg(0, std::ios::end);
		const auto prompt_audio_16k_end = f.tellg();
		if (prompt_audio_16k_end == std::streampos(-1))
		{
			print_error_log("Error: failed to query prompt audio file \"%s\".\n", options.prompt_audio_16k.c_str());
			return 1;
		}
		const auto prompt_audio_16k_size = static_cast<std::streamoff>(prompt_audio_16k_end);
		if (prompt_audio_16k_size <= 0)
		{
			print_error_log("Error: prompt audio file \"%s\" is empty.\n", options.prompt_audio_16k.c_str());
			return 1;
		}
		if (prompt_audio_16k_size % static_cast<std::streamoff>(sizeof(float)) != 0)
		{
			print_error_log("Error: prompt audio file \"%s\" size is not aligned to float samples.\n", options.prompt_audio_16k.c_str());
			return 1;
		}
		if (prompt_audio_16k_size / static_cast<std::streamoff>(sizeof(float)) > static_cast<std::streamoff>(std::numeric_limits<uint32_t>::max()))
		{
			print_error_log("Error: prompt audio file \"%s\" is too large.\n", options.prompt_audio_16k.c_str());
			return 1;
		}
		uint32_t prompt_audio_16k_length = static_cast<uint32_t>(prompt_audio_16k_size / static_cast<std::streamoff>(sizeof(float)));
		auto prompt_audio_16k_data = std::make_unique<float[]>(prompt_audio_16k_length);
		f.seekg(0, std::ios::beg);
		f.read(reinterpret_cast<char*>(prompt_audio_16k_data.get()), static_cast<std::streamsize>(prompt_audio_16k_length) * static_cast<std::streamsize>(sizeof(float)));
		if (!f)
		{
			print_error_log("Error: failed to read prompt audio file \"%s\".\n", options.prompt_audio_16k.c_str());
			return 1;
		}

		f = open_ifstream_utf8(options.prompt_audio_24k.c_str());
		if (!f)
		{
			print_error_log("Error: failed to open prompt audio file \"%s\".\n", options.prompt_audio_24k.c_str());
			return 1;
		}
		f.seekg(0, std::ios::end);
		const auto prompt_audio_24k_end = f.tellg();
		if (prompt_audio_24k_end == std::streampos(-1))
		{
			print_error_log("Error: failed to query prompt audio file \"%s\".\n", options.prompt_audio_24k.c_str());
			return 1;
		}
		const auto prompt_audio_24k_size = static_cast<std::streamoff>(prompt_audio_24k_end);
		if (prompt_audio_24k_size <= 0)
		{
			print_error_log("Error: prompt audio file \"%s\" is empty.\n", options.prompt_audio_24k.c_str());
			return 1;
		}
		if (prompt_audio_24k_size % static_cast<std::streamoff>(sizeof(float)) != 0)
		{
			print_error_log("Error: prompt audio file \"%s\" size is not aligned to float samples.\n", options.prompt_audio_24k.c_str());
			return 1;
		}
		if (prompt_audio_24k_size / static_cast<std::streamoff>(sizeof(float)) > static_cast<std::streamoff>(std::numeric_limits<uint32_t>::max()))
		{
			print_error_log("Error: prompt audio file \"%s\" is too large.\n", options.prompt_audio_24k.c_str());
			return 1;
		}
		uint32_t prompt_audio_24k_length = static_cast<uint32_t>(prompt_audio_24k_size / static_cast<std::streamoff>(sizeof(float)));
		auto prompt_audio_24k_data = std::make_unique<float[]>(prompt_audio_24k_length);
		f.seekg(0, std::ios::beg);
		f.read(reinterpret_cast<char*>(prompt_audio_24k_data.get()), static_cast<std::streamsize>(prompt_audio_24k_length) * static_cast<std::streamsize>(sizeof(float)));
		if (!f)
		{
			print_error_log("Error: failed to read prompt audio file \"%s\".\n", options.prompt_audio_24k.c_str());
			return 1;
		}

		prompt_speech.reset(cosyvoice_frontend_prompt_speech_direct(frontend.get(), prompt_audio_16k_data.get(), prompt_audio_16k_length, prompt_audio_24k_data.get(), prompt_audio_24k_length, options.prompt_text.c_str()));
		if (!prompt_speech)
		{
			print_error_log("Error: failed to generate prompt_speech from frontend inputs.\n");
			return 1;
		}
	#else
		float* prompt_audio_data = nullptr;
		uint32_t prompt_audio_length = 0;
		uint32_t prompt_audio_sample_rate = 0;
		if (!cosyvoice_audio_load_from_file(options.prompt_audio.c_str(), &prompt_audio_data, &prompt_audio_length, &prompt_audio_sample_rate))
		{
			print_error_log("Error: failed to load prompt audio file \"%s\".\n", options.prompt_audio.c_str());
			return 1;
		}

		audio_buffer_handle audio(prompt_audio_data);
		prompt_speech.reset(cosyvoice_frontend_prompt_speech(frontend.get(), audio.get(), prompt_audio_length, prompt_audio_sample_rate, options.prompt_text.c_str()));
		if (!prompt_speech)
		{
			print_error_log("Error: failed to generate prompt_speech from frontend inputs.\n");
			return 1;
		}
	#endif
	}

	if (!options.prompt_speech_output.empty() && !cosyvoice_prompt_speech_save_to_file(prompt_speech.get(), options.prompt_speech_output.c_str()))
	{
		print_error_log("Error: failed to save prompt_speech file \"%s\".\n", options.prompt_speech_output.c_str());
		return 1;
	}

	if (options.frontend_only)
		return 0;
#endif

	cosyvoice_context_params_t params;
	cosyvoice_init_default_context_params(&params);
	params.inference_buffer_policy = COSYVOICE_INFERENCE_BUFFER_POLICY_SHARED;
	params.n_max_seq = options.max_llm_len;
	cosyvoice_context_handle ctx(cosyvoice_load_from_file_with_params(options.model.c_str(), &params));
	if (!ctx)
	{
		print_error_log("Error: failed to load model file \"%s\".\n", options.model.c_str());
		return 1;
	}

	cosyvoice_prompt_handle prompt(cosyvoice_prompt_init_from_prompt_speech(ctx.get(), prompt_speech.get()));
	if (!prompt)
	{
		print_error_log("Error: failed to initialize prompt from prompt_speech.\n");
		return 1;
	}

	cosyvoice_tts_context_handle tts_ctx(cosyvoice_tts_context_new(ctx.get(), prompt.get()));
	if (!tts_ctx)
	{
		print_error_log("Error: failed to create TTS context.\n");
		return 1;
	}
#ifndef COSYVOICE_NO_ICU
	cosyvoice_tts_context_set_text_normalization_enabled(tts_ctx.get(), options.text_normalization_enabled);
#endif

	cosyvoice_generated_speech result = {};
	bool ok = false;
	if (options.mode == "cross-lingual")
		ok = cosyvoice_tts_cross_lingual(tts_ctx.get(), options.text.c_str(), options.speed, &result);
	else if (options.mode == "zero-shot")
		ok = cosyvoice_tts_zero_shot(tts_ctx.get(), options.text.c_str(), options.speed, &result);
	else
		ok = cosyvoice_tts_instruct(tts_ctx.get(), options.text.c_str(), options.instruction.c_str(), options.speed, &result);

	if (!ok && (!result.data || result.length == 0))
	{
		print_error_log("Error: TTS generation failed.\n");
		return 1;
	}

	if (!cosyvoice_audio_save_to_file(options.output.c_str(), result.data, result.length, cosyvoice_get_sample_rate(ctx.get())))
	{
		print_error_log("Error: failed to save output audio file \"%s\".\n", options.output.c_str());
		return 1;
	}

	return 0;
}
