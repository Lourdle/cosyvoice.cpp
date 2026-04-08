#ifdef _MSC_VER
	#define _CRT_SECURE_NO_WARNINGS
#endif

#include "common.h"

#include <cstring>
#include <cinttypes>
#include <memory>
#include <string>
#include <map>
#include <array>
#include <tuple>
#include <thread>
#include <atomic>

#include <ggml.h>
#include <gguf.h>
#include <ggml-cpp.h>

static ggml_type get_rollback_type(ggml_type type, int64_t ne)
{
	if (ne % ggml_blck_size(type) == 0) return type;
	switch (type)
	{
	case GGML_TYPE_Q4_0:
	case GGML_TYPE_Q4_1:
	case GGML_TYPE_Q5_0:
	case GGML_TYPE_Q5_1:
	case GGML_TYPE_Q8_0:
	case GGML_TYPE_Q8_1:
		return GGML_TYPE_F16;
	case GGML_TYPE_Q2_K:
	case GGML_TYPE_Q3_K:
		return get_rollback_type(GGML_TYPE_Q4_0, ne);
	case GGML_TYPE_Q4_K:
		return get_rollback_type(GGML_TYPE_Q5_0, ne);
	case GGML_TYPE_Q5_K:
		return get_rollback_type(GGML_TYPE_Q5_1, ne);
	case GGML_TYPE_Q6_K:
		return get_rollback_type(GGML_TYPE_Q8_0, ne);
	default:
		GGML_ABORT("get_rollback_type: unsupported type %s", ggml_type_name(type));
	}
}

#ifdef _WIN32
int wmain(int argc, wchar_t** argv)
{
	setup_console_utf8();
#else
int main(int argc, char** argv)
{
#endif

	tchar* input = nullptr;
	tchar* output = nullptr;
	ggml_ftype ftype = static_cast<ggml_ftype>(-2);
	std::map<std::string, std::string> custom_strings;
	for (int i = 1; i != argc; ++i)
	{
		auto arg = argv[i];
		auto get_arg_value = [&]()
		{
			if (++i == argc)
			{
				fprintf(stderr, "Error: missing value for the command-line option \"%s\".\n", tchar_to_utf8(arg).c_str());
				exit(1);
			}

			return argv[i];
		};

		if (tchar_casecmp(arg, COSYVOICE_TEXT("--help")) == 0 || tchar_casecmp(arg, COSYVOICE_TEXT("-h")) == 0)
		{
			printf(
				R"(gguf quantize - a tool for quantizing gguf files.
Usage: %s [options]"
Options:\n
  --help, -h                            Show this help message and exit.
  --file, -f <path>                     Specify the input file path.
  --output-file, -o <path>              Specify the output file path.
  --type, -t <type>                     Specify the quantization type. Supported types: F16, Q8_0, Q5_0, Q5_1, Q4_0, Q4_1, Q6_K, Q5_K, Q4_K, Q3_K, Q2_K, COPY.
  --custom-string, -c <key> <value>     Specify a custom string key-value pair to be included in the output file. This option can be used multiple times for different key-value pairs.
Example:
%s -f /path/to/file.gguf -o /path/to/output.gguf -t Q4_K -c general.quantized_by Lourdle -c general.description model_description
)", tchar_to_utf8(argv[0]).c_str(), tchar_to_utf8(argv[0]).c_str());
			return 0;
		}
		else if (tchar_casecmp(arg, COSYVOICE_TEXT("--file")) == 0 || tchar_casecmp(arg, COSYVOICE_TEXT("-f")) == 0)
			input = get_arg_value();
		else if (tchar_casecmp(arg, COSYVOICE_TEXT("--output-file")) == 0 || tchar_casecmp(arg, COSYVOICE_TEXT("-o")) == 0)
			output = get_arg_value();
		else if (tchar_casecmp(arg, COSYVOICE_TEXT("--type")) == 0 || tchar_casecmp(arg, COSYVOICE_TEXT("-t")) == 0)
		{
			arg = get_arg_value();

			if (tchar_casecmp(arg, COSYVOICE_TEXT("F16")) == 0) ftype = GGML_FTYPE_MOSTLY_F16;
			else if (tchar_casecmp(arg, COSYVOICE_TEXT("Q8_0")) == 0) ftype = GGML_FTYPE_MOSTLY_Q8_0;
			else if (tchar_casecmp(arg, COSYVOICE_TEXT("Q5_0")) == 0) ftype = GGML_FTYPE_MOSTLY_Q5_0;
			else if (tchar_casecmp(arg, COSYVOICE_TEXT("Q5_1")) == 0) ftype = GGML_FTYPE_MOSTLY_Q5_1;
			else if (tchar_casecmp(arg, COSYVOICE_TEXT("Q4_0")) == 0) ftype = GGML_FTYPE_MOSTLY_Q4_0;
			else if (tchar_casecmp(arg, COSYVOICE_TEXT("Q4_1")) == 0) ftype = GGML_FTYPE_MOSTLY_Q4_1;
			else if (tchar_casecmp(arg, COSYVOICE_TEXT("Q6_K")) == 0) ftype = GGML_FTYPE_MOSTLY_Q6_K;
			else if (tchar_casecmp(arg, COSYVOICE_TEXT("Q5_K")) == 0) ftype = GGML_FTYPE_MOSTLY_Q5_K;
			else if (tchar_casecmp(arg, COSYVOICE_TEXT("Q4_K")) == 0) ftype = GGML_FTYPE_MOSTLY_Q4_K;
			else if (tchar_casecmp(arg, COSYVOICE_TEXT("Q3_K")) == 0) ftype = GGML_FTYPE_MOSTLY_Q3_K;
			else if (tchar_casecmp(arg, COSYVOICE_TEXT("Q2_K")) == 0) ftype = GGML_FTYPE_MOSTLY_Q2_K;
			else if (tchar_casecmp(arg, COSYVOICE_TEXT("COPY")) == 0) ftype = GGML_FTYPE_UNKNOWN;
			else
			{
				fprintf(stderr, "Error: unsupported quantization type \"%s\".\n", tchar_to_utf8(arg).c_str());
				return 1;
			}
		}
		else if (tchar_casecmp(arg, COSYVOICE_TEXT("--custom-string")) == 0 || tchar_casecmp(arg, COSYVOICE_TEXT("-c")) == 0)
		{
			auto key = get_arg_value();
			auto value = get_arg_value();
			custom_strings[tchar_to_utf8(key)] = tchar_to_utf8(value);
		}
		else
		{
			fprintf(stderr, "Error: the program doesn't recognize the command-line option \"%s\".\n", tchar_to_utf8(arg).c_str());
			return 1;
		}
	}

	if (!input)
		fprintf(stderr, "Error: you must specify the input file path with --file or -f.\n");
	if (!output)
		fprintf(stderr, "Error: you must specify the output file path with --output-file or -o.\n");
	if (ftype == static_cast<ggml_ftype>(-2))
		fprintf(stderr, "Error: you must specify the quantization type with --type or -t.\n");
	if (!input || !output || ftype == static_cast<ggml_ftype>(-2))
		return 1;

	ggml_context* ctx;
	gguf_context_ptr input_gguf_ctx(gguf_init_from_file(tchar_to_utf8(input).c_str(), gguf_init_params{ .no_alloc = false, .ctx = &ctx }));
	if (!input_gguf_ctx)
	{
		fprintf(stderr, "Error: failed to load the input file \"%s\".\n", tchar_to_utf8(input).c_str());
		fprintf(stderr, "Reason: %s\n", strerror(errno));
		return 1;
	}

	std::map<std::string, std::tuple<std::unique_ptr<char[]>, ggml_type, std::array<int64_t, GGML_MAX_DIMS>>> quantized_tensors;
	if (ftype == GGML_FTYPE_UNKNOWN)
		for (int64_t i = 0; i != gguf_get_n_tensors(input_gguf_ctx.get()); ++i)
		{
			auto tensor = ggml_get_tensor(ctx, gguf_get_tensor_name(input_gguf_ctx.get(), i));
			auto nbytes = ggml_nbytes(tensor);
			auto data = std::make_unique<char[]>(nbytes);
			memcpy(data.get(), tensor->data, nbytes);
			quantized_tensors[tensor->name] = std::make_tuple(std::move(data), tensor->type, *reinterpret_cast<std::array<int64_t, GGML_MAX_DIMS>*>(&tensor->ne));
		}
	else
	{
		int64_t n_parameters = 0;
		ggml_type type = ggml_ftype_to_ggml_type(ftype);
		for (int64_t i = 0; i != gguf_get_n_tensors(input_gguf_ctx.get()); ++i)
		{
			auto tensor = ggml_get_tensor(ctx, gguf_get_tensor_name(input_gguf_ctx.get(), i));
			if (tensor->type != GGML_TYPE_F32)
			{
				for (auto& [key, value] : quantized_tensors)
					std::get<0>(value).release();
				fprintf(stderr, "Error: the tensor \"%s\" has unsupported data type %s, only F32 is supported as input.\n", tensor->name, ggml_type_name(tensor->type));
				ggml_free(ctx);
				return 1;
			}

			const int64_t nelements = ggml_nelements(tensor);
			if (ggml_is_vector(tensor))
				quantized_tensors[tensor->name] = std::make_tuple(std::unique_ptr<char[]>(reinterpret_cast<char*>(tensor->data)), tensor->type, *reinterpret_cast<std::array<int64_t, GGML_MAX_DIMS>*>(&tensor->ne));
			else if (ggml_is_matrix(tensor))
			{
				auto cur_type = get_rollback_type(type, tensor->ne[0]);
				if (cur_type != type)
					printf("Warning: the tensor \"%s\"'s ne0 is %" PRId64 ", which is not divisible by the block size of the target quantization type %s, so it will be quantized to %s instead.\n", tensor->name, tensor->ne[0], ggml_type_name(type), ggml_type_name(cur_type));
				quantized_tensors[tensor->name] = std::make_tuple(std::unique_ptr<char[]>(reinterpret_cast<char*>(tensor->data)), cur_type, *reinterpret_cast<std::array<int64_t, GGML_MAX_DIMS>*>(&tensor->ne));
			}
			else
			{
				printf("Warning: the tensor \"%s\" is not a matrix or vector. It will be quantized to F16.\n", tensor->name);
				quantized_tensors[tensor->name] = std::make_tuple(std::unique_ptr<char[]>(reinterpret_cast<char*>(tensor->data)), GGML_TYPE_F16, *reinterpret_cast<std::array<int64_t, GGML_MAX_DIMS>*>(&tensor->ne));
			}
			n_parameters += nelements;
		}

		printf("Total number of parameters: %" PRId64 "\n", n_parameters);
		auto n_threads = std::max(1u, std::thread::hardware_concurrency());
		auto threads = std::make_unique<std::thread[]>(n_threads);
		std::atomic_size_t i = 0;
		std::atomic_size_t processed_parameters = 0;
		auto worker_thread = [&](auto report_progress)
		{
			auto iter = quantized_tensors.begin();
			size_t last = 0;
			for (;;)
			{
				size_t cur = i++;
				if (cur >= quantized_tensors.size())
					break;
				std::advance(iter, cur - last);
				last = cur;
				auto& [data, type, ne] = iter->second;
				auto n_rows = ne[1] * ne[2] * ne[3];
				auto quantized_data = std::make_unique<char[]>(ggml_row_size(type, ne[0]) * n_rows);
				ggml_quantize_chunk(type, reinterpret_cast<float*>(data.get()), quantized_data.get(), 0, n_rows, ne[0], nullptr);
				data.release();
				data.swap(quantized_data);
				processed_parameters += ne[0] * n_rows;
				report_progress();
			}
		};
		auto report_progress = [&]()
		{
			auto n_processed = processed_parameters.load();
			printf("Quantized %zu / %zu parameters (%.2f%%)\r", n_processed, n_parameters, 100.0 * n_processed / n_parameters);
		};
		for (auto j = 0u; j != n_threads; ++j)
			threads[j] = j == 0 ? std::thread(worker_thread, report_progress) : std::thread(worker_thread, []() {});
		for (auto j = 0u; j != n_threads; ++j)
			threads[j].join();
		report_progress();
		printf("\n");
	}

	ggml_init_params params = { .mem_size = ggml_tensor_overhead() * quantized_tensors.size(), .no_alloc = true };
	ggml_context_ptr gguf_ggml_ctx(ggml_init(params));
	gguf_context_ptr output_gguf_ctx(gguf_init_empty());
	int64_t id = gguf_find_key(input_gguf_ctx.get(), "general.file_type");
	gguf_set_kv(output_gguf_ctx.get(), input_gguf_ctx.get());
	ggml_free(ctx);
	if (id != -1 && ftype != GGML_FTYPE_UNKNOWN)
		gguf_set_val_u32(output_gguf_ctx.get(), "general.file_type", ftype);

	for (const auto& [key, value] : custom_strings)
		gguf_set_val_str(output_gguf_ctx.get(), key.c_str(), value.c_str());

	for (const auto& [name, tensor_info] : quantized_tensors)
	{
		const auto& [data, type, ne] = tensor_info;
		auto tensor = ggml_new_tensor(gguf_ggml_ctx.get(), type, GGML_MAX_DIMS, ne.data());
		tensor->data = data.get();
		ggml_set_name(tensor, name.c_str());
		gguf_add_tensor(output_gguf_ctx.get(), tensor);
	}

	if (!gguf_write_to_file(output_gguf_ctx.get(), tchar_to_utf8(output).c_str(), false))
	{
		fprintf(stderr, "Error: failed to save the output file \"%s\".\n", tchar_to_utf8(output).c_str());
		fprintf(stderr, "Reason: %s\n", strerror(errno));
		return 1;
	}
}
