#ifdef _MSC_VER
    #define _CRT_SECURE_NO_WARNINGS
#endif

#include "tool_common.h"

#include <cstring>
#include <cinttypes>
#include <memory>
#include <string>
#include <map>
#include <set>
#include <array>
#include <tuple>
#include <thread>
#include <atomic>

#include <ggml.h>
#include <gguf.h>
#include <ggml-cpp.h>

#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>

#include <nlohmann/json.hpp>

enum OptionID {
    OPT_HELP,
    OPT_FILE,
    OPT_OUTPUT_FILE,
    OPT_TYPE,
    OPT_CUSTOM_STRING,
    OPT_TENSOR_MAP,
    OPT_COUNT
};

struct OptionDef {
    const char* long_name;
    char short_name;
    int num_args;
    const char* help;
};

static const OptionDef s_options[OPT_COUNT] = {
    { "--help",         'h', 0, "Show this help message and exit." },
    { "--file",         'f', 1, "Input GGUF file path." },
    { "--output-file",  'o', 1, "Output GGUF file path." },
    { "--type",         't', 1, "Default quantization type. Supported: F16, Q8_0, Q5_0, Q5_1, Q4_0, Q4_1, Q6_K, Q5_K, Q4_K, Q3_K, Q2_K, COPY." },
    { "--custom-string",'c', 2, "Custom metadata key-value pair (repeatable). Usage: -c <key> <value>" },
    { "--tensor-map",   'M', 1, "JSON file mapping tensor name regex to quantization type. Unlisted tensors use --type default." },
};

static void print_help(const char* program)
{
    printf("gguf quantize - a tool for quantizing gguf files.\n");
    printf("Usage: %s [options]\n", program);
    printf("\nOptions:\n");
    for (int i = 0; i < OPT_COUNT; i++)
    {
        const auto& opt = s_options[i];
        printf("  %s, -%c", opt.long_name, opt.short_name);
        for (int a = 0; a < opt.num_args; a++)
            printf(" <arg>");
        printf("\n    %s\n", opt.help);
    }
    printf("\nExamples:\n");
    printf("  %s -f model.gguf -o model-q4k.gguf -t Q4_K\n", program);
    printf("  %s -f model.gguf -o model-partial.gguf -t COPY -M tensor_map.json\n", program);
    printf("  %s -f model.gguf -o model-with-meta.gguf -t Q4_K -c general.quantized_by Lourdle\n", program);
}

static ggml_ftype parse_ftype(const char* s)
{
    if (str_casecmp(s, "F16") == 0)  return GGML_FTYPE_MOSTLY_F16;
    if (str_casecmp(s, "Q8_0") == 0) return GGML_FTYPE_MOSTLY_Q8_0;
    if (str_casecmp(s, "Q5_0") == 0) return GGML_FTYPE_MOSTLY_Q5_0;
    if (str_casecmp(s, "Q5_1") == 0) return GGML_FTYPE_MOSTLY_Q5_1;
    if (str_casecmp(s, "Q4_0") == 0) return GGML_FTYPE_MOSTLY_Q4_0;
    if (str_casecmp(s, "Q4_1") == 0) return GGML_FTYPE_MOSTLY_Q4_1;
    if (str_casecmp(s, "Q6_K") == 0) return GGML_FTYPE_MOSTLY_Q6_K;
    if (str_casecmp(s, "Q5_K") == 0) return GGML_FTYPE_MOSTLY_Q5_K;
    if (str_casecmp(s, "Q4_K") == 0) return GGML_FTYPE_MOSTLY_Q4_K;
    if (str_casecmp(s, "Q3_K") == 0) return GGML_FTYPE_MOSTLY_Q3_K;
    if (str_casecmp(s, "Q2_K") == 0) return GGML_FTYPE_MOSTLY_Q2_K;
    if (str_casecmp(s, "COPY") == 0) return GGML_FTYPE_UNKNOWN;
    return static_cast<ggml_ftype>(-2);
}

static ggml_type get_fallback_type(ggml_type type, int64_t ne)
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
        return get_fallback_type(GGML_TYPE_Q4_0, ne);
    case GGML_TYPE_Q4_K:
        return get_fallback_type(GGML_TYPE_Q5_0, ne);
    case GGML_TYPE_Q5_K:
        return get_fallback_type(GGML_TYPE_Q5_1, ne);
    case GGML_TYPE_Q6_K:
        return get_fallback_type(GGML_TYPE_Q8_0, ne);
    default:
        GGML_ABORT("get_fallback_type: unsupported type %s", ggml_type_name(type));
    }
}

struct TensorMapEntry {
    std::string pattern;
    std::unique_ptr<pcre2_code_8, decltype(&pcre2_code_free_8)> code;
    ggml_ftype ftype;
    bool is_literal;
    bool matched = false;
};

static bool is_literal_pattern(const std::string& pattern)
{
    for (size_t i = 0; i < pattern.size(); i++)
    {
        if (pattern[i] == '\\')
        {
            i++; // skip escaped char
            continue;
        }
        if (strchr(".*+?[({|^$", pattern[i]))
            return false;
    }
    return true;
}

static std::vector<TensorMapEntry> load_tensor_map(const char* path)
{
    auto file = open_ifstream_utf8(path);
    if (!file.is_open())
    {
        fprintf(stderr, "Error: failed to open tensor map file \"%s\".\n", path);
        return {};
    }

    std::vector<TensorMapEntry> entries;
    try
    {
        auto j = nlohmann::json::parse(file);
        for (auto it = j.begin(); it != j.end(); ++it)
        {
            auto& pattern_str = it.key();
            auto& type_str = it.value();

            if (!type_str.is_string())
            {
                fprintf(stderr, "Error: tensor map value must be a string for pattern \"%s\".\n", pattern_str.c_str());
                return {};
            }

            auto ftype = parse_ftype(type_str.get<std::string>().c_str());
            if (static_cast<int>(ftype) == -2)
            {
                fprintf(stderr, "Error: unsupported quantization type \"%s\" for pattern \"%s\".\n",
                    type_str.get<std::string>().c_str(), pattern_str.c_str());
                return {};
            }

            int error_code = 0;
            PCRE2_SIZE error_offset = 0;
            pcre2_code_8* raw = pcre2_compile_8(
                reinterpret_cast<PCRE2_SPTR8>(pattern_str.c_str()),
                pattern_str.size(), 0, &error_code, &error_offset, nullptr);
            if (!raw)
            {
                PCRE2_UCHAR buffer[256];
                pcre2_get_error_message_8(error_code, buffer, sizeof(buffer));
                fprintf(stderr, "Error: invalid regex \"%s\": %s\n",
                    pattern_str.c_str(), reinterpret_cast<const char*>(buffer));
                return {};
            }

            entries.push_back({
                pattern_str,
                std::unique_ptr<pcre2_code_8, decltype(&pcre2_code_free_8)>(raw, pcre2_code_free_8),
                ftype,
                is_literal_pattern(pattern_str),
                false
            });
        }
    }
    catch (const nlohmann::json::parse_error& e)
    {
        fprintf(stderr, "Error: failed to parse tensor map JSON: %s\n", e.what());
        return {};
    }

    return entries;
}

static bool match_tensor_name(pcre2_code_8* code, const char* name)
{
    auto* match_data = pcre2_match_data_create_from_pattern_8(code, nullptr);
    if (!match_data)
        return false;
    int rc = pcre2_match_8(code, reinterpret_cast<PCRE2_SPTR8>(name), strlen(name),
                           0, 0, match_data, nullptr);
    pcre2_match_data_free_8(match_data);
    return rc >= 0;
}

int tool_entry(int argc, char** argv)
{
    char* input = nullptr;
    char* output = nullptr;
    char* tensor_map_path = nullptr;
    ggml_ftype default_ftype = static_cast<ggml_ftype>(-2);
    std::map<std::string, std::string> custom_strings;

    auto get_required_arg = [&](int& i, const char* opt_name) -> const char*
    {
        if (++i >= argc)
        {
            fprintf(stderr, "Error: option \"%s\" requires an argument.\n", opt_name);
            exit(1);
        }
        return argv[i];
    };

    for (int i = 1; i < argc; i++)
    {
        const char* arg = argv[i];
        bool matched = false;

        for (int opt_id = 0; opt_id < OPT_COUNT; opt_id++)
        {
            const auto& opt = s_options[opt_id];
            auto short_form = std::string("-") + opt.short_name;
            if (str_casecmp(arg, opt.long_name) == 0 || arg == short_form)
            {
                matched = true;

                if (opt_id == OPT_HELP)
                {
                    print_help(argv[0]);
                    return 0;
                }

                if (opt.num_args < 1)
                {
                    fprintf(stderr, "Error: internal error - option \"%s\" has no handler.\n", arg);
                    return 1;
                }

                const char* v = get_required_arg(i, arg);

                switch (opt_id)
                {
                case OPT_FILE:
                    input = const_cast<char*>(v);
                    break;
                case OPT_OUTPUT_FILE:
                    output = const_cast<char*>(v);
                    break;
                case OPT_TYPE:
                    default_ftype = parse_ftype(v);
                    if (static_cast<int>(default_ftype) == -2)
                    {
                        fprintf(stderr, "Error: unsupported quantization type \"%s\".\n", v);
                        return 1;
                    }
                    break;
                case OPT_CUSTOM_STRING:
                {
                    auto key = v;
                    auto value = get_required_arg(i, arg);
                    custom_strings[key] = value;
                    break;
                }
                case OPT_TENSOR_MAP:
                    tensor_map_path = const_cast<char*>(v);
                    break;
                default:
                    break;
                }

                break;
            }
        }

        if (!matched)
        {
            fprintf(stderr, "Error: unknown option \"%s\".\n", arg);
            return 1;
        }
    }

    if (!input)
        fprintf(stderr, "Error: you must specify the input file path with --file or -f.\n");
    if (!output)
        fprintf(stderr, "Error: you must specify the output file path with --output-file or -o.\n");
    if (default_ftype == static_cast<ggml_ftype>(-2))
        fprintf(stderr, "Error: you must specify the quantization type with --type or -t.\n");
    if (!input || !output || default_ftype == static_cast<ggml_ftype>(-2))
        return 1;

    ggml_context* ctx;
    gguf_context_ptr input_gguf_ctx(gguf_init_from_file(input, gguf_init_params{ .no_alloc = false, .ctx = &ctx }));
    if (!input_gguf_ctx)
    {
        fprintf(stderr, "Error: failed to load the input file \"%s\".\n", input);
        fprintf(stderr, "Reason: %s\n", strerror(errno));
        return 1;
    }

    // Load tensor map
    std::vector<TensorMapEntry> tensor_map;
    if (tensor_map_path)
    {
        tensor_map = load_tensor_map(tensor_map_path);
        if (tensor_map.empty() && tensor_map_path)
            return 1;
    }

    // Resolve ftype per tensor:
    //   1. Literal (exact) patterns take priority over regex patterns.
    //   2. Among regex patterns, the longest textual pattern wins (most specific).
    //   3. Ties broken by JSON insertion order.
    auto resolve_ftype = [&](const char* name) -> ggml_ftype
    {
        int best = -1;
        bool best_is_literal = false;
        size_t best_len = 0;
        for (int i = 0; i < static_cast<int>(tensor_map.size()); i++)
        {
            if (!match_tensor_name(tensor_map[i].code.get(), name))
                continue;
            auto& entry = tensor_map[i];
            // Literal always beats regex
            if (!best_is_literal && entry.is_literal)
            {
                best = i;
                best_is_literal = true;
                best_len = entry.pattern.size();
                continue;
            }
            if (entry.is_literal)
                continue; // already have a literal, can't beat
            if (best_is_literal)
                continue; // literal already won
            // Both are regex: prefer longer pattern
            if (best < 0 || entry.pattern.size() > best_len)
            {
                best = i;
                best_len = entry.pattern.size();
            }
        }
        if (best >= 0)
        {
            tensor_map[best].matched = true;
            return tensor_map[best].ftype;
        }
        return default_ftype;
    };

    std::map<std::string, std::tuple<std::unique_ptr<char[]>, ggml_type, std::array<int64_t, GGML_MAX_DIMS>>> quantized_tensors;
    std::set<std::string> tensors_to_quantize;
    int64_t n_parameters = 0;

    for (int64_t i = 0; i != gguf_get_n_tensors(input_gguf_ctx.get()); ++i)
    {
        auto tensor = ggml_get_tensor(ctx, gguf_get_tensor_name(input_gguf_ctx.get(), i));
        auto tensor_name = tensor->name;
        auto cur_ftype = resolve_ftype(tensor_name);

        if (cur_ftype == GGML_FTYPE_UNKNOWN)
        {
            // COPY: must preserve the original data, allocate independent copy
            auto nbytes = ggml_nbytes(tensor);
            auto data = std::make_unique<char[]>(nbytes);
            memcpy(data.get(), tensor->data, nbytes);
            quantized_tensors[tensor_name] = std::make_tuple(std::move(data), tensor->type,
                *reinterpret_cast<std::array<int64_t, GGML_MAX_DIMS>*>(&tensor->ne));
            printf("Tensor \"%s\": keep as %s (mapped to COPY)\n", tensor_name, ggml_type_name(tensor->type));
        }
        else
        {
            // Quantize: must be F32
            if (tensor->type != GGML_TYPE_F32)
            {
                fprintf(stderr, "Error: tensor \"%s\" has type %s, only F32 tensors can be quantized.\n",
                    tensor_name, ggml_type_name(tensor->type));
                ggml_free(ctx);
                return 1;
            }

            ggml_type target_type = ggml_ftype_to_ggml_type(cur_ftype);
            const int64_t nelements = ggml_nelements(tensor);

            if (ggml_is_vector(tensor))
            {
                // Vectors are small: copy to avoid dangling ptr issues
                auto nbytes = ggml_nbytes(tensor);
                auto data = std::make_unique<char[]>(nbytes);
                memcpy(data.get(), tensor->data, nbytes);
                quantized_tensors[tensor_name] = std::make_tuple(std::move(data), tensor->type,
                    *reinterpret_cast<std::array<int64_t, GGML_MAX_DIMS>*>(&tensor->ne));
                printf("Tensor \"%s\": keep as F32 (vector, no quantization)\n", tensor_name);
            }
            else if (ggml_is_matrix(tensor))
            {
                // Steal pointer from ggml context – worker thread will replace data after quantization
                auto cur_type = get_fallback_type(target_type, tensor->ne[0]);
                if (cur_type != target_type)
                    printf("Warning: tensor \"%s\"'s ne0 is %" PRId64 ", not divisible by block size of %s, rolling back to %s.\n",
                        tensor_name, tensor->ne[0], ggml_type_name(target_type), ggml_type_name(cur_type));
                quantized_tensors[tensor_name] = std::make_tuple(
                    std::unique_ptr<char[]>(reinterpret_cast<char*>(tensor->data)),
                    cur_type,
                    *reinterpret_cast<std::array<int64_t, GGML_MAX_DIMS>*>(&tensor->ne));
                tensors_to_quantize.insert(tensor_name);
                n_parameters += nelements;
            }
            else
            {
                printf("Warning: tensor \"%s\" is not a matrix or vector. It will be quantized to F16.\n", tensor_name);
                quantized_tensors[tensor_name] = std::make_tuple(
                    std::unique_ptr<char[]>(reinterpret_cast<char*>(tensor->data)),
                    GGML_TYPE_F16,
                    *reinterpret_cast<std::array<int64_t, GGML_MAX_DIMS>*>(&tensor->ne));
                tensors_to_quantize.insert(tensor_name);
                n_parameters += nelements;
            }
        }
    }

    // Warn about unused tensor map patterns
    for (const auto& entry : tensor_map)
        if (!entry.matched)
            printf("Warning: tensor map pattern \"%s\" never matched any tensor.\n", entry.pattern.c_str());

    if (!tensors_to_quantize.empty())
    {
        printf("\nTotal number of quantized parameters: %" PRId64 "\n", n_parameters);
        auto n_threads = std::max(1u, std::thread::hardware_concurrency());
        auto threads = std::make_unique<std::thread[]>(n_threads);
        std::atomic_size_t idx = 0;
        std::atomic_size_t processed_parameters = 0;

        auto worker_thread = [&](auto report_progress)
        {
            auto iter = quantized_tensors.begin();
            size_t last = 0;
            for (;;)
            {
                size_t cur = idx++;
                if (cur >= quantized_tensors.size())
                    break;
                std::advance(iter, cur - last);
                last = cur;
                if (tensors_to_quantize.find(iter->first) == tensors_to_quantize.end())
                    continue;
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
            printf("\rQuantized %zu / %zu parameters (%.2f%%)", n_processed, static_cast<size_t>(n_parameters), 100.0 * n_processed / n_parameters);
        };

        for (auto j = 0u; j != n_threads; ++j)
            threads[j] = j == 0 ? std::thread(worker_thread, report_progress) : std::thread(worker_thread, []() {});
        for (auto j = 0u; j != n_threads; ++j)
            threads[j].join();

        report_progress();
        printf("\n");
    }
    else
        printf("No tensors to quantize (all tensors are copied as-is).\n");

    ggml_init_params params = { .mem_size = ggml_tensor_overhead() * quantized_tensors.size(), .no_alloc = true };
    ggml_context_ptr gguf_ggml_ctx(ggml_init(params));
    gguf_context_ptr output_gguf_ctx(gguf_init_empty());
    int64_t id = gguf_find_key(input_gguf_ctx.get(), "general.file_type");
    gguf_set_kv(output_gguf_ctx.get(), input_gguf_ctx.get());
    ggml_free(ctx);

    if (id != -1 && default_ftype != GGML_FTYPE_UNKNOWN)
        gguf_set_val_u32(output_gguf_ctx.get(), "general.file_type", default_ftype);

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

    if (!gguf_write_to_file(output_gguf_ctx.get(), output, false))
    {
        fprintf(stderr, "Error: failed to save the output file \"%s\".\n", output);
        fprintf(stderr, "Reason: %s\n", strerror(errno));
        return 1;
    }

    return 0;
}
