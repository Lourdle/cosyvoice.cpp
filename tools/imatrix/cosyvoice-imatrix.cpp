#ifdef _MSC_VER
    #define _CRT_SECURE_NO_WARNINGS
#endif

#include "tool_common.h"
#include "tool_common_cosyvoice.h"
#include "imatrix.h"

#include <ggml.h>
#include <ggml-backend.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <format>
#include <fstream>
#include <iterator>
#include <map>
#include <string>
#include <string_view>
#include <vector>

// ---------------------------------------------------------------------------
// Importance-matrix collection
// ---------------------------------------------------------------------------

namespace
{

// Skip tail-row batches and single-token decodes; too few rows to be useful.
constexpr int64_t kMinRows = 16;

struct collector
{
    std::map<std::string, imatrix_tensor_stats> stats;
    std::vector<float> act_buffer; // Host scratch for activation readbacks.
    uint32_t chunk_count = 0;
    // Collection gate: on during the LLM prefill and token2wav, off from the
    // first accepted token (decode steps only ever see single-row batches)
    // and back on for the explicit full-forward replay of the generated part.
    bool collect = true;
};

// Scheduler copies of weights are named "<backend>#<name>#<copy>"; strip the
// decoration back to the original tensor name.
std::string_view filter_tensor_name(std::string_view name)
{
    const auto first = name.find('#');
    if (first == std::string_view::npos)
        return name;
    const auto last = name.rfind('#');
    if (first == last)
        return name.substr(first + 1);
    return name.substr(first + 1, last - first - 1);
}

bool collect_eval_callback(ggml_tensor* t, bool ask, void* user_data)
{
    auto& c = *static_cast<collector*>(user_data);

    if (!ask)
    {
        // The node was just evaluated; accumulate the per-column sums of
        // squared activations of its input.
        const auto* src0 = t->src[0];
        const auto* src1 = t->src[1];
        if (!src0 || !src1 || t->op != GGML_OP_MUL_MAT || src1->type != GGML_TYPE_F32)
            return true;

        const auto name = std::string(filter_tensor_name(src0->name));
        const auto ncols = static_cast<size_t>(src1->ne[0]);

        auto& entry = c.stats[name];
        if (entry.in_sum2.empty())
        {
            entry.in_sum2.assign(ncols, 0.f);
            entry.in_counts.assign(ncols, 0.f);
        }
        if (entry.in_sum2.size() != ncols)
            return true; // Shape changed; skip defensively.

        const auto nrows = src1->ne[1] * src1->ne[2] * src1->ne[3];
        if (nrows <= 0)
            return true;

        c.act_buffer.resize(ggml_nbytes(src1) / sizeof(float));
        ggml_backend_tensor_get(src1, c.act_buffer.data(), 0, ggml_nbytes(src1));

        const auto* x = c.act_buffer.data();
        for (int64_t k = 0; k < src1->ne[3]; ++k)
            for (int64_t j = 0; j < src1->ne[2]; ++j)
                for (int64_t i = 0; i < src1->ne[1]; ++i)
                {
                    const auto* row = x + (k * src1->nb[3] + j * src1->nb[2] + i * src1->nb[1]) / sizeof(float);
                    for (size_t col = 0; col < ncols; ++col)
                    {
                        const float v = row[col];
                        entry.in_sum2[col] += v * v;
                    }
                }
        for (auto& count : entry.in_counts)
            count += static_cast<float>(nrows);

        return true;
    }

    if (t->op != GGML_OP_MUL_MAT)
        return false;

    const auto* src0 = t->src[0];
    const auto* src1 = t->src[1];
    if (!src0 || !src1)
        return false;

    // Only linear layers matter: src0 must be a loaded weight (tensor names of
    // weights end in ".weight"; scheduler copies keep the suffix) and src1 must
    // be an F32 activation batch with enough rows.
    if (src1->type != GGML_TYPE_F32 || src1->ne[1] < kMinRows)
        return false;

    return c.collect && filter_tensor_name(src0->name).ends_with(".weight");
}

} // namespace

// ---------------------------------------------------------------------------
// Stage callback for full-mode collection
// ---------------------------------------------------------------------------

namespace
{

// Generation collects the LLM prefill, pauses collection from the first
// accepted token (decode steps only ever see single-row batches), and collects
// the flow/DiT tensors during token2wav. The KV length saved at the first token
// marks the boundary between the prompt and the generated tokens; run_full()
// replays the generated part from there with collection enabled, so the final
// layer and the output projection contribute statistics too.
struct stage_state
{
    cosyvoice_context_t ctx = nullptr;
    collector* col = nullptr;
    uint32_t max_new_tokens = 0;
    bool first_token_seen = false;
    uint32_t prompt_kv_len = 0;
};

bool stage_callback(cosyvoice_stage_t stage, int32_t value, uint32_t /*worker_no*/, void* user_data)
{
    auto& st = *static_cast<stage_state*>(user_data);

    switch (stage)
    {
    case COSYVOICE_STAGE_LLM_BEGIN:
        st.col->collect = true;
        st.first_token_seen = false;
        st.prompt_kv_len = 0;
        return true;

    case COSYVOICE_STAGE_LLM_TOKEN:
        // The first accepted token marks the start of the generated part; the
        // KV cache still holds exactly the prompt at this point, so save its
        // length as the replay boundary. Collection pauses from here on:
        // decode steps only ever see single-row batches.
        if (!st.first_token_seen)
        {
            st.first_token_seen = true;
            st.prompt_kv_len = cosyvoice_llm_get_kv_cache_len(st.ctx);
        }
        st.col->collect = false;
        // Token budget: stop generation once the accepted count reaches the cap.
        return st.max_new_tokens == 0 || static_cast<uint32_t>(value) < st.max_new_tokens;

    case COSYVOICE_STAGE_TOKEN2WAV_BEGIN:
        st.col->collect = true;
        return true;

    default:
        return true;
    }
}

} // namespace

// ---------------------------------------------------------------------------
// CLI options
// ---------------------------------------------------------------------------

namespace
{

enum OptionID {
    OPT_HELP,
    OPT_MODEL,
    OPT_OUTPUT_FILE,
    OPT_BACKEND,
    OPT_THREADS,
    OPT_SEED,
    OPT_TEXT,
    OPT_TEXT_FILE,
    OPT_CORPUS_FILE,
    OPT_PROMPT_SPEECH,
    OPT_LLM_ONLY,
    OPT_CHUNK_SIZE,
    OPT_MAX_NEW_TOKENS,
    OPT_COUNT
};

struct OptionDef {
    const char* long_name;
    char short_name;
    int num_args;
    const char* help;
};

static const OptionDef s_options[OPT_COUNT] = {
    { "--help",           'h', 0, "Show this help message and exit." },
    { "--model",          'm', 1, "Input F32 GGUF model file path." },
    { "--output-file",    'o', 1, "Output importance-matrix GGUF file path. Default: imatrix.gguf" },
    { "--backend",        'b', 1, "GGML backend name. Default: auto (best available)." },
    { "--threads",        't', 1, "CPU threads for the backend. Default: 4." },
    { "--seed",           's', 1, "Sampler and noise generator seed. Default: 42." },
    { "--text",           'T', 1, "Calibration text, zero-shot (repeatable)." },
    { "--text-file",      'F', 1, "Calibration text file, one zero-shot sample per non-empty line (repeatable)." },
    { "--corpus-file",    'C', 1, "Calibration corpus file (repeatable). TSV lines: 'zs<TAB>text', 'xl<TAB>text' or 'instr<TAB>instruction<TAB>text'. UTF-8, '#' comments allowed." },
    { "--prompt-speech",  'p', 1, "Prompt speech GGUF file path. Required unless --llm-only is given." },
    { "--llm-only",       'l', 0, "Only collect LLM statistics via fast prefill passes (no Flow/DiT statistics)." },
    { "--chunk-size",     'c', 1, "Tokens per LLM prefill chunk in --llm-only mode. Default: the model context batch size." },
    { "--max-new-tokens", 'n', 1, "Cap on generated speech tokens per sample in full mode. Default: 400." },
};

static void print_help(const char* program)
{
    printf("cosyvoice-imatrix - a tool for collecting importance matrices for quantization.\n");
    printf("Usage: %s [options]\n", program);
    printf("\nOptions:\n");
    for (int i = 0; i < OPT_COUNT; i++)
    {
        const auto& opt = s_options[i];
        printf("  %s, -%c", opt.long_name, opt.short_name);
        for (int a = 0; a < opt.num_args; a++)
            printf(" <arg>");
        printf("%-2s %s\n", "", opt.help);
    }
    printf("\nExamples:\n");
    printf("  %s -m model.gguf -p prompt_speech.gguf -F calib.txt -o imatrix.gguf\n", program);
    printf("  %s -m model.gguf -p prompt_speech.gguf -C corpus.tsv -o imatrix.gguf\n", program);
    printf("  %s -m model.gguf -F calib.txt --llm-only -o imatrix-llm.gguf\n", program);
}

// Calibration sample modes. Modes mirror the high-level TTS behavior.
enum class sample_mode
{
    zero_shot,
    cross_lingual,
    instruct
};

struct calibration_sample
{
    sample_mode mode = sample_mode::zero_shot;
    std::string instruction;
    std::string text;
};

const char* sample_mode_name(sample_mode mode)
{
    switch (mode)
    {
    case sample_mode::zero_shot: return "zs";
    case sample_mode::cross_lingual: return "xl";
    case sample_mode::instruct: return "instr";
    }
    return "?";
}

bool parse_sample_mode(std::string_view token, sample_mode& mode)
{
    if (token == "zs" || token == "zero" || token == "zero-shot")
    {
        mode = sample_mode::zero_shot;
        return true;
    }
    if (token == "xl" || token == "cross" || token == "cross-lingual")
    {
        mode = sample_mode::cross_lingual;
        return true;
    }
    if (token == "instr" || token == "instruct")
    {
        mode = sample_mode::instruct;
        return true;
    }
    return false;
}

struct options
{
    const char* model = nullptr;
    const char* output = "imatrix.gguf";
    std::string backend = "auto";
    uint32_t threads = 4;
    uint32_t seed = 42;
    std::vector<calibration_sample> samples;
    std::vector<std::string> text_files;
    std::vector<std::string> corpus_files;
    const char* prompt_speech = nullptr;
    bool llm_only = false;
    uint32_t chunk_size = 0;    // 0: use the model context batch size.
    uint32_t max_new_tokens = 400;
};

// Reads a UTF-8 text file and returns every line (including empty ones) with a
// leading BOM and trailing '\r' removed.
bool read_text_file(const char* path, std::vector<std::string>& lines, std::string& error)
{
    auto in = open_ifstream_utf8(path, std::ios::binary);
    if (!in)
    {
        error = std::format("failed to open the text file \"{}\"", path);
        return false;
    }

    std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());

    // Strip a UTF-8 BOM at the start of the file.
    if (content.size() >= 3
        && static_cast<unsigned char>(content[0]) == 0xEF
        && static_cast<unsigned char>(content[1]) == 0xBB
        && static_cast<unsigned char>(content[2]) == 0xBF)
        content.erase(0, 3);

    size_t start = 0;
    while (start <= content.size())
    {
        auto end = content.find('\n', start);
        if (end == std::string::npos)
            end = content.size();
        auto line = std::string_view(content).substr(start, end - start);
        if (!line.empty() && line.back() == '\r')
            line.remove_suffix(1);
        lines.emplace_back(line);
        if (end == content.size())
            break;
        start = end + 1;
    }

    if (!lines.empty() && lines.back().empty())
        lines.pop_back();

    return true;
}

// Appends one zero-shot sample per non-empty line.
bool load_text_samples(const char* path, std::vector<calibration_sample>& out, std::string& error)
{
    std::vector<std::string> lines;
    if (!read_text_file(path, lines, error))
        return false;

    size_t added = 0;
    for (auto& line : lines)
    {
        if (line.empty())
            continue;
        out.push_back({ sample_mode::zero_shot, {}, std::move(line) });
        ++added;
    }

    if (added == 0)
    {
        error = std::format("no non-empty lines found in the text file \"{}\"", path);
        return false;
    }

    return true;
}

// Appends samples from a TSV corpus file:
//   zs<TAB>text
//   xl<TAB>text
//   instr<TAB>instruction<TAB>text
// Empty lines and lines starting with '#' are ignored.
bool load_corpus_samples(const char* path, std::vector<calibration_sample>& out, std::string& error)
{
    std::vector<std::string> lines;
    if (!read_text_file(path, lines, error))
        return false;

    std::vector<std::string_view> fields;
    size_t added = 0;
    for (size_t i = 0; i != lines.size(); ++i)
    {
        std::string_view line = lines[i];
        if (line.empty() || line.front() == '#')
            continue;

        fields.clear();
        size_t pos = 0;
        while (true)
        {
            const auto tab = line.find('\t', pos);
            fields.push_back(line.substr(pos, tab == std::string_view::npos ? std::string_view::npos : tab - pos));
            if (tab == std::string_view::npos)
                break;
            pos = tab + 1;
        }

        calibration_sample sample;
        if (!parse_sample_mode(fields[0], sample.mode))
        {
            error = std::format("{}:{}: unsupported mode \"{}\" (expected zs, xl or instr)", path, i + 1, fields[0]);
            return false;
        }

        if (fields.size() == 2)
        {
            if (sample.mode == sample_mode::instruct)
            {
                error = std::format("{}:{}: instruct lines must be \"instr<TAB>instruction<TAB>text\"", path, i + 1);
                return false;
            }
            sample.text = fields[1];
        }
        else if (fields.size() == 3)
        {
            if (sample.mode != sample_mode::instruct)
            {
                error = std::format("{}:{}: only instruct lines may carry an instruction field", path, i + 1);
                return false;
            }
            sample.instruction = fields[1];
            sample.text = fields[2];
        }
        else
        {
            error = std::format("{}:{}: expected 2 or 3 tab-separated fields, got {}", path, i + 1, fields.size());
            return false;
        }

        if (sample.text.empty())
        {
            error = std::format("{}:{}: empty text", path, i + 1);
            return false;
        }
        if (sample.mode == sample_mode::instruct && sample.instruction.empty())
        {
            error = std::format("{}:{}: empty instruction", path, i + 1);
            return false;
        }

        out.push_back(std::move(sample));
        ++added;
    }

    if (added == 0)
    {
        error = std::format("no samples found in the corpus file \"{}\"", path);
        return false;
    }

    return true;
}

} // namespace

// ---------------------------------------------------------------------------
// Collection drivers
// ---------------------------------------------------------------------------

namespace
{

bool run_llm_only(const options& opt, cosyvoice_context_t ctx, collector& col, uint32_t& chunk_size_out)
{
    const auto tokenizer = cosyvoice_get_tokenizer(ctx);
    const auto result = cosyvoice_tokenization_result_create();
    const auto* emb = cosyvoice_get_word_token_embed_weight(ctx);

    cosyvoice_context_params params = {};
    cosyvoice_get_context_params(ctx, &params);
    auto chunk_size = opt.chunk_size != 0 ? opt.chunk_size : params.n_batch;
    chunk_size = std::min<uint32_t>(chunk_size, params.n_batch);
    if (params.n_max_seq > 1)
        chunk_size = std::min<uint32_t>(chunk_size, params.n_max_seq - 1);
    if (chunk_size == 0)
        chunk_size = params.n_batch;
    chunk_size_out = chunk_size;

    const auto row_bytes = emb->nb[1];
    const auto emb_host = emb->buffer && ggml_backend_buffer_is_host(emb->buffer);
    std::vector<char> rows;

    bool ok = true;
    size_t sample_index = 0;
    for (const auto& sample : opt.samples)
    {
        const auto& text = sample.text;
        const auto t0 = std::chrono::steady_clock::now();
        const auto n_tokens = cosyvoice_tokenize_ext(tokenizer, text.c_str(), static_cast<uint32_t>(text.size()), result, true);
        const auto* tokens = cosyvoice_tokenization_result_get_tokens(result);

        for (uint32_t offset = 0; offset < n_tokens; offset += chunk_size)
        {
            const auto n = std::min(chunk_size, n_tokens - offset);
            if (!cosyvoice_llm_set_kv_cache_len(ctx, 0))
            {
                printf("cosyvoice_llm_set_kv_cache_len failed.\n");
                ok = false;
                break;
            }

            rows.resize(n * row_bytes);
            for (uint32_t i = 0; i != n; ++i)
            {
                const auto row_offset = static_cast<size_t>(tokens[offset + i]) * row_bytes;
                if (emb_host)
                    memcpy(rows.data() + i * row_bytes, static_cast<const char*>(emb->data) + row_offset, row_bytes);
                else
                    ggml_backend_tensor_get(emb, rows.data() + i * row_bytes, row_offset, row_bytes);
            }

            if (!cosyvoice_llm_prefill(ctx, emb->type, rows.data(), n))
            {
                printf("cosyvoice_llm_prefill failed.\n");
                ok = false;
                break;
            }
            ++col.chunk_count;
        }

        printf("[%zu/%zu] %s %u tokens (%.1fs)\n", ++sample_index, opt.samples.size(),
            sample_mode_name(sample.mode), n_tokens,
            elapsed_ms(t0, std::chrono::steady_clock::now()) / 1000.0);
        if (!ok)
            break;
    }

    cosyvoice_tokenization_result_free(result);
    return ok;
}

bool reset_llm_state(cosyvoice_context_t ctx)
{
    cosyvoice_llm_clear_accepted_tokens(ctx);
    if (!cosyvoice_llm_set_kv_cache_len(ctx, 0))
    {
        printf("cosyvoice_llm_set_kv_cache_len failed.\n");
        return false;
    }
    return true;
}

bool run_full(const options& opt, cosyvoice_context_t ctx, collector& col)
{
    const cosyvoice_prompt_speech_handle ps(cosyvoice_prompt_speech_load_from_file(opt.prompt_speech));
    if (!ps)
    {
        printf("failed to load the prompt speech file \"%s\".\n", opt.prompt_speech);
        return false;
    }

    const cosyvoice_prompt_handle prompt(cosyvoice_prompt_init_from_prompt_speech(ctx, ps.get()));
    if (!prompt)
    {
        printf("failed to initialize a prompt from \"%s\".\n", opt.prompt_speech);
        return false;
    }

    const cosyvoice_tts_context_handle tts_ctx(cosyvoice_tts_context_new(ctx, prompt.get()));
    if (!tts_ctx)
    {
        printf("failed to create a TTS context.\n");
        return false;
    }

    // One corpus line is one generation: disable text splitting. Text
    // normalization stays enabled, matching the previous per-line tokenization.
    cosyvoice_tts_context_set_split_text_enabled(tts_ctx.get(), false);

    stage_state st;
    st.ctx = ctx;
    st.col = &col;
    st.max_new_tokens = opt.max_new_tokens;
    cosyvoice_set_stage_callback(ctx, stage_callback, &st);

    size_t sample_index = 0;
    for (const auto& sample : opt.samples)
    {
        if (!reset_llm_state(ctx))
            return false;
        st.first_token_seen = false;
        st.prompt_kv_len = 0;

        printf("[%zu/%zu] %s: ", ++sample_index, opt.samples.size(), sample_mode_name(sample.mode));
        fflush(stdout);
        const auto t0 = std::chrono::steady_clock::now();

        cosyvoice_generated_speech speech = {};
        bool ok = false;
        switch (sample.mode)
        {
        case sample_mode::zero_shot:
            ok = cosyvoice_tts_zero_shot(tts_ctx.get(), sample.text.c_str(), 1.0f, &speech);
            break;
        case sample_mode::cross_lingual:
            ok = cosyvoice_tts_cross_lingual(tts_ctx.get(), sample.text.c_str(), 1.0f, &speech);
            break;
        case sample_mode::instruct:
            ok = cosyvoice_tts_instruct(tts_ctx.get(), sample.text.c_str(), sample.instruction.c_str(), 1.0f, &speech);
            break;
        }

        const auto n_generated = cosyvoice_llm_get_n_accepted_tokens(ctx);

        // Replay the generated tokens through every layer with collection on:
        // truncate the KV cache to the prompt boundary saved at the first
        // token, materialize the accepted tokens' embedding rows, and run them
        // in batch-sized slices. The final layer's attention/FFN and the output
        // projection only ever run with single-row batches during generation,
        // so their statistics come from this pass.
        if (ok && n_generated != 0)
        {
            if (!cosyvoice_llm_set_kv_cache_len(ctx, st.prompt_kv_len))
            {
                printf("cosyvoice_llm_set_kv_cache_len failed.\n");
                return false;
            }

            const auto* speech_table = cosyvoice_get_speech_token_embed_weight(ctx);
            if (!speech_table)
            {
                printf("failed to get the speech-token embedding table.\n");
                return false;
            }
            const auto row_size = static_cast<size_t>(speech_table->nb[1]);
            const auto* ids = cosyvoice_llm_get_accepted_tokens(ctx);
            std::vector<char> rows(static_cast<size_t>(n_generated) * row_size);
            for (uint32_t i = 0; i != n_generated; ++i)
            {
                const auto id = ids[i];
                if (id < 0 || static_cast<uint64_t>(id) >= static_cast<uint64_t>(speech_table->ne[1]))
                {
                    printf("speech token id %d out of range.\n", id);
                    return false;
                }
                memcpy(rows.data() + static_cast<size_t>(i) * row_size,
                    static_cast<const char*>(speech_table->data) + static_cast<size_t>(id) * row_size,
                    row_size);
            }

            cosyvoice_context_params_t params;
            cosyvoice_get_context_params(ctx, &params);
            col.collect = true;
            for (uint32_t off = 0; off < n_generated; )
            {
                const auto chunk = std::min<uint32_t>(n_generated - off, params.n_batch);
                if (!cosyvoice_llm_full_forward(ctx, speech_table->type,
                        rows.data() + static_cast<size_t>(off) * row_size, chunk, nullptr, 0, nullptr))
                {
                    printf("cosyvoice_llm_full_forward failed.\n");
                    return false;
                }
                off += chunk;
            }
        }

        printf("%u speech tokens (%.1fs)\n", n_generated, elapsed_ms(t0, std::chrono::steady_clock::now()) / 1000.0);
        if (!ok)
        {
            printf("TTS generation failed.\n");
            return false;
        }
        ++col.chunk_count;
    }

    cosyvoice_set_stage_callback(ctx, nullptr, nullptr);
    return true;
}

} // namespace

int tool_entry(int argc, char** argv)
{
    options opt;

    auto get_required_arg = [&](int& i, const char* opt_name) -> const char*
    {
        if (++i >= argc)
        {
            fprintf(stderr, "Error: option \"%s\" requires an argument.\n", opt_name);
            exit(1);
        }
        return argv[i];
    };

    auto parse_uint32 = [&](const char* v, const char* opt_name, uint32_t& out) -> bool
    {
        if (!parse_uint32_arg(v, &out))
        {
            fprintf(stderr, "Error: option \"%s\" requires a 32-bit unsigned integer, got \"%s\".\n", opt_name, v);
            return false;
        }
        return true;
    };

    for (int i = 1; i < argc; i++)
    {
        const char* arg = argv[i];
        bool matched = false;

        for (int opt_id = 0; opt_id < OPT_COUNT; opt_id++)
        {
            const auto& def = s_options[opt_id];
            auto short_form = std::string("-") + def.short_name;
            if (str_casecmp(arg, def.long_name) == 0 || arg == short_form)
            {
                matched = true;

                if (opt_id == OPT_HELP)
                {
                    print_help(argv[0]);
                    return 0;
                }

                if (def.num_args < 1)
                {
                    if (opt_id == OPT_LLM_ONLY)
                        opt.llm_only = true;
                    continue;
                }

                const char* v = get_required_arg(i, arg);

                switch (opt_id)
                {
                case OPT_MODEL:
                    opt.model = v;
                    break;
                case OPT_OUTPUT_FILE:
                    opt.output = v;
                    break;
                case OPT_BACKEND:
                    opt.backend = v;
                    break;
                case OPT_THREADS:
                    if (!parse_uint32(v, arg, opt.threads)) return 1;
                    if (opt.threads == 0) opt.threads = 1;
                    break;
                case OPT_SEED:
                    if (!parse_uint32(v, arg, opt.seed)) return 1;
                    break;
                case OPT_TEXT:
                    opt.samples.push_back({ sample_mode::zero_shot, {}, v });
                    opt.text_files.push_back("inline");
                    break;
                case OPT_TEXT_FILE:
                    opt.text_files.push_back(v);
                    break;
                case OPT_CORPUS_FILE:
                    opt.corpus_files.push_back(v);
                    break;
                case OPT_PROMPT_SPEECH:
                    opt.prompt_speech = v;
                    break;
                case OPT_CHUNK_SIZE:
                    if (!parse_uint32(v, arg, opt.chunk_size)) return 1;
                    break;
                case OPT_MAX_NEW_TOKENS:
                    if (!parse_uint32(v, arg, opt.max_new_tokens)) return 1;
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

    if (!opt.model)
        fprintf(stderr, "Error: you must specify the model file path with --model or -m.\n");
    if (opt.samples.empty() && opt.text_files.empty() && opt.corpus_files.empty())
        fprintf(stderr, "Error: you must specify calibration text with --text, --text-file or --corpus-file.\n");
    if (!opt.llm_only && !opt.prompt_speech)
        fprintf(stderr, "Error: --prompt-speech is required in full mode (or pass --llm-only).\n");
    if (opt.llm_only && opt.prompt_speech)
        fprintf(stderr, "Error: --prompt-speech is ignored with --llm-only; remove one of the conflicting options.\n");
    if (!opt.model || (opt.samples.empty() && opt.text_files.empty() && opt.corpus_files.empty())
        || (!opt.llm_only && !opt.prompt_speech) || (opt.llm_only && opt.prompt_speech))
        return 1;

    std::string error;
    for (const auto& path : opt.text_files)
    {
        if (path == "inline")
            continue;
        if (!load_text_samples(path.c_str(), opt.samples, error))
        {
            fprintf(stderr, "Error: %s.\n", error.c_str());
            return 1;
        }
    }
    for (const auto& path : opt.corpus_files)
    {
        if (!load_corpus_samples(path.c_str(), opt.samples, error))
        {
            fprintf(stderr, "Error: %s.\n", error.c_str());
            return 1;
        }
    }
    if (opt.samples.empty())
    {
        fprintf(stderr, "Error: no calibration text samples found.\n");
        return 1;
    }

    printf("cosyvoice-imatrix: model=%s backend=%s mode=%s samples=%zu seed=%u\n",
        opt.model, opt.backend.c_str(), opt.llm_only ? "llm-only" : "full", opt.samples.size(), opt.seed);

    cosyvoice_init_backend_from_path(nullptr);
    ggml_backend_t backend = nullptr;
    if (opt.backend == "auto")
        backend = ggml_backend_init_best();
    else
        backend = ggml_backend_init_by_name(opt.backend.c_str(), nullptr);
    if (!backend)
    {
        fprintf(stderr, "Error: backend \"%s\" unavailable.\n", opt.backend.c_str());
        return 1;
    }

    cosyvoice_context_params_v4_cpp params_v4 = {};
    cosyvoice_init_default_context_params(&params_v4);
    params_v4.inference_buffer_policy = COSYVOICE_INFERENCE_BUFFER_POLICY_DEDICATED;
    params_v4.n_workers = 1;
    params_v4.seed = opt.seed;
    params_v4.diffusion_steps = 4;

    const cosyvoice_context_handle ctx(cosyvoice_load_from_file_ext(opt.model, &params_v4, backend, opt.threads));
    if (!ctx)
    {
        fprintf(stderr, "Error: failed to load the model \"%s\".\n", opt.model);
        return 1;
    }

    collector col;
    cosyvoice_set_sched_eval_callback(ctx.get(), collect_eval_callback, &col);

    const auto t0 = std::chrono::steady_clock::now();

    uint32_t chunk_size = 0;
    bool ok = opt.llm_only
        ? run_llm_only(opt, ctx.get(), col, chunk_size)
        : run_full(opt, ctx.get(), col);
    if (!ok)
        return 1;

    const auto total_s = elapsed_ms(t0, std::chrono::steady_clock::now()) / 1000.0;

    printf("\nCollected statistics for %zu tensors (%u chunks, %.1fs):\n",
        col.stats.size(), col.chunk_count, total_s);
    for (const auto& [name, entry] : col.stats)
        printf("  %s: ne0=%zu rows=%llu\n", name.c_str(), entry.in_sum2.size(),
            static_cast<unsigned long long>(entry.in_counts.empty() ? 0 : entry.in_counts.front()));

    std::vector<std::string> datasets;
    for (const auto* paths : { &opt.text_files, &opt.corpus_files })
        for (const auto& path : *paths)
            if (std::find(datasets.begin(), datasets.end(), path) == datasets.end())
                datasets.push_back(path);

    if (!write_imatrix(opt.output, col.stats, datasets, col.chunk_count, chunk_size, error))
    {
        fprintf(stderr, "Error: %s.\n", error.c_str());
        return 1;
    }
    printf("Saved the importance matrix to \"%s\".\n", opt.output);

    return 0;
}
