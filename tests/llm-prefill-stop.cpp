// Model-backed regression for stopping the LLM while it prefills a prompt.
//
// A long text (thousands of tokens) makes the prefill span several n_batch
// batches. A worker thread requests a stop while the prefill is running. The
// stopped job must fail without generating tokens and must leave the KV cache
// in a clean state: decoding the same short prompt afterwards has to produce
// exactly the tokens it produced before the stop.
//
// Usage (positional; leave trailing arguments out to take defaults):
//   test-llm-prefill-stop [model.gguf] [backend] [prompt_speech.gguf] [threads]
//
//   model           GGUF model to load.                    default: CosyVoice3-2512_F32.gguf
//   backend         Backend name (CUDA0/CPU/MTL0).          default: CUDA0
//   prompt          Path to prompt_speech.gguf.             default: prompt_speech.gguf
//   threads         CPU threads for the backend.            default: 4
#include "tool_common_cosyvoice.h"
#include <ggml-backend.h>

#include <atomic>
#include <charconv>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <thread>
#include <vector>

namespace
{
struct options
{
    const char* model = "CosyVoice3-2512_F32.gguf";
    const char* device = "CUDA0";
    const char* prompt_speech = "prompt_speech.gguf";
    uint32_t threads = 4;
};

template <typename T>
bool parse(const char* s, T& out)
{
    if (!s || !*s) return false;
    const char* end = s + std::strlen(s);
    const auto [ptr, ec] = std::from_chars(s, end, out);
    return ec == std::errc{} && ptr == end;
}

int run(const options& opt)
{
    cosyvoice_context_params_v4_cpp params = {};
    cosyvoice_init_default_context_params(&params);
    params.n_max_seq = 8192;
    params.n_batch = 256;
    params.strict_seed_mode = true;
    params.seed = 42;

    ggml_backend_t backend = ggml_backend_init_by_name(opt.device, nullptr);
    if (!backend)
    {
        std::fprintf(stderr, "backend %s unavailable\n", opt.device);
        return 2;
    }

    cosyvoice_context_handle ctx(cosyvoice_load_from_file_ext(opt.model, &params, backend, opt.threads));
    if (!ctx)
    {
        std::fprintf(stderr, "failed to load %s\n", opt.model);
        return 2;
    }

    cosyvoice_prompt_speech_handle prompt_speech(cosyvoice_prompt_speech_load_from_file(opt.prompt_speech));
    if (!prompt_speech)
    {
        std::fprintf(stderr, "failed to load %s\n", opt.prompt_speech);
        return 2;
    }
    cosyvoice_prompt_handle prompt(cosyvoice_prompt_init_from_prompt_speech(ctx.get(), prompt_speech.get()));
    if (!prompt)
    {
        std::fprintf(stderr, "failed to load %s\n", opt.prompt_speech);
        return 2;
    }

    // Token ids are fabricated: the test only needs deterministic text input,
    // so any ids inside the model vocabulary work.
    const int short_text[] = { 100, 101, 102, 103 };
    std::vector<int> long_text(7000);
    for (size_t i = 0; i != long_text.size(); ++i)
        long_text[i] = 100 + static_cast<int>(i % 500);

    auto decode_short = [&](std::vector<int>& out) -> bool
    {
        cosyvoice_llm_clear_accepted_tokens(ctx.get());
        cosyvoice_set_sampler_seed(ctx.get(), 42);
        const bool ok = cosyvoice_llm_job_ext(
            ctx.get(), short_text, static_cast<uint32_t>(std::size(short_text)), prompt.get(), 8, nullptr);
        const auto n = cosyvoice_llm_get_n_accepted_tokens(ctx.get());
        const auto* tokens = cosyvoice_llm_get_accepted_tokens(ctx.get());
        out.assign(tokens, tokens + n);
        return ok;
    };

    int rc = 1;
    std::vector<int> baseline;
    std::vector<int> after;
    do
    {
        cosyvoice_llm_set_kv_cache_len(ctx.get(), 0);
        if (!decode_short(baseline) || baseline.empty())
        {
            std::printf("FAIL: baseline decode failed\n");
            break;
        }
        std::printf("baseline: %zu tokens\n", baseline.size());

        cosyvoice_llm_clear_accepted_tokens(ctx.get());
        cosyvoice_llm_set_kv_cache_len(ctx.get(), 0);

        std::atomic<bool> started { false };
        std::thread stopper([&]()
        {
            while (!started.load(std::memory_order_acquire))
                std::this_thread::yield();
            // The long prefill takes far longer than this on every backend.
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            cosyvoice_request_stop(ctx.get());
        });

        started.store(true, std::memory_order_release);
        const bool completed = cosyvoice_llm_job_ext(
            ctx.get(), long_text.data(), static_cast<uint32_t>(long_text.size()), prompt.get(), 8, nullptr);
        stopper.join();

        if (completed)
        {
            std::printf("FAIL: long prefill completed before the stop was requested\n");
            break;
        }
        if (cosyvoice_llm_get_n_accepted_tokens(ctx.get()) != 0)
        {
            std::printf("FAIL: stopped prefill accepted %u tokens\n", cosyvoice_llm_get_n_accepted_tokens(ctx.get()));
            break;
        }
        const auto kv_len = cosyvoice_llm_get_kv_cache_len(ctx.get());
        if (kv_len != 0)
        {
            std::printf("FAIL: stopped prefill left %u KV cache entries\n", kv_len);
            break;
        }
        std::printf("stopped prefill: KV cache cleared\n");

        // Deliberately do not reset the KV cache here: the stop handler must
        // have done it already.
        if (!decode_short(after))
        {
            std::printf("FAIL: decode after the stop failed\n");
            break;
        }
        if (after != baseline)
        {
            std::printf("FAIL: tokens changed after the prefill stop (%zu vs %zu tokens)\n",
                after.size(), baseline.size());
            break;
        }

        std::printf("PASS\n");
        rc = 0;
    } while (false);

    return rc;
}
}

int tool_entry(int argc, char** argv)
{
    options opt;
    if (argc > 1 && *argv[1]) opt.model = argv[1];
    if (argc > 2 && *argv[2]) opt.device = argv[2];
    if (argc > 3 && *argv[3]) opt.prompt_speech = argv[3];
    if (argc > 4 && *argv[4] && !parse(argv[4], opt.threads))
    {
        std::fprintf(stderr, "invalid thread count: %s\n", argv[4]);
        return 2;
    }

    cosyvoice_init_backend_from_path(nullptr);
    return run(opt);
}
