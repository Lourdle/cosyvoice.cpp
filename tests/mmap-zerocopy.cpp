// CPU mmap zero-copy weights regression.
//
// The runtime uses the model file mapping in place on CPU backends, so the
// weights are not copied into a separate CPU buffer. This test loads the model
// twice, once through that path and once with COSYVOICE_NO_MMAP_ZEROCOPY=1
// forcing the copy path, and checks that both produce bit-identical audio for
// the same text and seed.
//
// Usage (positional; leave trailing arguments out to take defaults):
//   test-mmap-zerocopy [model.gguf] [backend] [prompt_speech.gguf] TEXT [threads]
//
//   model           GGUF model to load.                    default: CosyVoice3-2512_F32.gguf
//   backend         Backend name.                          default: CPU
//   prompt          Path to prompt_speech.gguf.            default: prompt_speech.gguf
//   text            Text to synthesize (required).
//   threads         CPU threads for the backend.           default: 8
#include "cosyvoice.h"
#include "cosyvoice-lowlevel.h"
#include "tool_common.h"
#include <ggml-backend.h>

#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <system_error>
#include <vector>

namespace
{
struct options
{
    const char* model = "CosyVoice3-2512_F32.gguf";
    const char* device = "CPU";
    const char* prompt_speech = "prompt_speech.gguf";
    const char* text = nullptr;
    uint32_t threads = 8;
};

struct captured
{
    std::vector<float> pcm;
    size_t samples = 0;
};

bool collect(const float* audio, uint32_t n, void* user_data)
{
    auto& c = *static_cast<captured*>(user_data);
    c.pcm.insert(c.pcm.end(), audio, audio + n);
    c.samples += n;
    return true;
}

struct result
{
    std::vector<float> pcm;
    size_t parameters = 0;
};

void set_env(const char* name, const char* value)
{
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value)
        setenv(name, value, 1);
    else
        unsetenv(name);
#endif
}

bool run(const options& opt, result& out)
{
    cosyvoice_context_params_v4_cpp params = {};
    cosyvoice_init_default_context_params(&params);
    params.n_max_seq = 256;
    params.n_batch = 256;
    params.diffusion_steps = 4;
    params.strict_seed_mode = true;
    params.seed = 42;
    params.dit_kv_cache_length = 512;

    ggml_backend_t backend = ggml_backend_init_by_name(opt.device, nullptr);
    if (!backend)
    {
        std::fprintf(stderr, "backend %s unavailable\n", opt.device);
        return false;
    }

    const cosyvoice_context_t ctx = cosyvoice_load_from_file_ext(opt.model, &params, backend, opt.threads);
    if (!ctx)
    {
        std::fprintf(stderr, "failed to load %s\n", opt.model);
        return false;
    }

    const cosyvoice_prompt_speech_t ps = cosyvoice_prompt_speech_load_from_file(opt.prompt_speech);
    cosyvoice_prompt_t prompt = ps ? cosyvoice_prompt_init_from_prompt_speech(ctx, ps) : nullptr;
    if (ps) cosyvoice_prompt_speech_free(ps);
    if (!prompt)
    {
        std::fprintf(stderr, "failed to load %s\n", opt.prompt_speech);
        cosyvoice_free(ctx);
        return false;
    }

    const cosyvoice_tts_context_t tts = cosyvoice_tts_context_new(ctx, prompt);
    cosyvoice_prompt_free(prompt);
    if (!tts)
    {
        std::fprintf(stderr, "failed to create tts context\n");
        cosyvoice_free(ctx);
        return false;
    }

    captured cap;
    const bool ok = cosyvoice_set_sampler_seed(ctx, params.seed) &&
        cosyvoice_tts_zero_shot_stream(tts, opt.text, 1.0f, &collect, &cap);

    cosyvoice_memory_usage_t usage = {};
    cosyvoice_get_memory_usage(ctx, &usage);

    cosyvoice_tts_context_free(tts);
    cosyvoice_free(ctx);

    if (!ok)
    {
        std::fprintf(stderr, "synthesis failed\n");
        return false;
    }

    out.pcm = std::move(cap.pcm);
    out.parameters = usage.parameters;
    return true;
}
}

int tool_entry(int argc, char** argv)
{
    options opt;
    if (argc > 1 && *argv[1]) opt.model = argv[1];
    if (argc > 2 && *argv[2]) opt.device = argv[2];
    if (argc > 3 && *argv[3]) opt.prompt_speech = argv[3];
    if (argc > 4 && *argv[4]) opt.text = argv[4];
    if (argc > 5 && *argv[5])
    {
        const char* end = argv[5] + std::strlen(argv[5]);
        const auto [ptr, ec] = std::from_chars(argv[5], end, opt.threads);
        if (ec != std::errc{} || ptr != end)
        {
            std::fprintf(stderr, "invalid thread count: %s\n", argv[5]);
            return 2;
        }
    }
    if (!opt.text)
    {
        std::fprintf(stderr, "usage: %s [model.gguf] [backend] [prompt_speech.gguf] TEXT [threads]\n", argv[0]);
        return 2;
    }

    cosyvoice_init_backend_from_path(nullptr);

    set_env("COSYVOICE_NO_MMAP_ZEROCOPY", nullptr);
    result zero_copy;
    if (!run(opt, zero_copy)) return 1;

    set_env("COSYVOICE_NO_MMAP_ZEROCOPY", "1");
    result copied;
    if (!run(opt, copied)) return 1;
    set_env("COSYVOICE_NO_MMAP_ZEROCOPY", nullptr);

    size_t file_size = 0;
    if (std::error_code ec; !std::filesystem::exists(opt.model, ec))
        std::fprintf(stderr, "warning: cannot stat %s\n", opt.model);
    else
        file_size = static_cast<size_t>(std::filesystem::file_size(opt.model));

    std::printf("zero-copy: %zu samples, parameters=%zu bytes\n", zero_copy.pcm.size(), zero_copy.parameters);
    std::printf("copy:      %zu samples, parameters=%zu bytes\n", copied.pcm.size(), copied.parameters);

    bool ok = true;
    if (zero_copy.pcm.empty() || zero_copy.pcm.size() != copied.pcm.size())
    {
        std::printf("FAIL sample count mismatch (%zu vs %zu)\n", zero_copy.pcm.size(), copied.pcm.size());
        ok = false;
    }
    else if (std::memcmp(zero_copy.pcm.data(), copied.pcm.data(), zero_copy.pcm.size() * sizeof(float)) != 0)
    {
        size_t first = 0;
        while (first < zero_copy.pcm.size() && zero_copy.pcm[first] == copied.pcm[first])
            ++first;
        std::printf("FAIL audio differs at sample %zu (%g vs %g)\n",
            first, zero_copy.pcm[first], copied.pcm[first]);
        ok = false;
    }

    if (file_size != 0 && zero_copy.parameters < file_size)
    {
        std::printf("FAIL zero-copy parameters (%zu) are smaller than the model file (%zu)\n",
            zero_copy.parameters, file_size);
        ok = false;
    }

    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
