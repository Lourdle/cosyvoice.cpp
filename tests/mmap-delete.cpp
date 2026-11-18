// Deleting the model file while a zero-copy context is alive must be safe.
//
// The test copies the given model to a work file, loads that copy on a CPU
// backend (weights are used in place from the file mapping), synthesizes once,
// deletes the work file, and synthesizes again. The second synthesis must still
// succeed and produce bit-identical audio, because the mapping remains valid
// until the context is released. After freeing the context the work file name
// must be gone.
//
// Usage (positional; leave trailing arguments out to take defaults):
//   test-mmap-delete [model.gguf] [work.gguf] [backend] [prompt_speech.gguf] TEXT [threads]
//
//   model           Source GGUF copied to the work path.   default: CosyVoice3-2512_F32.gguf
//   work            Work file that gets deleted.           default: mmap-delete-work.gguf
//   backend         Backend name (must be a CPU device).   default: CPU
//   prompt          Path to prompt_speech.gguf.            default: prompt_speech.gguf
//   text            Text to synthesize (required).
//   threads         CPU threads for the backend.           default: 8

#ifdef _WIN32
    #define WIN32_LEAN_AND_MEAN
    #define NOMINMAX
    #include <windows.h>
#endif

#include "cosyvoice.h"
#include "cosyvoice-lowlevel.h"
#include "tool_common.h"
#include <ggml-backend.h>

#include <charconv>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <system_error>
#include <vector>

namespace
{
struct options
{
    const char* model = "CosyVoice3-2512_F32.gguf";
    const char* work = "mmap-delete-work.gguf";
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

bool delete_file(const char* path)
{
#ifdef _WIN32
    return DeleteFileW(utf8_to_wstr(path).c_str()) != 0;
#else
    return std::remove(path) == 0;
#endif
}

bool run(const options& opt, bool& zero_copy, size_t& file_size, captured& cap)
{
    cosyvoice_context_params_v4_cpp params = {};
    cosyvoice_init_default_context_params(&params);
    params.n_max_seq = 256;
    params.n_batch = 256;
    params.diffusion_steps = 4;
    params.strict_seed_mode = true;
    params.seed = 42;
    params.dit_kv_cache_length = 512;

    std::error_code ec;
    file_size = static_cast<size_t>(std::filesystem::file_size(opt.work, ec));
    if (ec)
    {
        std::fprintf(stderr, "cannot stat %s: %s\n", opt.work, ec.message().c_str());
        return false;
    }

    ggml_backend_t backend = ggml_backend_init_by_name(opt.device, nullptr);
    if (!backend)
    {
        std::fprintf(stderr, "backend %s unavailable\n", opt.device);
        return false;
    }

    const cosyvoice_context_t ctx = cosyvoice_load_from_file_ext(opt.work, &params, backend, opt.threads);
    if (!ctx)
    {
        std::fprintf(stderr, "failed to load %s\n", opt.work);
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

    cosyvoice_memory_usage_t usage = {};
    cosyvoice_get_memory_usage(ctx, &usage);
    zero_copy = usage.parameters >= file_size;
    std::printf("loaded %s: parameters=%zu bytes (file=%zu), zero-copy=%s\n",
        opt.work, usage.parameters, file_size, zero_copy ? "yes" : "no");

    if (!cosyvoice_set_sampler_seed(ctx, params.seed) ||
        !cosyvoice_tts_zero_shot_stream(tts, opt.text, 1.0f, &collect, &cap))
    {
        std::fprintf(stderr, "first synthesis failed\n");
        cosyvoice_tts_context_free(tts);
        cosyvoice_free(ctx);
        return false;
    }
    std::printf("run1: %zu samples\n", cap.samples);

    // Delete the mapped file while the context is alive.
    if (!delete_file(opt.work))
    {
        std::fprintf(stderr, "failed to delete %s while it is mapped\n", opt.work);
        cosyvoice_tts_context_free(tts);
        cosyvoice_free(ctx);
        return false;
    }
    std::printf("deleted %s while mapped\n", opt.work);

    // The mapping must still be readable: synthesize again with the same seed.
    captured second;
    const bool second_ok = cosyvoice_set_sampler_seed(ctx, params.seed) &&
        cosyvoice_tts_zero_shot_stream(tts, opt.text, 1.0f, &collect, &second);
    std::printf("run2: %zu samples\n", second.samples);

    cosyvoice_tts_context_free(tts);
    cosyvoice_free(ctx);

    if (!second_ok)
    {
        std::fprintf(stderr, "synthesis after deletion failed\n");
        return false;
    }

    if (second.pcm.size() != cap.pcm.size() ||
        std::memcmp(second.pcm.data(), cap.pcm.data(), second.pcm.size() * sizeof(float)) != 0)
    {
        std::fprintf(stderr, "audio differs after deleting the mapped file\n");
        return false;
    }

    if (std::filesystem::exists(opt.work, ec))
    {
        std::fprintf(stderr, "%s still exists after releasing the context\n", opt.work);
        return false;
    }

    return true;
}
}

int tool_entry(int argc, char** argv)
{
    options opt;
    if (argc > 1 && *argv[1]) opt.model = argv[1];
    if (argc > 2 && *argv[2]) opt.work = argv[2];
    if (argc > 3 && *argv[3]) opt.device = argv[3];
    if (argc > 4 && *argv[4]) opt.prompt_speech = argv[4];
    if (argc > 5 && *argv[5]) opt.text = argv[5];
    if (argc > 6 && *argv[6])
    {
        const char* end = argv[6] + std::strlen(argv[6]);
        const auto [ptr, ec] = std::from_chars(argv[6], end, opt.threads);
        if (ec != std::errc{} || ptr != end)
        {
            std::fprintf(stderr, "invalid thread count: %s\n", argv[6]);
            return 2;
        }
    }
    if (!opt.text)
    {
        std::fprintf(stderr,
            "usage: %s [model.gguf] [work.gguf] [backend] [prompt_speech.gguf] TEXT [threads]\n", argv[0]);
        return 2;
    }

    cosyvoice_init_backend_from_path(nullptr);

    std::error_code ec;
    // An existing work file is used as is; this allows hard-linking it to the source
    // model for space-constrained drives (deleting the link while it is mapped).
    if (!std::filesystem::exists(opt.work, ec))
    {
        std::filesystem::copy_file(opt.model, opt.work, std::filesystem::copy_options::overwrite_existing, ec);
        if (ec)
        {
            std::fprintf(stderr, "failed to copy %s to %s: %s\n", opt.model, opt.work, ec.message().c_str());
            return 2;
        }
    }

    bool zero_copy = false;
    size_t file_size = 0;
    captured cap;
    const bool ok = run(opt, zero_copy, file_size, cap);

    if (ok && !zero_copy)
    {
        std::printf("FAIL weights were not mapped zero-copy (use a CPU backend)\n");
        return 1;
    }

    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
