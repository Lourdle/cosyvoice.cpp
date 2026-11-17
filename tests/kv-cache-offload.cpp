// Model-backed regression for the KV cache offload paths.
//
// Phase 1 loads the model with the dedicated buffer policy and two offloadable
// DiT KV slots: it checks that the offloaded DiT cache is reported by
// cosyvoice_get_memory_usage() and released by cosyvoice_empty_buffer_cache().
//
// Phase 2 loads the model with the balanced buffer policy so the LLM KV cache
// gets an offload slot, then clears the buffer cache and synthesizes again. This
// used to use a freed ggml context when offloading the cache after the clear
// (and double-freed it on destruction).
//
// Usage (positional; leave trailing arguments out to take defaults):
//   test-kv-offload [model.gguf] [backend] [prompt_speech.gguf] [text] [threads]
//
//   model           GGUF model to load.                    default: CosyVoice3-2512_F32.gguf
//   backend         Backend name (CUDA0/CPU/MTL0).          default: CUDA0
//   prompt          Path to prompt_speech.gguf.             default: prompt_speech.gguf
//   text            Text to synthesize (required).
//   threads         CPU threads for the backend.            default: 4
#include "cosyvoice.h"
#include "cosyvoice-lowlevel.h"
#include "tool_common.h"
#include <ggml-backend.h>

#include <charconv>
#include <cstdio>
#include <cstring>

namespace
{
struct audio_sink
{
    size_t samples = 0;
};

bool collect_audio(const float*, uint32_t n, void* user_data)
{
    static_cast<audio_sink*>(user_data)->samples += n;
    return true;
}

struct options
{
    const char* model = "CosyVoice3-2512_F32.gguf";
    const char* device = "CUDA0";
    const char* prompt_speech = "prompt_speech.gguf";
    const char* text = nullptr;   // required: pass the text as the 4th argument
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

int run_phase(const options& opt, const char* name, bool dit_offload)
{
    cosyvoice_context_params_v4_cpp params = {};
    cosyvoice_init_default_context_params(&params);
    params.n_max_seq = 256;
    params.n_batch = 256;
    params.diffusion_steps = 4;
    params.strict_seed_mode = true;
    params.seed = 42;
    // Keep the DiT KV cache small: the test only needs the offload round trip.
    params.dit_kv_cache_length = 512;
    params.dit_kv_fixed_slots = 0;
    params.dit_kv_offloadable_slots = dit_offload ? 2 : 0;
    params.dit_kv_actual_offloadable_slots = dit_offload ? 1 : 0;
    params.inference_buffer_policy = dit_offload
        ? COSYVOICE_INFERENCE_BUFFER_POLICY_DEDICATED
        : COSYVOICE_INFERENCE_BUFFER_POLICY_BALANCED;

    ggml_backend_t backend = ggml_backend_init_by_name(opt.device, nullptr);
    if (!backend)
    {
        std::fprintf(stderr, "%s: backend %s unavailable\n", name, opt.device);
        return 2;
    }
    const cosyvoice_context_t ctx = cosyvoice_load_from_file_ext(opt.model, &params, backend, opt.threads);
    if (!ctx)
    {
        std::fprintf(stderr, "%s: failed to load %s\n", name, opt.model);
        return 2;
    }

    const cosyvoice_prompt_speech_t ps = cosyvoice_prompt_speech_load_from_file(opt.prompt_speech);
    cosyvoice_prompt_t prompt = ps ? cosyvoice_prompt_init_from_prompt_speech(ctx, ps) : nullptr;
    if (ps) cosyvoice_prompt_speech_free(ps);
    if (!prompt)
    {
        std::fprintf(stderr, "%s: failed to load %s\n", name, opt.prompt_speech);
        cosyvoice_free(ctx);
        return 2;
    }
    const cosyvoice_tts_context_t tts = cosyvoice_tts_context_new(ctx, prompt);
    cosyvoice_prompt_free(prompt);
    if (!tts)
    {
        std::fprintf(stderr, "%s: failed to create tts context\n", name);
        cosyvoice_free(ctx);
        return 2;
    }

    // The DiT KV cache offload only runs in streaming mode.
    auto synthesize = [&](const char* tag)
    {
        cosyvoice_set_sampler_seed(ctx, params.seed);
        audio_sink sink;
        if (!cosyvoice_tts_zero_shot_stream(tts, opt.text, 1.0f, &collect_audio, &sink))
        {
            std::printf("%s: %s failed\n", name, tag);
            return false;
        }
        std::printf("%s: %s produced %zu samples\n", name, tag, sink.samples);
        return true;
    };

    auto offloaded = [&]()
    {
        cosyvoice_memory_usage_t usage = {};
        cosyvoice_get_memory_usage(ctx, &usage);
        return usage.offloaded_kv_cache;
    };

    bool ok = true;
    if (!synthesize("run1"))
    {
        cosyvoice_tts_context_free(tts);
        cosyvoice_free(ctx);
        return 1;
    }

    const size_t off1 = offloaded();
    std::printf("%s: offloaded KV cache after run1 = %zu bytes\n", name, off1);
    if (off1 == 0)
    {
        std::printf("%s: FAIL expected offloaded KV cache > 0\n", name);
        ok = false;
    }

    cosyvoice_empty_buffer_cache(ctx);
    const size_t cleared = offloaded();
    if (cleared != 0)
    {
        std::printf("%s: FAIL expected 0 after empty_buffer_cache, got %zu\n", name, cleared);
        ok = false;
    }

    if (ok && !dit_offload)
    {
        // Must be safe to call repeatedly and must leave the offload slot usable.
        cosyvoice_empty_buffer_cache(ctx);
        if (!synthesize("run2"))
            ok = false;
        else if (const size_t off2 = offloaded(); off2 != off1)
        {
            std::printf("%s: FAIL offloaded size changed %zu -> %zu\n", name, off1, off2);
            ok = false;
        }
    }

    cosyvoice_tts_context_free(tts);
    cosyvoice_free(ctx);
    return ok ? 0 : 1;
}
}

int tool_entry(int argc, char** argv)
{
    options opt;
    if (argc > 1 && *argv[1]) opt.model = argv[1];
    if (argc > 2 && *argv[2]) opt.device = argv[2];
    if (argc > 3 && *argv[3]) opt.prompt_speech = argv[3];
    if (argc > 4 && *argv[4]) opt.text = argv[4];
    if (!opt.text)
    {
        std::fprintf(stderr, "usage: %s [model.gguf] [backend] [prompt_speech.gguf] TEXT [threads]`n", argv[0]);
        return 2;
    }
    if (argc > 5 && *argv[5] && !parse(argv[5], opt.threads))
    {
        std::fprintf(stderr, "invalid thread count: %s\n", argv[5]);
        return 2;
    }

    cosyvoice_init_backend_from_path(nullptr);

    const int dit = run_phase(opt, "dit-offload", true);
    if (dit == 2) return 2;
    const int llm = run_phase(opt, "llm-offload", false);
    if (llm == 2) return 2;

    const bool ok = dit == 0 && llm == 0;
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
