// In-process streaming TTS test: no server, no HTTP.
// Runs one streaming generation and one non-streaming generation with the
// same seed/prompt, reports chunk boundary steps and the deviation from the
// non-streaming render, and saves both to wav.
//
// Usage (positional; leave trailing arguments out to take defaults):
//   test-tts-stream [model.gguf] [backend] [chunk_tokens] [out_prefix]
//                   text [instruction] [prompt_speech.gguf]
//                   [speed] [seed] [diffusion_steps] [threads] [reset|continue]
//                   [no-save]
//
//   model           GGUF model to load.                    default: CosyVoice3-2512_F32.gguf
//   backend         Backend name (CUDA0/CPU/MTL0).          default: CUDA0
//   chunk_tokens    Streaming LLM chunk size.               default: 28
//   out_prefix      Where to write *_stream.wav / *_ns.wav. default: cfv
//   text            Text to synthesize (required).
//   instruction     Instruct-mode instruction. Omit (or pass "-") for zero-shot.
//   prompt          Path to prompt_speech.gguf.             default: prompt_speech.gguf
//   speed           1.0 disables time stretching.           default: 1.0
//   seed            Sampler seed.                           default: 42
//   diffusion_steps Flow diffusion steps (0 = GGUF value).  default: 4
//   threads         CPU threads for the backend.            default: 4
//   reset|continue  Sampler RNG policy across sessions.     default: reset
//   no-save         Do not write the wav files.
#include "cosyvoice.h"
#include "cosyvoice-lowlevel.h"
#include "tool_common.h"
#include <ggml-backend.h>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace
{
struct options
{
    const char* model = "CosyVoice3-2512_F32.gguf";
    const char* device = "CUDA0";
    uint32_t chunk_tokens = 28;
    const char* prefix = "cfv";
    const char* text = nullptr;
    const char* instr = nullptr;
    const char* prompt_speech = "prompt_speech.gguf";
    float speed = 1.0f;
    uint32_t seed = 42;
    int32_t diffusion_steps = 4;
    uint32_t threads = 4;
    bool rng_continue = false;
    bool save = true;
};

template <typename T>
bool parse(const char* s, T& out)
{
    if (!s || !s[0]) return false;
    std::from_chars(s, s + strlen(s), out);
    return true;
}

bool eq(const char* s, const char* token)
{
    return s && std::strcmp(s, token) == 0;
}

struct capture
{
    std::vector<float> pcm;
    std::vector<uint32_t> chunk_len;
};

bool collect(const float* audio, uint32_t n, void* ud)
{
    auto& c = *static_cast<capture*>(ud);
    if ((c.pcm.size() + n) * 4 > (1 << 26))
        return false;
    c.chunk_len.push_back(n);
    c.pcm.insert(c.pcm.end(), audio, audio + n);
    return true;
}

uint64_t ms_at(const std::vector<float>& pcm, size_t i)
{
    return 1000 * i / 24000;
}

// |x[i] - x[i-1]| in float amplitude.
double step_at(const std::vector<float>& pcm, size_t i)
{
    return std::fabs(static_cast<double>(pcm[i]) - static_cast<double>(pcm[i - 1]));
}
}

int tool_entry(int argc, char** argv)
{
    options opt;

    auto pos = [&](int i)
    {
        return i < argc ? argv[i] : "";
    };

    if (const char* v = pos(1); *v) opt.model = v;
    if (const char* v = pos(2); *v) opt.device = v;
    if (const char* v = pos(3); *v) parse(v, opt.chunk_tokens);
    if (const char* v = pos(4); *v) opt.prefix = v;
    if (const char* v = pos(5); *v) opt.text = v;
    if (const char* v = pos(6); *v) opt.instr = v;
    if (const char* v = pos(7); *v) opt.prompt_speech = v;
    if (const char* v = pos(8); *v) parse(v, opt.speed);
    if (const char* v = pos(9); *v) parse(v, opt.seed);
    if (const char* v = pos(10); *v) parse(v, opt.diffusion_steps);
    if (const char* v = pos(11); *v) parse(v, opt.threads);
    if (const char* v = pos(12); *v && eq(v, "continue")) opt.rng_continue = true;

    for (int i = 13; i < argc; ++i)
    {
        if (eq(argv[i], "no-save")) opt.save = false;
        else
            std::fprintf(stderr, "warning: ignoring unknown token '%s'\n", argv[i]);
    }

    if (!opt.text || !opt.text[0])
    {
        std::fprintf(stderr,
            "usage: %s [model.gguf] [backend] [chunk_tokens] [out_prefix] text "
            "[instruction] [prompt_speech.gguf] [speed] [seed] [diffusion_steps] "
            "[threads] [reset|continue] [no-save]\n", argv[0]);
        return 2;
    }

    // An instruction selects instruct mode; no instruction (omitted or "-") is zero-shot.
    const bool zero_shot = !opt.instr || !opt.instr[0] || eq(opt.instr, "-");

    cosyvoice_init_backend_from_path(nullptr);
    const ggml_backend_t backend = ggml_backend_init_by_name(opt.device, nullptr);
    if (!backend)
    {
        std::fprintf(stderr, "backend %s unavailable\n", opt.device);
        return 2;
    }

    cosyvoice_context_params_v4_cpp params_v4 = {};
    cosyvoice_init_default_context_params(&params_v4);
    params_v4.dit_allow_kv_cache_fallback = true;
    params_v4.inference_buffer_policy = COSYVOICE_INFERENCE_BUFFER_POLICY_DEDICATED;
    params_v4.diffusion_steps = opt.diffusion_steps;
    params_v4.strict_seed_mode = true;
    params_v4.seed = opt.seed;
    const cosyvoice_context_t ctx = cosyvoice_load_from_file_ext(opt.model, &params_v4, backend, opt.threads);
    if (!ctx)
    {
        std::fprintf(stderr, "failed to load %s\n", opt.model);
        return 2;
    }
    cosyvoice_set_chunk_tokens(ctx, opt.chunk_tokens);
    if (opt.rng_continue)
        cosyvoice_set_builtin_sampler_rng_policy(ctx, COSYVOICE_BUILTIN_SAMPLER_RNG_POLICY_CONTINUE_ACROSS_SESSIONS);

    const cosyvoice_prompt_speech_t ps = cosyvoice_prompt_speech_load_from_file(opt.prompt_speech);
    cosyvoice_prompt_t prompt = ps ? cosyvoice_prompt_init_from_prompt_speech(ctx, ps) : nullptr;
    if (ps) cosyvoice_prompt_speech_free(ps);
    if (!prompt)
    {
        std::fprintf(stderr, "failed to load %s\n", opt.prompt_speech);
        return 2;
    }
    const cosyvoice_tts_context_t tts = cosyvoice_tts_context_new(ctx, prompt);
    cosyvoice_prompt_free(prompt);
    if (!tts)
    {
        std::fprintf(stderr, "failed to create tts context\n");
        return 2;
    }

    capture streamed;
    const auto t0 = std::chrono::steady_clock::now();
    const bool stream_ok = zero_shot
        ? cosyvoice_set_sampler_seed(ctx, opt.seed) && cosyvoice_tts_zero_shot_stream(tts, opt.text, opt.speed, &::collect, &streamed)
        : cosyvoice_set_sampler_seed(ctx, opt.seed) && cosyvoice_tts_instruct_stream(tts, opt.text, opt.instr, opt.speed, &::collect, &streamed);
    if (!stream_ok)
    {
        std::fprintf(stderr, "streaming generation failed\n");
        return 1;
    }
    const auto stream_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t0).count();

    if (!cosyvoice_set_sampler_seed(ctx, opt.seed))
        return 1;
    const auto t1 = std::chrono::steady_clock::now();
    cosyvoice_generated_speech ns = {};
    const bool ns_ok = zero_shot
        ? cosyvoice_tts_zero_shot(tts, opt.text, opt.speed, &ns)
        : cosyvoice_tts_instruct(tts, opt.text, opt.instr, opt.speed, &ns);
    if (!ns_ok)
    {
        std::fprintf(stderr, "non-streaming generation failed\n");
        return 1;
    }
    const auto ns_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t1).count();

    const size_t stream_n = streamed.pcm.size();
    const size_t ns_n = ns.data ? ns.length : 0;
    std::printf("stream: %zu samples (%.3f s), %zu chunks in %lld ms\n",
        stream_n, static_cast<double>(stream_n) / 24000.0, streamed.chunk_len.size(),
        static_cast<long long>(stream_ms));
    for (size_t k = 0; k != streamed.chunk_len.size(); ++k)
        std::printf("  chunk %zu: %u samples\n", k, streamed.chunk_len[k]);

    std::printf("non-stream: %zu samples (%.3f s) in %lld ms\n",
        ns_n, static_cast<double>(ns_n) / 24000.0, static_cast<long long>(ns_ms));

    double max_step = 0;
    size_t max_step_i = 0;
    for (size_t i = 1; i < stream_n; ++i)
    {
        const double s = step_at(streamed.pcm, i);
        if (s > max_step)
        {
            max_step = s;
            max_step_i = i;
        }
    }
    std::printf("max step overall: %.6f @ sample %zu (%llu ms)\n",
        max_step, max_step_i, static_cast<unsigned long long>(ms_at(streamed.pcm, max_step_i)));

    size_t acc = 0;
    for (size_t k = 0; k < streamed.chunk_len.size(); ++k)
    {
        acc += streamed.chunk_len[k];
        if (k + 1 == streamed.chunk_len.size())
            break;
        std::printf("seam %zu: step=%.6f rms64=%.6f\n", k, step_at(streamed.pcm, acc),
            [&, i = acc]()
            {
                double sum = 0;
                for (size_t j = i - 32; j != i + 32 && j < stream_n; ++j)
                    sum += step_at(streamed.pcm, j) * step_at(streamed.pcm, j);
                return std::sqrt(sum / 64.0);
            }());
    }

    const size_t n_cmp = std::min(stream_n, ns_n);
    double max_diff = 0, sum_sq = 0;
    size_t max_diff_i = 0;
    for (size_t i = 0; i != n_cmp; ++i)
    {
        const double d = static_cast<double>(streamed.pcm[i]) - static_cast<double>(ns.data[i]);
        const double ad = std::fabs(d);
        sum_sq += d * d;
        if (ad > max_diff)
        {
            max_diff = ad;
            max_diff_i = i;
        }
    }
    if (n_cmp != 0)
        std::printf("diff vs non-stream (n=%zu): rms=%.6f max=%.6f @ sample %zu (%llu ms)\n",
            n_cmp, std::sqrt(sum_sq / n_cmp), max_diff, max_diff_i,
            static_cast<unsigned long long>(ms_at(streamed.pcm, max_diff_i)));

    if (opt.save)
    {
        const bool saved =
            cosyvoice_save_wav((std::string(opt.prefix) + "_stream.wav").c_str(), streamed.pcm.data(),
                static_cast<uint32_t>(stream_n), 24000) &&
            cosyvoice_save_wav((std::string(opt.prefix) + "_ns.wav").c_str(), ns.data,
                static_cast<uint32_t>(ns_n), 24000);
        if (!saved)
        {
            std::fprintf(stderr, "wav save failed\n");
            return 1;
        }
        std::printf("saved %s_stream.wav / %s_ns.wav\n", opt.prefix, opt.prefix);
    }

    cosyvoice_tts_context_free(tts);
    return 0;
}
