#include "cosyvoice-internal.h"
#include "cosyvoice-stage-timing.h"
#include "cosyvoice-text-chunk.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <format>
#include <string_view>
#include <vector>

constexpr uint32_t COSYVOICE_TTS_FLAG_MASK = COSYVOICE_TTS_FLAG_TEXT_NORMALIZATION | COSYVOICE_TTS_FLAG_SPLIT_TEXT | COSYVOICE_TTS_FLAG_FAST_SPLIT | COSYVOICE_TTS_FLAG_FADE_IN;
constexpr double COSYVOICE_TTS_FADE_IN_SECONDS = 0.02;

namespace
{
struct text_stage_report
{
    double preprocess_ms = 0;
    double tokenize_ms = 0;
    uint32_t tokenize_calls = 0;

    ~text_stage_report()
    {
        if (!stage_timing_enabled())
            return;
        stage_timing_log(std::format(
            "stage timing: text: preprocess={:.1f}ms tokenize={:.1f}ms ({} calls)\n",
            preprocess_ms, tokenize_ms, tokenize_calls).c_str());
    }
};
}

static constexpr bool is_emoji_cp(uint32_t cp)
{
    if (cp >= 0x1F000 && cp <= 0x1FAFF) return true; // Emoji blocks
    if (cp >= 0x2600 && cp <= 0x27BF) return true;   // Miscellaneous Symbols, Dingbats
    if (cp >= 0x2B00 && cp <= 0x2BFF) return true;   // Miscellaneous Symbols and Arrows
    if ((cp >= 0x231A && cp <= 0x231B) // ⌚ ⌛
        || (cp >= 0x23E9 && cp <= 0x23EC) // ⏩ ⏪ ⏫ ⏬
        || cp == 0x23F0 || cp == 0x23F3 || (cp >= 0x23F8 && cp <= 0x23FA)) return true;
    if (cp == 0x25AA || cp == 0x25AB || cp == 0x25B6 || cp == 0x25C0
        || (cp >= 0x25FB && cp <= 0x25FE)) return true; // ▪ ▫ ▶ ◀ ◻ ◼ ◽ ◾
    if (cp >= 0x1F1E6 && cp <= 0x1F1FF) return true; // Regional indicator symbols (flag emoji)
    if (cp >= 0x1F3FB && cp <= 0x1F3FF) return true; // Emoji modifiers (skin tones)
    if (cp == 0xFE0F) return true;                   // Variation selector-16 (emoji presentation)
    return false;
}

static uint32_t decode_utf8_cp(std::string_view text, std::size_t i, std::size_t& len)
{
    const auto b0 = static_cast<unsigned char>(text[i]);
    if ((b0 & 0x80) == 0) { len = 1; return b0; }
    if ((b0 & 0xE0) == 0xC0) len = 2;
    else if ((b0 & 0xF0) == 0xE0) len = 3;
    else if ((b0 & 0xF8) == 0xF0) len = 4;
    else { len = 1; return b0; }
    if (i + len > text.size()) { len = 1; return b0; }
    for (std::size_t k = 1; k < len; ++k)
        if ((static_cast<unsigned char>(text[i + k]) & 0xC0) != 0x80) { len = 1; return b0; }
    uint32_t cp = b0 & (0x7F >> len);
    for (std::size_t k = 1; k < len; ++k)
        cp = (cp << 6) | (static_cast<unsigned char>(text[i + k]) & 0x3F);
    return cp;
}

static inline bool strip_emoji(std::string_view text, std::string& result)
{
    bool removed = false;
    bool prev_emoji = false;
    std::size_t i = 0;
    while (i < text.size())
    {
        std::size_t len = 0;
        const uint32_t cp = decode_utf8_cp(text, i, len);
        bool drop = is_emoji_cp(cp);
        if (!drop && (cp == 0x200D || cp == 0x20E3) && prev_emoji) // ZWJ / keycap combining mark inside an emoji sequence
            drop = true;
        if (!drop && (cp == 0x00A9 || cp == 0x00AE || cp == 0x2122) && i + len < text.size()) // ©️ ®️ ™️
        {
            std::size_t next_len = 0;
            if (decode_utf8_cp(text, i + len, next_len) == 0xFE0F)
            {
                drop = true;
                len += next_len;
            }
        }
        if (drop)
        {
            if (!removed)
            {
                result.append(text.data(), i);
                removed = true;
            }
            prev_emoji = true;
            i += len;
            continue;
        }
        prev_emoji = false;
        if (removed)
            result.append(text, i, len);
        i += len;
    }

    return removed;
}

struct cosyvoice_tts_context : cosyvoice_tokenization_result_impl, cosyvoice_prompt, std::string
{
    cosyvoice_tts_context(cosyvoice_context_t ctx, cosyvoice_prompt_t prompt)
        : cosyvoice_prompt(*prompt), ctx(ctx),
        flags(
#ifndef COSYVOICE_NO_ICU
            COSYVOICE_TTS_FLAG_TEXT_NORMALIZATION |
#endif
            COSYVOICE_TTS_FLAG_SPLIT_TEXT | COSYVOICE_TTS_FLAG_FAST_SPLIT | COSYVOICE_TTS_FLAG_FADE_IN
        ),
        fade_samples(static_cast<uint32_t>(std::ceil(static_cast<double>(ctx->get_sample_rate()) * COSYVOICE_TTS_FADE_IN_SECONDS)))
    {
        const auto instruction_prefix = ctx->get_instruction_prefix();
        if (instruction_prefix)
        {
            instruction_cache = instruction_prefix;
            prefix_len = instruction_cache.size();
        }
        else prefix_len = 0;
    }

    bool tts_job(const char* text, const char* instruction, float speed, cosyvoice_inference_mode mode, cosyvoice_generated_speech_ptr result, cosyvoice_tts_audio_callback_t callback, void* user_data)
    {
        text_stage_report text_report;

        instruction_cache.resize(prefix_len);
        if (instruction)
        {
            instruction_cache.push_back(' ');
            instruction_cache.append(instruction);
        }
        cosyvoice_prompt_set(
            ctx,
            this,
            mode,
            instruction_cache.c_str());

        // Remove all emoji from the synthesis text.
        const stage_mark preprocess_mark;
        auto effective_text = strip_emoji(text, *this) ? c_str() : text;

        if (flags & COSYVOICE_TTS_FLAG_TEXT_NORMALIZATION
            && cosyvoice_frontend_util_text_normalize(*this, effective_text, static_cast<uint32_t>(strlen(effective_text)), nullptr))
            effective_text = c_str();
        text_report.preprocess_ms += preprocess_mark.elapsed_ms();

        auto do_tokenize = [&](auto&&... args)
        {
            const stage_mark mark;
            ctx->tokenize(std::forward<decltype(args)>(args)...);
            text_report.tokenize_ms += mark.elapsed_ms();
            ++text_report.tokenize_calls;
        };

        // When splitting is disabled, do a single tokenize+synthesize pass.
        if (!(flags & COSYVOICE_TTS_FLAG_SPLIT_TEXT))
        {
            do_tokenize(effective_text, this, true);
            return result ? cosyvoice_tts_with_postprocess(get_tokens(), get_n_tokens(), speed, result)
                : cosyvoice_tts_stream_with_postprocess(get_tokens(), get_n_tokens(), speed, callback, user_data);
        }

        const auto fragments = cosyvoice_internal::split_into_fragments(effective_text);
        if (fragments.size() <= 1)
        {
            do_tokenize(effective_text, this, true);
            return result ? cosyvoice_tts_with_postprocess(get_tokens(), get_n_tokens(), speed, result)
                : cosyvoice_tts_stream_with_postprocess(get_tokens(), get_n_tokens(), speed, callback, user_data);
        }

        // Compute the maximum text-token budget per chunk:
        //   m = n_max_seq                                    (the LLM context window)
        //   o = SOS(1) + prompt_text + (task_token(1) + prompt_speech_tokens when present)
        //   r = max_token_text_ratio                         (decode-stage growth factor)
        // We need m > o + l * (1 + r) so the prefill plus decoded speech tokens fit; solving
        // for l yields the cap below. Falling back to a single call when the budget is
        // pathological keeps behavior consistent with the pre-chunking path.
        cosyvoice_context_params_t params;
        ctx->get_context_params(&params);
        cosyvoice_generation_config_t gen_config;
        ctx->get_generation_config(&gen_config);

        const uint32_t prompt_text_len = static_cast<uint32_t>(prompt_text.size());
        const uint32_t prompt_speech_len = llm_prompt_speech_tokens.second;
        const uint64_t o = 1ull + prompt_text_len
            + (prompt_speech_len != 0 ? 1ull + prompt_speech_len : 0ull);
        const float r = gen_config.max_token_text_ratio;

        if (params.n_max_seq <= o + 1u || !(r > 0.0f))
        {
            do_tokenize(effective_text, this, true);
            return result ? cosyvoice_tts_with_postprocess(get_tokens(), get_n_tokens(), speed, result)
                : cosyvoice_tts_stream_with_postprocess(get_tokens(), get_n_tokens(), speed, callback, user_data);
        }
        const auto max_text_tokens = static_cast<std::size_t>(
            static_cast<float>(params.n_max_seq - o - 1u) / (1.0f + r));

        if (max_text_tokens == 0)
        {
            do_tokenize(effective_text, this, true);
            return result ? cosyvoice_tts_with_postprocess(get_tokens(), get_n_tokens(), speed, result)
                : cosyvoice_tts_stream_with_postprocess(get_tokens(), get_n_tokens(), speed, callback, user_data);
        }

        // Fast-split path: tokenize each fragment once and merge token-ID vectors,
        // avoiding re-tokenization of each assembled chunk.
        if (flags & COSYVOICE_TTS_FLAG_FAST_SPLIT)
        {
            std::vector<std::vector<int>> fragment_tokens;
            fragment_tokens.reserve(fragments.size());
            {
                cosyvoice_tokenization_result_impl tok;
                for (const auto& fragment : fragments)
                {
                    tok.tokens.clear();
                    do_tokenize(fragment.data(), static_cast<uint32_t>(fragment.size()), &tok, true);
                    fragment_tokens.push_back(std::move(tok.tokens));
                }
            }
            auto chunk_token_list = cosyvoice_internal::reassemble_by_token_budget(
                fragment_tokens, max_text_tokens);

            if (chunk_token_list.size() <= 1)
            {
                const auto& t = chunk_token_list.empty() ? fragment_tokens[0] : chunk_token_list[0];
                tokens = t;
                return result ? cosyvoice_tts_with_postprocess(get_tokens(), get_n_tokens(), speed, result)
                    : cosyvoice_tts_stream_with_postprocess(get_tokens(), get_n_tokens(), speed, callback, user_data);
            }

            if (result)
            {
                combined_pcm.clear();
                for (auto& chunk_tokens : chunk_token_list)
                {
                    if (ctx->is_stop_requested())
                    {
                        result->data = nullptr;
                        result->length = 0;
                        return false;
                    }
                    tokens = std::move(chunk_tokens);
                    cosyvoice_generated_speech part = {};
                    if (!cosyvoice_tts_with_postprocess(get_tokens(), get_n_tokens(), speed, &part)
                        || !part.data || part.length == 0)
                    {
                        result->data = nullptr;
                        result->length = 0;
                        return false;
                    }
                    combined_pcm.insert(combined_pcm.end(), part.data, part.data + part.length);
                }
                result->data = combined_pcm.data();
                result->length = static_cast<uint32_t>(combined_pcm.size());
                return true;
            }

            for (auto& chunk_tokens : chunk_token_list)
            {
                if (ctx->is_stop_requested())
                    return false;
                tokens = std::move(chunk_tokens);
                if (!cosyvoice_tts_stream_with_postprocess(get_tokens(), get_n_tokens(), speed, callback, user_data))
                    return false;
            }
            return true;
        }

        // Slow-split path: tokenize each fragment only for counting,
        // reassemble text chunks, then re-tokenize each chunk.
        cosyvoice_tokenization_result_impl tokens_scratch;
        const auto token_count = [&](std::string_view fragment) -> std::size_t
            {
                tokens_scratch.tokens.clear();
                do_tokenize(fragment.data(), static_cast<uint32_t>(fragment.size()), &tokens_scratch, true);
                return tokens_scratch.get_n_tokens();
            };
        const auto chunks = cosyvoice_internal::reassemble_by_token_budget(
            fragments, max_text_tokens, token_count);

        if (chunks.size() <= 1)
        {
            do_tokenize(effective_text, this, true);
            return result ? cosyvoice_tts_with_postprocess(get_tokens(), get_n_tokens(), speed, result)
                : cosyvoice_tts_stream_with_postprocess(get_tokens(), get_n_tokens(), speed, callback, user_data);
        }

        // Multi-chunk path: synthesize each chunk and copy its PCM into a context-owned buffer.
        // cosyvoice_tts() points result->data at an internal token2wav buffer that is overwritten
        // on the next call, so the copy must happen before the following synthesis begins.
        combined_pcm.clear();
        if (result)
        {
            for (const auto& chunk : chunks)
            {
                if (ctx->is_stop_requested())
                {
                    result->data = nullptr;
                    result->length = 0;
                    return false;
                }
                do_tokenize(chunk.c_str(), this, true);
                cosyvoice_generated_speech part = {};
                if (!cosyvoice_tts_with_postprocess(get_tokens(), get_n_tokens(), speed, &part)
                    || !part.data || part.length == 0)
                {
                    result->data = nullptr;
                    result->length = 0;
                    return false;
                }
                combined_pcm.insert(combined_pcm.end(), part.data, part.data + part.length);
            }
            result->data = combined_pcm.data();
            result->length = static_cast<uint32_t>(combined_pcm.size());
        }
        else
            for (const auto& chunk : chunks)
            {
                if (ctx->is_stop_requested())
                    return false;
                do_tokenize(chunk.c_str(), this, true);
                if (!cosyvoice_tts_stream_with_postprocess(get_tokens(), get_n_tokens(), speed, callback, user_data))
                    return false;
            }
        return true;
    }

    bool cosyvoice_tts_stream_with_postprocess(const int* token_ids, uint32_t n_tokens, float speed, cosyvoice_tts_audio_callback_t callback, void* user_data)
    {
        bool ok;
        if (flags & COSYVOICE_TTS_FLAG_FADE_IN)
        {
            struct callback_wrapper_context
            {
                cosyvoice_tts_audio_callback_t callback;
                void* user_data;
                uint32_t fade_samples;
                uint32_t processed_samples = 0;
            } cb_ctx{ callback, user_data, fade_samples };

            auto callback_wrapper = [](const float* data, uint32_t length, void* user_data) -> bool
            {
                auto ctx = static_cast<callback_wrapper_context*>(user_data);
                if (ctx->processed_samples < ctx->fade_samples)
                {
                    const uint32_t fade_len = std::min<uint32_t>(ctx->fade_samples - ctx->processed_samples, length);
                    for (uint32_t i = 0; i < fade_len; ++i)
                    {
                        const float ramp = static_cast<float>(ctx->processed_samples + i + 1) / static_cast<float>(ctx->fade_samples + 1);
                        const_cast<float*>(data)[i] *= ramp;
                    }
                    ctx->processed_samples += fade_len;
                }
                return ctx->callback(data, length, ctx->user_data);
            };
            ok = cosyvoice_tts_stream(ctx, token_ids, n_tokens, speed, this, callback_wrapper, &cb_ctx);
        }
        else
            ok = cosyvoice_tts_stream(ctx, token_ids, n_tokens, speed, this, callback, user_data);

        return ok;
    }

    bool cosyvoice_tts_with_postprocess(const int* token_ids, uint32_t n_tokens, float speed, cosyvoice_generated_speech_ptr result)
    {
        if (!cosyvoice_tts(ctx, token_ids, n_tokens, speed, this, result))
            return false;
        if ((flags & COSYVOICE_TTS_FLAG_FADE_IN) && result->data && result->length != 0)
        {
            const uint32_t fade_len = std::min<uint32_t>(fade_samples, result->length);
            for (uint32_t i = 0; i < fade_len; ++i)
            {
                const float ramp = static_cast<float>(i + 1) / static_cast<float>(fade_len + 1);
                result->data[i] *= ramp;
            }
        }
        return true;
    }

    cosyvoice_context_t ctx;
    std::string instruction_cache;
    size_t prefix_len;
    uint32_t flags;
    uint32_t fade_samples;
    std::vector<float> combined_pcm;
};

cosyvoice_tts_context_t cosyvoice_tts_context_new(cosyvoice_context_t ctx, cosyvoice_prompt_t prompt)
{
    return new cosyvoice_tts_context(ctx, prompt);
}

void cosyvoice_tts_context_free(cosyvoice_tts_context_t ctx)
{
    delete ctx;
}

void cosyvoice_tts_context_set_prompt(cosyvoice_tts_context_t ctx, cosyvoice_prompt_t prompt)
{
    *static_cast<cosyvoice_prompt_t>(ctx) = *prompt;
}

static void set_flag(cosyvoice_tts_context_t ctx, uint32_t flag, bool enabled)
{
    if (enabled)
        ctx->flags |= flag;
    else
        ctx->flags &= ~flag;
}

#define COSYVOICE_TTS_CONTEXT_SET_FLAG_IMPL(name, flag) \
    bool cosyvoice_tts_context_set_##name##_enabled(cosyvoice_tts_context_t ctx, bool enabled) \
    { \
        set_flag(ctx, flag, enabled); \
        return true; \
    }

#define COSYVOICE_TTS_CONTEXT_GET_FLAG_IMPL(name, flag) \
    bool cosyvoice_tts_context_get_##name##_enabled(cosyvoice_tts_context_t ctx) \
    { \
        return (ctx->flags & flag) != 0; \
    }

#define COSYVOICE_TTS_CONTEXT_FLAG_ACCESSORS(name, flag) \
    COSYVOICE_TTS_CONTEXT_SET_FLAG_IMPL(name, flag) \
    COSYVOICE_TTS_CONTEXT_GET_FLAG_IMPL(name, flag)

bool cosyvoice_tts_context_set_text_normalization_enabled(cosyvoice_tts_context_t ctx, bool enabled)
{
#ifdef COSYVOICE_NO_ICU
    return !enabled;
#else
    set_flag(ctx, COSYVOICE_TTS_FLAG_TEXT_NORMALIZATION, enabled);
    return true;
#endif
}

COSYVOICE_TTS_CONTEXT_GET_FLAG_IMPL(text_normalization, COSYVOICE_TTS_FLAG_TEXT_NORMALIZATION)

COSYVOICE_TTS_CONTEXT_FLAG_ACCESSORS(split_text, COSYVOICE_TTS_FLAG_SPLIT_TEXT)

COSYVOICE_TTS_CONTEXT_FLAG_ACCESSORS(fast_split_text, COSYVOICE_TTS_FLAG_FAST_SPLIT)

COSYVOICE_TTS_CONTEXT_FLAG_ACCESSORS(fade_in, COSYVOICE_TTS_FLAG_FADE_IN)

uint32_t cosyvoice_tts_context_get_flags(cosyvoice_tts_context_t ctx)
{
    return ctx->flags;
}

uint32_t cosyvoice_tts_context_set_flags(cosyvoice_tts_context_t ctx, uint32_t flags)
{
    flags &= COSYVOICE_TTS_FLAG_MASK
#ifdef COSYVOICE_NO_ICU
        & ~COSYVOICE_TTS_FLAG_TEXT_NORMALIZATION
#endif
        ;
    ctx->flags = flags;
    return ctx->flags;
}

bool cosyvoice_tts_zero_shot(cosyvoice_tts_context_t ctx, const char* text, float speed, cosyvoice_generated_speech_ptr result)
{
    return ctx->tts_job(text, nullptr, speed, COSYVOICE_INFERENCE_MODE_ZERO_SHOT, result, nullptr, nullptr);
}

bool cosyvoice_tts_instruct(cosyvoice_tts_context_t ctx, const char* text, const char* instruction, float speed, cosyvoice_generated_speech_ptr result)
{
    return ctx->tts_job(text, instruction, speed, COSYVOICE_INFERENCE_MODE_INSTRUCT, result, nullptr, nullptr);
}

bool cosyvoice_tts_cross_lingual(cosyvoice_tts_context_t ctx, const char* text, float speed, cosyvoice_generated_speech_ptr result)
{
    return ctx->tts_job(text, nullptr, speed, COSYVOICE_INFERENCE_MODE_CROSS_LINGUAL, result, nullptr, nullptr);
}

bool cosyvoice_tts_zero_shot_stream(cosyvoice_tts_context_t ctx, const char* text, float speed, cosyvoice_tts_audio_callback_t callback, void* user_data)
{
    return ctx->tts_job(text, nullptr, speed, COSYVOICE_INFERENCE_MODE_ZERO_SHOT, nullptr, callback, user_data);
}

bool cosyvoice_tts_instruct_stream(cosyvoice_tts_context_t ctx, const char* text, const char* instruction, float speed, cosyvoice_tts_audio_callback_t callback, void* user_data)
{
    return ctx->tts_job(text, instruction, speed, COSYVOICE_INFERENCE_MODE_INSTRUCT, nullptr, callback, user_data);
}

bool cosyvoice_tts_cross_lingual_stream(cosyvoice_tts_context_t ctx, const char* text, float speed, cosyvoice_tts_audio_callback_t callback, void* user_data)
{
    return ctx->tts_job(text, nullptr, speed, COSYVOICE_INFERENCE_MODE_CROSS_LINGUAL, nullptr, callback, user_data);
}
