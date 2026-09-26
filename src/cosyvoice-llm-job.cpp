#include "cosyvoice-internal.h"
#include "cosyvoice-model.h"

#include <algorithm>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <vector>

bool cosyvoice_model_3::llm_job(const int* text, uint32_t text_len, cosyvoice_prompt_t prompt)
{
    llm_clear_accepted_tokens();
    return llm_job_ext(text, text_len, prompt, UINT32_MAX, nullptr);
}

bool cosyvoice_model_3::llm_job_ext(const int* text, uint32_t text_len, cosyvoice_prompt_t prompt, uint32_t max_new_tokens, bool* final)
{
    use_count_guard _guard(this);
    const auto& params = shared->params;
    auto& sched = worker->sched;
    const auto& llm = cv3_shared->llm;
    auto& batch_buffer = worker->batch_buffer;
    auto& prompt_crc32 = worker->prompt_crc32;
    auto last_prompt_crc32 = prompt_crc32;
    bool stop_reached = false;

    if (params.inference_buffer_policy == COSYVOICE_INFERENCE_BUFFER_POLICY_BALANCED)
    {
        ggml_backend_sched_reset(sched.get());
        worker->llm_kv_cache.load_cache(worker->backend.get(), sched.get());
    }

    try
    {
        const auto n_batch = params.n_batch;
        const auto speech_type = llm.speech_embedding_weight->type;
        const auto speech_row_size = static_cast<uint32_t>(llm.speech_embedding_weight->nb[1]);
        const auto speech_emb = reinterpret_cast<const char*>(llm.speech_embedding_weight->data);
        const auto token_emb = reinterpret_cast<const char*>(llm.embed_tokens_weight->data);
        const char* cur;
        uint32_t offset = 0;

        if (llm_get_n_accepted_tokens() == 0 || llm_get_kv_cache_len() == 0)
        {
            GGML_ASSERT(text);

            const auto token_type = llm.embed_tokens_weight->type;
            const auto token_row_size = static_cast<uint32_t>(llm.embed_tokens_weight->nb[1]);

            auto prefill_embedding = [&](const char* data, int token_id, uint32_t row_size, ggml_type type)
            {
                memcpy(batch_buffer.get() + offset++ * row_size, data + token_id * row_size, row_size);

                if (offset == n_batch)
                {
                    if (!llm_prefill(type, batch_buffer.get(), n_batch))
                        throw std::runtime_error("Failed to prefill LLM KV cache.\n");
                    offset = 0;
                }
            };

            if (llm_get_kv_cache_len() == 0)
            {
                if (speech_type == token_type)
                    prefill_embedding(speech_emb, llm.sos_token_id, speech_row_size, speech_type);
                else
                    llm_prefill(speech_type, speech_emb + llm.sos_token_id * speech_row_size, 1);
            prefill_prompt:
                for (const auto& i : prompt->prompt_text)
                    prefill_embedding(token_emb, i, token_row_size, token_type);
                prompt_crc32 = prompt->prompt_crc32;

                if (params.strict_seed_mode && offset != 0)
                {
                    if (!llm_prefill(token_type, batch_buffer.get(), offset))
                        throw std::runtime_error("Failed to prefill LLM KV cache.\n");
                    offset = 0;
                }
            }
            else if (prompt_crc32 != prompt->prompt_crc32
                || !llm_set_kv_cache_len(1 + static_cast<uint32_t>(prompt->prompt_text.size())))
            {
                // The first token is assumed to be the SOS token already stored in the KV cache.
                llm_set_kv_cache_len(1);
                goto prefill_prompt;
            }

            for (uint32_t i = 0; i != text_len; ++i)
                prefill_embedding(token_emb, text[i], token_row_size, token_type);

            if (token_type != speech_type)
            {
                if (offset != 0 && !llm_prefill(token_type, batch_buffer.get(), offset))
                    throw std::runtime_error("Failed to prefill LLM KV cache.\n");
                offset = 0;
            }

            if (prompt->llm_prompt_speech_tokens.second != 0)
            {
                prefill_embedding(speech_emb, llm.task_token_id, speech_row_size, speech_type);

                const auto end = prompt->llm_prompt_speech_tokens.second - 1;
                for (uint32_t i = 0; i != end; ++i)
                    prefill_embedding(speech_emb, prompt->llm_prompt_speech_tokens.first[i], speech_row_size, speech_type);
                cur = speech_emb + prompt->llm_prompt_speech_tokens.first[end] * speech_row_size;
            }
            else cur = speech_emb + llm.task_token_id * speech_row_size;

            if (const auto n_acc = llm_get_n_accepted_tokens(); n_acc > 0)
            {
                const auto* acc_tokens = llm_get_accepted_tokens();
                const auto end = n_acc - 1;
                prefill_embedding(cur, 0, speech_row_size, speech_type);
                for (uint32_t i = 0; i < end; ++i)
                    prefill_embedding(speech_emb, acc_tokens[i], speech_row_size, speech_type);
                cur = speech_emb + acc_tokens[end] * speech_row_size;
            }

            // Prefill the remaining tokens together with `cur` (the input of
            // the first decode step) in a single fused pass that also computes
            // the logits of the next token.
            memcpy(batch_buffer.get() + offset * speech_row_size, cur, speech_row_size);
            if (!llm_prefill_logits(speech_type, batch_buffer.get(), offset + 1))
                throw std::runtime_error("Failed to prefill LLM KV cache and compute logits.\n");
        }
        else
        {
            // Continue from existing KV cache — get last accepted token's embedding
            const auto last_token_id = llm_get_accepted_tokens()[llm_get_n_accepted_tokens() - 1];
            cur = speech_emb + last_token_id * speech_row_size;
            if (!llm_decode(speech_type, cur))
                throw std::runtime_error("Failed to decode LLM output.\n");
        }

        // First call: min/max from text input
        const auto min_len = static_cast<uint32_t>(text_len * worker->config.min_token_text_ratio);
        const auto max_len = static_cast<uint32_t>(text_len * worker->config.max_token_text_ratio);
        const auto limit = std::min(llm_get_n_accepted_tokens() + max_new_tokens, max_len);

        // FSQ silent/breath token filtering: allow up to 5 consecutive silent tokens,
        // then drop any further consecutive ones to avoid generating long silence.
        // Reset the counter whenever a non-silent token appears.
        uint32_t cur_silent_token_num = 0;
        constexpr uint32_t max_silent_token_num = 5;
        const auto& silent_tokens = cv3_shared->silent_tokens;

        for (uint32_t n = llm_get_n_accepted_tokens();; ++n)
        {
            llm_prepare_probs(n > min_len);
            const auto token_id = llm_sample_token();
            if (token_id == -1)
                throw std::runtime_error("Failed to sample token from LLM output. This might be wrong with the model or caused by an issue with the sampling parameters.\n");
            if (n > min_len && llm_is_stop_token(token_id))
            {
                stop_reached = true;
                break;
            }

            if (silent_tokens.count(token_id))
            {
                ++cur_silent_token_num;
                if (cur_silent_token_num > max_silent_token_num)
                {
                    // Drop excess silent tokens — the token is already in the
                    // KV cache (from llm_decode), so update cur to keep the
                    // model state consistent, but skip llm_accept_token so
                    // it does not appear in the output.
                    cur = speech_emb + token_id * speech_row_size;
                    --n;
                    continue;
                }
            }
            else
                cur_silent_token_num = 0;

            llm_accept_token(token_id);

            if (is_stop_requested())
            {
                worker->llm_input = nullptr;
                cosyvoice_call_ggml_log_callback(GGML_LOG_LEVEL_INFO, "LLM generation stopped by user.\n");
                if (params.builtin_sampler_rng_policy == COSYVOICE_BUILTIN_SAMPLER_RNG_POLICY_RESET_PER_SESSION)
                    reset_builtin_sampler_rng();
                return false;
            }

            if (n > limit)
                break;

            cur = speech_emb + token_id * speech_row_size;
            if (!llm_decode(speech_type, cur))
                throw std::runtime_error("Failed to decode LLM output.\n");
        }
    }
    catch (const std::exception& e)
    {
        worker->llm_input = nullptr;
        llm_set_kv_cache_len(0);
        cosyvoice_call_ggml_log_callback(GGML_LOG_LEVEL_ERROR, e.what());
        if (params.builtin_sampler_rng_policy == COSYVOICE_BUILTIN_SAMPLER_RNG_POLICY_RESET_PER_SESSION)
            reset_builtin_sampler_rng();
        return false;
    }

    worker->llm_input = nullptr;
    if (final)
        *final = stop_reached;

    if (params.inference_buffer_policy == COSYVOICE_INFERENCE_BUFFER_POLICY_BALANCED && text != nullptr && max_new_tokens == UINT32_MAX && last_prompt_crc32 != prompt_crc32)
    {
        llm_set_kv_cache_len(1 + static_cast<uint32_t>(prompt->prompt_text.size()));
        llm_offload_kv_cache();
    }
    else if (params.inference_buffer_policy == COSYVOICE_INFERENCE_BUFFER_POLICY_BALANCED && max_new_tokens != UINT32_MAX && !stop_reached)
        llm_offload_kv_cache();

    // Reset RNG on stop (matches non-streaming behavior)
    if (stop_reached && params.builtin_sampler_rng_policy == COSYVOICE_BUILTIN_SAMPLER_RNG_POLICY_RESET_PER_SESSION)
        reset_builtin_sampler_rng();

    return true;
}
