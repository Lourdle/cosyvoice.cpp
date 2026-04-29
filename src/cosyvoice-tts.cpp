#include "cosyvoice-internal.h"
#include "cosyvoice-model.h"
#include "cosyvoice-llm-kv-cache.h"

#include <cfloat>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <span>

bool cosyvoice_model_3::llm_job(const int* text, uint32_t text_len, cosyvoice_prompt_t prompt)
{
    if (params.inference_buffer_policy == COSYVOICE_INFERENCE_BUFFER_POLICY_BALANCED)
        ggml_backend_sched_synchronize(sched.get());

    try
    {
        const auto n_batch = params.n_batch;
        const auto speech_type = llm.speech_embedding_weight->type;
        const auto speech_row_size = static_cast<uint32_t>(llm.speech_embedding_weight->nb[1]);
        const auto speech_emb = reinterpret_cast<const char*>(llm.speech_embedding_weight->data);
        const auto token_emb = reinterpret_cast<const char*>(llm.embed_tokens_weight->data);
        const char* cur;
        uint32_t offset = 0;
        if (speech_type == llm.embed_tokens_weight->type)
        {
            auto prefill_embedding = [&](const char* data, int token_id)
                {
                    if (offset == n_batch)
                    {
                        if (!llm_prefill(speech_type, batch_buffer.get(), n_batch))
                            throw std::runtime_error("Failed to prefill LLM KV cache.\n");
                        offset = 0;
                    }

                    memcpy(batch_buffer.get() + offset++ * speech_row_size, data + token_id * speech_row_size, speech_row_size);
                };

            if (llm_get_kv_cache_len() == 0)
            {
                prefill_embedding(speech_emb, llm.sos_token_id);
                prompt_crc32 = 0;
            }
            // The first token is assumed to be the SOS token already stored in the KV cache.
            if (prompt_crc32 != prompt->prompt_crc32)
            {
                llm_set_kv_cache_len(1);
                for (const auto& i : prompt->prompt_text)
                    prefill_embedding(token_emb, i);
                prompt_crc32 = prompt->prompt_crc32;
            }
            else llm_set_kv_cache_len(1 + static_cast<uint32_t>(prompt->prompt_text.size()));

            for (uint32_t i = 0; i != text_len; ++i)
                prefill_embedding(token_emb, text[i]);

            if (prompt->llm_prompt_speech_tokens.second != 0)
            {
                prefill_embedding(speech_emb, llm.task_token_id);

                const auto end = prompt->llm_prompt_speech_tokens.second - 1;
                for (uint32_t i = 0; i != end; ++i)
                    prefill_embedding(speech_emb, prompt->llm_prompt_speech_tokens.first[i]);
                cur = speech_emb + prompt->llm_prompt_speech_tokens.first[end] * speech_row_size;
            }
            else cur = speech_emb + llm.task_token_id * speech_row_size;
        }
        else
        {
            const auto token_type = llm.embed_tokens_weight->type;
            const auto token_row_size = static_cast<uint32_t>(llm.embed_tokens_weight->nb[1]);

            auto prefill_embedding = [&](const char* data, int token_id, uint32_t row_size, ggml_type type)
                {
                    if (offset == n_batch)
                    {
                        if (!llm_prefill(type, batch_buffer.get(), n_batch))
                            throw std::runtime_error("Failed to prefill LLM KV cache.\n");
                        offset = 0;
                    }

                    memcpy(batch_buffer.get() + offset++ * row_size, data + token_id * row_size, row_size);
                };

            if (llm_get_kv_cache_len() == 0)
                llm_prefill(speech_type, speech_emb + llm.sos_token_id * speech_row_size, 1);
            // The first token is assumed to be the SOS token already stored in the KV cache.
            if (prompt_crc32 != prompt->prompt_crc32)
            {
                llm_set_kv_cache_len(1);
                for (const auto& i : prompt->prompt_text)
                    prefill_embedding(token_emb, i, token_row_size, token_type);
                prompt_crc32 = prompt->prompt_crc32;
            }
            else llm_set_kv_cache_len(1 + static_cast<uint32_t>(prompt->prompt_text.size()));

            for (uint32_t i = 0; i != text_len; ++i)
                prefill_embedding(token_emb, text[i], token_row_size, token_type);

            if (offset != 0 && !llm_prefill(token_type, batch_buffer.get(), offset))
                throw std::runtime_error("Failed to prefill LLM KV cache.\n");
            offset = 0;

            if (prompt->llm_prompt_speech_tokens.second != 0)
            {
                prefill_embedding(speech_emb, llm.task_token_id, speech_row_size, speech_type);

                const auto end = prompt->llm_prompt_speech_tokens.second - 1;
                for (uint32_t i = 0; i != end; ++i)
                    prefill_embedding(speech_emb, prompt->llm_prompt_speech_tokens.first[i], speech_row_size, speech_type);
                cur = speech_emb + prompt->llm_prompt_speech_tokens.first[end] * speech_row_size;
            }
            else cur = speech_emb + llm.task_token_id * speech_row_size;
        }

        if (offset != 0 && !llm_prefill(speech_type, batch_buffer.get(), offset))
            throw std::runtime_error("Failed to prefill LLM KV cache.\n");

        const auto min_len = static_cast<uint32_t>(text_len * config.min_token_text_ratio);
        const auto max_len = static_cast<uint32_t>(text_len * config.max_token_text_ratio);
        tokens.clear();
        for (uint32_t n = 0; n != max_len; ++n)
        {
            if (!llm_decode(speech_type, cur))
                throw std::runtime_error("Failed to decode LLM output.\n");

            llm_prepare_probs(n > min_len);
            const auto token_id = llm_sample_token();
            if (token_id == -1)
                throw std::runtime_error("Failed to sample token from LLM output. This might be wrong with the model or caused by an issue with the sampling parameters.\n");
            if (n > min_len && llm_is_stop_token(token_id))
                break;

            llm_accept_token(token_id);
            cur = speech_emb + token_id * speech_row_size;
        }
    }
    catch (const std::exception& e)
    {
        cosyvoice_call_ggml_log_callback(GGML_LOG_LEVEL_ERROR, e.what());
        if (params.builtin_sampler_rng_policy == COSYVOICE_BUILTIN_SAMPLER_RNG_POLICY_RESET_PER_SESSION)
            reset_builtin_sampler_rng();
        return false;
    }

    if (params.inference_buffer_policy == COSYVOICE_INFERENCE_BUFFER_POLICY_BALANCED)
    {
        ggml_backend_sched_reset(sched.get());
        kv_cache->offload_cache(backend.get(), sched.get(), 1 + static_cast<uint32_t>(prompt->prompt_text.size()));
    }
    if (params.builtin_sampler_rng_policy == COSYVOICE_BUILTIN_SAMPLER_RNG_POLICY_RESET_PER_SESSION)
        reset_builtin_sampler_rng();

    return true;
}

static void set_graph_backends(ggml_cgraph* gf, ggml_backend_sched_t sched, ggml_backend_t backend, ggml_backend_t cpu_backend, ggml_backend_op_capabilities op_caps, int end = 0)
{
    auto is_virtual = [](ggml_op op) {
        return op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
    };

    auto target_backend = backend;
    for (auto node : ggml_cgraph_node_iterator(gf, end))
    {
        if (node->buffer)
            continue;
        else if (!is_virtual(node->op))
            switch (node->op)
            {
            case GGML_OP_CUSTOM:
                target_backend = cpu_backend;
                break;
            case GGML_OP_PAD:
                target_backend = op_caps.pad ? backend : cpu_backend;
                break;
            case GGML_OP_PAD_REFLECT_1D:
                target_backend = op_caps.pad_reflect_1d ? backend : cpu_backend;
                break;
            default:
                target_backend = backend;
            }
        ggml_backend_sched_set_tensor_backend(sched, node, target_backend);
    }
}

bool cosyvoice_model_3::token2wav(const int* token_ids, uint32_t n_tokens, float speed, cosyvoice_prompt_t prompt, cosyvoice_generated_speech_ptr result)
{
    ggml_reset(ctx0.get());
    ggml_reset(ctx1.get());
    if (params.inference_buffer_policy == COSYVOICE_INFERENCE_BUFFER_POLICY_BALANCED)
        ggml_backend_sched_synchronize(sched.get());
    ggml_backend_sched_reset(sched.get());

    ggml_tensor* token = ggml_new_tensor_1d(ctx0.get(), GGML_TYPE_I32, n_tokens);
    ggml_tensor* prompt_token = ggml_new_tensor_1d(ctx0.get(), GGML_TYPE_I32, prompt->flow_prompt_speech_tokens.second);
    ggml_tensor* prompt_feat = ggml_new_tensor_2d(ctx0.get(), GGML_TYPE_F32, prompt->prompt_speech_feat.shape[1], prompt->prompt_speech_feat.shape[0]);
    ggml_tensor* embedding = ggml_new_tensor_2d(ctx0.get(), GGML_TYPE_F32, prompt->flow_embedding.shape[1], prompt->flow_embedding.shape[0]);

    ggml_cgraph* gf = ggml_new_graph_custom(ctx0.get(), GGML_DEFAULT_GRAPH_SIZE * 3 / 2, false);
    auto [mu, spks, conds, cut_len] = flow.build_cgraph_encode(ctx0.get(), token, prompt_token, prompt_feat, embedding, op_caps);
    auto ditctx = flow.decoder.prepare_context(ctx1.get(), mu, spks, conds);
    do
    {
        auto buft = ggml_backend_get_default_buffer_type(backend.get());
        auto size = ggml_backend_alloc_ctx_tensors_from_buft_size(ctx1.get(), buft);

        if (!token2wav_buffer || size > ggml_backend_buffer_get_size(token2wav_buffer.get()))
        {
            reset_shared_buffer(ggml_backend_alloc_ctx_tensors_from_buft(ctx1.get(), buft));
            break;
        }

        auto alignment = ggml_backend_buffer_get_alignment(token2wav_buffer.get());
        auto buffer_base = reinterpret_cast<char*>(ggml_backend_buffer_get_base(token2wav_buffer.get()));
        for (auto tensor : std::span(reinterpret_cast<ggml_tensor**>(&ditctx), sizeof(ditctx) / sizeof(ggml_tensor*)))
        {
            ggml_backend_tensor_alloc(token2wav_buffer.get(), tensor, buffer_base);
            size = ggml_backend_buffer_get_alloc_size(token2wav_buffer.get(), tensor);
            size = get_aligned_size(size, alignment);
            buffer_base += size;
        }
    } while (false);

    uint32_t noise_len = static_cast<uint32_t>(ggml_nelements(ditctx.x));
    float* noise_buffer = noise_callback(COSYVOICE_NOISE_CALLBACK_STAGE_BEFORE_FLOW, noise_len, nullptr, noise_callback_ctx);
    ggml_backend_tensor_set_async(backend.get(), ditctx.x, noise_buffer, 0, ggml_nbytes(ditctx.x));

    auto feat = flow.decoder.build_cgraph_one_step(ctx0.get(), ditctx, 1, op_caps, cut_len);
    ggml_build_forward_expand(gf, feat);
    set_graph_backends(gf, sched.get(), backend.get(), cpu_backend.get(), op_caps);
    ggml_backend_sched_synchronize(sched.get());
    ggml_backend_sched_alloc_graph(sched.get(), gf);

    ggml_backend_tensor_set_async(backend.get(), token, token_ids, 0, token->nb[1]);
    ggml_backend_tensor_set_async(backend.get(), prompt_token, prompt->flow_prompt_speech_tokens.first.get(), 0, prompt_token->nb[1]);
    ggml_backend_tensor_set_async(backend.get(), prompt_feat, prompt->prompt_speech_feat.data, 0, prompt_feat->nb[2]);
    ggml_backend_tensor_set_async(backend.get(), embedding, prompt->flow_embedding.data, 0, embedding->nb[2]);

    status = ggml_backend_sched_graph_compute(sched.get(), gf);
    noise_callback(COSYVOICE_NOISE_CALLBACK_STAGE_AFTER_FLOW, noise_len, noise_buffer, noise_callback_ctx);
    if (params.inference_buffer_policy == COSYVOICE_INFERENCE_BUFFER_POLICY_SHARED)
        llm_set_kv_cache_len(0); // Reset the visible KV length when sharing the buffer.
    if (status != GGML_STATUS_SUCCESS)
        return false;

    for (auto tensor : std::span(&ditctx.mu_in, 3))
    {
        tensor->op = GGML_OP_NONE;
        tensor->view_src = nullptr;
        tensor->view_offs = 0;
        memset(tensor->src, 0, sizeof(tensor->src));
    }

    ggml_tensor* t_leaf;
    ggml_backend_tensor_copy_async(backend.get(), backend.get(), feat, ditctx.x);

    ggml_reset(ctx0.get());
    ggml_backend_sched_reset(sched.get());
    gf = ggml_new_graph(ctx0.get());
    feat = flow.decoder.build_cgraph_one_step(ctx0.get(), ditctx, 2, op_caps, cut_len, &t_leaf);

    ggml_build_forward_expand(gf, feat);
    set_graph_backends(gf, sched.get(), backend.get(), cpu_backend.get(), op_caps);

    ggml_backend_sched_alloc_graph(sched.get(), gf);
    status = ggml_backend_sched_graph_compute(sched.get(), gf);
    if (status != GGML_STATUS_SUCCESS) return false;

    auto scale_node = feat->src[1];
    GGML_ASSERT(scale_node->op == GGML_OP_SCALE);

    for (int step = 3; step != flow.decoder.t_span.size() - 1; ++step)
    {
        ggml_backend_tensor_copy_async(backend.get(), backend.get(), feat, ditctx.x);

        auto [t, dt] = flow.decoder.get_t_and_dt(ctx0.get(), step);
        reinterpret_cast<float*>(t_leaf->op_params)[0] = t;
        reinterpret_cast<float*>(scale_node->op_params)[0] = dt;

        status = ggml_backend_sched_graph_compute(sched.get(), gf);
        if (status != GGML_STATUS_SUCCESS)
            return false;
    }

    ggml_backend_tensor_copy_async(backend.get(), backend.get(), feat, ditctx.x);
    ggml_reset(ctx0.get());
    ggml_backend_sched_reset(sched.get());
    gf = ggml_new_graph(ctx0.get());
    feat = flow.decoder.build_cgraph_one_step(ctx0.get(), ditctx, static_cast<int>(flow.decoder.t_span.size() - 1), op_caps, cut_len, nullptr);

    ggml_build_forward_expand(gf, feat);
    set_graph_backends(gf, sched.get(), backend.get(), cpu_backend.get(), op_caps);

    ggml_backend_sched_alloc_graph(sched.get(), gf);
    status = ggml_backend_sched_graph_compute(sched.get(), gf);
    if (status != GGML_STATUS_SUCCESS) return false;

    ggml_reset(ctx1.get());
    ggml_tensor* speech_feat = ggml_new_tensor(ctx1.get(), feat->type, GGML_MAX_DIMS, feat->ne);
    ggml_backend_tensor_alloc(token2wav_buffer.get(), speech_feat, ggml_backend_buffer_get_base(token2wav_buffer.get()));
    ggml_backend_tensor_copy_async(backend.get(), backend.get(), feat, speech_feat);

    ggml_reset(ctx0.get());
    ggml_backend_sched_reset(sched.get());
    gf = ggml_new_graph(ctx0.get());

    if (std::abs(speed - 1.f) > FLT_EPSILON)
        speech_feat = ggml_interpolate(ctx0.get(), speech_feat,
            static_cast<int64_t>(speech_feat->ne[0] / speed), speech_feat->ne[1], speech_feat->ne[2], speech_feat->ne[3],
            GGML_SCALE_MODE_BILINEAR);

    auto [generated_speech, noise] = hift.build_cgraph(ctx0.get(), speech_feat);
    ggml_build_forward_expand(gf, generated_speech);
    set_graph_backends(gf, sched.get(), backend.get(), cpu_backend.get(), op_caps, -1);

    ggml_backend_sched_set_tensor_backend(sched.get(), generated_speech, cpu_backend.get());
    ggml_backend_sched_alloc_graph(sched.get(), gf);

    for (auto node : ggml_cgraph_node_iterator(gf))
        if (node->op == GGML_OP_IM2COL && node->ne[1] > 0xFFFF)
            node->op = GGML_OP_NONE;

    noise_len = static_cast<uint32_t>(ggml_nelements(noise));
    noise_buffer = noise_callback(COSYVOICE_NOISE_CALLBACK_STAGE_BEFORE_HIFT, noise_len, nullptr, noise_callback_ctx);
    ggml_backend_tensor_set_async(backend.get(), noise, noise_buffer, 0, noise->nb[2]);

    result->data = reinterpret_cast<float*>(generated_speech->data);
    result->length = static_cast<uint32_t>(generated_speech->ne[0]);
    status = ggml_backend_sched_graph_compute(sched.get(), gf);
    noise_callback(COSYVOICE_NOISE_CALLBACK_STAGE_AFTER_HIFT, noise_len, noise_buffer, noise_callback_ctx);
    if (params.inference_buffer_policy == COSYVOICE_INFERENCE_BUFFER_POLICY_BALANCED)
    {
        ggml_backend_sched_reset(sched.get());
        kv_cache->load_cache(backend.get(), sched.get());
    }
    return status == GGML_STATUS_SUCCESS;
}

struct cosyvoice_tts_context : cosyvoice_tokenization_result_impl, cosyvoice_prompt, std::string
{
    cosyvoice_tts_context(cosyvoice_context_t ctx, cosyvoice_prompt_t prompt)
        : cosyvoice_prompt(*prompt), ctx(ctx), text_normalization_enabled(true)
    {
        const auto instruction_prefix = ctx->get_instruction_prefix();
        if (instruction_prefix)
        {
            instruction_cache = instruction_prefix;
            prefix_len = instruction_cache.size();
        }
        else prefix_len = 0;
    }

    bool tts_job(const char* text, const char* instruction, float speed, cosyvoice_inference_mode mode, cosyvoice_generated_speech_ptr result)
    {
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

        bool normalized = false;
        if (text_normalization_enabled)
            normalized = cosyvoice_frontend_util_text_normalize(*this, text, static_cast<uint32_t>(strlen(text)), nullptr);
        cosyvoice_tokenize(
            ctx,
            normalized ? c_str() : text,
            this,
            true);

        return cosyvoice_tts(ctx, get_tokens(), get_n_tokens(), speed, this, result);
    }

    cosyvoice_context_t ctx;
    std::string instruction_cache;
    size_t prefix_len;
    bool text_normalization_enabled;
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

void cosyvoice_tts_context_set_text_normalization_enabled(cosyvoice_tts_context_t ctx, bool enabled)
{
    ctx->text_normalization_enabled = enabled;
}

bool cosyvoice_tts_context_get_text_normalization_enabled(cosyvoice_tts_context_t ctx)
{
    return ctx->text_normalization_enabled;
}

bool cosyvoice_tts_zero_shot(cosyvoice_tts_context_t ctx, const char* text, float speed, cosyvoice_generated_speech_ptr result)
{
    return ctx->tts_job(text, nullptr, speed, COSYVOICE_INFERENCE_MODE_ZERO_SHOT, result);
}

bool cosyvoice_tts_instruct(cosyvoice_tts_context_t ctx, const char* text, const char* instruction, float speed, cosyvoice_generated_speech_ptr result)
{
    return ctx->tts_job(text, instruction, speed, COSYVOICE_INFERENCE_MODE_INSTRUCT, result);
}

bool cosyvoice_tts_cross_lingual(cosyvoice_tts_context_t ctx, const char* text, float speed, cosyvoice_generated_speech_ptr result)
{
    return ctx->tts_job(text, nullptr, speed, COSYVOICE_INFERENCE_MODE_CROSS_LINGUAL, result);
}
