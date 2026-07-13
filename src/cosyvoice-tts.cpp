#include "cosyvoice-internal.h"
#include "cosyvoice-model.h"
#include "cosyvoice-kv-cache.h"
#include "cosyvoice-text-chunk.h"
#include "ggml-cpu-flag.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <span>
#include <vector>

constexpr uint32_t COSYVOICE_TTS_FLAG_MASK = COSYVOICE_TTS_FLAG_TEXT_NORMALIZATION | COSYVOICE_TTS_FLAG_SPLIT_TEXT | COSYVOICE_TTS_FLAG_FAST_SPLIT | COSYVOICE_TTS_FLAG_FADE_IN;
constexpr double COSYVOICE_TTS_FADE_IN_SECONDS = 0.02;

bool cosyvoice_model_3::llm_job(const int* text, uint32_t text_len, cosyvoice_prompt_t prompt)
{
    llm_clear_accepted_tokens();
    return llm_job_ext(text, text_len, prompt, UINT32_MAX, nullptr);
}

bool cosyvoice_model_3::llm_job_ext(const int* text, uint32_t text_len, cosyvoice_prompt_t prompt, uint32_t max_new_tokens, bool* final)
{
    const auto& params = shared->params;
    auto& sched = worker->sched;
    const auto& llm = cv3_shared->llm;
    auto& batch_buffer = worker->batch_buffer;
    auto& prompt_crc32 = worker->prompt_crc32;
    auto last_prompt_crc32 = prompt_crc32;
    bool stop_reached = false;

    try
    {
        const auto n_batch = params.n_batch;
        const auto speech_type = llm.speech_embedding_weight->type;
        const auto speech_row_size = static_cast<uint32_t>(llm.speech_embedding_weight->nb[1]);
        const auto speech_emb = reinterpret_cast<const char*>(llm.speech_embedding_weight->data);
        const auto token_emb = reinterpret_cast<const char*>(llm.embed_tokens_weight->data);
        const char* cur;
        uint32_t offset = 0;

        if (text != nullptr)
        {
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
        }
        else
        {
            // Continue from existing KV cache — get last accepted token's embedding
            const auto n_acc = llm_get_n_accepted_tokens();
            if (n_acc == 0)
                throw std::runtime_error("llm_job_ext: cannot continue generation with empty token history.\n");
            const auto last_token_id = llm_get_accepted_tokens()[n_acc - 1];
            cur = speech_emb + last_token_id * speech_row_size;
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

        for (uint32_t n = llm_get_n_accepted_tokens(); n != limit; ++n)
        {
            if (!llm_decode(speech_type, cur))
                throw std::runtime_error("Failed to decode LLM output.\n");

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
            cur = speech_emb + token_id * speech_row_size;
        }
    }
    catch (const std::exception& e)
    {
        worker->llm_input = nullptr;
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

    // Reset RNG on stop (matches non-streaming behavior)
    if (stop_reached && params.builtin_sampler_rng_policy == COSYVOICE_BUILTIN_SAMPLER_RNG_POLICY_RESET_PER_SESSION)
        reset_builtin_sampler_rng();

    return true;
}

static void set_graph_backends(ggml_cgraph* gf, ggml_backend_sched_t sched, ggml_backend_t backend, ggml_backend_t cpu_backend, ggml_backend_op_capabilities op_caps, int end = 0)
{
    auto is_virtual = [](ggml_op op) {
        return op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
    };

    for (auto node : ggml_cgraph_node_iterator(gf, end))
    {
        if (node->data)
            continue;

        ggml_backend_t target_backend;
        if (ggml_cpu_fallback(node))
            target_backend = cpu_backend;
        else
        {
            auto op = node->op;
            auto src = node;
            if (is_virtual(op))
            {
                src = node;
                while (src && is_virtual(src->op))
                    src = src->src[0];
                op = src->op;
            }

            switch (op)
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
            case GGML_OP_CUMSUM:
                target_backend = op_caps.cumsum ? backend : cpu_backend;
                break;
            case GGML_OP_LEAKY_RELU:
                target_backend = op_caps.leaky_relu ? backend : cpu_backend;
                break;
            case GGML_OP_SIN:
                target_backend = op_caps.sin ? backend : cpu_backend;
                break;
            case GGML_OP_COS:
                target_backend = op_caps.cos ? backend : cpu_backend;
                break;
            case GGML_OP_ARANGE:
                target_backend = op_caps.arange ? backend : cpu_backend;
                break;
            case GGML_OP_ACC:
                target_backend = op_caps.acc ? backend : cpu_backend;
                break;
            case GGML_OP_UNARY:
                switch (ggml_get_unary_op(src))
                {
                case GGML_UNARY_OP_ELU:
                    target_backend = op_caps.elu ? backend : cpu_backend;
                    break;
                case GGML_UNARY_OP_ABS:
                    target_backend = op_caps.abs ? backend : cpu_backend;
                    break;
                case GGML_UNARY_OP_FLOOR:
                    target_backend = op_caps.floor ? backend : cpu_backend;
                    break;
                default:
                    target_backend = backend;
                }
                break;
            case GGML_OP_CPY:
                if (node->type == GGML_TYPE_I32 && node->src[0]->type == GGML_TYPE_F32)
                {
                    target_backend = cpu_backend;
                    break;
                }
            default:
                target_backend = backend;
            }
        }
        ggml_backend_sched_set_tensor_backend(sched, node, target_backend);
    }
}

bool cosyvoice_model_3::token2wav(const int* token_ids, uint32_t n_tokens, float speed, cosyvoice_prompt_t prompt, cosyvoice_generated_speech_ptr result)
{
    return token2wav_ext(token_ids, n_tokens, speed, prompt, nullptr, false, true, result);
}

template<int n>
struct dit_sched_config
{
    struct graph_config_t
    {
        bool rebuild;
        bool cache_kv;
        bool load;
        bool offload;
        bool slide;
        bool slice;
        bool mask;
        int64_t cut_len;
    };

    graph_config_t graph_config[n];

    const auto& operator[](int i) const { return graph_config[i]; }

    dit_sched_config(const cosyvoice_context_params_v3_cpp& params, int64_t cut_len, uint32_t offset, bool streaming, bool kv_slidable)
    {
        if (!streaming)
        {
            for (int i = 0; i != n; ++i)
            {
                graph_config[i].rebuild = false;
                graph_config[i].cache_kv = false;
                graph_config[i].load = false;
                graph_config[i].offload = false;
                graph_config[i].slide = false;
                graph_config[i].slice = false;
                graph_config[i].mask = false;
                graph_config[i].cut_len = 0;
            }

            graph_config[0].rebuild = true;
            graph_config[1].rebuild = true;
            graph_config[n - 1].rebuild = true;
            graph_config[n - 1].cut_len = cut_len;
        }
        else
        {
            if (offset != 0) cut_len = 0;

            const auto n_offloadable_steps = params.dit_kv_offloadable_slots;
            const auto n_fixed_steps = params.dit_kv_fixed_slots;
            const auto n_no_cache_steps = static_cast<uint32_t>(n) - n_offloadable_steps - n_fixed_steps;

            for (int i = 0; i != n; ++i)
            {
                graph_config[i].rebuild = i == 0
                    || i == 1
                    || offset != 0 && i == n_no_cache_steps - 1
                    || i == n_no_cache_steps
                    || i == n - 1 && cut_len != 0
                    || !kv_slidable && i >= n_no_cache_steps;
                graph_config[i].cache_kv = i >= n_no_cache_steps;
                graph_config[i].offload = i >= n_no_cache_steps && i < n_no_cache_steps + n_offloadable_steps;
                graph_config[i].load = graph_config[i].offload && offset != 0;
                graph_config[i].mask = i < n_no_cache_steps && offset != 0;
                graph_config[i].cut_len = i == n - 1 && offset == 0 ? cut_len : 0;
                if (i == n_no_cache_steps - 1 && offset != 0)
                    graph_config[i].cut_len = offset;
                graph_config[i].slice = offset != 0 && i == n_no_cache_steps;
                graph_config[i].slide = kv_slidable && graph_config[i].cache_kv && !graph_config[i].offload && i != n - 1;
            }
        }
    }
};

bool cosyvoice_model_3::token2wav_ext(const int* token_ids, uint32_t n_tokens, float speed, cosyvoice_prompt_t prompt, uint32_t* offset, bool streaming, bool finalize, cosyvoice_generated_speech_ptr result)
{
    const auto& params = shared->params;
    auto& sched = worker->sched;
    auto& flow = cv3_shared->flow;
    auto& hift = cv3_shared->hift;
    auto& batch_buffer = worker->batch_buffer;
    auto& prompt_crc32 = worker->prompt_crc32;
    auto& ctx0 = worker->ctx0;
    auto& ctx1 = cv3_worker->ctx1;
    auto& backend = worker->backend;
    auto& cpu_backend = worker->cpu_backend;
    auto& token2wav_buffer = cv3_worker->token2wav_buffer;
    auto op_caps = shared->op_caps;

    ggml_reset(ctx0.get());
    ggml_reset(ctx1.get());
    ggml_backend_sched_reset(sched.get());

    ggml_tensor* token = ggml_new_tensor_1d(ctx0.get(), GGML_TYPE_I32, n_tokens);
    ggml_tensor* prompt_token = ggml_new_tensor_1d(ctx0.get(), GGML_TYPE_I32, prompt->flow_prompt_speech_tokens.second);
    ggml_tensor* prompt_feat = ggml_new_tensor_2d(ctx0.get(), GGML_TYPE_F32, prompt->prompt_speech_feat.shape[1], prompt->prompt_speech_feat.shape[0]);
    ggml_tensor* embedding = ggml_new_tensor_2d(ctx0.get(), GGML_TYPE_F32, prompt->flow_embedding.shape[1], prompt->flow_embedding.shape[0]);
    auto kv_cache = &worker->dit_kv_cache;

    // Phase 1: Flow encode
    ggml_cgraph* gf = new_cgraph(ctx0.get());
    auto [mu, spks, conds, cut_len] = flow.build_cgraph_encode(ctx0.get(), token, prompt_token, prompt_feat, embedding, op_caps, (flow.decoder.diffusion_steps - params.dit_kv_fixed_slots - params.dit_kv_offloadable_slots) == 0 && offset ? *offset : 0, streaming);
    const auto chunk_len = mu->ne[1];
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

    dit_sched_config<flow.decoder.diffusion_steps> config(params, cut_len, offset ? *offset : 0, streaming, kv_cache->can_reuse());
    if (offset)
        if (flow.decoder.diffusion_steps == params.dit_kv_fixed_slots + params.dit_kv_offloadable_slots)
            *offset += chunk_len;
        else
            *offset = chunk_len;

    int position_offset = kv_cache ? static_cast<int>(kv_cache->cur_len) : 0;
    uint32_t noise_len = static_cast<uint32_t>(ggml_nelements(ditctx.x));
    float* noise_buffer = shared->noise_callback(COSYVOICE_NOISE_CALLBACK_STAGE_BEFORE_FLOW, noise_len, nullptr, shared->noise_callback_ctx);
    ggml_backend_tensor_set_async(backend.get(), ditctx.x, noise_buffer, 0, ggml_nbytes(ditctx.x));

    // Phase 2: Flow decode steps
    ggml_tensor* t_leaf;
    ggml_tensor* position_ids;
    ggml_tensor* attn_mask;
    if (config[0].cache_kv) kv_cache->bind_slot(0);
    auto feat = flow.decoder.build_cgraph_one_step(ctx0.get(), ditctx, 1, op_caps, config[0].cut_len, t_leaf, position_ids, gf, config[0].cache_kv ? kv_cache : nullptr, config[0].mask ? &attn_mask : nullptr);
    ggml_build_forward_expand(gf, feat);
    set_graph_backends(gf, sched.get(), backend.get(), cpu_backend.get(), op_caps);
    ggml_backend_sched_synchronize(sched.get());
    ggml_backend_sched_alloc_graph(sched.get(), gf);

    auto post_process = [&](int step)
    {
        if (!op_caps.fill) ggml_set_zero(t_leaf);
        for (int64_t i = 0; i < position_ids->ne[1]; ++i)
        {
            auto cur_row = reinterpret_cast<int32_t*>(position_ids->data) + i * position_ids->ne[0];
            for (int32_t j = 0; j < position_ids->ne[0]; ++j)
                cur_row[j] = j + position_offset;
        }
        if (config[step].mask)
        {
            const auto cs = flow.decoder.estimator.static_chunk_size;
            for (int64_t i = 0; i < attn_mask->ne[0]; ++i)
            {
                int64_t block_end = ((i / cs) + 1) * cs;
                auto row = reinterpret_cast<ggml_fp16_t*>(attn_mask->data) + i * attn_mask->ne[1];
                for (int64_t j = 0; j < attn_mask->ne[1]; ++j)
                    row[j] = j < block_end ? 0 : 0xFC00;
            }
        }
    };

    post_process(0);

    ggml_backend_tensor_set_async(backend.get(), token, token_ids, 0, token->nb[1]);
    ggml_backend_tensor_set_async(backend.get(), prompt_token, prompt->flow_prompt_speech_tokens.first.get(), 0, prompt_token->nb[1]);
    ggml_backend_tensor_set_async(backend.get(), prompt_feat, prompt->prompt_speech_feat.data, 0, prompt_feat->nb[2]);
    ggml_backend_tensor_set_async(backend.get(), embedding, prompt->flow_embedding.data, 0, embedding->nb[2]);

    worker->status = ggml_backend_sched_graph_compute(sched.get(), gf);
    shared->noise_callback(COSYVOICE_NOISE_CALLBACK_STAGE_AFTER_FLOW, noise_len, noise_buffer, shared->noise_callback_ctx);
    if (params.inference_buffer_policy == COSYVOICE_INFERENCE_BUFFER_POLICY_SHARED)
        llm_set_kv_cache_len(0);
    if (worker->status != GGML_STATUS_SUCCESS)
        return false;

    for (auto tensor : std::span(&ditctx.mu_in, 3))
    {
        tensor->op = GGML_OP_NONE;
        tensor->view_src = nullptr;
        tensor->view_offs = 0;
        memset(tensor->src, 0, sizeof(tensor->src));
    }

    int offload_slot = 0;
    int kv_slot = config[0].cache_kv && !config[0].offload;
    for (int step = 1; step != flow.decoder.diffusion_steps; ++step)
    {
        if (config[step].slice)
        {
            const auto cut_len = config[step - 1].cut_len;
            reinterpret_cast<char*&>(ditctx.cond_in->data) += static_cast<size_t>(ditctx.cond_in->nb[1]) * cut_len;
            ditctx.cond_in->ne[1] -= cut_len;
            reinterpret_cast<char*&>(ditctx.mu_in->data) += static_cast<size_t>(ditctx.mu_in->nb[1]) * cut_len;
            ditctx.mu_in->ne[1] -= cut_len;

            memcpy(ditctx.x->ne, feat->ne, sizeof(ditctx.x->ne));
            memcpy(ditctx.x->nb, feat->nb, sizeof(ditctx.x->nb));
        }
        ggml_backend_tensor_copy_async(backend.get(), backend.get(), feat, ditctx.x);

        if (config[step].rebuild)
        {
            if (config[step].cache_kv && !config[step].offload)
                kv_cache->bind_slot(kv_slot++);
            ggml_reset(ctx0.get());
            ggml_backend_sched_reset(sched.get());

            if (config[step].load)
            {
                kv_cache->load_slot(backend.get(), sched.get(), offload_slot);
                ggml_backend_sched_reset(sched.get());
            }

            gf = new_cgraph(ctx0.get());
            feat = flow.decoder.build_cgraph_one_step(ctx0.get(), ditctx, step + 1, op_caps, config[step].cut_len, t_leaf, position_ids, gf, config[step].cache_kv ? kv_cache : nullptr, config[step].mask ? &attn_mask : nullptr);
            ggml_build_forward_expand(gf, feat);
            set_graph_backends(gf, sched.get(), backend.get(), cpu_backend.get(), op_caps);
            ggml_backend_sched_alloc_graph(sched.get(), gf);
            post_process(step);
        }
        else
        {
            auto scale_node = feat->src[1];
            GGML_ASSERT(scale_node->op == GGML_OP_SCALE);
            auto [t, dt] = flow.decoder.get_t_and_dt(ctx0.get(), step);
            reinterpret_cast<float*>(t_leaf->op_params)[op_caps.fill ? 0 : 1] = t;
            reinterpret_cast<float*>(scale_node->op_params)[0] = dt;

            if (config[step].load)
                kv_cache->load_slot(backend.get(), sched.get(), offload_slot);
        }

        worker->status = ggml_backend_sched_graph_compute(sched.get(), gf);
        if (worker->status != GGML_STATUS_SUCCESS)
            return false;

        if (config[step].offload)
        {
            if (step != flow.decoder.diffusion_steps - 1 && config[step + 1].rebuild)
                ggml_backend_sched_reset(sched.get());
            kv_cache->offload_slot(backend.get(), sched.get(), offload_slot++);
        }
        if (config[step].slide)
            kv_cache->slide_kv_slot();
    }

    // Phase 3: Copy flow output to speech_feat (persistent buffer)
    ggml_reset(ctx1.get());
    auto cache_length = static_cast<int64_t>(worker->flow_cache.size() / feat->ne[0]);
    ggml_tensor* speech_feat = ggml_new_tensor_2d(ctx1.get(), feat->type, feat->ne[0], feat->ne[1] + cache_length);
    if (cache_length != 0)
    {
        auto buft = ggml_backend_get_default_buffer_type(backend.get());
        auto size = ggml_backend_buft_get_alloc_size(buft, speech_feat);
        if (size > ggml_backend_buffer_get_size(token2wav_buffer.get()))
            reset_shared_buffer(ggml_backend_alloc_ctx_tensors_from_buft(ctx1.get(), buft));
        else
            ggml_backend_tensor_alloc(token2wav_buffer.get(), speech_feat,
                ggml_backend_buffer_get_base(token2wav_buffer.get()));

        ggml_backend_tensor_set_async(backend.get(), speech_feat, worker->flow_cache.data(), 0, speech_feat->nb[2] - feat->nb[2]);
        ggml_backend_tensor_copy_async(backend.get(), backend.get(), feat,
            ggml_view_2d(ctx0.get(), speech_feat, speech_feat->ne[0], feat->ne[1], speech_feat->nb[1], speech_feat->nb[1] * cache_length));
    }
    else
    {
        ggml_backend_tensor_alloc(token2wav_buffer.get(), speech_feat,
            ggml_backend_buffer_get_base(token2wav_buffer.get()));
        ggml_backend_tensor_copy_async(backend.get(), backend.get(), feat, speech_feat);
    }

    if (!finalize)
    {
        auto n_cached_elements = cache_length * feat->ne[0];
        worker->flow_cache.resize(n_cached_elements + static_cast<size_t>(ggml_nelements(feat)));
        ggml_backend_tensor_get_async(backend.get(), feat, worker->flow_cache.data() + n_cached_elements, 0, feat->nb[2]);
    }

    if (config[flow.decoder.diffusion_steps - 1].cache_kv)
        if (finalize)
            kv_cache->cur_len = 0;
        else
            kv_cache->cur_len += static_cast<uint32_t>(position_ids->ne[0]);
    ggml_reset(ctx0.get());
    ggml_backend_sched_reset(sched.get());
    gf = new_cgraph(ctx0.get());

    speech_feat = ggml_permute(ctx0.get(), speech_feat, 1, 0, 2, 3);
    if (auto ne0 = static_cast<int64_t>(speech_feat->ne[0] / speed); ne0 != speech_feat->ne[0])
        speech_feat = ggml_interpolate(ctx0.get(), speech_feat,
            ne0, speech_feat->ne[1], speech_feat->ne[2], speech_feat->ne[3],
            GGML_SCALE_MODE_BILINEAR);
    else
        speech_feat = ggml_cont(ctx0.get(), speech_feat);

    auto [generated_speech, noise] = hift.build_cgraph(ctx0.get(), speech_feat, finalize);
    ggml_build_forward_expand(gf, generated_speech);
    set_graph_backends(gf, sched.get(), backend.get(), cpu_backend.get(), op_caps);

    ggml_backend_sched_alloc_graph(sched.get(), gf);

    for (auto node : ggml_cgraph_node_iterator(gf))
        if (node->op == GGML_OP_IM2COL && node->ne[1] > 0xFFFF)
            node->op = GGML_OP_NONE;

    noise_len = static_cast<uint32_t>(ggml_nelements(noise));
    noise_buffer = shared->noise_callback(COSYVOICE_NOISE_CALLBACK_STAGE_BEFORE_HIFT, noise_len, nullptr, shared->noise_callback_ctx);
    ggml_backend_tensor_set_async(backend.get(), noise, noise_buffer, 0, noise->nb[2]);

    result->data = reinterpret_cast<float*>(generated_speech->data);
    result->length = static_cast<uint32_t>(generated_speech->ne[0]);
    worker->status = ggml_backend_sched_graph_compute(sched.get(), gf);
    shared->noise_callback(COSYVOICE_NOISE_CALLBACK_STAGE_AFTER_HIFT, noise_len, noise_buffer, shared->noise_callback_ctx);

    if (finalize)
        worker->flow_cache.clear();

    if (params.inference_buffer_policy == COSYVOICE_INFERENCE_BUFFER_POLICY_BALANCED)
    {
        ggml_backend_sched_reset(sched.get());
        worker->llm_kv_cache.load_cache(backend.get(), sched.get());
    }
    return worker->status == GGML_STATUS_SUCCESS;
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
        if (flags & COSYVOICE_TTS_FLAG_TEXT_NORMALIZATION)
            normalized = cosyvoice_frontend_util_text_normalize(*this, text, static_cast<uint32_t>(strlen(text)), nullptr);
        const char* effective_text = normalized ? c_str() : text;

        // When splitting is disabled, do a single tokenize+synthesize pass.
        if (!(flags & COSYVOICE_TTS_FLAG_SPLIT_TEXT))
        {
            ctx->tokenize(effective_text, this, true);
            return cosyvoice_tts_with_postprocess(get_tokens(), get_n_tokens(), speed, result);
        }

        const auto fragments = cosyvoice_internal::split_into_fragments(effective_text);
        if (fragments.size() <= 1)
        {
            ctx->tokenize(effective_text, this, true);
            return cosyvoice_tts_with_postprocess(get_tokens(), get_n_tokens(), speed, result);
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
            ctx->tokenize(effective_text, this, true);
            return result ? cosyvoice_tts_with_postprocess(get_tokens(), get_n_tokens(), speed, result)
                : cosyvoice_tts_stream_with_postprocess(get_tokens(), get_n_tokens(), speed, callback, user_data);
        }
        const auto max_text_tokens = static_cast<std::size_t>(
            static_cast<float>(params.n_max_seq - o - 1u) / (1.0f + r));

        if (max_text_tokens == 0)
        {
            ctx->tokenize(effective_text, this, true);
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
                    ctx->tokenize(fragment.c_str(), &tok, true);
                    fragment_tokens.push_back(tok.tokens);
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
                tokens = std::move(chunk_tokens);
                if (!cosyvoice_tts_stream_with_postprocess(get_tokens(), get_n_tokens(), speed, callback, user_data))
                    return false;
            }
        }

        // Slow-split path: tokenize each fragment only for counting,
        // reassemble text chunks, then re-tokenize each chunk.
        cosyvoice_tokenization_result_impl tokens_scratch;
        const auto token_count = [&](std::string_view fragment) -> std::size_t
        {
            tokens_scratch.tokens.clear();
            ctx->tokenize(fragment.data(), static_cast<uint32_t>(fragment.size()), &tokens_scratch, true);
            return tokens_scratch.get_n_tokens();
        };
        const auto chunks = cosyvoice_internal::reassemble_by_token_budget(
            fragments, max_text_tokens, token_count);

        if (chunks.size() <= 1)
        {
            ctx->tokenize(effective_text, this, true);
            return cosyvoice_tts_with_postprocess(get_tokens(), get_n_tokens(), speed, result);
        }

        // Multi-chunk path: synthesize each chunk and copy its PCM into a context-owned buffer.
        // cosyvoice_tts() points result->data at an internal token2wav buffer that is overwritten
        // on the next call, so the copy must happen before the following synthesis begins.
        combined_pcm.clear();
        for (const auto& chunk : chunks)
        {
            ctx->tokenize(chunk.c_str(), this, true);
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

    bool cosyvoice_tts_stream_with_postprocess(const int* token_ids, uint32_t n_tokens, float speed, cosyvoice_tts_audio_callback_t callback, void* user_data)
    {
        bool ok;
        if (flags & COSYVOICE_TTS_FLAG_FADE_IN)
            ok = cosyvoice_tts_stream(ctx, token_ids, n_tokens, speed, this, callback, user_data);
        else
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
