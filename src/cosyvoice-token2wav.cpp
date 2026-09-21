#include "cosyvoice-internal.h"
#include "cosyvoice-model.h"
#include "ggml-cpu-flag.h"

#include <algorithm>
#include <cstring>
#include <span>
#include <vector>

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
    return token2wav_ext(token_ids, n_tokens, speed, prompt, false, true, result);
}

struct dit_sched_config
{
    struct graph_config_t
    {
        bool rebuild;
        bool cache_kv;
        bool offload;
        bool load;
        bool store;
        bool slide;
        bool slice;
        bool mask;
        int phys_slot;
        int off_group;
        int64_t cut_len;
    };

    static constexpr int n_max = CausalConditionalCFM::MAX_DIFFUSION_STEPS;
    int n;
    graph_config_t graph_config[n_max];

    const auto& operator[](int i) const { return graph_config[i]; }

    dit_sched_config(const cosyvoice_context_params_v4_cpp& params, int64_t cut_len, uint32_t offset, bool streaming, bool kv_slidable, int diffusion_steps)
        : n(diffusion_steps)
    {
        if (!streaming)
        {
            for (int i = 0; i != n; ++i)
            {
                graph_config[i].rebuild = false;
                graph_config[i].cache_kv = false;
                graph_config[i].offload = false;
                graph_config[i].load = false;
                graph_config[i].store = false;
                graph_config[i].slide = false;
                graph_config[i].slice = false;
                graph_config[i].mask = false;
                graph_config[i].phys_slot = 0;
                graph_config[i].off_group = 0;
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
            const auto n_actual_fixed = params.dit_kv_actual_fixed_slots;
            const auto n_actual_offloadable = params.dit_kv_actual_offloadable_slots;
            // Device slot 0 is the scratch buffer for the whole offloadable region;
            // fixed steps map onto `scratch_slot + group`, so the slot index is
            // non-decreasing and increases by at most one per step.
            const auto scratch_slot = n_offloadable_steps != 0 ? 1 : 0;
            const auto o_start = n_no_cache_steps;
            const auto o_end = n_no_cache_steps + n_offloadable_steps;
            const auto f_start = o_end;

            // Adjacent steps mapped to the same group share one physical KV cache.
            const auto off_group_of = [&](int i)
            {
                return (i - static_cast<int>(o_start)) * n_actual_offloadable / n_offloadable_steps;
            };
            const auto phys_slot_of = [&](int i)
            {
                if (i < static_cast<int>(o_end)) return 0;
                return scratch_slot + static_cast<int>((i - static_cast<int>(f_start)) * n_actual_fixed / n_fixed_steps);
            };

            for (int i = 0; i != n; ++i)
            {
                graph_config[i].rebuild = i <= 1
                    || offset != 0 && i == n_no_cache_steps - 1
                    || i == n_no_cache_steps
                    || i == n - 1 && cut_len != 0
                    || !kv_slidable && i >= n_no_cache_steps;
                graph_config[i].cache_kv = i >= n_no_cache_steps;
                graph_config[i].offload = i >= static_cast<int>(o_start) && i < static_cast<int>(o_end);
                graph_config[i].phys_slot = graph_config[i].cache_kv ? phys_slot_of(i) : 0;
                graph_config[i].off_group = graph_config[i].offload ? off_group_of(i) : 0;
                graph_config[i].load = graph_config[i].offload && offset != 0
                    && (i == static_cast<int>(o_start) || off_group_of(i - 1) != off_group_of(i));
                graph_config[i].store = graph_config[i].offload
                    && (i == static_cast<int>(o_end) - 1 || off_group_of(i + 1) != off_group_of(i));
                graph_config[i].mask = i < n_no_cache_steps && offset != 0;
                graph_config[i].cut_len = i == n - 1 && offset == 0 ? cut_len : 0;
                if (i == n_no_cache_steps - 1 && offset != 0)
                    graph_config[i].cut_len = offset;
                graph_config[i].slice = offset != 0 && i == n_no_cache_steps;
                graph_config[i].slide = kv_slidable && graph_config[i].cache_kv && i != n - 1
                    && phys_slot_of(i + 1) == graph_config[i].phys_slot + 1;
            }
        }
    }
};

bool cosyvoice_model_3::token2wav_ext(const int* token_ids, uint32_t n_tokens, float speed, cosyvoice_prompt_t prompt, bool streaming, bool finalize, cosyvoice_generated_speech_ptr result)
{
    use_count_guard _guard(this);

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
    auto kv_cache = &worker->dit_kv_cache;

    if (streaming && worker->offset == 0)
    {
        kv_cache->cur_len = 0;
        worker->chunk_boundaries.clear();
        worker->flow_cache.clear();
    }

    ggml_reset(ctx0.get());
    ggml_reset(ctx1.get());
    ggml_backend_sched_reset(sched.get());

    ggml_tensor* token = ggml_new_tensor_1d(ctx0.get(), GGML_TYPE_I32, n_tokens);
    ggml_tensor* prompt_token = ggml_new_tensor_1d(ctx0.get(), GGML_TYPE_I32, prompt->flow_prompt_speech_tokens.second);
    ggml_tensor* prompt_feat = ggml_new_tensor_2d(ctx0.get(), GGML_TYPE_F32, prompt->prompt_speech_feat.shape[1], prompt->prompt_speech_feat.shape[0]);
    ggml_tensor* embedding = ggml_new_tensor_2d(ctx0.get(), GGML_TYPE_F32, prompt->flow_embedding.shape[1], prompt->flow_embedding.shape[0]);
    ggml_set_input(token);
    ggml_set_input(prompt_token);
    ggml_set_input(prompt_feat);
    ggml_set_input(embedding);

    // Phase 1: Flow encode
    ggml_cgraph* gf = new_cgraph(ctx0.get());
    auto [mu, spks, conds, cut_len] = flow.build_cgraph_encode(ctx0.get(), token, prompt_token, prompt_feat, embedding, op_caps, (flow.decoder.diffusion_steps - params.dit_kv_fixed_slots - params.dit_kv_offloadable_slots) == 0 && streaming ? worker->offset : 0, streaming && !finalize);
    const auto seq_len = mu->ne[1];
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

    dit_sched_config config(params, cut_len, streaming ? worker->offset : 0, streaming, kv_cache->can_reuse(), flow.decoder.diffusion_steps);
    uint32_t noise_len = static_cast<uint32_t>(ggml_nelements(ditctx.x));
    uint32_t noise_req = noise_len;
    int64_t full_len;
    if (streaming)
    {
        if (flow.decoder.diffusion_steps == params.dit_kv_fixed_slots + params.dit_kv_offloadable_slots)
        {
            full_len = worker->offset + seq_len;
            noise_req = static_cast<uint32_t>(full_len * ditctx.x->ne[0]);
        }
        else
            full_len = seq_len;

        worker->offset = finalize ? 0 : full_len;
        worker->chunk_boundaries.push_back(full_len);

        const auto chunk_len = full_len - (worker->chunk_boundaries.size() > 1 ? worker->chunk_boundaries.rbegin()[1] : 0);
        if (kv_cache->cur_len + chunk_len >= params.dit_kv_cache_length)
            kv_cache->cur_len = params.dit_kv_cache_length - chunk_len;
    }
    else
        full_len = seq_len;

    float* noise_buffer = shared->noise_callback(COSYVOICE_NOISE_CALLBACK_STAGE_BEFORE_FLOW, noise_req, nullptr, shared->noise_callback_ctx);
    ggml_backend_tensor_set_async(backend.get(), ditctx.x, noise_buffer + noise_req - noise_len, 0, ditctx.x->nb[2]);

    // Phase 2: Flow decode steps
    ggml_tensor* t_leaf;
    ggml_tensor* position_ids;
    ggml_tensor* attn_mask;
    if (config[0].cache_kv)
        kv_cache->bind_slot(config[0].phys_slot);
    auto feat = flow.decoder.build_cgraph_one_step(ctx0.get(), ditctx, 1, op_caps, config[0].cut_len, t_leaf, position_ids, gf, config[0].cache_kv ? kv_cache : nullptr, config[0].mask ? &attn_mask : nullptr);
    ggml_build_forward_expand(gf, feat);
    set_graph_backends(gf, sched.get(), backend.get(), cpu_backend.get(), op_caps);
    ggml_backend_sched_synchronize(sched.get());
    ggml_backend_sched_alloc_graph(sched.get(), gf);

    auto stage_build_after = [&](int step)
    {
        if (!op_caps.fill) ggml_set_zero(t_leaf);

        const auto base = config[step].cache_kv ? kv_cache->cur_len : 0;
        for (int64_t i = 0; i < position_ids->ne[1]; ++i)
        {
            auto cur_row = reinterpret_cast<int32_t*>(position_ids->data) + i * position_ids->ne[0];
            for (int32_t j = 0; j < position_ids->ne[0]; ++j)
                cur_row[j] = j + base;
        }

        if (config[step].mask)
        {
            const auto& bounds = worker->chunk_boundaries;
            for (int64_t i = 0; i < attn_mask->ne[0]; ++i)
            {
                auto it = std::upper_bound(bounds.begin(), bounds.end(), static_cast<uint32_t>(i));
                int64_t block_end = it != bounds.end()
                    ? static_cast<int64_t>(*it)
                    : attn_mask->ne[0];
                auto row = reinterpret_cast<ggml_fp16_t*>(attn_mask->data) + i * attn_mask->ne[1];
                for (int64_t j = 0; j < attn_mask->ne[1]; ++j)
                    row[j] = j < block_end ? 0 : 0xFC00;
            }
        }

        if (config[step].cache_kv)
            kv_cache->set_input_v_idxs(backend.get(), reinterpret_cast<const int32_t*>(position_ids->data), static_cast<uint32_t>(position_ids->ne[0] * position_ids->ne[1]), static_cast<uint32_t>(position_ids->ne[0]));
    };

    stage_build_after(0);

    memcpy(token->data, token_ids, token->nb[1]);
    memcpy(prompt_token->data, prompt->flow_prompt_speech_tokens.first.get(), prompt_token->nb[1]);
    memcpy(prompt_feat->data, prompt->prompt_speech_feat.data, prompt_feat->nb[2]);
    memcpy(embedding->data, prompt->flow_embedding.data, embedding->nb[2]);

    auto load_kv = [&](int step)
    {
        if (config[step].load)
        {
            const auto kv_len = kv_cache->cur_len;
            kv_cache->load_slot(backend.get(), sched.get(), config[step].off_group);
            kv_cache->cur_len = kv_len;
            return true;
        }
        return false;
    };

    load_kv(0);

    worker->status = ggml_backend_sched_graph_compute(sched.get(), gf);
    shared->noise_callback(COSYVOICE_NOISE_CALLBACK_STAGE_AFTER_FLOW, noise_req, noise_buffer, shared->noise_callback_ctx);
    if (params.inference_buffer_policy != COSYVOICE_INFERENCE_BUFFER_POLICY_DEDICATED)
        llm_set_kv_cache_len(0);
    if (worker->status != GGML_STATUS_SUCCESS)
        return false;

    auto stage_compute_after = [&](int step)
    {
        if (config[step].offload)
        {
            memcpy(ditctx.x->ne, feat->ne, sizeof(ditctx.x->ne));
            memcpy(ditctx.x->nb, feat->nb, sizeof(ditctx.x->nb));
            ggml_backend_tensor_copy_async(backend.get(), backend.get(), feat, ditctx.x);
            if (step != flow.decoder.diffusion_steps - 1 && config[step + 1].rebuild)
                ggml_backend_sched_reset(sched.get());
            // Only the last step of an offload group round-trips to CPU; earlier
            // steps leave their state in the scratch slot for the group to overwrite.
            if (config[step].store && full_len < params.dit_kv_cache_length)
                kv_cache->offload_slot(backend.get(), sched.get(), config[step].off_group, kv_cache->cur_len + static_cast<uint32_t>(position_ids->ne[0]));
        }
        if (config[step].slide)
            kv_cache->slide_kv_slot();
    };

    stage_compute_after(0);

    for (auto tensor : std::span(&ditctx.mu_in, 3))
    {
        tensor->op = GGML_OP_NONE;
        tensor->view_src = nullptr;
        tensor->view_offs = 0;
        memset(tensor->src, 0, sizeof(tensor->src));
    }

    for (int step = 1; step != flow.decoder.diffusion_steps; ++step)
    {
        if (is_stop_requested())
        {
            ggml_reset(ctx0.get());
            ggml_backend_sched_reset(sched.get());
            cosyvoice_call_ggml_log_callback(GGML_LOG_LEVEL_INFO, "token2wav stopped during DiT steps.\n");
            return false;
        }

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
        if (!config[step - 1].offload)
            ggml_backend_tensor_copy_async(backend.get(), backend.get(), feat, ditctx.x);

        if (config[step].rebuild)
        {
            if (config[step].cache_kv)
                kv_cache->bind_slot(config[step].phys_slot);
            ggml_reset(ctx0.get());
            ggml_backend_sched_reset(sched.get());

            if (load_kv(step) && !kv_cache->can_reuse())
                ggml_backend_sched_reset(sched.get());

            gf = new_cgraph(ctx0.get());
            feat = flow.decoder.build_cgraph_one_step(ctx0.get(), ditctx, step + 1, op_caps, config[step].cut_len, t_leaf, position_ids, gf, config[step].cache_kv ? kv_cache : nullptr, config[step].mask ? &attn_mask : nullptr);
            ggml_build_forward_expand(gf, feat);
            set_graph_backends(gf, sched.get(), backend.get(), cpu_backend.get(), op_caps);
            ggml_backend_sched_alloc_graph(sched.get(), gf);
            stage_build_after(step);
        }
        else
        {
            auto scale_node = feat->src[1];
            GGML_ASSERT(scale_node->op == GGML_OP_SCALE);
            // Match the rebuild branch, which advances this iteration to step + 1.
            auto [t, dt] = flow.decoder.get_t_and_dt(ctx0.get(), step + 1);
            reinterpret_cast<float*>(t_leaf->op_params)[op_caps.fill ? 0 : 1] = t;
            reinterpret_cast<float*>(scale_node->op_params)[0] = dt;

            load_kv(step);
        }

        worker->status = ggml_backend_sched_graph_compute(sched.get(), gf);
        if (worker->status != GGML_STATUS_SUCCESS)
            return false;

        stage_compute_after(step);
    }

    // Phase 3: Copy flow output to speech_feat (persistent buffer)
    ggml_reset(ctx1.get());
    const auto cache_length = streaming ? static_cast<int64_t>(worker->flow_cache.size() / feat->ne[0]) : 0;
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

        auto speech_feat_view = ggml_view_2d(ctx0.get(), speech_feat, speech_feat->ne[0], feat->ne[1], speech_feat->nb[1], speech_feat->nb[1] * cache_length);
        speech_feat_view->buffer = speech_feat->buffer;
        ggml_backend_tensor_set_async(backend.get(), speech_feat, worker->flow_cache.data(), 0, speech_feat->nb[2] - feat->nb[2]);
        ggml_backend_tensor_copy_async(backend.get(), backend.get(), feat, speech_feat_view);
    }
    else
    {
        ggml_backend_tensor_alloc(token2wav_buffer.get(), speech_feat,
            ggml_backend_buffer_get_base(token2wav_buffer.get()));
        ggml_backend_tensor_copy_async(backend.get(), backend.get(), feat, speech_feat);
    }

    if (streaming && !finalize)
    {
        const auto overlap = static_cast<int64_t>(shared->hift_overlap);
        const auto n_feat_frames = feat->ne[1];
        const auto frames_to_keep = std::min(overlap, n_feat_frames);
        const auto elements_to_keep = static_cast<size_t>(frames_to_keep * feat->ne[0]);

        worker->flow_cache.resize(elements_to_keep);
        if (frames_to_keep > 0)
        {
            const auto byte_offset = static_cast<size_t>((n_feat_frames - frames_to_keep) * feat->nb[1]);
            const auto byte_size = static_cast<size_t>(frames_to_keep * feat->nb[1]);
            ggml_backend_tensor_get_async(backend.get(), feat, worker->flow_cache.data(), byte_offset, byte_size);
        }
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
    noise_req = static_cast<uint32_t>(hift.f0_predictor.condnet_8.output_length(
        hift.f0_predictor.condnet_6.output_length(
            hift.f0_predictor.condnet_4.output_length(
                hift.f0_predictor.condnet_2.output_length(
                    hift.f0_predictor.condnet_0.output_length(speech_feat->ne[0],
                        finalize), true), true), true), true) * hift.scale_factor * noise->ne[0]);
    noise_req = std::max(noise_req, noise_len);
    noise_buffer = shared->noise_callback(COSYVOICE_NOISE_CALLBACK_STAGE_BEFORE_HIFT, noise_req, nullptr, shared->noise_callback_ctx);
    ggml_backend_tensor_set_async(backend.get(), noise, noise_buffer + noise_req - noise_len, 0, noise->nb[2]);

    result->data = reinterpret_cast<float*>(generated_speech->data);
    result->length = static_cast<uint32_t>(generated_speech->ne[0]);
    worker->status = ggml_backend_sched_graph_compute(sched.get(), gf);
    shared->noise_callback(COSYVOICE_NOISE_CALLBACK_STAGE_AFTER_HIFT, noise_req, noise_buffer, shared->noise_callback_ctx);

    return worker->status == GGML_STATUS_SUCCESS;
}
