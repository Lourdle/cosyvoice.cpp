#include "cosyvoice-model.h"
#include "ggml-fft.h"
#include "ggml-cpu-flag.h"

#include <cstring>
#include <span>

static void split_tensor(ggml_context* ctx, ggml_tensor* tensor, int dim, ggml_tensor** tensors, uint16_t chunks)
{
    GGML_ASSERT(dim >= 0 && dim < GGML_MAX_DIMS);
    GGML_ASSERT(tensor->ne[dim] % chunks == 0);

    int64_t ne[GGML_MAX_DIMS];
    memcpy(ne, tensor->ne, sizeof(ne));
    ne[dim] /= chunks;
    const size_t offset_per_chunk = tensor->nb[dim] * ne[dim];

    for (uint16_t i = 0; i != chunks; ++i)
        tensors[i] = ggml_view_4d(
            ctx,
            tensor,
            ne[0],
            ne[1],
            ne[2],
            ne[3],
            tensor->nb[1],
            tensor->nb[2],
            tensor->nb[3],
            i * offset_per_chunk);
}

template<uint16_t chunks>
std::array<ggml_tensor*, chunks> split_tensor(ggml_context* ctx, ggml_tensor* tensor, int dim = 0)
{
    std::array<ggml_tensor*, chunks> result;
    split_tensor(ctx, tensor, dim, result.data(), chunks);
    return result;
}

static ggml_tensor* view_tensor(ggml_context* ctx, ggml_tensor* tensor, ggml_tensor* src = nullptr)
{
    auto view = ggml_view_tensor(ctx, tensor);
    view->op = GGML_OP_VIEW;
    view->src[0] = src ? src : tensor;
    return view;
}

static ggml_tensor* concat_tensors(ggml_context* ctx, ggml_tensor** tensors, size_t chunks, int dim, ggml_backend_op_capabilities capabilities)
{
    GGML_ASSERT(chunks > 1);
    GGML_ASSERT(dim >= 0 && dim < GGML_MAX_DIMS);

    if (chunks == 2)
        if (tensors[0]->type == GGML_TYPE_F32 && tensors[1]->type == GGML_TYPE_F32
            || tensors[0]->type == GGML_TYPE_I32 && tensors[1]->type == GGML_TYPE_I32 && capabilities.concat_i32)
            return ggml_concat(ctx, tensors[0], tensors[1], dim);
        else if (tensors[0]->type == GGML_TYPE_I32 && tensors[1]->type == GGML_TYPE_I32)
        {
            auto src0 = view_tensor(ctx, tensors[0]);
            auto src1 = view_tensor(ctx, tensors[1]);

            src0->type = GGML_TYPE_F32;
            src1->type = GGML_TYPE_F32;

            auto res = ggml_concat(ctx, src0, src1, dim);
            res = view_tensor(ctx, res);
            res->type = GGML_TYPE_I32;
            return res;
        }

    auto type = tensors[0]->type;
    int64_t ne[GGML_MAX_DIMS];
    memcpy(ne, tensors[0]->ne, sizeof(ne));

    for (const auto tensor : std::span(tensors + 1, chunks - 1))
    {
        GGML_ASSERT(tensor->type == type);
        for (int d = 0; d != GGML_MAX_DIMS; ++d)
        {
            if (d == dim)
                continue;
            GGML_ASSERT(tensor->ne[d] == ne[d]);
        }
        ne[dim] += tensor->ne[dim];
    }

    auto result = ggml_new_tensor_4d(ctx, type, ne[0], ne[1], ne[2], ne[3]);
    auto prev_part = result;
    size_t offset = 0;
    for (auto tensor : std::span(tensors, chunks))
    {
        auto dest_part = ggml_view_4d(ctx, result,
            tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3],
            result->nb[1], result->nb[2], result->nb[3],
            offset);
        dest_part->src[0] = prev_part;

        // GGML may produce incorrect results when copying a non-contiguous tensor,
        // so make it contiguous first.
        if (!ggml_is_contiguous(tensor))
            tensor = ggml_cont(ctx, tensor);
        prev_part = ggml_cpy(ctx, tensor, dest_part);

        offset += tensor->ne[dim] * result->nb[dim];
    }

    return view_tensor(ctx, result, prev_part);
}

template<size_t chunks>
static ggml_tensor* concat_tensors(ggml_context* ctx, std::array<ggml_tensor*, chunks> tensors, int dim, ggml_backend_op_capabilities capabilities = {})
{
    return concat_tensors(ctx, tensors.data(), chunks, dim, capabilities);
}

static ggml_tensor* unsqueeze(ggml_context* ctx, ggml_tensor* x, int dim)
{
    if (dim == 0)
        return ggml_view_4d(ctx, x, 1, x->ne[0], x->ne[1], x->ne[2], x->nb[0], x->nb[1], x->nb[2], 0);
    else if (dim == 1)
        return ggml_view_4d(ctx, x, x->ne[0], 1, x->ne[1], x->ne[2], x->nb[1], x->nb[1], x->nb[2], 0);
    else if (dim == 2)
        return ggml_view_4d(ctx, x, x->ne[0], x->ne[1], 1, x->ne[2], x->nb[1], x->nb[2], x->nb[2], 0);
    else GGML_ABORT("unsqueeze: invalid dim %d for 3D tensor", dim);
}

static ggml_tensor* mish(ggml_context* ctx, ggml_tensor* x)
{
    auto out = ggml_softplus(ctx, x);
    out = ggml_tanh(ctx, out);
    out = ggml_mul(ctx, x, out);
    return out;
}

ggml_tensor* Conv1d::build_cgraph(ggml_context* ctx, ggml_tensor* x, int s, int p, int d, int g, ggml_backend_op_capabilities capabilities) const
{
    GGML_ASSERT(g >= 1);
    GGML_ASSERT(x->ne[3] == 1 && weight->ne[3] == 1);

    ggml_tensor* im2col;
    ggml_tensor* weight = this->weight;
    // Backends without F16 IM2COL support (e.g. Metal) need F32 input + F32/F16
    // output and contiguous src[1]. Fall back only when the probe says so.
    const bool needs_f32_fallback = !capabilities.im2col_f16;
    if (needs_f32_fallback && x->type == GGML_TYPE_F16)
        x = ggml_cast(ctx, x, GGML_TYPE_F32);
    if (g != 1)
    {
        GGML_ASSERT(weight->ne[2] % g == 0 && weight->ne[1] * g == x->ne[1]);

        auto xs = reinterpret_cast<ggml_tensor**>(alloca(sizeof(ggml_tensor*) * g));
        auto ws = reinterpret_cast<ggml_tensor**>(alloca(sizeof(ggml_tensor*) * g));
        if (needs_f32_fallback && weight->type == GGML_TYPE_F16)
            weight = ggml_cast(ctx, weight, GGML_TYPE_F32);
        split_tensor(ctx, x, 1, xs, g);
        split_tensor(ctx, weight, 2, ws, g);
        const auto out_type = needs_f32_fallback ? GGML_TYPE_F32 : weight->type;
        for (int i = 0; i != g; i++)
            xs[i] = unsqueeze(ctx,
                ggml_im2col(ctx,
                    ws[i],
                    ggml_cont(ctx, xs[i]),
                    s, 0, p, 0, d, 0,
                    false, out_type),
                2);
        im2col = concat_tensors(ctx, xs, g, 2, capabilities);
    }
    else
    {
        const auto out_type = needs_f32_fallback
            ? ((weight->type == GGML_TYPE_F16) ? GGML_TYPE_F16 : GGML_TYPE_F32)
            : weight->type;
        im2col = ggml_im2col(ctx, weight, x, s, 0, p, 0, d, 0, false, out_type);
        if (im2col->ne[1] > 0xFFFF)
        {
            // Split the im2col output along the output-width dimension.
            // Keep the op as GGML_OP_IM2COL until graph allocation, then switch to
            // GGML_OP_NONE. This avoids a GGML allocation-size bug that can reserve
            // much more memory than necessary.
            constexpr int chunk_max_size = 65528;
            auto prev_chunk = im2col;
            memset(im2col->op_params, 0, sizeof(im2col->op_params));
            memset(im2col->src, 0, sizeof(im2col->src));

            for (int64_t start = 0; start < im2col->ne[1]; start += chunk_max_size)
            {
                auto end = std::min(im2col->ne[1], start + chunk_max_size);

                auto logical_start = start * s - p;
                auto logical_end = (end - 1) * s - p + d * (weight->ne[0] - 1) + 1;

                int64_t phys_start = std::max<int64_t>(0, logical_start);
                int64_t phys_end = std::min(x->ne[0], logical_end);
                int64_t chunk_iw = phys_end - phys_start;

                int chunk_p0 = std::max(0, -static_cast<int>(logical_start));

                auto x_view = ggml_view_3d(
                    ctx, x,
                    chunk_iw, x->ne[1], x->ne[2],
                    x->nb[1], x->nb[2],
                    phys_start * x->nb[0]
                );

                auto chunk_im2col = ggml_im2col(
                    ctx, weight,
                    needs_f32_fallback ? ggml_cont(ctx, x_view) : x_view,
                    s, 1,
                    chunk_p0, 0,
                    d, 1,
                    false,
                    im2col->type
                );

                chunk_im2col->view_src = im2col;
                chunk_im2col->view_offs = start * im2col->nb[1];
                chunk_im2col->src[2] = prev_chunk;
                prev_chunk = chunk_im2col;
            }

            im2col = view_tensor(ctx, im2col, prev_chunk);
        }
    }

    weight = ggml_reshape_3d(ctx, weight, weight->ne[0] * weight->ne[1], weight->ne[2] / g, g);
    if (x->ne[2] != 1)
        if (weight->type == GGML_TYPE_F16 && !capabilities.repeat_f16)
        {
            weight->ne[3] = x->ne[2];
            weight->nb[3] = 0;
        }
        else
            weight = ggml_repeat_4d(ctx, weight, weight->ne[0], weight->ne[1], g, x->ne[2]);

    ggml_tensor* result = ggml_mul_mat(ctx, im2col, weight);
    result = ggml_reshape_3d(ctx, result, im2col->ne[1], this->weight->ne[2], x->ne[2]);

    if (bias)
        result = ggml_add(ctx, result, unsqueeze(ctx, bias, 0));

    return result;
}

ggml_tensor* Linear::build_cgraph(ggml_context* ctx, ggml_tensor* x) const
{
    auto out = ggml_mul_mat(ctx, weight, x);
    if (bias)
        out = ggml_add(ctx, out, bias);
    return out;
}

ggml_tensor* LayerNorm::build_cgraph(ggml_context* ctx, ggml_tensor* x) const
{
    x = ggml_norm(ctx, x, eps);

    if (weight)
    {
        x = ggml_mul(ctx, weight, x);
        if (bias)
            x = ggml_add(ctx, x, bias);
    }
    return x;
}

ggml_tensor* CausalConvPositionEmbedding::build_cgraph(ggml_context* ctx, ggml_tensor* x, ggml_backend_op_capabilities capabilities) const
{
    x = ggml_permute(ctx, x, 1, 0, 2, 3);
    x = ggml_cont(ctx, x);
    x = ggml_pad_ext(ctx, x, static_cast<int>(conv1.weight->ne[0] - 1), 0, 0, 0, 0, 0, 0, 0);
    x = conv1.build_cgraph(ctx, x, 1, 0, 1, 16, capabilities);
    x = mish(ctx, x);
    x = ggml_pad_ext(ctx, x, static_cast<int>(conv2.weight->ne[0] - 1), 0, 0, 0, 0, 0, 0, 0);
    x = conv2.build_cgraph(ctx, x, 1, 0, 1, 16, capabilities);
    x = mish(ctx, x);
    return ggml_permute(ctx, x, 1, 0, 2, 3);
}

ggml_tensor* InputEmbedding::build_cgraph(ggml_context* ctx, ggml_tensor* x, ggml_tensor* cond, ggml_tensor* text_embed, ggml_tensor* spks, ggml_backend_op_capabilities capabilities) const
{
    spks = ggml_repeat(ctx, spks, x);
    x = concat_tensors(ctx, std::array{ x, cond, text_embed, spks }, 0);
    x = proj.build_cgraph(ctx, x);
    x = ggml_add(ctx,
        ggml_cont(ctx,
            conv_pos_embed.build_cgraph(ctx, x, capabilities)),
        x);
    return x;
}

ggml_tensor* SinusPositionEmbedding::build_cgraph(ggml_context* ctx, ggml_tensor* x) const
{
    x = unsqueeze(ctx, x, 0);
    x = ggml_repeat_4d(ctx, x, emb->ne[0], x->ne[1], x->ne[2], x->ne[3]);
    auto emb = ggml_mul(ctx, x, this->emb);
    auto sin_emb = ggml_sin(ctx, emb);
    auto cos_emb = ggml_cos(ctx, emb);
    return concat_tensors(ctx, std::array{ sin_emb, cos_emb }, 0);
}

ggml_tensor* TimestepEmbedding::build_cgraph(ggml_context* ctx, ggml_tensor* t) const
{
    auto time_hidden = time_embed.build_cgraph(ctx, t);
    auto time = time_mlp_0.build_cgraph(ctx, time_hidden);
    time = ggml_silu(ctx, time);
    time = time_mlp_2.build_cgraph(ctx, time);
    return time;
}

std::array<ggml_tensor*, 5> AdaLayerNormZero::build_cgraph(ggml_context* ctx, ggml_tensor* x, ggml_tensor* emb) const
{
    emb = ggml_silu(ctx, emb);
    emb = linear.build_cgraph(ctx, emb);
    auto [shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp] = split_tensor<6>(ctx, emb, 0);

    scale_msa = ggml_scale_bias(ctx, ggml_cont(ctx, scale_msa), 1.f, 1.f);
    scale_msa = unsqueeze(ctx, scale_msa, 1);
    shift_msa = unsqueeze(ctx, shift_msa, 1);

    x = norm.build_cgraph(ctx, x);
    x = ggml_mul(ctx, x, scale_msa);
    x = ggml_add(ctx, x, shift_msa);

    return { x, gate_msa, shift_mlp, scale_mlp, gate_mlp };
}

ggml_tensor* Attention::build_cgraph(ggml_context* ctx, ggml_tensor* x, ggml_tensor* position_ids, int64_t cut_len, cosyvoice_kv_cache* kv_cache, ggml_cgraph* gf, ggml_tensor* attn_mask, int layer_idx) const
{
    const auto full_seq_len = position_ids->ne[0];
    const auto seq_len = full_seq_len - (cut_len > 0 ? cut_len : 0);
    const auto batch_size = x->ne[2];

    auto key = to_k.build_cgraph(ctx, x);
    auto value = to_v.build_cgraph(ctx, x);

    auto original_position_ids = position_ids;
    auto full_position_ids = position_ids;
    if (cut_len > 0)
    {
        x = ggml_view_3d(ctx, x, x->ne[0], x->ne[1] - cut_len, x->ne[2], x->nb[1], x->nb[2], x->nb[1] * cut_len);
        position_ids = ggml_view_2d(ctx, position_ids, position_ids->ne[0] - cut_len, position_ids->ne[1], position_ids->nb[1], position_ids->nb[0] * cut_len);
        position_ids = ggml_cont(ctx, position_ids);
    }
    auto query = to_q.build_cgraph(ctx, x);

    // Follow the original DiT implementation and apply RoPE before reshaping and
    // permuting. The ggml RoPE op does not support 4D tensors yet, so reshape to
    // 3D with an explicit single-head dimension first.
    query = ggml_reshape_3d(ctx, query, query->ne[0], 1, seq_len * batch_size);
    key = ggml_reshape_3d(ctx, key, key->ne[0], 1, full_seq_len * batch_size);
    position_ids = ggml_reshape_1d(ctx, position_ids, seq_len * batch_size);
    full_position_ids = ggml_reshape_1d(ctx, full_position_ids, full_seq_len * batch_size);

    const auto head_dim = static_cast<int>(key->ne[0] / heads);
    query = ggml_rope(ctx, query, position_ids, head_dim, GGML_ROPE_TYPE_NORMAL);
    key = ggml_rope(ctx, key, full_position_ids, head_dim, GGML_ROPE_TYPE_NORMAL);

    query = ggml_reshape_4d(ctx, query, head_dim, heads, seq_len, batch_size);
    key = ggml_reshape_4d(ctx, key, head_dim, heads, full_seq_len, batch_size);
    value = ggml_reshape_4d(ctx, value, value->ne[0] / heads, heads, full_seq_len, batch_size);

    query = ggml_permute(ctx, query, 0, 2, 1, 3);
    key = ggml_permute(ctx, key, 0, 2, 1, 3);
    value = ggml_permute(ctx, value, 0, 2, 1, 3);

    ggml_tensor* attn_output;
    if (kv_cache)
    {
        kv_cache->update_cache(ctx, gf, key, value, original_position_ids, layer_idx);
        attn_output = kv_cache->attention_forward(ctx, query, key, value, attn_mask);
    }
    else
    {
        query = ggml_cont(ctx, query);
        key = ggml_cont(ctx, key);

        if (fattn)
            attn_output = ggml_flash_attn_ext(ctx, query, key, value, attn_mask, 1.f / std::sqrt(static_cast<float>(head_dim)), 0.f, 0.f);
        else
        {
            value = ggml_permute(ctx, value, 1, 0, 2, 3);
            value = ggml_cont(ctx, value);
            auto attn_scores = ggml_mul_mat(ctx, key, query);
            auto attn_weights = ggml_soft_max_ext_inplace(ctx, attn_scores, attn_mask, 1.f / std::sqrt(static_cast<float>(head_dim)), 0.f);
            attn_output = ggml_mul_mat(ctx, value, attn_weights);
            attn_output = ggml_permute(ctx, attn_output, 0, 2, 1, 3);
            attn_output = ggml_cont(ctx, attn_output);
        }
    }
    x = ggml_reshape_3d(ctx,
        attn_output,
        attn_output->ne[0] * attn_output->ne[1],
        seq_len,
        batch_size);

    x = to_out.build_cgraph(ctx, x);
    return x;
}

ggml_tensor* FeedForward::build_cgraph(ggml_context* ctx, ggml_tensor* x) const
{
    x = ff_0_0.build_cgraph(ctx, x);
    x = ggml_gelu_erf(ctx, x);  // Use exact erf-based GELU to match PyTorch (tanh approx diverges over 22 DiT blocks)
    x = ff_2.build_cgraph(ctx, x);
    return x;
}

ggml_tensor* DiTBlock::build_cgraph(ggml_context* ctx, ggml_tensor* x, ggml_tensor* time_emb, ggml_tensor* position_ids, int64_t cut_len, cosyvoice_kv_cache* kv_cache, ggml_cgraph* gf, ggml_tensor* attn_mask, int layer_idx) const
{
    auto [norm, gate_msa, shift_mlp, scale_mlp, gate_mlp] = attn_norm.build_cgraph(ctx, x, time_emb);

    auto attn_output = attn.build_cgraph(ctx, norm, position_ids, cut_len, kv_cache, gf, attn_mask, layer_idx);
    gate_msa = unsqueeze(ctx, gate_msa, 1);
    attn_output = ggml_mul(ctx, attn_output, gate_msa);
    if (cut_len > 0)
        x = ggml_view_3d(ctx, x, x->ne[0], x->ne[1] - cut_len, x->ne[2], x->nb[1], x->nb[2], x->nb[1] * cut_len);
    x = ggml_add(ctx, x, attn_output);

    auto ff_norm = this->ff_norm.build_cgraph(ctx, x);
    scale_mlp = ggml_scale_bias(ctx, ggml_cont(ctx, scale_mlp), 1.f, 1.f);
    ff_norm = ggml_mul(ctx, ff_norm, unsqueeze(ctx, scale_mlp, 1));
    ff_norm = ggml_add(ctx, ff_norm, unsqueeze(ctx, shift_mlp, 1));

    auto ff_output = ff.build_cgraph(ctx, ff_norm);
    ff_output = ggml_mul(ctx, ff_output, unsqueeze(ctx, gate_mlp, 1));
    x = ggml_add(ctx, x, ff_output);
    return x;
}

ggml_tensor* AdaLayerNorm_Final::build_cgraph(ggml_context* ctx, ggml_tensor* x, ggml_tensor* emb) const
{
    emb = ggml_silu(ctx, emb);
    emb = linear.build_cgraph(ctx, emb);
    auto [scale, shift] = split_tensor<2>(ctx, emb, 0);

    scale = ggml_scale_bias(ctx, ggml_cont(ctx, scale), 1.f, 1.f);
    x = norm.build_cgraph(ctx, x);
    x = ggml_mul(ctx, x, unsqueeze(ctx, scale, 1));
    x = ggml_add(ctx, x, unsqueeze(ctx, shift, 1));
    return x;
}

ggml_tensor* DiT::build_cgraph(ggml_context* ctx, ggml_tensor* x, ggml_tensor* mu, ggml_tensor* t, ggml_tensor* spks, ggml_tensor* cond, int64_t cut_len, ggml_tensor*& ref_position_ids, ggml_backend_op_capabilities capabilities, cosyvoice_kv_cache* kv_cache, ggml_tensor** attn_mask, ggml_cgraph* gf) const
{
    t = time_embed.build_cgraph(ctx, t);
    x = input_embed.build_cgraph(ctx, x, cond, mu, spks, capabilities);

    ggml_tensor* position_ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, x->ne[1], x->ne[2]);
    ggml_set_input(position_ids);
    ref_position_ids = position_ids;
    position_ids = ggml_dup(ctx, position_ids);

    ggml_tensor* mask = nullptr;
    if (attn_mask)
    {
        const auto seq_len = x->ne[1];
        mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, seq_len, seq_len);
        ggml_set_input(mask);
        *attn_mask = mask;
    }

    for (int i = 0; i < transformer_blocks.size() - 1; ++i)
        x = transformer_blocks[i].build_cgraph(ctx, x, t, position_ids, 0, kv_cache, gf, mask, i);
    // Apply `cut_len` only on the final block.
    x = transformer_blocks.back().build_cgraph(ctx, x, t, position_ids, cut_len, kv_cache, gf, mask, static_cast<int>(transformer_blocks.size()) - 1);

    x = norm_out.build_cgraph(ctx, x, t);

    auto output = proj_out.build_cgraph(ctx, x);
    return output;
}

CausalConditionalCFM::DiTContext CausalConditionalCFM::prepare_context(ggml_context* ctx, ggml_tensor* mu, ggml_tensor* spks, ggml_tensor* cond) const
{
    DiTContext ditctx{
        .x = ggml_new_tensor_2d(
        ctx,
        GGML_TYPE_F32,
        80, mu->ne[1]),

        .mu_in = ggml_pad(ctx, mu, 0, 0, 1, 0),

        .spks_in = ggml_pad(ctx, spks, 0, 0, 1, 0),
        .cond_in = ggml_pad(ctx, cond, 0, 0, 1, 0)
    };

    return ditctx;
}

std::array<float, 2> CausalConditionalCFM::get_t_and_dt(ggml_context* ctx, int step) const
{
    GGML_ASSERT(step > 0 && step < t_span.size());

    auto t = t_span[0];

    for (int cur = 1; cur != step; ++cur)
        t += t_span[cur] - t_span[cur - 1];

    auto dt = t_span[step] - t_span[step - 1];

    return { t, dt };
}

ggml_tensor* CausalConditionalCFM::build_cgraph_one_step(ggml_context* ctx, const DiTContext& ditctx, int step, ggml_backend_op_capabilities capabilities, int64_t cut_len, ggml_tensor*& t_tensor, ggml_tensor*& position_ids, ggml_cgraph* gf, cosyvoice_kv_cache* kv_cache, ggml_tensor** attn_mask) const
{
    auto x = ditctx.x;
    auto [t, dt] = get_t_and_dt(ctx, step);

    auto t_in = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2);
    t_in = capabilities.fill ? ggml_fill_inplace(ctx, t_in, t) : ggml_scale_bias_inplace(ctx, t_in, 0.f, t);
    auto x_in = ggml_repeat_4d(ctx, x, x->ne[0], x->ne[1], 2, x->ne[3]);

    auto [dphi_dt, cfg_dphi_dt] = split_tensor<2>(
        ctx, estimator.build_cgraph(
            ctx,
            x_in,
            ditctx.mu_in, t_in,
            ditctx.spks_in,
            ditctx.cond_in,
            cut_len,
            position_ids,
            capabilities,
            kv_cache,
            attn_mask,
            gf),
        2);

    cfg_dphi_dt->nb[3] = dphi_dt->nb[3] = dphi_dt->nb[2];
    cfg_dphi_dt = ggml_scale(ctx, cfg_dphi_dt, inference_cfg_rate);
    dphi_dt = ggml_scale(ctx, dphi_dt, 1.f + inference_cfg_rate);
    dphi_dt = ggml_sub(ctx,
        dphi_dt,
        cfg_dphi_dt);

    if (cut_len > 0)
        x = ggml_view_3d(ctx, x, x->ne[0], x->ne[1] - cut_len, x->ne[2], x->nb[1], x->nb[2], x->nb[1] * cut_len);

    x = ggml_add(ctx, x,
        ggml_scale(ctx, dphi_dt, dt));

    t_tensor = t_in;
    return x;
}

ggml_tensor* PreLookaheadLayer::build_cgraph(ggml_context* ctx, ggml_tensor* inputs, bool streaming, uint32_t cut_len) const
{
    auto outputs = ggml_permute(ctx, inputs, 1, 0, 2, 3);
    outputs = ggml_cont(ctx, outputs);
    if (streaming)
        inputs = ggml_view_2d(ctx, inputs, inputs->ne[0], inputs->ne[1] - pre_lookahead_len, inputs->nb[1], 0);
    else
        outputs = ggml_pad(ctx, outputs, pre_lookahead_len, 0, 0, 0);
    outputs = conv1.build_cgraph(ctx, outputs, 1, 0, 1, 1, {});
    outputs = ggml_leaky_relu(ctx, outputs, 0.01f, false);

    outputs = ggml_pad_ext(ctx, outputs, static_cast<int>(conv2.weight->ne[0] - 1), 0, 0, 0, 0, 0, 0, 0);
    outputs = conv2.build_cgraph(ctx, outputs, 1, 0, 1, 1, {});
    outputs = ggml_permute(ctx, outputs, 1, 0, 2, 3);

    outputs = ggml_cont(ctx, outputs);
    if (cut_len)
    {
        outputs = ggml_view_3d(ctx, outputs, outputs->ne[0], outputs->ne[1] - cut_len, outputs->ne[2], outputs->nb[1], outputs->nb[2], outputs->nb[1] * cut_len);
        inputs = ggml_view_3d(ctx, inputs, inputs->ne[0], inputs->ne[1] - cut_len, inputs->ne[2], inputs->nb[1], inputs->nb[2], inputs->nb[1] * cut_len);
    }

    outputs = ggml_add(ctx, outputs, inputs);
    return outputs;
}

CausalMaskedDiffWithDiT::EncodeResult CausalMaskedDiffWithDiT::build_cgraph_encode(ggml_context* ctx, ggml_tensor* token, ggml_tensor* prompt_token, ggml_tensor* prompt_feat, ggml_tensor* embedding, ggml_backend_op_capabilities capabilities, uint32_t cut_len, bool streaming) const
{
    embedding = ggml_l2_norm(ctx, embedding, 1e-6f);
    embedding = spk_embed_affine_layer.build_cgraph(ctx, embedding);

    token = concat_tensors(ctx, std::array{ prompt_token, token }, 0, capabilities);
    token = ggml_get_rows(ctx, input_embedding, token);

    ggml_tensor* h = pre_lookahead_layer.build_cgraph(ctx, token, streaming, cut_len / token_mel_ratio);
    h = unsqueeze(ctx, h, 1);
    h = ggml_repeat_4d(ctx, h, h->ne[0], token_mel_ratio, h->ne[2], h->ne[3]);
    h = ggml_reshape_3d(ctx, h, h->ne[0], h->ne[1] * h->ne[2], h->ne[3]);

    const auto mel_len1 = prompt_feat->ne[1];
    const auto mel_len2 = h->ne[1] - mel_len1 + cut_len;
    auto conds = ggml_pad(ctx, prompt_feat, 0, static_cast<int>(mel_len2), 0, 0);
    if (cut_len)
        conds = ggml_view_3d(ctx, conds, conds->ne[0], conds->ne[1] - cut_len, conds->ne[2], conds->nb[1], conds->nb[2], conds->nb[1] * cut_len);

    return EncodeResult{
        .mu = h,
        .spks = embedding,
        .conds = conds,
        .cut_len = cut_len ? 0 : mel_len1
    };
}

int CausalConv1d::causal_padding() const
{
    const auto kernel_size = weight->ne[0];
    const auto causal_padding = static_cast<int>((kernel_size * d - d) / 2 * 2 + (kernel_size + 1) % 2);
    return causal_padding;
}

ggml_tensor* CausalConv1d::build_cgraph(ggml_context* ctx, ggml_tensor* x, bool finalize) const
{
    GGML_ASSERT(finalize || causal_type == right);

    if (!ggml_is_contiguous(x))
        x = ggml_cont(ctx, x);
    if (causal_type == left)
        x = ggml_pad_ext(ctx, x, causal_padding(), 0, 0, 0, 0, 0, 0, 0);
    else if (causal_type == right)
    {
        if (finalize)
            x = ggml_pad_ext(ctx, x, 0, causal_padding(), 0, 0, 0, 0, 0, 0);
    }
    else
        GGML_ABORT("CausalConv1d: invalid causal type %d", static_cast<int>(causal_type));

    x = Conv1d::build_cgraph(ctx, x, 1, 0, d, 1, {});
    return x;
}

CausalConvRNNF0Predictor::CausalConvRNNF0Predictor()
{
    condnet_0.causal_type = CausalConv1d::right;
    condnet_2.causal_type = CausalConv1d::left;
    condnet_4.causal_type = CausalConv1d::left;
    condnet_6.causal_type = CausalConv1d::left;
    condnet_8.causal_type = CausalConv1d::left;
}

ggml_tensor* CausalConvRNNF0Predictor::build_cgraph(ggml_context* ctx, ggml_tensor* x, bool finalize) const
{
    x = condnet_0.build_cgraph(ctx, x, finalize);
    x = ggml_elu(ctx, x);
    x = condnet_2.build_cgraph(ctx, x);
    x = ggml_elu(ctx, x);
    x = condnet_4.build_cgraph(ctx, x);
    x = ggml_elu(ctx, x);
    x = condnet_6.build_cgraph(ctx, x);
    x = ggml_elu(ctx, x);
    x = condnet_8.build_cgraph(ctx, x);
    x = ggml_elu(ctx, x);

    x = ggml_permute(ctx, x, 1, 0, 2, 3);
    x = ggml_cont(ctx, x);
    x = classifier.build_cgraph(ctx, x);
    x = ggml_reshape_2d(ctx, x, x->ne[1], x->ne[2]);
    return ggml_abs(ctx, x);
}

std::array<ggml_tensor*, 2> SineGen2::build_cgraph(ggml_context* ctx, ggml_tensor* f0, int harmonic_num, int sampling_rate, int upsample_scale, float sine_amp, int voiced_threshold, float noise_std) const
{
    auto fn = ggml_arange(ctx, 1.f, static_cast<float>(harmonic_num + 2), 1.f);
    fn = ggml_mul(ctx, ggml_repeat_4d(ctx, f0, harmonic_num + 1, f0->ne[1], f0->ne[2], 1), fn);

    // _f02sine
    auto rad_values = ggml_scale(ctx, fn, 1.f / sampling_rate);
    ggml_tensor* sine_waves;
    {
        rad_values = ggml_sub(ctx,
            rad_values,
            ggml_floor(ctx, rad_values));

        rad_values = view_tensor(ctx, rad_values,
            ggml_acc(ctx, rad_values, rand_ini, rad_values->nb[1], rad_values->nb[2], rad_values->nb[3], 0));

        rad_values = ggml_permute(ctx, rad_values, 1, 0, 2, 3);
        rad_values = ggml_interpolate(ctx, rad_values, rad_values->ne[0] / upsample_scale, rad_values->ne[1], rad_values->ne[2], 1, GGML_SCALE_MODE_BILINEAR);
        auto phase = ggml_cumsum(ctx, rad_values);
        phase = ggml_scale(ctx, phase, 2.0f * 3.14159265358979323846f * upsample_scale);
        phase = ggml_permute(ctx, phase, 1, 0, 2, 3);
        phase = ggml_interpolate(ctx, phase, phase->ne[0], phase->ne[1] * upsample_scale, phase->ne[2], 1, GGML_SCALE_MODE_NEAREST);
        auto sines = ggml_sin(ctx, phase);
        sine_waves = ggml_scale(ctx, sines, sine_amp);
    }

    // _f02uv
    ggml_tensor* uv;
    {
        constexpr float eps = 1e-7f;
        auto x = ggml_scale_bias(ctx, f0, 1.f, static_cast<float>(-voiced_threshold));
        auto pos = ggml_relu(ctx, x);
        auto denom = ggml_scale_bias(ctx, ggml_abs(ctx, x), 1.f, eps);
        uv = ggml_div(ctx, pos, denom);
    }

    auto noise_amp = ggml_scale_bias(ctx, uv, noise_std - sine_amp / 3, sine_amp / 3);
    auto noise_orig = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, sine_waves->ne[0], sine_waves->ne[1]);
    auto noise = ggml_mul(ctx, noise_orig, noise_amp);

    sine_waves = ggml_mul(ctx, sine_waves, uv);
    sine_waves = ggml_add(ctx, sine_waves, noise);
    return { sine_waves, noise_orig };
}

std::array<ggml_tensor*, 2> SourceModuleHnNSF::build_cgraph(ggml_context* ctx, ggml_tensor* x, int harmonic_num, int sampling_rate, int upsample_scale, float sine_amp, int voiced_threshold, float noise_std) const
{
    auto [sine_wavs, noise] = l_sin_gen.build_cgraph(ctx, x, harmonic_num, sampling_rate, upsample_scale, sine_amp, voiced_threshold, noise_std);
    auto sine_merge = l_linear.build_cgraph(ctx, sine_wavs);
    sine_merge = ggml_tanh(ctx, sine_merge);
    return { sine_merge, noise };
}

ggml_tensor* Snake::build_cgraph(ggml_context* ctx, ggml_tensor* x) const
{
    auto alpha = unsqueeze(ctx, this->alpha, 0);
    auto x_sin = ggml_sin(ctx, ggml_mul(ctx, x, alpha));
    x_sin = ggml_mul(ctx, x_sin, x_sin);
    return ggml_add(ctx, x,
        ggml_div(ctx, x_sin, alpha));
}

ggml_tensor* ResBlock::build_cgraph(ggml_context* ctx, ggml_tensor* x) const
{
    for (const auto& [act1, conv1, act2, conv2] : convs)
    {
        auto xt = act1.build_cgraph(ctx, x);
        xt = conv1.build_cgraph(ctx, xt);
        xt = act2.build_cgraph(ctx, xt);
        xt = conv2.build_cgraph(ctx, xt);
        x = ggml_add(ctx, x, xt);
    }
    return x;
}

ggml_tensor* CausalConv1dDownSample::build_cgraph(ggml_context* ctx, ggml_tensor* x) const
{
    x = ggml_pad_ext(ctx, x, s - 1, 0, 0, 0, 0, 0, 0, 0);
    return Conv1d::build_cgraph(ctx, x, s, 0, 1, 1, {});
}

ggml_tensor* CausalConv1dUpsample::build_cgraph(ggml_context* ctx, ggml_tensor* x) const
{
    x = ggml_interpolate(ctx, x, x->ne[0] * s, x->ne[1], x->ne[2], 1, GGML_SCALE_MODE_NEAREST);
    x = ggml_pad_ext(ctx, x, static_cast<int>(weight->ne[0] - 1), 0, 0, 0, 0, 0, 0, 0);
    return Conv1d::build_cgraph(ctx, x, 1, 0, 1, 1, {});
}

std::array<ggml_tensor*, 2> CausalHiFTGenerator::build_cgraph(ggml_context* ctx, ggml_tensor* speech_feat, bool finalize) const
{
    auto f0 = f0_predictor.build_cgraph(ctx, speech_feat, finalize);
    auto s_input = ggml_permute(ctx, f0, 1, 0, 2, 3);
    s_input = ggml_interpolate(ctx, s_input, s_input->ne[0], s_input->ne[1] * scale_factor, s_input->ne[2], 1, GGML_SCALE_MODE_NEAREST);
    auto [s, noise] = m_source.build_cgraph(ctx, s_input, nb_harmonics, sampling_rate, scale_factor, nsf_alpha, nsf_voiced_threshold, nsf_sigma);
    s = ggml_permute(ctx, s, 1, 0, 2, 3);

    // decode
    ggml_tensor* x;
    {
        if (!finalize)
            speech_feat = ggml_view_2d(ctx, speech_feat, speech_feat->ne[0] - f0_predictor.condnet_0.causal_padding(), speech_feat->ne[1], speech_feat->nb[1], 0);
        x = conv_pre.build_cgraph(ctx, speech_feat, finalize);

        auto s_stft = ggml_stft(ctx, s, window, hop_len, true, fctx.get());
        s_stft = ggml_cont(ctx, s_stft);
        s_stft = ggml_reshape_2d(ctx, s_stft, s_stft->ne[0], s_stft->ne[1] * 2);
        if (!finalize)
            s_stft = ggml_view_2d(ctx, s_stft, s_stft->ne[0] - (scale_factor / hop_len * conv_pre.causal_padding()), s_stft->ne[1], s_stft->nb[1], 0);

        const auto num_upsamples = ups.size();
        const auto num_kernels = resblocks.size() / num_upsamples;
        for (size_t i = 0; i != num_upsamples; ++i)
        {
            x = ggml_leaky_relu(ctx, x, lrelu_slope, false);
            x = ups[i].build_cgraph(ctx, x);

            if (i == num_upsamples - 1)
                x = ggml_pad_reflect_1d(ctx, x, 1, 0);

            auto si = source_downs[i]->build_cgraph(ctx, s_stft);
            si = source_resblocks[i].build_cgraph(ctx, si);
            x = ggml_add(ctx, x, si);

            auto xs = resblocks[i * num_kernels].build_cgraph(ctx, x);
            for (size_t j = 1; j != num_kernels; ++j)
                xs = ggml_add(ctx, xs,
                    resblocks[i * num_kernels + j].build_cgraph(ctx, x));
            x = ggml_scale(ctx, xs, 1.f / num_kernels);
        }
        x = ggml_leaky_relu(ctx, x, 0.01f, false);
        x = conv_post.build_cgraph(ctx, x);

        auto magnitude = ggml_view_3d(ctx, x, x->ne[0], nfft / 2 + 1, x->ne[2], x->nb[1], x->nb[2], 0);
        magnitude = ggml_exp(ctx, magnitude);
        auto phase = ggml_view_3d(ctx, x, x->ne[0], nfft / 2 + 1, x->ne[2], x->nb[1], x->nb[2], (nfft / 2 + 1) * x->nb[1]);
        phase = ggml_sin(ctx, phase);

        // istft
        {
            magnitude = ggml_clamp(ctx, magnitude, 0.f, 1e2f);
            auto real = ggml_mul(ctx, magnitude,
                ggml_cos(ctx, phase));
            auto imag = ggml_mul(ctx, magnitude,
                ggml_sin(ctx, phase));

            x = ggml_istft(ctx, real, imag, window, hop_len, true, ictx.get());
            if (!finalize)
                x = ggml_view_2d(ctx, x, x->ne[0] - scale_factor, x->ne[1], x->nb[1], 0);
            x = ggml_clamp(ctx, x, -audio_limit, audio_limit);
            ggml_set_cpu(x);
            return { x, noise };
        }
    }
}

ggml_tensor* Qwen2MLP::build_cgraph(ggml_context* ctx, ggml_tensor* x) const
{
    auto gate = gate_proj.build_cgraph(ctx, x);
    auto up_states = up_proj.build_cgraph(ctx, x);

    up_states = ggml_swiglu_split(ctx, gate, up_states);
    return down_proj.build_cgraph(ctx, up_states);
}

ggml_tensor* Qwen2RMSNorm::build_cgraph(ggml_context* ctx, ggml_tensor* hidden_states, float variance_epsilon) const
{
    hidden_states = ggml_rms_norm(ctx, hidden_states, variance_epsilon);
    hidden_states = ggml_mul(ctx, hidden_states, weight);
    return hidden_states;
}	
