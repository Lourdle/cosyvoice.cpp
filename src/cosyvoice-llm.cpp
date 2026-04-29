#include "cosyvoice-internal.h"
#include "cosyvoice-model.h"
#include "cosyvoice-llm-kv-cache.h"

#include <algorithm>
#include <span>
#include <ranges>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#elif defined(__aarch64__) || defined(_M_ARM64)
#define SIMDE_ENABLE_NATIVE_ALIASES
#include <simde/x86/avx2.h>
#include <simde/x86/fma.h>
#else
#error "src/cosyvoice-llm.cpp requires x86_64 SIMD intrinsics or SIMDe on ARM64; unsupported architecture"
#endif

static void build_causal_mask(ggml_fp16_t* mask, uint32_t n_batch, uint32_t seq_len)
{
    uint32_t visible_prefix_end = seq_len - n_batch;
    for (uint32_t i = 0; i != n_batch; ++i)
    {
        for (uint32_t j = 0; j != seq_len; ++j)
            *mask++ = j > visible_prefix_end ? 0xFC00 /* -inf */ : 0;
        ++visible_prefix_end;
    }
}

static ggml_tensor* build_qwen2_decoder_layer(const Qwen2DecoderLayer& layer, ggml_context* ctx0, ggml_cgraph* gf, ggml_tensor* hidden_states, ggml_tensor* position_ids, cosyvoice_llm_kv_cache& kv_cache, ggml_tensor* attention_mask, float rope_theta, float rms_norm_eps, int num_attention_heads, int num_key_value_heads, int layer_idx)
{
    auto residual = hidden_states;
    hidden_states = layer.input_layernorm.build_cgraph(ctx0, hidden_states, rms_norm_eps);

    auto query_states = layer.self_attn.q_proj.build_cgraph(ctx0, hidden_states);
    auto key_states = layer.self_attn.k_proj.build_cgraph(ctx0, hidden_states);
    auto value_states = layer.self_attn.v_proj.build_cgraph(ctx0, hidden_states);

    query_states = ggml_reshape_3d(
        ctx0, query_states,
        query_states->ne[0] / num_attention_heads,
        num_attention_heads,
        query_states->ne[1]);
    key_states = ggml_reshape_3d(
        ctx0, key_states,
        key_states->ne[0] / num_key_value_heads,
        num_key_value_heads,
        key_states->ne[1]);
    value_states = ggml_reshape_3d(
        ctx0, value_states,
        value_states->ne[0] / num_key_value_heads,
        num_key_value_heads,
        value_states->ne[1]);

    query_states = ggml_rope_ext(ctx0, query_states, position_ids, nullptr, static_cast<int>(query_states->ne[0]), GGML_ROPE_TYPE_NEOX, 0, rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    key_states = ggml_rope_ext(ctx0, key_states, position_ids, nullptr, static_cast<int>(key_states->ne[0]), GGML_ROPE_TYPE_NEOX, 0, rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    query_states = ggml_permute(ctx0, query_states, 0, 2, 1, 3);
    key_states = ggml_permute(ctx0, key_states, 0, 2, 1, 3);
    value_states = ggml_permute(ctx0, value_states, 0, 2, 1, 3);

    query_states = ggml_cont(ctx0, query_states);
    kv_cache.update_cache(ctx0, gf, key_states, value_states, position_ids, static_cast<int>(layer_idx));

    ggml_tensor* attn_output = kv_cache.attention_forward(ctx0, gf, query_states, key_states, value_states, attention_mask, layer_idx);
    attn_output = ggml_reshape_2d(
        ctx0, attn_output,
        attn_output->ne[0] * attn_output->ne[1],
        attn_output->ne[2]);
    attn_output = layer.self_attn.o_proj.build_cgraph(ctx0, attn_output);

    hidden_states = ggml_add(ctx0, residual, attn_output);

    residual = hidden_states;
    hidden_states = layer.post_attention_layernorm.build_cgraph(ctx0, hidden_states, rms_norm_eps);
    hidden_states = layer.mlp.build_cgraph(ctx0, hidden_states);
    hidden_states = ggml_add(ctx0, residual, hidden_states);
    return hidden_states;
}

static void set_graph_backend(ggml_cgraph* gf, ggml_backend_sched_t sched, ggml_backend_t backend)
{
    for (auto node : ggml_cgraph_node_iterator(gf))
        ggml_backend_sched_set_tensor_backend(sched, node, backend);
}

bool cosyvoice_model_3::llm_prefill(
    ggml_type type,
    const void* data,
    uint32_t n_tokens
)
{
    auto total_len = n_tokens + kv_cache->cur_len;
    GGML_ASSERT(total_len <= params.n_max_seq - 1);
    GGML_ASSERT(n_tokens <= params.n_batch);

    auto causal_mask = n_tokens == 1 ? nullptr : this->causal_mask;
    if (causal_mask)
    {
        causal_mask->ne[0] = total_len;
        causal_mask->ne[1] = n_tokens;
        causal_mask->nb[1] = total_len * sizeof(ggml_fp16_t);
        causal_mask->nb[3] = causal_mask->nb[2] = causal_mask->ne[1] * causal_mask->nb[1];
        build_causal_mask(causal_mask_buffer.get(), n_tokens, total_len);
        ggml_backend_tensor_set_async(
            backend.get(),
            causal_mask,
            causal_mask_buffer.get(),
            0,
            causal_mask->nb[2]
        );
    }

    position_ids->ne[0] = n_tokens;
    ggml_backend_tensor_set_async(backend.get(), position_ids, full_position_ids.get() + kv_cache->cur_len, 0, n_tokens * sizeof(int32_t));
    
    if (gf && llm_input && !llm_probs
        && llm_input->type == type && n_tokens == llm_input->ne[1]
        && kv_cache->can_reuse(true))
    {
        ggml_backend_tensor_set_async(backend.get(), llm_input, data, 0, ggml_nbytes(llm_input));
        kv_cache->shift_kv_node_pos(n_tokens);
    }
    else
    {
        auto ctx0 = this->ctx0.get();
        ggml_reset(ctx0);
        ggml_backend_sched_reset(sched.get());

        gf = ggml_new_graph(ctx0);
        this->gf = gf;
        llm_input = ggml_new_tensor_2d(
            ctx0,
            type,
            llm.speech_embedding_weight->ne[0],
            n_tokens);
        llm_probs = nullptr;

        auto hidden_states = type == GGML_TYPE_F32 ? llm_input : ggml_cast(ctx0, llm_input, GGML_TYPE_F32);

        auto num_attention_heads = llm.num_attention_heads;
        auto num_key_value_heads = llm.num_key_value_heads;
        for (auto end = static_cast<int>(llm.layers.size() - 1), i = 0; i != end; ++i)
            hidden_states = build_qwen2_decoder_layer(
                llm.layers[i],
                ctx0, gf,
                hidden_states, position_ids, *kv_cache, causal_mask,
                llm.rope_theta,
                llm.rms_norm_eps,
                num_attention_heads, num_key_value_heads,
                i);

        // Final layer.
        const auto& layer = llm.layers.back();

        hidden_states = layer.input_layernorm.build_cgraph(ctx0, hidden_states, llm.rms_norm_eps);

        auto key_states = layer.self_attn.k_proj.build_cgraph(ctx0, hidden_states);
        auto value_states = layer.self_attn.v_proj.build_cgraph(ctx0, hidden_states);

        key_states = ggml_reshape_3d(
            ctx0, key_states,
            key_states->ne[0] / num_key_value_heads,
            num_key_value_heads,
            key_states->ne[1]);
        value_states = ggml_reshape_3d(
            ctx0, value_states,
            value_states->ne[0] / num_key_value_heads,
            num_key_value_heads,
            value_states->ne[1]);

        key_states = ggml_rope_ext(ctx0, key_states, position_ids, nullptr, static_cast<int>(key_states->ne[0]), GGML_ROPE_TYPE_NEOX, 0, llm.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        key_states = ggml_permute(ctx0, key_states, 0, 2, 1, 3);
        value_states = ggml_permute(ctx0, value_states, 0, 2, 1, 3);

        kv_cache->update_cache(ctx0, gf, key_states, value_states, position_ids, static_cast<int>(llm.layers.size() - 1));
        kv_cache->cur_len += n_tokens;

        set_graph_backend(gf, sched.get(), backend.get());
        ggml_backend_sched_alloc_graph(sched.get(), gf);
        ggml_backend_tensor_set_async(backend.get(), llm_input, data, 0, ggml_nbytes(llm_input));
    }

    status = ggml_backend_sched_graph_compute(sched.get(), gf);
    return status == GGML_STATUS_SUCCESS;
}

bool cosyvoice_model_3::llm_decode(ggml_type type, const void* data)
{
    GGML_ASSERT(kv_cache->cur_len + 1 <= params.n_max_seq);

    position_ids->ne[0] = 1;
    ggml_backend_tensor_set_async(backend.get(), position_ids, full_position_ids.get() + kv_cache->cur_len, 0, sizeof(int32_t));

    if (gf && llm_input && llm_probs
        && llm_input->type == type && 1 == llm_input->ne[1]
        && kv_cache->can_reuse(false))
    {
        ggml_backend_tensor_set_async(backend.get(), llm_input, data, 0, ggml_nbytes(llm_input));
        kv_cache->shift_kv_node_pos(1);
    }
    else
    {
        auto ctx0 = this->ctx0.get();
        ggml_reset(ctx0);
        ggml_backend_sched_reset(sched.get());

        gf = ggml_new_graph(ctx0);
        llm_input = ggml_new_tensor_1d(
            ctx0,
            type,
            llm.speech_embedding_weight->ne[0]);
        llm_probs = nullptr;

        auto hidden_states = type == GGML_TYPE_F32 ? llm_input : ggml_cast(ctx0, llm_input, GGML_TYPE_F32);

        auto num_attention_heads = llm.num_attention_heads;
        auto num_key_value_heads = llm.num_key_value_heads;
        for (auto end = static_cast<int>(llm.layers.size()), i = 0; i != end; ++i)
            hidden_states = build_qwen2_decoder_layer(
                llm.layers[i],
                ctx0, gf,
                hidden_states, position_ids, *kv_cache, nullptr,
                llm.rope_theta,
                llm.rms_norm_eps,
                num_attention_heads, num_key_value_heads,
                i);
        ++kv_cache->cur_len;

        hidden_states = llm.norm.build_cgraph(ctx0, hidden_states, llm.rms_norm_eps);

        auto logits = llm.llm_decoder.build_cgraph(ctx0, hidden_states);
        auto probs = ggml_soft_max_ext(ctx0, logits, nullptr, 1.f / config.temperature, 0.f);
        auto top_k = ggml_top_k(ctx0, probs, config.sampling.top_k);

        probs = ggml_reshape_2d(ctx0, probs, 1, probs->ne[0]);
        probs = ggml_get_rows(ctx0, probs, top_k);
        top_k = ggml_reshape_2d(ctx0, top_k, 1, top_k->ne[0]);
        top_k->type = GGML_TYPE_F32;
        probs = ggml_concat(ctx0, top_k, probs, 0);
        this->llm_probs = probs;

        ggml_build_forward_expand(gf, probs);
        set_graph_backend(gf, sched.get(), backend.get());
        ggml_backend_sched_alloc_graph(sched.get(), gf);
        ggml_backend_tensor_set_async(backend.get(), llm_input, data, 0, ggml_nbytes(llm_input));
    }

    status = ggml_backend_sched_graph_compute_async(sched.get(), gf);
    if (status == GGML_STATUS_SUCCESS)
    {
        auto probs = reinterpret_cast<cosyvoice_llm_token_prob_t*>(nucleus_probs.get() + 1);
        ggml_backend_tensor_get_async(backend.get(), llm_probs, probs, 0, ggml_nbytes(llm_probs));
        ggml_backend_tensor_get_async(
            backend.get(),
            llm_probs->src[1]->src[0]->src[0],
            this->probs.get(),
            0,
            sizeof(float) * llm.llm_decoder.weight->ne[1]);
        ggml_backend_synchronize(backend.get());
        return true;
    }
    return false;
}

void cosyvoice_model_3::llm_prepare_probs(bool allow_stop_tokens)
{
    GGML_ASSERT(llm_probs);

    auto probs = reinterpret_cast<cosyvoice_llm_token_prob_t*>(nucleus_probs.get() + 1);
    int k = config.sampling.top_k;
    const float top_p = config.sampling.top_p;
    std::sort(probs, probs + k, [](const cosyvoice_llm_token_prob_t& a, const cosyvoice_llm_token_prob_t& b)
        { return a.prob > b.prob; });

    float p = 0.f;
    for (int i = 0; i != k; ++i)
    {
        if (!allow_stop_tokens && llm_is_stop_token(probs[i].token_id))
            probs[i].prob = 0.f;
        p += probs[i].prob;
        if (p >= top_p)
        {
            k = i + 1;
            break;
        }
    }

    reinterpret_cast<int*>(probs)[-1] = k;
    for (int i = 0; i != k; ++i)
        probs[i].prob /= p;

    if (!allow_stop_tokens)
    {
        auto raw_probs = this->probs.get();
        auto vocab_size = static_cast<uint32_t>(get_speech_token_embed_weight()->ne[1]);
        for (auto token_id : stop_tokens)
            raw_probs[token_id] = 0;

        uint32_t i = 0;
        __m256 sum256 = _mm256_setzero_ps();
        for (; i + 7 < vocab_size; i += 8)
        {
            __m256 prob256 = _mm256_loadu_ps(raw_probs + i);
            sum256 = _mm256_add_ps(sum256, prob256);
        }

        __m128 vlow = _mm256_castps256_ps128(sum256);
        __m128 vhigh = _mm256_extractf128_ps(sum256, 1);
        __m128 sum128 = _mm_add_ps(vlow, vhigh);
        for (; i + 4 < vocab_size; ++i)
        {
            __m128 prob256 = _mm_loadu_ps(raw_probs + i);
            sum128 = _mm_add_ps(sum128, prob256);
        }

        __m128 shuf = _mm_movehdup_ps(sum128);
        __m128 sums = _mm_add_ps(sum128, shuf);
        shuf = _mm_movehl_ps(shuf, sums);
        sums = _mm_add_ss(sums, shuf);

        float sum = _mm_cvtss_f32(sums);
        sum256 = _mm256_set1_ps(sum);
        for (i = 0; i + 7 < vocab_size; i += 8)
        {
            __m256 prob256 = _mm256_loadu_ps(raw_probs + i);
            prob256 = _mm256_div_ps(prob256, sum256);
            _mm256_storeu_ps(raw_probs + i, prob256);
        }
        for (sum128 = _mm_set_ps1(sum); i + 3 < vocab_size; i += 4)
        {
            __m128 prob256 = _mm_loadu_ps(raw_probs + i);
            prob256 = _mm_div_ps(prob256, sum128);
            _mm_storeu_ps(raw_probs + i, prob256);
        }
        for (; i < vocab_size; ++i)
            raw_probs[i] /= sum;
    }
}

int cosyvoice_llm_sampler(
    cosyvoice_llm_token_prob_t* nucleus_probs,
    int k,
    float* probs,
    uint32_t size,
    const cosyvoice_sampling_params_t* sampling_params,
    int* accepted_tokens,
    uint32_t n_accepted_tokens,
    std::mt19937* rng)
{
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float fallback_random = dist(*rng);
    float nucleus_random = fallback_random;
    for (int i = 0; i != k; ++i)
    {
        nucleus_random -= nucleus_probs[i].prob;
        if (nucleus_random <= 0.0f)
        {
            int sampled_token = nucleus_probs[i].token_id;
            int repeat_count = 0;
            for (auto t : std::ranges::views::reverse(std::span(accepted_tokens, n_accepted_tokens)) | std::ranges::views::take(sampling_params->win_size))
                if (t == sampled_token)
                    ++repeat_count;
            if (repeat_count < sampling_params->win_size * sampling_params->tau_r) return sampled_token;

            for (i = 0; i != size; ++i)
            {
                fallback_random -= probs[i];
                if (fallback_random <= 0.0f)
                    return i;
            }
            break;
        }
    }
    return -1;
}
