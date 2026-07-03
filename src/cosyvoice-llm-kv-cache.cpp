#include "cosyvoice-model.h"
#include "cosyvoice-llm-kv-cache.h"

#include <ggml-backend.h>

#include <utility>
#include <span>

void cosyvoice_llm_kv_cache::build_kv_cache(
    ggml_backend_t backend,
    ggml_backend_buffer_ptr& shared_buffer,
    int layers,
    int k_head_dim,
    int v_head_dim,
    int num_attention_heads,
    int num_key_value_heads,
    uint32_t max_seq,
    ggml_type k_type,
    ggml_type v_type,
    bool fattn)
{
    cur_len = 0;
    this->layers = layers;
    this->fattn = fattn;
    this->k_type = k_type;
    this->v_type = v_type;
    offloaded_cache = nullptr;
    this->num_attention_heads = num_attention_heads;
    kv_cache_layers = new kv_cache_layer[layers];

    ggml_init_params params = {
        .mem_size = layers * 2 * ggml_tensor_overhead(),
        .no_alloc = true
    };
    ctx = ggml_init(params);

    shared_buffer.reset(initialize_buffer(backend, k_head_dim, v_head_dim, num_attention_heads, num_key_value_heads, max_seq, k_type, v_type, fattn));
}

ggml_backend_buffer* cosyvoice_llm_kv_cache::initialize_buffer(ggml_backend_t backend, int k_head_dim, int v_head_dim, int num_attention_heads, int num_key_value_heads, uint32_t max_seq, ggml_type k_type, ggml_type v_type, bool fattn)
{
    int64_t k_ne[3] = { k_head_dim, max_seq, num_key_value_heads };
    int64_t v_ne[3] = { v_head_dim, max_seq, num_key_value_heads };
    if (!fattn) std::swap(v_ne[0], v_ne[1]);
    ggml_reset(ctx);

    for (auto& [k, v, k_view, v_view] : std::span(kv_cache_layers, layers))
    {
        k = ggml_new_tensor(ctx, k_type, 3, k_ne);
        v = ggml_new_tensor(ctx, v_type, 3, v_ne);
        k_view = nullptr;
        v_view = nullptr;
    }
    return ggml_backend_alloc_ctx_tensors(ctx, backend);
}

uint32_t cosyvoice_llm_kv_cache::reset_buffer(ggml_backend_buffer* buffer)
{
    cur_len = 0;
    auto alignment = ggml_backend_buffer_get_alignment(buffer);
    auto max_seq_len = static_cast<uint32_t>(kv_cache_layers[0].k->ne[1]);
    auto k_ne = *reinterpret_cast<std::array<int64_t, GGML_MAX_DIMS>*>(&kv_cache_layers[0].k->ne);
    auto v_ne = *reinterpret_cast<std::array<int64_t, GGML_MAX_DIMS>*>(&kv_cache_layers[0].v->ne);

    auto current_k_size = get_aligned_size(ggml_backend_buffer_get_alloc_size(buffer, kv_cache_layers[0].k), alignment);
    auto current_v_size = get_aligned_size(ggml_backend_buffer_get_alloc_size(buffer, kv_cache_layers[0].v), alignment);
    GGML_ASSERT(ggml_backend_buffer_get_size(buffer) >= static_cast<size_t>(layers) * (current_k_size + current_v_size));

    // Rebind the existing KV layout into the larger buffer without changing the
    // cached sequence capacity.
    if (k_ne[0] == v_ne[0])
        v_ne[1] = k_ne[1];
    else
        v_ne[0] = k_ne[1];

    ggml_reset(ctx);
    auto buffer_base = reinterpret_cast<char*>(ggml_backend_buffer_get_base(buffer));
    for (auto& [k, v, k_view, v_view] : std::span(kv_cache_layers, layers))
    {
        k = ggml_new_tensor(ctx, k_type, GGML_MAX_DIMS, k_ne.data());
        v = ggml_new_tensor(ctx, v_type, GGML_MAX_DIMS, v_ne.data());
        k_view = nullptr;
        v_view = nullptr;

        ggml_backend_tensor_alloc(buffer, k, buffer_base);
        buffer_base += get_aligned_size(ggml_backend_buffer_get_alloc_size(buffer, k), alignment);
        ggml_backend_tensor_alloc(buffer, v, buffer_base);
        buffer_base += get_aligned_size(ggml_backend_buffer_get_alloc_size(buffer, v), alignment);
    }
    return max_seq_len;
}

#pragma pack(push)
#pragma pack(1)
struct offloaded_kv_cache
{
    uint32_t len;
    ggml_context* ctx;
    struct offloaded_kv_layer
    {
        ggml_tensor* k_view;
        ggml_tensor* v_view;
        void* k;
        void* v;
    } offloaded_kv_layers[];
};
#pragma pack(pop)

void cosyvoice_llm_kv_cache::offload_cache(ggml_backend_t backend, ggml_backend_sched* sched, uint32_t n_tokens)
{
    char* buffer_base;
    size_t nbytes = kv_cache_layers[0].k->ne[0] * sizeof(float) * n_tokens * kv_cache_layers[0].v->ne[2];
    if (!offloaded_cache)
    {
        ggml_init_params params =
        {
            .mem_size = (ggml_tensor_overhead() + ggml_graph_overhead()) * layers * 4,
            .no_alloc = true
        };
        offloaded_cache = reinterpret_cast<offloaded_kv_cache*>(malloc(sizeof(offloaded_kv_cache::offloaded_kv_layer) * layers + sizeof(uint32_t) + sizeof(ggml_context*)));
        offloaded_cache->ctx = ggml_init(params);
        goto alloc_buffer;
    }
    else
    {
        ggml_reset(offloaded_cache->ctx);
        if (offloaded_cache->len < n_tokens)
        {
            free(offloaded_cache->offloaded_kv_layers[0].k);
        alloc_buffer:
            buffer_base = reinterpret_cast<char*>(malloc(nbytes * 2 * layers));
        }
        else buffer_base = reinterpret_cast<char*>(offloaded_cache->offloaded_kv_layers[0].k);
    }

    offloaded_cache->len = n_tokens;
    auto gf = ggml_new_graph_custom(offloaded_cache->ctx, layers * 4, false);
    ggml_tensor k_placeholder_tensor{ .ne{ kv_cache_layers[0].k->ne[0], n_tokens, kv_cache_layers[0].k->ne[2], 1 } };
    ggml_tensor v_placeholder_tensor{ .ne{ fattn ? kv_cache_layers[0].v->ne[0] : n_tokens, fattn ? n_tokens : kv_cache_layers[0].v->ne[0], k_placeholder_tensor.ne[2], 1 } };
    for (int i = 0; i != layers; ++i)
    {
        auto& offloaded_layer = offloaded_cache->offloaded_kv_layers[i];
        auto& layer = kv_cache_layers[i];

        ggml_tensor* k_view = ggml_view_3d(offloaded_cache->ctx, layer.k, layer.k->ne[0], n_tokens, layer.k->ne[2], layer.k->nb[1], layer.k->nb[2], 0);
        ggml_tensor* v_view;
        if (fattn)
            v_view = ggml_view_3d(offloaded_cache->ctx, layer.v, layer.v->ne[0], n_tokens, layer.v->ne[2], layer.v->nb[1], layer.v->nb[2], 0);
        else
            v_view = ggml_view_3d(offloaded_cache->ctx, layer.v, n_tokens, layer.v->ne[1], layer.v->ne[2], layer.v->nb[1], layer.v->nb[2], 0);

        // The ggml_cast uses GGML_OP_CPY to implement the cast, which has a bug that the non-contiguous tensor causes the bad result.
        // But most of the backends use cpy to implement the dup, so we have to use ggml_cast and then set the op to GGML_OP_DUP to avoid the bug.
        offloaded_layer.k_view = ggml_cast(offloaded_cache->ctx, k_view, GGML_TYPE_F32);
        offloaded_layer.v_view = ggml_cast(offloaded_cache->ctx, v_view, GGML_TYPE_F32);
        offloaded_layer.k_view->op = GGML_OP_DUP;
        offloaded_layer.v_view->op = GGML_OP_DUP;
        // IMPORTANT: src[1] must be set to the placeholder tensor
        // or it will cause the bug that the non-contiguous tensor causes the bad result.
        offloaded_layer.k_view->src[1] = &k_placeholder_tensor;
        offloaded_layer.v_view->src[1] = &v_placeholder_tensor;

        ggml_build_forward_expand(gf, offloaded_layer.k_view);
        ggml_build_forward_expand(gf, offloaded_layer.v_view);
    }

    ggml_backend_sched_alloc_graph(sched, gf);
    ggml_backend_sched_graph_compute_async(sched, gf);

    for (int i = 0; i != layers; ++i)
    {
        auto& offloaded_layer = offloaded_cache->offloaded_kv_layers[i];
        offloaded_layer.k = buffer_base;
        offloaded_layer.v = buffer_base + nbytes;
        ggml_backend_tensor_get_async(backend, offloaded_layer.k_view, offloaded_layer.k, 0, nbytes);
        ggml_backend_tensor_get_async(backend, offloaded_layer.v_view, offloaded_layer.v, 0, nbytes);
        buffer_base += nbytes * 2;
    }
}

void cosyvoice_llm_kv_cache::load_cache(ggml_backend_t backend, ggml_backend_sched* sched)
{
    cur_len = offloaded_cache->len;
    ggml_reset(offloaded_cache->ctx);
    auto gf = ggml_new_graph_custom(offloaded_cache->ctx, layers * 4, false);
    for (int i = 0; i != layers; ++i)
    {
        auto& offloaded_layer = offloaded_cache->offloaded_kv_layers[i];
        auto& layer = kv_cache_layers[i];
        ggml_tensor* k_view = ggml_view_3d(offloaded_cache->ctx, layer.k, layer.k->ne[0], offloaded_cache->len, layer.k->ne[2], layer.k->nb[1], layer.k->nb[2], 0);
        ggml_tensor* v_view;
        if (fattn)
            v_view = ggml_view_3d(offloaded_cache->ctx, layer.v, layer.v->ne[0], offloaded_cache->len, layer.v->ne[2], layer.v->nb[1], layer.v->nb[2], 0);
        else
            v_view = ggml_view_3d(offloaded_cache->ctx, layer.v, offloaded_cache->len, layer.v->ne[1], layer.v->ne[2], layer.v->nb[1], layer.v->nb[2], 0);

        offloaded_layer.k_view = ggml_new_tensor(offloaded_cache->ctx, GGML_TYPE_F32, 3, k_view->ne);
        offloaded_layer.v_view = ggml_new_tensor(offloaded_cache->ctx, GGML_TYPE_F32, 3, v_view->ne);
        ggml_backend_sched_set_tensor_backend(sched, k_view, backend);
        ggml_backend_sched_set_tensor_backend(sched, v_view, backend);
        ggml_backend_sched_set_tensor_backend(sched, offloaded_layer.k_view, backend);
        ggml_backend_sched_set_tensor_backend(sched, offloaded_layer.v_view, backend);
        ggml_build_forward_expand(gf, ggml_cpy(offloaded_cache->ctx, offloaded_layer.k_view, k_view));
        ggml_build_forward_expand(gf, ggml_cpy(offloaded_cache->ctx, offloaded_layer.v_view, v_view));
    }

    ggml_backend_sched_alloc_graph(sched, gf);
    for (int i = 0; i != layers; ++i)
    {
        auto& offloaded_layer = offloaded_cache->offloaded_kv_layers[i];
        ggml_backend_tensor_set_async(backend, offloaded_layer.k_view, offloaded_layer.k, 0, offloaded_layer.k_view->nb[3]);
        ggml_backend_tensor_set_async(backend, offloaded_layer.v_view, offloaded_layer.v, 0, offloaded_layer.v_view->nb[3]);
    }

    ggml_backend_sched_graph_compute_async(sched, gf);
}

size_t cosyvoice_llm_kv_cache::get_offloaded_cache_size() const
{
    if (offloaded_cache)
        return offloaded_cache->offloaded_kv_layers[0].k_view->nb[3] * 2 * layers;
    else return 0;
}

void cosyvoice_llm_kv_cache::clear_offloaded_cache()
{
    if (offloaded_cache)
    {
        ggml_free(offloaded_cache->ctx);
        free(offloaded_cache->offloaded_kv_layers[0].k);
        free(offloaded_cache);
        offloaded_cache = nullptr;
    }
}

cosyvoice_llm_kv_cache::~cosyvoice_llm_kv_cache()
{
    ggml_free(ctx);
    delete[] kv_cache_layers;
    clear_offloaded_cache();
}

void cosyvoice_llm_kv_cache::update_cache(ggml_context* ctx0, ggml_cgraph* gf, ggml_tensor*& k, ggml_tensor*& v, ggml_tensor* position_ids, int layer_idx)
{
    GGML_ASSERT(ggml_are_same_shape(k, v));

    auto& layer = kv_cache_layers[layer_idx];

    if (fattn)
    {
        layer.k_view = ggml_set_rows(ctx0, layer.k, k, position_ids);
        layer.k_view = ggml_view_3d(ctx0, layer.k_view, k->ne[0], cur_len + position_ids->ne[0], k->ne[2], layer.k_view->nb[1], layer.k_view->nb[2], 0);

        layer.v_view = ggml_set_rows(ctx0, layer.v, v, position_ids);
        layer.v_view = ggml_view_3d(ctx0, layer.v_view, v->ne[0], cur_len + position_ids->ne[0], v->ne[2], layer.v_view->nb[1], layer.v_view->nb[2], 0);
    }
    else
    {
        auto k_view = ggml_view_3d(ctx0, layer.k, k->ne[0], k->ne[1], k->ne[2], layer.k->nb[1], layer.k->nb[2], layer.k->nb[1] * cur_len);
        k_view = ggml_cpy(ctx0, k, k_view);
        layer.k_view = ggml_view_3d(ctx0, layer.k, k->ne[0], cur_len + k->ne[1], k->ne[2], layer.k->nb[1], layer.k->nb[2], 0);
        layer.k_view->src[0] = k_view;

        v = ggml_permute(ctx0, v, 1, 0, 2, 3);
        auto v_view = ggml_view_3d(ctx0, layer.v, v->ne[0], v->ne[1], v->ne[2], layer.v->nb[1], layer.v->nb[2], layer.v->nb[0] * cur_len);
        v_view = ggml_cpy(ctx0, v, v_view);
        layer.v_view = ggml_view_3d(ctx0, layer.v, cur_len + v->ne[0], v->ne[1], v->ne[2], layer.v->nb[1], layer.v->nb[2], 0);
        layer.v_view->src[0] = v_view;
    }

    k = layer.k_view;
    ggml_build_forward_expand(gf, k);

    v = layer.v_view;
    ggml_build_forward_expand(gf, v);
}

ggml_tensor* cosyvoice_llm_kv_cache::attention_forward(ggml_context* ctx0, ggml_cgraph* gf, ggml_tensor* query_states, ggml_tensor* key_states, ggml_tensor* value_states, ggml_tensor* attention_mask, int layer_idx) const
{
    if (fattn)
        return ggml_flash_attn_ext(ctx0, query_states, key_states, value_states, attention_mask, 1.f / std::sqrt(static_cast<float>(key_states->ne[0])), 0.f, 0.f);
    else
    {
        auto attn_scores = ggml_mul_mat(ctx0, key_states, query_states);
        auto attn_weights = ggml_soft_max_ext_inplace(ctx0, attn_scores, attention_mask, 1.f / std::sqrt(static_cast<float>(key_states->ne[0])), 0.f);
        auto attn_output = ggml_mul_mat(ctx0, value_states, attn_weights);
        attn_output = ggml_permute(ctx0, attn_output, 0, 2, 1, 3);
        return ggml_cont(ctx0, attn_output);
    }
}

void cosyvoice_llm_kv_cache::shift_kv_node_pos(uint32_t shift_pos)
{
    GGML_ASSERT(fattn);
    cur_len += shift_pos;

    for (auto& layer : std::span(kv_cache_layers, layers))
    {
        layer.k_view->ne[1] += shift_pos;
        layer.v_view->ne[1] += shift_pos;
    }
}

bool cosyvoice_llm_kv_cache::can_reuse(bool prefill) const
{
    return fattn;
}
