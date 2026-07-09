#include "cosyvoice-model.h"
#include "cosyvoice-kv-cache.h"

#include <ggml-backend.h>

#include <utility>
#include <span>

void cosyvoice_kv_cache::build_kv_cache(
    ggml_backend_t backend,
    ggml_backend_buffer_ptr& buffer,
    int layers,
    int k_head_dim,
    int v_head_dim,
    int num_key_value_heads,
    uint32_t max_seq,
    ggml_type k_type,
    ggml_type v_type,
    int batch_size,
    bool fattn)
{
    cur_len = 0;
    this->layers = layers;
    this->fattn = fattn;
    this->k_type = k_type;
    this->v_type = v_type;
    offloaded_cache = nullptr;
    this->num_heads = num_key_value_heads;
    kv_cache_layers = new kv_cache_layer[layers];

    ggml_init_params params = {
        .mem_size = layers * 2 * ggml_tensor_overhead(),
        .no_alloc = true
    };
    ctx = ggml_init(params);

    buffer.reset(initialize_buffer(backend, k_head_dim, v_head_dim, num_key_value_heads, max_seq, k_type, v_type, batch_size, fattn));
}

ggml_backend_buffer* cosyvoice_kv_cache::initialize_buffer(ggml_backend_t backend, int k_head_dim, int v_head_dim, int num_key_value_heads, uint32_t max_seq, ggml_type k_type, ggml_type v_type, int batch_size, bool fattn)
{
    int64_t k_ne[4] = { k_head_dim, max_seq, num_key_value_heads, batch_size };
    int64_t v_ne[4] = { v_head_dim, max_seq, num_key_value_heads, batch_size };
    if (!fattn) std::swap(v_ne[0], v_ne[1]);
    ggml_reset(ctx);

    for (auto& [k, v, k_view, v_view] : std::span(kv_cache_layers, layers))
    {
        k = ggml_new_tensor(ctx, k_type, 4, k_ne);
        v = ggml_new_tensor(ctx, v_type, 4, v_ne);
        k_view = nullptr;
        v_view = nullptr;
    }
    return ggml_backend_alloc_ctx_tensors(ctx, backend);
}

uint32_t cosyvoice_kv_cache::reset_buffer(ggml_backend_buffer* buffer)
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
    size_t buffer_size;
    ggml_context* ctx;
    struct offloaded_kv_layer
    {
        ggml_tensor* v_tensor;
        char* k;
        char* v;
    } offloaded_kv_layers[];
};
#pragma pack(pop)

void cosyvoice_kv_cache::offload_cache(ggml_backend_t backend, ggml_backend_sched* sched, uint32_t n_tokens)
{
    char* buffer_base;
    const auto batch_size = kv_cache_layers[0].k->ne[3];
    const size_t k_head_nbytes = kv_cache_layers[0].k->nb[1] * n_tokens;
    const size_t v_head_nbytes = fattn ? kv_cache_layers[0].v->nb[1] * n_tokens : ggml_row_size(kv_cache_layers[0].v->type, n_tokens) * kv_cache_layers[0].v->ne[1];
    const size_t kv_nbytes = (k_head_nbytes + v_head_nbytes) * num_heads * batch_size;
    if (!offloaded_cache)
    {
        ggml_init_params params =
        {
            .mem_size = (ggml_tensor_overhead() + ggml_graph_overhead()) * layers * 4,
            .no_alloc = true
        };
        offloaded_cache = reinterpret_cast<offloaded_kv_cache*>(malloc(sizeof(offloaded_kv_cache::offloaded_kv_layer) * layers + sizeof(uint32_t) + sizeof(ggml_context*) + sizeof(size_t)));
        offloaded_cache->ctx = ggml_init(params);
        offloaded_cache->buffer_size = 0;
        goto alloc_buffer;
    }
    else
    {
        ggml_reset(offloaded_cache->ctx);
        if (offloaded_cache->buffer_size < kv_nbytes * layers)
        {
            free(offloaded_cache->offloaded_kv_layers[0].k);
        alloc_buffer:
            offloaded_cache->buffer_size = kv_nbytes * layers;
            buffer_base = reinterpret_cast<char*>(malloc(offloaded_cache->buffer_size));
        }
        else buffer_base = reinterpret_cast<char*>(offloaded_cache->offloaded_kv_layers[0].k);
    }

    offloaded_cache->len = n_tokens;
    const auto total_heads = num_heads * batch_size;
    for (int i = 0; i != layers; ++i)
    {
        auto& offloaded_layer = offloaded_cache->offloaded_kv_layers[i];
        auto& layer = kv_cache_layers[i];

        offloaded_layer.k = buffer_base + i * kv_nbytes;
        offloaded_layer.v = offloaded_layer.k + k_head_nbytes * total_heads;
        for (int h = 0; h != total_heads; ++h)
        {
            ggml_backend_tensor_get_async(backend, layer.k, offloaded_layer.k + h * k_head_nbytes, h * layer.k->nb[2], k_head_nbytes);
            if (fattn)
                ggml_backend_tensor_get_async(backend, layer.v, offloaded_layer.v + h * v_head_nbytes, h * layer.v->nb[2], v_head_nbytes);
        }
    }

    if (!fattn)
    {
        ggml_reset(offloaded_cache->ctx);
        auto gf = ggml_new_graph_custom(offloaded_cache->ctx, layers * 3, false);
        for (int i = 0; i != layers; ++i)
        {
            auto& offloaded_layer = offloaded_cache->offloaded_kv_layers[i];
            auto& layer = kv_cache_layers[i];
            auto v = layer.v;
            v = ggml_view_4d(offloaded_cache->ctx, v, n_tokens, v->ne[1], v->ne[2], v->ne[3], v->nb[1], v->nb[2], v->nb[3], 0);
            v = ggml_cont(offloaded_cache->ctx, v);
            ggml_backend_sched_set_tensor_backend(sched, v, backend);
            ggml_build_forward_expand(gf, v);
            offloaded_layer.v_tensor = v;
        }

        ggml_backend_sched_alloc_graph(sched, gf);
        ggml_backend_sched_graph_compute_async(sched, gf);

        auto v_nbytes = v_head_nbytes * total_heads;
        for (auto& layer : std::span(offloaded_cache->offloaded_kv_layers, layers))
            ggml_backend_tensor_get_async(backend, layer.v_tensor, layer.v, 0, v_nbytes);
    }

    cur_len = 0;
    ggml_backend_sched_synchronize(sched);
}

void cosyvoice_kv_cache::load_cache(ggml_backend_t backend, ggml_backend_sched* sched)
{
    const auto batch_size = kv_cache_layers[0].k->ne[3];
    const auto total_heads = num_heads * batch_size;
    const size_t k_head_nbytes = kv_cache_layers[0].k->nb[1] * offloaded_cache->len;
    const size_t v_head_nbytes = fattn ? kv_cache_layers[0].v->nb[1] * offloaded_cache->len : ggml_row_size(kv_cache_layers[0].v->type, offloaded_cache->len) * kv_cache_layers[0].v->ne[1];
    cur_len = offloaded_cache->len;

    if (!fattn)
    {
        ggml_reset(offloaded_cache->ctx);
        auto gf = ggml_new_graph_custom(offloaded_cache->ctx, layers * 4, false);
        for (int i = 0; i != layers; ++i)
        {
            auto& offloaded_layer = offloaded_cache->offloaded_kv_layers[i];
            auto& layer = kv_cache_layers[i];
            ggml_tensor* v = ggml_new_tensor_4d(offloaded_cache->ctx, v_type, cur_len, layer.v->ne[1], num_heads, batch_size);
            ggml_backend_sched_set_tensor_backend(sched, v, backend);
            offloaded_layer.v_tensor = v;
            ggml_tensor* v_view = ggml_view_4d(offloaded_cache->ctx, layer.v, cur_len, layer.v->ne[1], num_heads, batch_size, layer.v->nb[1], layer.v->nb[2], layer.v->nb[3], 0);
            ggml_build_forward_expand(gf, ggml_cpy(offloaded_cache->ctx, v, v_view));
        }

        ggml_backend_sched_alloc_graph(sched, gf);
        const auto v_nbytes = v_head_nbytes * total_heads;
        for (auto& layer : std::span(offloaded_cache->offloaded_kv_layers, layers))
            ggml_backend_tensor_set_async(backend, layer.v_tensor, layer.v, 0, v_nbytes);

        ggml_backend_sched_graph_compute_async(sched, gf);
    }

    for (int i = 0; i != layers; ++i)
    {
        auto& offloaded_layer = offloaded_cache->offloaded_kv_layers[i];
        auto& layer = kv_cache_layers[i];
        for (int h = 0; h != total_heads; ++h)
        {
            ggml_backend_tensor_set_async(backend, layer.k, offloaded_layer.k + h * k_head_nbytes, h * layer.k->nb[2], k_head_nbytes);
            if (fattn)
                ggml_backend_tensor_set_async(backend, layer.v, offloaded_layer.v + h * v_head_nbytes, h * layer.v->nb[2], v_head_nbytes);
        }
    }

    offloaded_cache->len = 0;
    ggml_backend_sched_synchronize(sched);
}

size_t cosyvoice_kv_cache::get_offloaded_cache_size() const
{
    if (offloaded_cache)
        return offloaded_cache->buffer_size;
    else return 0;
}

void cosyvoice_kv_cache::clear_offloaded_cache()
{
    if (offloaded_cache)
    {
        ggml_free(offloaded_cache->ctx);
        free(offloaded_cache->offloaded_kv_layers[0].k);
        free(offloaded_cache);
        offloaded_cache = nullptr;
    }
}

cosyvoice_kv_cache::~cosyvoice_kv_cache()
{
    ggml_free(ctx);
    delete[] kv_cache_layers;
    clear_offloaded_cache();
}

void cosyvoice_kv_cache::update_cache(ggml_context* ctx0, ggml_cgraph* gf, ggml_tensor*& k, ggml_tensor*& v, ggml_tensor* position_ids, int layer_idx)
{
    GGML_ASSERT(ggml_are_same_shape(k, v));

    auto& layer = kv_cache_layers[layer_idx];

    if (fattn)
    {
        layer.k_view = ggml_set_rows(ctx0, layer.k, k, position_ids);
        layer.k_view = ggml_view_4d(ctx0, layer.k_view, k->ne[0], cur_len + position_ids->ne[0], k->ne[2], k->ne[3], layer.k_view->nb[1], layer.k_view->nb[2], layer.k_view->nb[3], 0);

        layer.v_view = ggml_set_rows(ctx0, layer.v, v, position_ids);
        layer.v_view = ggml_view_4d(ctx0, layer.v_view, v->ne[0], cur_len + position_ids->ne[0], v->ne[2], v->ne[3], layer.v_view->nb[1], layer.v_view->nb[2], layer.v_view->nb[3], 0);
    }
    else
    {
        auto k_view = ggml_view_4d(ctx0, layer.k, k->ne[0], k->ne[1], k->ne[2], k->ne[3], layer.k->nb[1], layer.k->nb[2], layer.k->nb[3], layer.k->nb[1] * cur_len);
        k_view = ggml_cpy(ctx0, k, k_view);
        layer.k_view = ggml_view_4d(ctx0, layer.k, k->ne[0], cur_len + k->ne[1], k->ne[2], k->ne[3], layer.k->nb[1], layer.k->nb[2], layer.k->nb[3], 0);
        layer.k_view->src[0] = k_view;

        v = ggml_permute(ctx0, v, 1, 0, 2, 3);
        auto v_view = ggml_view_4d(ctx0, layer.v, v->ne[0], v->ne[1], v->ne[2], v->ne[3], layer.v->nb[1], layer.v->nb[2], layer.v->nb[3], layer.v->nb[0] * cur_len);
        v_view = ggml_cpy(ctx0, v, v_view);
        layer.v_view = ggml_view_4d(ctx0, layer.v, cur_len + v->ne[0], v->ne[1], v->ne[2], v->ne[3], layer.v->nb[1], layer.v->nb[2], layer.v->nb[3], 0);
        layer.v_view->src[0] = v_view;
    }

    k = layer.k_view;
    ggml_build_forward_expand(gf, k);

    v = layer.v_view;
    ggml_build_forward_expand(gf, v);
}

ggml_tensor* cosyvoice_kv_cache::attention_forward(ggml_context* ctx0, ggml_tensor* query_states, ggml_tensor* key_states, ggml_tensor* value_states, ggml_tensor* attention_mask) const
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

void cosyvoice_kv_cache::shift_kv_node_pos(uint32_t shift_pos)
{
    GGML_ASSERT(fattn);
    cur_len += shift_pos;

    for (auto& layer : std::span(kv_cache_layers, layers))
    {
        layer.k_view->ne[1] += shift_pos;
        layer.v_view->ne[1] += shift_pos;
    }
}

bool cosyvoice_kv_cache::can_reuse(bool prefill) const
{
    if (!fattn) return false;
    for (auto& layer : std::span(kv_cache_layers, layers))
        if (!layer.k_view || !layer.v_view)
            return false;
    return true;
}

void cosyvoice_kv_cache::slide_kv_layers(int layer_idx, int stride)
{
    GGML_ASSERT(layer_idx + stride <= layers);
    if (fattn)
    {
        const auto end = layer_idx + stride;
        for (int cur = layer_idx; cur != end; ++cur)
        {
            auto& layer = kv_cache_layers[cur];
            auto& next_layer = kv_cache_layers[cur + stride];

            auto k = next_layer.k;
            auto k_view = layer.k_view;
            next_layer.k_view = k_view;
            k_view->data = k->data;
            k_view->view_src = k;
            k_view = k_view->src[0];
            k_view->data = k->data;
            k_view->src[2] = k_view->view_src = k;

            auto v = next_layer.v;
            auto v_view = layer.v_view;
            next_layer.v_view = v_view;
            v_view->data = v->data;
            v_view->view_src = v;
            v_view = v_view->src[0];
            v_view->data = v->data;
            v_view->src[2] = v_view->view_src = v;
        }
    }
    else
        GGML_ABORT("slide_kv_layers is not supported for non-flash attention");
}
