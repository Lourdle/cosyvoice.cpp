#include "cosyvoice-model.h"
#include "cosyvoice-kv-cache.h"

#include <ggml-backend.h>

#include <utility>
#include <span>
#include <vector>

struct kv_cache_layer
{
    ggml_tensor* k;
    ggml_tensor* v;

    ggml_tensor* k_view;
    ggml_tensor* v_view;
};

struct offloaded_kv_cache
{
    ~offloaded_kv_cache() = default;

    uint32_t len;
    std::vector<char> buffer;
    ggml_context* ctx;
    struct offloaded_kv_layer
    {
        ggml_tensor* v_tensor;
        char* k;
        char* v;
    } offloaded_kv_layers[];
};

static constexpr size_t get_offloaded_kv_cache_struct_size(int layers)
{
    return sizeof(offloaded_kv_cache) + sizeof(offloaded_kv_cache::offloaded_kv_layer) * layers;
}

template<typename T>
static constexpr T* advance_ptr(T* ptr, std::ptrdiff_t offset)
{
    return reinterpret_cast<T*>(reinterpret_cast<char*>(ptr) + offset);
}

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
    int n_slots,
    int n_offloaded_kv_slots,
    bool fattn)
{
    cur_len = 0;
    this->layers = layers;
    this->fattn = fattn;
    this->k_type = k_type;
    this->v_type = v_type;
    this->num_heads = num_key_value_heads;
    this->n_slots = n_slots;
    this->n_offloaded_kv_slots = n_offloaded_kv_slots;
    cur_slot_idx = 0;
    kv_cache_layers = new kv_cache_layer[layers * n_slots];

    {
        ggml_init_params params = {
            .mem_size = layers * 2 * n_slots * ggml_tensor_overhead(),
            .no_alloc = true
        };
        ctx = ggml_init(params);
    }

    buffer.reset(initialize_buffer(backend, k_head_dim, v_head_dim, max_seq, batch_size));

    if (n_offloaded_kv_slots > 0)
    {
        ggml_init_params params =
        {
            .mem_size = (ggml_tensor_overhead() + ggml_graph_overhead()) * layers * 4 * n_offloaded_kv_slots,
            .no_alloc = true
        };
        const auto object_size = get_offloaded_kv_cache_struct_size(layers);
        offloaded_cache = reinterpret_cast<offloaded_kv_cache*>(malloc(object_size * n_offloaded_kv_slots));

        for (int i = 0; i != n_offloaded_kv_slots; ++i)
        {
            auto slot = advance_ptr(offloaded_cache, object_size * i);
            new(slot) offloaded_kv_cache();
            slot->len = 0;
            slot->buffer.resize(0);
            slot->ctx = ggml_init(params);
        }
    }
    else
        offloaded_cache = nullptr;
}

ggml_backend_buffer* cosyvoice_kv_cache::initialize_buffer(ggml_backend_t backend, int k_head_dim, int v_head_dim, uint32_t max_seq, int batch_size)
{
    int64_t k_ne[3] = { k_head_dim * num_heads, max_seq, batch_size };
    int64_t v_ne[3] = { v_head_dim * num_heads, max_seq, batch_size };
    if (!fattn) std::swap(v_ne[0], v_ne[1]);
    ggml_reset(ctx);

    for (auto& [k, v, k_view, v_view] : std::span(kv_cache_layers, layers* n_slots))
    {
        k = ggml_new_tensor(ctx, k_type, std::size(k_ne), k_ne);
        v = ggml_new_tensor(ctx, v_type, std::size(v_ne), v_ne);
        k_view = nullptr;
        v_view = nullptr;
    }
    return ggml_backend_alloc_ctx_tensors(ctx, backend);
}

void cosyvoice_kv_cache::reset_buffer(ggml_backend_buffer* buffer)
{
    cur_len = 0;
    auto alignment = ggml_backend_buffer_get_alignment(buffer);
    auto k_ne = *reinterpret_cast<std::array<int64_t, GGML_MAX_DIMS>*>(&kv_cache_layers[0].k->ne);
    auto v_ne = *reinterpret_cast<std::array<int64_t, GGML_MAX_DIMS>*>(&kv_cache_layers[0].v->ne);

    auto current_k_size = get_aligned_size(ggml_backend_buffer_get_alloc_size(buffer, kv_cache_layers[0].k), alignment);
    auto current_v_size = get_aligned_size(ggml_backend_buffer_get_alloc_size(buffer, kv_cache_layers[0].v), alignment);
    GGML_ASSERT(ggml_backend_buffer_get_size(buffer) >= static_cast<size_t>(layers * n_slots) * (current_k_size + current_v_size));

    ggml_reset(ctx);
    auto buffer_base = reinterpret_cast<char*>(ggml_backend_buffer_get_base(buffer));
    for (auto& [k, v, k_view, v_view] : std::span(kv_cache_layers, layers * n_slots))
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
}

bool cosyvoice_kv_cache::offload_cache(ggml_backend_t backend, ggml_backend_sched* sched, uint32_t n_tokens)
{
    const auto batch_size = kv_cache_layers[0].k->ne[2];
    const size_t k_nbytes = kv_cache_layers[0].k->nb[1] * n_tokens;
    const size_t v_nbytes = fattn ? kv_cache_layers[0].v->nb[1] * n_tokens : ggml_row_size(kv_cache_layers[0].v->type, n_tokens) * kv_cache_layers[0].v->ne[1];
    const size_t kv_nbytes = (k_nbytes + v_nbytes) * batch_size;

    ggml_reset(offloaded_cache->ctx);
    offloaded_cache->buffer.resize(kv_nbytes * layers);
    char* buffer_base = offloaded_cache->buffer.data();

    offloaded_cache->len = n_tokens;
    for (int i = 0; i != layers; ++i)
    {
        auto& offloaded_layer = offloaded_cache->offloaded_kv_layers[i];
        auto& layer = kv_cache_layers[i];

        offloaded_layer.k = buffer_base + i * kv_nbytes;
        offloaded_layer.v = offloaded_layer.k + k_nbytes * batch_size;
        for (uint32_t b = 0; b != batch_size; ++b)
        {
            ggml_backend_tensor_get_async(backend, layer.k, offloaded_layer.k + b * k_nbytes, b * layer.k->nb[2], k_nbytes);
            if (fattn)
                ggml_backend_tensor_get_async(backend, layer.v, offloaded_layer.v + b * v_nbytes, b * layer.v->nb[2], v_nbytes);
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

        if (!ggml_backend_sched_alloc_graph(sched, gf)
            || ggml_backend_sched_graph_compute_async(sched, gf) != GGML_STATUS_SUCCESS)
        {
            offloaded_cache->len = 0;
            return false;
        }

        for (auto& layer : std::span(offloaded_cache->offloaded_kv_layers, layers))
            ggml_backend_tensor_get_async(backend, layer.v_tensor, layer.v, 0, v_nbytes * batch_size);
    }

    ggml_backend_sched_synchronize(sched);
    return true;
}

bool cosyvoice_kv_cache::load_cache(ggml_backend_t backend, ggml_backend_sched* sched)
{
    cur_len = offloaded_cache->len;
    const auto batch_size = kv_cache_layers[0].k->ne[2];
    const size_t k_nbytes = kv_cache_layers[0].k->nb[1] * cur_len;
    const size_t v_nbytes = fattn ? kv_cache_layers[0].v->nb[1] * cur_len : ggml_row_size(kv_cache_layers[0].v->type, cur_len) * kv_cache_layers[0].v->ne[1];
    const size_t kv_nbytes = (k_nbytes + v_nbytes) * batch_size;

    if (cur_len == 0) return true;

    if (!fattn)
    {
        ggml_reset(offloaded_cache->ctx);
        auto gf = ggml_new_graph_custom(offloaded_cache->ctx, layers * 4, false);
        for (int i = 0; i != layers; ++i)
        {
            auto& offloaded_layer = offloaded_cache->offloaded_kv_layers[i];
            auto& layer = kv_cache_layers[i + cur_slot_idx * layers];
            ggml_tensor* v = ggml_new_tensor_3d(offloaded_cache->ctx, v_type, cur_len, layer.v->ne[1], batch_size);
            ggml_backend_sched_set_tensor_backend(sched, v, backend);
            offloaded_layer.v_tensor = v;
            ggml_tensor* v_view = ggml_view_3d(offloaded_cache->ctx, layer.v, cur_len, layer.v->ne[1], batch_size, layer.v->nb[1], layer.v->nb[2], 0);
            ggml_build_forward_expand(gf, ggml_cpy(offloaded_cache->ctx, v, v_view));
        }

        if (!ggml_backend_sched_alloc_graph(sched, gf))
        {
            cur_len = 0;
            return false;
        }
        for (auto& layer : std::span(offloaded_cache->offloaded_kv_layers, layers))
            ggml_backend_tensor_set_async(backend, layer.v_tensor, layer.v, 0, v_nbytes * batch_size);

        if (ggml_backend_sched_graph_compute_async(sched, gf) != GGML_STATUS_SUCCESS)
        {
            cur_len = 0;
            return false;
        }
    }

    for (int i = 0; i != layers; ++i)
    {
        auto& offloaded_layer = offloaded_cache->offloaded_kv_layers[i];
        auto& layer = kv_cache_layers[i + cur_slot_idx * layers];
        for (uint32_t b = 0; b != batch_size; ++b)
        {
            ggml_backend_tensor_set_async(backend, layer.k, offloaded_layer.k + b * k_nbytes, b * layer.k->nb[2], k_nbytes);
            if (fattn)
                ggml_backend_tensor_set_async(backend, layer.v, offloaded_layer.v + b * v_nbytes, b * layer.v->nb[2], v_nbytes);
        }
    }

    ggml_backend_sched_synchronize(sched);
    return true;
}

size_t cosyvoice_kv_cache::get_offloaded_cache_size() const
{
    if (offloaded_cache)
    {
        size_t nbytes = 0;
        auto object_size = get_offloaded_kv_cache_struct_size(layers);
        for (int i = 0; i != n_offloaded_kv_slots; ++i)
            nbytes += advance_ptr(offloaded_cache, static_cast<std::ptrdiff_t>(object_size * i))->buffer.capacity();
        return nbytes;
    }
    else return 0;
}

void cosyvoice_kv_cache::clear_offloaded_cache()
{
    if (!offloaded_cache) return;

    auto object_size = get_offloaded_kv_cache_struct_size(layers);
    for (int i = 0; i != n_offloaded_kv_slots; ++i)
    {
        auto cur = advance_ptr(offloaded_cache, static_cast<std::ptrdiff_t>(object_size * i));
        std::vector<char>().swap(cur->buffer);
        cur->len = 0;
    }
}

bool cosyvoice_kv_cache::offload_slot(ggml_backend_t backend, ggml_backend_sched* sched, int offloaded_slot_idx, uint32_t n_tokens)
{
    auto offset = static_cast<std::ptrdiff_t>(get_offloaded_kv_cache_struct_size(layers) * offloaded_slot_idx);
    offloaded_cache = advance_ptr(offloaded_cache, offset);
    const bool ok = offload_cache(backend, sched, n_tokens);
    offloaded_cache = advance_ptr(offloaded_cache, -offset);
    return ok;
}

bool cosyvoice_kv_cache::load_slot(ggml_backend_t backend, ggml_backend_sched* sched, int offloaded_slot_idx)
{
    auto offset = static_cast<std::ptrdiff_t>(get_offloaded_kv_cache_struct_size(layers) * offloaded_slot_idx);
    offloaded_cache = advance_ptr(offloaded_cache, offset);
    const bool ok = load_cache(backend, sched);
    offloaded_cache = advance_ptr(offloaded_cache, -offset);
    return ok;
}

cosyvoice_kv_cache::~cosyvoice_kv_cache()
{
    if (!ctx) return;

    ggml_free(ctx);
    delete[] kv_cache_layers;
    auto object_size = get_offloaded_kv_cache_struct_size(layers);
    for (int i = 0; i != n_offloaded_kv_slots; ++i)
    {
        auto cur = advance_ptr(offloaded_cache, object_size * i);
        ggml_free(cur->ctx);
        cur->~offloaded_kv_cache();
    }
    free(offloaded_cache);
    delete v_idxs_data;
}

void cosyvoice_kv_cache::update_cache(ggml_context* ctx0, ggml_cgraph* gf, ggml_tensor*& k, ggml_tensor*& v, ggml_tensor* position_ids, int layer_idx)
{
    GGML_ASSERT(ggml_are_same_shape(k, v));

    auto& layer = kv_cache_layers[cur_slot_idx * layers + layer_idx];

    k = ggml_reshape_3d(ctx0, k, k->ne[0] * num_heads, k->ne[2], k->ne[3]);
    layer.k_view = ggml_set_rows(ctx0, layer.k, k, position_ids);
    layer.k_view = ggml_view_4d(ctx0, layer.k_view, k->ne[0] / num_heads, num_heads, cur_len + position_ids->ne[0], k->ne[2], layer.k_view->nb[1] / num_heads, layer.k_view->nb[1], layer.k_view->nb[2], 0);
    layer.k_view = ggml_permute(ctx0, layer.k_view, 0, 2, 1, 3);
    if (fattn)
    {
        v = ggml_reshape_3d(ctx0, v, v->ne[0] * num_heads, v->ne[2], v->ne[3]);
        layer.v_view = ggml_set_rows(ctx0, layer.v, v, position_ids);
        layer.v_view = ggml_view_4d(ctx0, layer.v_view, v->ne[0] / num_heads, num_heads, cur_len + position_ids->ne[0], v->ne[2], layer.v_view->nb[1] / num_heads, layer.v_view->nb[1], layer.v_view->nb[2], 0);
        layer.v_view = ggml_permute(ctx0, layer.v_view, 0, 2, 1, 3);
    }
    else
    {
        const auto head_dim = v->ne[0];
        const auto seq_len  = v->ne[2];
        const auto n_batch  = v->ne[3];

        if (layer_idx == 0)
        {
            // Create the index tensor in the (just-reset) graph context and reuse it for all
            // layers. It is recreated for every graph build so its memory can be revalidated;
            // the length matches the flattened source, i.e. (v_head_dim*num_heads)*n_tokens.
            v_idxs = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, head_dim * num_heads * seq_len * n_batch);
        }

        // V is stored transposed, so a row-oriented set_rows cannot be used directly.
        // Flatten both sides to {1, N} columns; each element is then scattered to its
        // absolute position via v_idxs.
        auto v_view = ggml_reshape_2d(ctx0, layer.v, 1, ggml_nelements(layer.v));
        v = ggml_reshape_2d(ctx0, v, 1, ggml_nelements(v));
        v_view = ggml_set_rows(ctx0, v_view, v, v_idxs);

        layer.v_view = ggml_view_4d(ctx0, v_view, cur_len + seq_len, head_dim, num_heads, n_batch, layer.v->nb[1], head_dim * layer.v->nb[1], layer.v->nb[2], 0);
    }

    k = layer.k_view;
    ggml_build_forward_expand(gf, k);

    v = layer.v_view;
    ggml_build_forward_expand(gf, v);
}

void cosyvoice_kv_cache::set_input_v_idxs(ggml_backend_t backend, const int32_t* positions, uint32_t n_tokens, uint32_t n_tokens_per_batch)
{
    if (!v_idxs) return;

    const auto n_embd = kv_cache_layers[0].v->ne[1];
    const auto max_seq = kv_cache_layers[0].v->ne[0];
    if (!v_idxs_data)
        v_idxs_data = new std::vector<int32_t>(n_tokens * n_embd);
    else
        v_idxs_data->resize(n_tokens * n_embd);

    for (uint32_t i = 0; i != n_tokens; ++i)
    {
        const uint32_t batch_idx = i / n_tokens_per_batch;
        const int32_t base = static_cast<int32_t>(batch_idx * max_seq * n_embd) + positions[i % n_tokens_per_batch];
        for (uint32_t j = 0; j != n_embd; ++j)
            (*v_idxs_data)[static_cast<size_t>(i) * n_embd + j] = base + static_cast<int32_t>(j * max_seq);
    }

    ggml_backend_tensor_set_async(backend, v_idxs, v_idxs_data->data(), 0, v_idxs_data->size() * sizeof(int32_t));
}

ggml_tensor* cosyvoice_kv_cache::attention_forward(ggml_context* ctx0, ggml_tensor* query_states, ggml_tensor* key_states, ggml_tensor* value_states, ggml_tensor* attention_mask) const
{
    if (fattn)
    {
        auto attn_output = ggml_flash_attn_ext(ctx0, query_states, key_states, value_states, attention_mask, 1.f / std::sqrt(static_cast<float>(key_states->ne[0])), 0.f, 0.f);
        ggml_prec_set_acc(attn_output, GGML_PREC_F32);
        return attn_output;
    }
    else
    {
        auto attn_scores = ggml_mul_mat(ctx0, key_states, query_states);
        ggml_prec_set_acc(attn_scores, GGML_PREC_F32);
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

    for (auto& layer : std::span(kv_cache_layers + cur_slot_idx * layers, layers))
    {
        layer.k_view->ne[1] += shift_pos;
        layer.v_view->ne[1] += shift_pos;
    }
}

bool cosyvoice_kv_cache::can_reuse() const
{
    return fattn;
}

bool cosyvoice_kv_cache::bind_slot(int slot_idx)
{
    if (slot_idx >= n_slots) return false;
    cur_slot_idx = slot_idx;
    return true;
}

void cosyvoice_kv_cache::slide_kv_slot()
{
    GGML_ASSERT(cur_slot_idx + 1 < n_slots);
    if (fattn)
    {
        auto layer_idx = cur_slot_idx++ * layers;
        const auto end = layer_idx + layers;
        for (int cur = layer_idx; cur != end; ++cur)
        {
            auto& layer = kv_cache_layers[cur];
            auto& next_layer = kv_cache_layers[cur + layers];

            auto k = next_layer.k;
            auto k_view = layer.k_view;
            next_layer.k_view = k_view;
            // permute node
            k_view->data = k->data;
            k_view->view_src = k;
            k_view = k_view->src[0];
            // view node
            k_view->data = k->data;
            k_view->view_src = k;
            k_view = k_view->src[0];
            // set_rows (writer) node
            k_view->data = k->data;
            k_view->src[2] = k_view->view_src = k;

            auto v = next_layer.v;
            auto v_view = layer.v_view;
            next_layer.v_view = v_view;
            // permute node
            v_view->data = v->data;
            v_view->view_src = v;
            v_view = v_view->src[0];
            // view node
            v_view->data = v->data;
            v_view->view_src = v;
            v_view = v_view->src[0];
            // set_rows (writer) node
            v_view->data = v->data;
            v_view->src[2] = v_view->view_src = v;
        }
    }
    else
        GGML_ABORT("slide_kv_slot is not supported for non-flash attention");
}
