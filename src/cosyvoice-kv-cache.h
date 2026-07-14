#pragma once

#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-cpp.h>

// Non-FlashAttention mode currently rebuilds parts of the graph instead of reusing one.
class cosyvoice_kv_cache
{
public:
    cosyvoice_kv_cache() = default;
    ~cosyvoice_kv_cache();

    void build_kv_cache(
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
        int n_kv_slots,
        int n_offloaded_kv_slots,
        bool fattn);

    void update_cache(ggml_context* ctx0, ggml_cgraph* gf, ggml_tensor*& k, ggml_tensor*& v, ggml_tensor* position_ids, int layer_idx);

    ggml_tensor* attention_forward(ggml_context* ctx0, ggml_tensor* query_states, ggml_tensor* key_states, ggml_tensor* value_states, ggml_tensor* attention_mask) const;

    void shift_kv_node_pos(uint32_t shift_pos);
    bool can_reuse() const;

    bool bind_slot(int slot_idx);
    void slide_kv_slot();

    ggml_backend_buffer* initialize_buffer(ggml_backend_t backend, int k_head_dim, int v_head_dim, uint32_t max_seq, int batch_size);
    uint32_t reset_buffer(ggml_backend_buffer* buffer);

    void offload_cache(ggml_backend_t backend, ggml_backend_sched* sched, uint32_t n_tokens);
    void load_cache(ggml_backend_t backend, ggml_backend_sched* sched);
    size_t get_offloaded_cache_size() const;
    void clear_offloaded_cache();

    void offload_slot(ggml_backend_t backend, ggml_backend_sched* sched, int offloaded_slot_idx, uint32_t n_tokens);
    void load_slot(ggml_backend_t backend, ggml_backend_sched* sched, int offloaded_slot_idx);

    uint32_t cur_len;
private:
    int layers;
    int num_heads;
    bool fattn;
    int n_slots;
    int n_offloaded_kv_slots;
    ggml_type k_type;
    ggml_type v_type;
    int cur_slot_idx;
    ggml_context* ctx;
    struct kv_cache_layer* kv_cache_layers;
    struct offloaded_kv_cache* offloaded_cache;
};
