#pragma once

#include <ggml.h>
#include <ggml-alloc.h>

// Non-FlashAttention mode currently rebuilds parts of the graph instead of reusing one.
class cosyvoice_llm_kv_cache
{
public:
	cosyvoice_llm_kv_cache(
		ggml_backend_t backend,
		ggml_backend_buffer_ptr& shared_buffer,
		int layers,
		int k_head_dim,
		int v_head_dim,
		int num_attention_heads,
		int num_key_value_heads,
		uint32_t max_seq,
		ggml_type kv_type,
		bool fattn);
	~cosyvoice_llm_kv_cache();

	struct kv_cache_layer
	{
		ggml_tensor* k;
		ggml_tensor* v;

		ggml_tensor* k_view;
		ggml_tensor* v_view;
	};

	void update_cache(ggml_context* ctx0, ggml_cgraph* gf, ggml_tensor*& k, ggml_tensor*& v, ggml_tensor* position_ids, int layer_idx);

	ggml_tensor* attention_forward(ggml_context* ctx0, ggml_cgraph* gf, ggml_tensor* query_states, ggml_tensor* key_states, ggml_tensor* value_states, ggml_tensor* attention_mask, int layer_idx) const;

	void shift_kv_node_pos(uint32_t shift_pos);

	bool can_reuse(bool prefill) const;

	ggml_backend_buffer* initialize_buffer(ggml_backend_t backend, int k_head_dim, int v_head_dim, int num_attention_heads, int num_key_value_heads, uint32_t max_seq, ggml_type kv_type, bool fattn);
	uint32_t reset_buffer(ggml_backend_buffer* buffer);

	void offload_cache(ggml_backend_t backend, ggml_backend_sched* sched, uint32_t n_tokens);
	void load_cache(ggml_backend_t backend, ggml_backend_sched* sched);
	size_t get_offloaded_cache_size() const;
	void clear_offloaded_cache();

	uint32_t cur_len;
private:
	int layers;
	int num_attention_heads;
	bool fattn;
	ggml_context* ctx;
	kv_cache_layer* kv_cache_layers;
	struct offloaded_kv_cache* offloaded_cache;
};
