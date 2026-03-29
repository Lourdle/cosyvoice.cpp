#pragma once

#include "cosyvoice-modules.h"
#include "cosyvoice-lowlevel.h"
#include "cosyvoice-interface.h"

#include <ggml-cpp.h>

#include <set>
#include <random>

class cosyvoice_llm_kv_cache;

struct cosyvoice_model : virtual cosyvoice_model_context
{
	cosyvoice_model(ggml_backend_t backend, const cosyvoice_context_params_t& params);
	~cosyvoice_model();

	virtual void load(gguf_loader& loader) = 0;

	uint32_t llm_get_kv_cache_len();
	bool llm_set_kv_cache_len(uint32_t len);

	int llm_sample_token();
	void llm_accept_token(int token);
	void llm_clear_accepted_tokens();
	uint32_t llm_get_n_accepted_tokens();
	const int* llm_get_accepted_tokens();

	ggml_status get_last_status();
	virtual void empty_buffer_cache();

	uint32_t get_sample_rate();

	void set_prompt(
		cosyvoice_prompt_t         prompt,
		cosyvoice_inference_mode_t mode,
		const int*                 instruction,
		uint32_t                   instruction_length
	);

	void get_generation_config(cosyvoice_generation_config_t* config);
	bool set_generation_config(const cosyvoice_generation_config_t* config);
	const char* get_instruction_prefix();
	void get_context_params(cosyvoice_context_params_t* params);

	void get_sampler(cosyvoice_sampler_t* sampler, void** sampler_ctx);
	void set_sampler(cosyvoice_sampler_t sampler, void* sampler_ctx);
	bool using_builtin_sampler() const;
	bool reset_builtin_sampler_rng();
	cosyvoice_builtin_sampler_rng_policy_t get_builtin_sampler_rng_policy();
	bool set_builtin_sampler_rng_policy(cosyvoice_builtin_sampler_rng_policy_t policy);
	bool set_sampler_seed(uint32_t seed);

	void set_noise_callback(cosyvoice_noise_callback_t callback, void* callback_ctx);
	void get_noise_callback(cosyvoice_noise_callback_t* callback, void** callback_ctx);

	ggml_context_ptr ctx;

	ggml_backend_ptr backend;
	ggml_backend_ptr cpu_backend;
	ggml_backend_buffer_ptr buffer;
	ggml_backend_sched_ptr sched;

	ggml_cgraph* gf;
	ggml_context_ptr ctx0;

	ggml_tensor* llm_input;
	ggml_tensor* llm_probs;
	ggml_tensor* position_ids;
	ggml_tensor* causal_mask;

	std::mt19937 sampler_rng, noise_rng;

	std::vector<int> tokens;
	std::unique_ptr<int> full_position_ids;
	std::unique_ptr<ggml_fp16_t[]> causal_mask_buffer;

	cosyvoice_llm_kv_cache* kv_cache;

	cosyvoice_noise_callback_t noise_callback;
	void* noise_callback_ctx;

	cosyvoice_context_params_t params;
	cosyvoice_generation_config_t config;
	uint32_t sample_rate;
	ggml_backend_op_capabilities op_caps;

	ggml_status status;
	uint32_t prompt_crc32;

	uint32_t rand_noise_len;
	std::unique_ptr<float[]> rand_noise;
	std::unique_ptr<char> batch_buffer;
	std::unique_ptr<float[]> nucleus_probs;
	std::unique_ptr<float[]> probs;
	std::unique_ptr<char[]> instruction_prefix;
	ggml_backend_buffer_ptr kv_buffer;
};

struct cosyvoice_model_3 : cosyvoice_model
{
	cosyvoice_model_3(ggml_backend_t backend, const cosyvoice_context_params_t& params);

	CausalMaskedDiffWithDiT flow;
	CausalHiFTGenerator hift;
	CosyVoice3LM llm;

	void load(gguf_loader& loader);

	bool llm_decode(ggml_type type, const void* data);
	bool llm_prefill(ggml_type type, const void* data, uint32_t seq_len);

	bool llm_is_stop_token(int token_id);

	const ggml_tensor* get_word_token_embed_weight();
	const ggml_tensor* get_speech_token_embed_weight();

	uint32_t get_hift_rand_ini_len();
	void set_hift_rand_ini(const float* data);

	bool llm_job(const int* text, uint32_t text_len, cosyvoice_prompt_t prompt);
	bool token2wav(const int* token_ids, uint32_t n_tokens, float speed, cosyvoice_prompt_t prompt, cosyvoice_generated_speech_ptr result);

	void empty_buffer_cache();
	void get_memory_usage(cosyvoice_memory_usage_t* usage);
	void reset_shared_buffer(ggml_backend_buffer* new_buffer);

	ggml_context_ptr ctx1;
	ggml_backend_buffer_ptr token2wav_buffer;

	uint32_t orig_max_seq_len;
	ggml_type kv_type;

	std::set<int> stop_tokens;
};
