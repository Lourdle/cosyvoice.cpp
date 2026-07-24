#pragma once

#if defined(_WIN32) && !defined(COSYVOICE_STATIC)
    #define COSYVOICE_API __declspec(dllexport)
#endif

#include "cosyvoice.h"
#include "cosyvoice-lowlevel.h"
#include "cosyvoice-interface.h"

#include <vector>
#include <utility>
#include <memory>
#include <string>
#include <atomic>

constexpr int kCosyVoiceGraphSize = GGML_DEFAULT_GRAPH_SIZE + 512;
constexpr int kCosyVoiceSchedGraphSize = kCosyVoiceGraphSize + 256;

inline
ggml_cgraph* new_cgraph(ggml_context* ctx)
{
    return ggml_new_graph_custom(ctx, kCosyVoiceGraphSize, false);
}

struct ggml_backend_op_capabilities
{
    bool concat_i32    : 1;
    bool repeat_f16    : 1;
    bool pad           : 1;
    bool pad_reflect_1d: 1;
    bool im2col_f16    : 1;
    bool fill          : 1;
    bool cumsum        : 1;
    bool emb_cast_f32  : 1;
    bool top_k         : 1;
    bool leaky_relu    : 1;
    bool sin           : 1;
    bool cos           : 1;
    bool arange        : 1;
    bool elu           : 1;
    bool abs           : 1;
    bool floor         : 1;
    bool acc           : 1;
};

struct ggml_cgraph_node_iterator
{
    ggml_cgraph_node_iterator(ggml_cgraph* gf) : ggml_cgraph_node_iterator(gf, 0) {}
    ggml_cgraph_node_iterator(ggml_cgraph* gf, int end) : nodes(ggml_graph_nodes(gf)), n_nodes(end > 0 ? end : ggml_graph_n_nodes(gf) + end) {}

    ggml_tensor** begin() const { return nodes; }
    ggml_tensor** end() const { return nodes + n_nodes; }

    ggml_tensor** nodes;
    int n_nodes;
};

struct matrix
{
    matrix() : shape{ 0, 0 }, stride(0), data(nullptr) {}
    matrix(uint32_t dim0, uint32_t dim1) : shape{ dim0, dim1 }, stride(dim1), data(new float[dim0 * dim1]())
    {
        orig_data.reset(data);
    }

    matrix operator[](uint32_t index) const
    {
        matrix result;
        result.shape[0] = 1;
        result.shape[1] = shape[1];
        result.stride = stride;

        result.data = data + index * stride;
        result.orig_data = orig_data;
        return result;
    }

    matrix slice(uint32_t start_dim0, uint32_t end_dim0, uint32_t start_dim1, uint32_t end_dim1) const
    {
        matrix result;
        result.shape[0] = end_dim0 - start_dim0;
        result.shape[1] = end_dim1 - start_dim1;
        result.data = data + start_dim0 * stride + start_dim1;
        result.stride = stride;
        result.orig_data = orig_data;
        return result;
    }

    inline
    float& operator()(uint32_t dim0, uint32_t dim1)
    {
        return data[dim0 * stride + dim1];
    }

    inline
    const float& operator()(uint32_t dim0, uint32_t dim1) const
    {
        return data[dim0 * stride + dim1];
    }

    uint32_t shape[2];
    size_t stride;
    float* data;
    std::shared_ptr<float[]> orig_data;
};

class cosyvoice_object_ref_counter
{
public:
    cosyvoice_object_ref_counter() : ref_count(new std::atomic_uint32_t(1)) {}
    cosyvoice_object_ref_counter(cosyvoice_object_ref_counter& ref_obj) : ref_count(ref_obj.ref_count) { ref_count->fetch_add(1, std::memory_order_relaxed); }
    ~cosyvoice_object_ref_counter() { if (ref_count->fetch_sub(1, std::memory_order_acq_rel) == 1) delete ref_count; }

    uint32_t get_ref_count() const { return ref_count->load(std::memory_order_relaxed); }
private:
    std::atomic_uint32_t* ref_count;
};

using tokens_t = std::pair<std::shared_ptr<int[]>, uint32_t>;
using text_t = std::pair<std::shared_ptr<char[]>, uint32_t>;

struct cosyvoice_prompt_speech
{
    text_t text;
    matrix feat;
    matrix embedding;
    tokens_t tokens;
    uint32_t crc32;

    void calculate_crc32();
};

struct cosyvoice_prompt
{
    std::vector<int> prompt_text;
    tokens_t orig_prompt_text;
    tokens_t llm_prompt_speech_tokens;
    tokens_t flow_prompt_speech_tokens;
    matrix prompt_speech_feat;
    matrix llm_embedding;
    matrix flow_embedding;

    uint32_t prompt_crc32;
    void calculate_crc32();
};

struct cosyvoice_worker_context;
struct cosyvoice_model;

struct use_count_guard
{
    cosyvoice_worker_context* worker;

    use_count_guard(cosyvoice_context* ctx);
    use_count_guard(cosyvoice_model* model);

    ~use_count_guard();

    use_count_guard(const use_count_guard&) = delete;
    use_count_guard& operator=(const use_count_guard&) = delete;
};

struct cosyvoice_tokenization_result_impl : cosyvoice_tokenization_result {
    cosyvoice_tokenization_result_impl() {}

    int* get_tokens() { return tokens.data(); }
    uint32_t get_n_tokens() { return static_cast<uint32_t>(tokens.size()); }

    std::vector<int> tokens;
};

bool cosyvoice_frontend_util_text_normalize(
    std::string& text,
    const char* orig_text,
    uint32_t text_len,
    const char* locale
);

int cosyvoice_llm_sampler(
    cosyvoice_llm_token_prob_t* nucleus_probs,
    int k,
    float* probs,
    uint32_t size,
    const cosyvoice_sampling_params_t* sampling_params,
    int* accepted_tokens,
    uint32_t n_accepted_tokens,
    cosyvoice_worker_context* workers,
    uint32_t worker_no
);

void cosyvoice_call_ggml_log_callback(
    ggml_log_level level,
    const char* message
);
