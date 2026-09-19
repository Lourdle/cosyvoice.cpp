#pragma once

#include "cosyvoice-internal.h"
#include "cosyvoice-kv-cache.h"
#include "ggml-fft.h"

#include <array>
#include <memory>
#include <vector>
#include <string>

struct gguf_loader;

template<bool up = true>
constexpr size_t get_aligned_size(size_t size, size_t alignment)
{
    if constexpr (up)
        return (size + alignment - 1) & ~(alignment - 1);
    else
        return size & ~(alignment - 1);
}

struct BasicModule
{
    void OnLoad(gguf_loader& loader, const std::string& prefix);

    ggml_tensor* weight;
    ggml_tensor* bias;
};

struct Conv1d : BasicModule
{
    ggml_tensor* build_cgraph(ggml_context* ctx, ggml_tensor* x, int s, int p, int d, int g, ggml_backend_op_capabilities capabilities) const;
};

struct Linear : BasicModule
{
    ggml_tensor* build_cgraph(ggml_context* ctx, ggml_tensor* x) const;
};

struct LayerNorm : BasicModule
{
    constexpr static float eps = 1e-6f;

    void OnLoad(gguf_loader& loader, const std::string& prefix);

    ggml_tensor* build_cgraph(ggml_context* ctx, ggml_tensor* x) const;
};


struct CausalConvPositionEmbedding
{
    Conv1d conv1;
    Conv1d conv2;

    void OnLoad(gguf_loader& loader, const std::string& prefix);

    ggml_tensor* build_cgraph(ggml_context* ctx, ggml_tensor* x, ggml_backend_op_capabilities capabilities) const;
};

struct InputEmbedding
{
    Linear proj;

    CausalConvPositionEmbedding conv_pos_embed;

    void OnLoad(gguf_loader& loader, const std::string& prefix);

    ggml_tensor* build_cgraph(ggml_context* ctx, ggml_tensor* x, ggml_tensor* cond, ggml_tensor* text_embed, ggml_tensor* spks, ggml_backend_op_capabilities capabilities) const;
};

struct SinusPositionEmbedding
{
    ggml_tensor* emb = nullptr;

    ggml_tensor* build_cgraph(ggml_context* ctx, ggml_tensor* x) const;
};

struct TimestepEmbedding
{
    SinusPositionEmbedding time_embed;
    Linear time_mlp_0;
    Linear time_mlp_2;

    void OnLoad(gguf_loader& loader, const std::string& prefix);

    ggml_tensor* build_cgraph(ggml_context* ctx, ggml_tensor* t) const;
};

struct AdaLayerNormZero
{
    Linear linear;

    LayerNorm norm;

    void OnLoad(gguf_loader& loader, const std::string& prefix);

    std::array<ggml_tensor*, 5> build_cgraph(ggml_context* ctx, ggml_tensor* x, ggml_tensor* emb) const;
};

struct Attention
{
    int heads;
    bool fattn;

    Linear to_q;
    Linear to_k;
    Linear to_v;
    Linear to_out;

    void OnLoad(gguf_loader& loader, const std::string& prefix);

    ggml_tensor* build_cgraph(ggml_context* ctx, ggml_tensor* x, ggml_tensor* position_ids, int64_t cut_len, cosyvoice_kv_cache* kv_cache, ggml_cgraph* gf, ggml_tensor* attn_mask, int layer_idx) const;
};

struct FeedForward
{
    Linear ff_0_0;
    Linear ff_2;

    void OnLoad(gguf_loader& loader, const std::string& prefix);

    ggml_tensor* build_cgraph(ggml_context* ctx, ggml_tensor* x) const;
};

struct DiTBlock
{
    AdaLayerNormZero attn_norm;
    Attention attn;

    LayerNorm ff_norm;
    FeedForward ff;

    void OnLoad(gguf_loader& loader, const std::string& prefix);

    ggml_tensor* build_cgraph(ggml_context* ctx, ggml_tensor* x, ggml_tensor* time_emb, ggml_tensor* position_ids, int64_t cut_len, cosyvoice_kv_cache* kv_cache, ggml_cgraph* gf, ggml_tensor* attn_mask, int layer_idx) const;
};

struct AdaLayerNorm_Final
{
    Linear linear;

    LayerNorm norm;

    void OnLoad(gguf_loader& loader, const std::string& prefix);

    ggml_tensor* build_cgraph(ggml_context* ctx, ggml_tensor* x, ggml_tensor* emb) const;
};

struct DiT
{
    TimestepEmbedding time_embed;
    InputEmbedding input_embed;

    std::vector<DiTBlock> transformer_blocks;

    AdaLayerNorm_Final norm_out;
    Linear proj_out;

    int mel_dim;
    constexpr static int static_chunk_size = 50;

    void OnLoad(gguf_loader& loader, const std::string& prefix);

    ggml_tensor* build_cgraph(ggml_context* ctx, ggml_tensor* x, ggml_tensor* mu, ggml_tensor* t, ggml_tensor* spks, ggml_tensor* cond, int64_t cut_len, ggml_tensor*& position_ids, ggml_backend_op_capabilities capabilities, cosyvoice_kv_cache* kv_cache, ggml_tensor** ref_attn_mask, ggml_cgraph* gf) const;
};

struct CausalConditionalCFM
{
    constexpr static int MAX_DIFFUSION_STEPS = 50;
    std::vector<float> t_span;
    int diffusion_steps = 10;
    float inference_cfg_rate;

    DiT estimator;

    void OnLoad(gguf_loader& loader, const std::string& prefix);

    struct DiTContext
    {
        ggml_tensor* x;

        ggml_tensor* mu_in;
        ggml_tensor* spks_in;
        ggml_tensor* cond_in;
    };

    DiTContext prepare_context(ggml_context* ctx, ggml_tensor* mu, ggml_tensor* spks, ggml_tensor* cond) const;
    std::array<float, 2> get_t_and_dt(ggml_context* ctx, int step) const;
    ggml_tensor* build_cgraph_one_step(ggml_context* ctx, const DiTContext& ditctx, int step, ggml_backend_op_capabilities capabilities, int64_t cut_len, ggml_tensor*& t_tensor, ggml_tensor*& position_ids, ggml_cgraph* gf, cosyvoice_kv_cache* kv_cache, ggml_tensor** attn_mask) const;
};

struct PreLookaheadLayer
{
    int pre_lookahead_len;

    Conv1d conv1;
    Conv1d conv2;

    void OnLoad(gguf_loader& loader, const std::string& prefix);

    ggml_tensor* build_cgraph(ggml_context* ctx, ggml_tensor* inputs, bool streaming, uint32_t cut_len) const;
};

struct CausalMaskedDiffWithDiT
{
    int token_mel_ratio;

    ggml_tensor* input_embedding;
    Linear spk_embed_affine_layer;
    PreLookaheadLayer pre_lookahead_layer;
    CausalConditionalCFM decoder;

    void OnLoad(gguf_loader& loader, const std::string& prefix);

    struct EncodeResult
    {
        ggml_tensor* mu;
        ggml_tensor* spks;
        ggml_tensor* conds;
        int64_t cut_len;
    };

    EncodeResult build_cgraph_encode(ggml_context* ctx, ggml_tensor* token, ggml_tensor* prompt_token, ggml_tensor* prompt_feat, ggml_tensor* embedding, ggml_backend_op_capabilities capabilities, uint32_t cut_len = 0, bool streaming = false) const;
};

struct CausalConv1dBase : Conv1d
{
    virtual ~CausalConv1dBase() = default;
    virtual ggml_tensor* build_cgraph(ggml_context* ctx, ggml_tensor* x) const = 0;
};

struct CausalConv1d : CausalConv1dBase
{
    enum causal_type_t : char
    {
        left,
        right
    } causal_type;
    int d = 1;

    int causal_padding() const;
    int64_t output_length(int64_t input_length, bool finalize) const;

    ggml_tensor* build_cgraph(ggml_context* ctx, ggml_tensor* x) const { return build_cgraph(ctx, x, true); }
    ggml_tensor* build_cgraph(ggml_context* ctx, ggml_tensor* x, bool finalize) const;
};

struct CausalConvRNNF0Predictor
{
    CausalConvRNNF0Predictor();

    void OnLoad(gguf_loader& loader, const std::string& prefix);

    CausalConv1d condnet_0;
    CausalConv1d condnet_2;
    CausalConv1d condnet_4;
    CausalConv1d condnet_6;
    CausalConv1d condnet_8;

    Linear classifier;

    ggml_tensor* build_cgraph(ggml_context* ctx, ggml_tensor* x, bool finalize) const;
};

struct SineGen2
{
    ggml_tensor* rand_ini;

    std::array<ggml_tensor*, 2> build_cgraph(ggml_context* ctx, ggml_tensor* f0, int harmonic_num, int sampling_rate, int upsample_scale, float sine_amp, int voiced_threshold, float noise_std) const;
};

struct SourceModuleHnNSF
{
    void OnLoad(gguf_loader& loader, const std::string& prefix);

    SineGen2 l_sin_gen;
    Linear l_linear;

    std::array<ggml_tensor*, 2> build_cgraph(ggml_context* ctx, ggml_tensor* x, int harmonic_num, int sampling_rate, int upsample_scale, float sine_amp, int voiced_threshold, float noise_std) const;
};

struct Snake
{
    void OnLoad(gguf_loader& loader, const std::string& prefix);

    static constexpr float no_div_by_zero = 0.000000001f;
    ggml_tensor* alpha;

    ggml_tensor* build_cgraph(ggml_context* ctx, ggml_tensor* x) const;
};

struct ResBlock
{
    void OnLoad(gguf_loader& loader, const std::string& prefix);

    std::vector<std::tuple<Snake, CausalConv1d, Snake, CausalConv1d>> convs;

    ggml_tensor* build_cgraph(ggml_context* ctx, ggml_tensor* x) const;
};

struct CausalConv1dDownSample : CausalConv1dBase
{
    int s;
    ggml_tensor* build_cgraph(ggml_context* ctx, ggml_tensor* x) const;
};

struct CausalConv1dUpsample : CausalConv1dBase
{
    int s = 1;
    ggml_tensor* build_cgraph(ggml_context* ctx, ggml_tensor* x) const;
};

struct CausalHiFTGenerator
{
    void OnLoad(gguf_loader& loader, const std::string& prefix);

    CausalConvRNNF0Predictor f0_predictor;
    SourceModuleHnNSF m_source;
    CausalConv1d conv_pre;
    std::vector<CausalConv1dUpsample> ups;
    std::vector<std::unique_ptr<CausalConv1dBase>> source_downs;
    std::vector<ResBlock> source_resblocks;
    std::vector<ResBlock> resblocks;
    CausalConv1d conv_post;

    std::array<ggml_tensor*, 2> build_cgraph(ggml_context* ctx, ggml_tensor* speech_feat, bool finalize) const;

    int overlap_length() const;
    void set_rand_ini(const float* data) const;

    float lrelu_slope;
    int scale_factor;
    int nb_harmonics;
    int sampling_rate;
    float nsf_alpha;
    int nsf_voiced_threshold;
    float nsf_sigma;
    float audio_limit;
    int nfft;
    int hop_len;
    ggml_tensor* window;
    fft_context_ptr fctx;
    istft_context_ptr ictx;
};

struct Qwen2MLP
{
    void OnLoad(gguf_loader& loader, const std::string& prefix);

    ggml_tensor* build_cgraph(ggml_context* ctx, ggml_tensor* x) const;

    Linear gate_proj;
    Linear up_proj;
    Linear down_proj;
};

struct Qwen2RMSNorm
{
    void OnLoad(gguf_loader& loader, const std::string& prefix);

    ggml_tensor* build_cgraph(ggml_context* ctx, ggml_tensor* hidden_states, float variance_epsilon) const;

    ggml_tensor* weight;
};

struct Qwen2Attention
{
    void OnLoad(gguf_loader& loader, const std::string& prefix);

    Linear q_proj;
    Linear k_proj;
    Linear v_proj;
    Linear o_proj;
};

struct Qwen2DecoderLayer
{
    void OnLoad(gguf_loader& loader, const std::string& prefix);

    Qwen2Attention self_attn;
    Qwen2MLP mlp;
    Qwen2RMSNorm input_layernorm;
    Qwen2RMSNorm post_attention_layernorm;
};

struct CosyVoice3LM
{
    void OnLoad(gguf_loader& loader, const std::string& prefix);

    ggml_tensor* embed_tokens_weight;
    ggml_tensor* speech_embedding_weight;
    std::vector<Qwen2DecoderLayer> layers;
    Qwen2RMSNorm norm;
    Linear llm_decoder;

    int num_attention_heads;
    int num_key_value_heads;
    float rms_norm_eps;
    float rope_theta;

    int sos_token_id;
    int task_token_id;
};
