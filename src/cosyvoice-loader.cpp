#include "cosyvoice-model.h"
#include "cosyvoice-loader.h"
#include "cosyvoice-kv-cache.h"

#include <tuple>
#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <format>
#include <span>
#include <ranges>

namespace
{
#if defined(__APPLE__) && defined(__aarch64__)
constexpr bool backend_looks_uma(ggml_backend_t backend, ggml_backend_buffer* buffer) { return true; }
#else
constexpr size_t kUmaProbeBytes = 128ull * 1024ull * 1024ull;
constexpr size_t kUmaProbeMinBytes = 64ull * 1024ull * 1024ull;
constexpr int kUmaProbeIters = 4;

struct transfer_fit
{
    double bandwidth_bytes_per_us = 0.0;
    double overhead_us = 0.0;
};

static transfer_fit fit_transfer(size_t size_1, double time_1_us, size_t size_2, double time_2_us)
{
    transfer_fit result;
    if (size_1 <= size_2 || time_1_us <= time_2_us)
        return result;

    const auto bw = static_cast<double>(size_1 - size_2) / (time_1_us - time_2_us);
    if (bw <= 0.0)
        return result;

    result.bandwidth_bytes_per_us = bw;
    result.overhead_us = time_1_us - static_cast<double>(size_1) / bw;
    if (result.overhead_us < 0.0)
        result.overhead_us = 0.0;
    return result;
}

template <typename F>
static double measure_min_us(F&& fn)
{
    double min_us = (std::numeric_limits<double>::max)();
    for (int i = 0; i != kUmaProbeIters; ++i) {
        const auto start = ggml_time_us();
        fn();
        const auto elapsed = ggml_time_us() - start;
        if (elapsed > 0 && static_cast<double>(elapsed) < min_us)
            min_us = static_cast<double>(elapsed);
    }
    return min_us < (std::numeric_limits<double>::max)() ? min_us : 0.0;
}

bool backend_looks_uma(ggml_backend_t backend, ggml_backend_buffer* buffer)
{
    ggml_backend_dev_props props;
    ggml_backend_dev_get_props(ggml_backend_get_device(backend), &props);
    if (props.type == GGML_BACKEND_DEVICE_TYPE_IGPU || props.type == GGML_BACKEND_DEVICE_TYPE_CPU)
        return true;
    if (strncmp(props.name, "Vulkan", 6) == 0 && props.type == GGML_BACKEND_DEVICE_TYPE_GPU)
        return false;

    constexpr auto size_1 = kUmaProbeBytes - (kUmaProbeBytes % sizeof(float));
    constexpr auto size_2 = kUmaProbeMinBytes - (kUmaProbeMinBytes % sizeof(float));
    if constexpr (size_1 == 0 || size_2 == 0 || size_1 <= size_2)
        return false;

    if (!backend || !buffer)
        return false;

    const auto buffer_size = ggml_backend_buffer_get_size(buffer);
    if (buffer_size < kUmaProbeBytes)
        return false;

    ggml_tensor tensor =
    {
        .type = GGML_TYPE_I8,
        .ne = { static_cast<int64_t>(size_1), 1, 1, 1 },
        .nb = { 1, size_1, size_1, size_1 }
    };
    ggml_backend_tensor_alloc(buffer, &tensor, ggml_backend_buffer_get_base(buffer));
    auto host_src = std::make_unique<char[]>(size_1 * 2);

    auto* src = host_src.get();
    auto* dst = src + size_1;

    memcpy(dst, src, size_1);

    const auto memcpy_1_us = measure_min_us([&] { memcpy(dst, src, size_1); });
    const auto memcpy_2_us = measure_min_us([&] { memcpy(dst, src, size_2); });

    ggml_backend_tensor_set(&tensor, src, 0, size_1);
    const auto backend_1_us = measure_min_us([&] { ggml_backend_tensor_set(&tensor, src, 0, size_1); });

    ggml_backend_tensor_set(&tensor, src, 0, size_2);
    const auto backend_2_us = measure_min_us([&] { ggml_backend_tensor_set(&tensor, src, 0, size_2); });

    if (memcpy_1_us <= 0.0 || memcpy_2_us <= 0.0 || backend_1_us <= 0.0 || backend_2_us <= 0.0)
        return false;

    const auto memcpy_fit = fit_transfer(size_1, memcpy_1_us, size_2, memcpy_2_us);
    const auto backend_fit = fit_transfer(size_1, backend_1_us, size_2, backend_2_us);
    if (memcpy_fit.bandwidth_bytes_per_us <= 0.0 || backend_fit.bandwidth_bytes_per_us <= 0.0)
        return false;

    constexpr double mib_per_us_to_mib_per_s = 1000000.0 / (1024.0 * 1024.0);
    const auto memcpy_mib_s = memcpy_fit.bandwidth_bytes_per_us * mib_per_us_to_mib_per_s;
    const auto backend_mib_s = backend_fit.bandwidth_bytes_per_us * mib_per_us_to_mib_per_s;
    const auto uma = backend_fit.bandwidth_bytes_per_us >= memcpy_fit.bandwidth_bytes_per_us * 0.7;
    cosyvoice_call_ggml_log_callback(
        GGML_LOG_LEVEL_INFO,
        std::format(
            "UMA probe: memcpy={:.1f} MiB/s (overhead {:.1f} us), backend={:.1f} MiB/s (overhead {:.1f} us), guess: UMA={}\n",
            memcpy_mib_s,
            memcpy_fit.overhead_us,
            backend_mib_s,
            backend_fit.overhead_us,
            uma
        ).c_str()
    );

    return uma;
}
#endif
}

#define LOAD_SUBMODULE_EX(name, module) do {\
    auto& _module = module;\
    _module.OnLoad(loader, combine_prefix(prefix, name));\
} while (false)
#define LOAD_SUBMODULE(name) LOAD_SUBMODULE_EX(#name, name)

#define LOAD_TENSOR_EX(name, obj) do {\
    this->obj = loader.get_gguf_tensor(prefix, name);\
    loader.register_tensor(prefix, name, &this->obj);\
} while (false)
#define LOAD_TENSOR(name) LOAD_TENSOR_EX(#name, name)

#define LOAD_OPTIONAL_TENSOR_EX(name, obj) do {\
    this->obj = loader.get_gguf_tensor(prefix, name, true);\
    if (this->obj)\
        loader.register_tensor(prefix, name, &this->obj);\
} while (false)
#define LOAD_OPTIONAL_TENSOR(name) LOAD_OPTIONAL_TENSOR_EX(#name, name)

#define LOAD_METADATA(name) GGML_ASSERT(loader.get_metadata(prefix, #name, name))
#define LOAD_METADATA_NOPREFIX(name) GGML_ASSERT(loader.get_metadata(#name, name))

constexpr uint32_t GGML_TENSOR_FLAG_NEED_EPS_ADJUST = 28u << 0;

static inline void set_tensor_eps(ggml_tensor* tensor, float eps)
{
    GGML_ASSERT(tensor && tensor->type == GGML_TYPE_F32);

    tensor->flags |= GGML_TENSOR_FLAG_NEED_EPS_ADJUST;
    *reinterpret_cast<float*>(tensor->op_params) = eps;
}

static inline float get_tensor_eps(const ggml_tensor* tensor)
{
    return (tensor->flags & GGML_TENSOR_FLAG_NEED_EPS_ADJUST)
        ? *reinterpret_cast<const float*>(tensor->op_params)
        : 0.0f;
}

void BasicModule::OnLoad(gguf_loader& loader, const std::string& prefix)
{
    LOAD_TENSOR(weight);
    LOAD_OPTIONAL_TENSOR(bias);
}

void LayerNorm::OnLoad(gguf_loader& loader, const std::string& prefix)
{
    LOAD_OPTIONAL_TENSOR(weight);
    LOAD_OPTIONAL_TENSOR(bias);
}

void CausalConvPositionEmbedding::OnLoad(gguf_loader& loader, const std::string& prefix)
{
    LOAD_SUBMODULE_EX("conv1.0", conv1);
    LOAD_SUBMODULE_EX("conv2.0", conv2);
}

void InputEmbedding::OnLoad(gguf_loader& loader, const std::string& prefix)
{
    LOAD_SUBMODULE(proj);
    LOAD_SUBMODULE(conv_pos_embed);
}

void TimestepEmbedding::OnLoad(gguf_loader& loader, const std::string& prefix)
{
    LOAD_SUBMODULE_EX("time_mlp.0", time_mlp_0);
    LOAD_SUBMODULE_EX("time_mlp.2", time_mlp_2);
}

void AdaLayerNormZero::OnLoad(gguf_loader& loader, const std::string& prefix)
{
    LOAD_SUBMODULE(linear);
    LOAD_SUBMODULE(norm);
}

void Attention::OnLoad(gguf_loader& loader, const std::string& prefix)
{
    LOAD_SUBMODULE(to_q);
    LOAD_SUBMODULE(to_k);
    LOAD_SUBMODULE(to_v);
    LOAD_SUBMODULE_EX("to_out.0", to_out);
}

void FeedForward::OnLoad(gguf_loader& loader, const std::string& prefix)
{
    LOAD_SUBMODULE_EX("ff.0.0", ff_0_0);
    LOAD_SUBMODULE_EX("ff.2", ff_2);
}

void DiTBlock::OnLoad(gguf_loader& loader, const std::string& prefix)
{
    LOAD_SUBMODULE(attn_norm);
    LOAD_SUBMODULE(attn);
    LOAD_SUBMODULE(ff_norm);
    LOAD_SUBMODULE(ff);
}

void AdaLayerNorm_Final::OnLoad(gguf_loader& loader, const std::string& prefix)
{
    LOAD_SUBMODULE(linear);
    LOAD_SUBMODULE(norm);
}

void DiT::OnLoad(gguf_loader& loader, const std::string& prefix)
{
    LOAD_SUBMODULE(time_embed);
    LOAD_SUBMODULE(input_embed);

    int heads;
    int depth;
    LOAD_METADATA(heads);
    LOAD_METADATA(depth);
    LOAD_METADATA(mel_dim);

    transformer_blocks.resize(depth);
    for (int i = 0; i != depth; ++i)
    {
        auto& block = transformer_blocks[i];
        auto name = std::format("{}.transformer_blocks.{}", prefix, i);
        block.OnLoad(loader, name);
        block.attn.heads = heads;
    }

    LOAD_SUBMODULE(norm_out);
    LOAD_SUBMODULE(proj_out);
}

void CausalConditionalCFM::OnLoad(gguf_loader& loader, const std::string& prefix)
{
    LOAD_SUBMODULE(estimator);

    LOAD_METADATA(inference_cfg_rate);

    t_span.resize(diffusion_steps + 1);
    for (int i = 0; i <= diffusion_steps; ++i)
        t_span[i] = 1.f - std::cos(0.5f * 3.14159265358979323846f * i / diffusion_steps);
}

void PreLookaheadLayer::OnLoad(gguf_loader& loader, const std::string& prefix)
{
    LOAD_METADATA(pre_lookahead_len);

    LOAD_SUBMODULE(conv1);
    LOAD_SUBMODULE(conv2);
}

void CausalMaskedDiffWithDiT::OnLoad(gguf_loader& loader, const std::string& prefix)
{
    LOAD_METADATA(token_mel_ratio);

    LOAD_TENSOR_EX("input_embedding.weight", input_embedding);
    LOAD_SUBMODULE(spk_embed_affine_layer);
    LOAD_SUBMODULE(pre_lookahead_layer);
    LOAD_SUBMODULE(decoder);
}

void CausalConvRNNF0Predictor::OnLoad(gguf_loader& loader, const std::string& prefix)
{
    LOAD_SUBMODULE_EX("condnet.0", condnet_0);
    LOAD_SUBMODULE_EX("condnet.2", condnet_2);
    LOAD_SUBMODULE_EX("condnet.4", condnet_4);
    LOAD_SUBMODULE_EX("condnet.6", condnet_6);
    LOAD_SUBMODULE_EX("condnet.8", condnet_8);
    LOAD_SUBMODULE(classifier);
}

void SourceModuleHnNSF::OnLoad(gguf_loader& loader, const std::string& prefix)
{
    LOAD_SUBMODULE(l_linear);
}

void Snake::OnLoad(gguf_loader& loader, const std::string& prefix)
{
    LOAD_TENSOR(alpha);

    set_tensor_eps(alpha, 1e-6f);
}

void ResBlock::OnLoad(gguf_loader& loader, const std::string& prefix)
{
    int64_t id;
    GGML_ASSERT(loader.find_metadata_key(combine_prefix(prefix, "dilations").c_str(), id));
    GGML_ASSERT(loader.parser.arr_type(id) == GGUF_TYPE_INT32);

    const auto n_dilations = loader.parser.arr_n(id);
    const int* dilations = reinterpret_cast<const int*>(loader.parser.arr_data(id));
    convs.resize(n_dilations);
    for (size_t i = 0; i != n_dilations; ++i)
    {
        auto& [snk1, conv1, snk2, conv2] = convs[i];
        conv1.causal_type = CausalConv1d::causal_type_t::left;
        conv1.d = dilations[i];
        conv2.causal_type = CausalConv1d::causal_type_t::left;
        conv2.d = 1;

        LOAD_SUBMODULE_EX(std::format("convs1.{}", i).c_str(), conv1);
        LOAD_SUBMODULE_EX(std::format("convs2.{}", i).c_str(), conv2);
        LOAD_SUBMODULE_EX(std::format("activations1.{}", i).c_str(), snk1);
        LOAD_SUBMODULE_EX(std::format("activations2.{}", i).c_str(), snk2);
    }
}

void CausalHiFTGenerator::OnLoad(gguf_loader& loader, const std::string& prefix)
{
    LOAD_SUBMODULE(f0_predictor);
    LOAD_SUBMODULE(m_source);
    LOAD_SUBMODULE(conv_pre);
    conv_pre.causal_type = CausalConv1d::causal_type_t::right;

    int num_kernels;
    GGML_ASSERT(loader.get_metadata("sample_rate", reinterpret_cast<uint32_t&>(sampling_rate)));
    LOAD_METADATA(num_kernels);
    LOAD_METADATA(nb_harmonics);
    LOAD_METADATA(nsf_alpha);
    LOAD_METADATA(nsf_voiced_threshold);
    LOAD_METADATA(nsf_sigma);
    LOAD_METADATA(lrelu_slope);
    LOAD_METADATA(audio_limit);
    GGML_ASSERT(loader.get_metadata(prefix, "istft_params.n_fft", nfft));
    GGML_ASSERT(loader.get_metadata(prefix, "istft_params.hop_len", hop_len));

    int64_t id;
    GGML_ASSERT(loader.find_metadata_key(combine_prefix(prefix, "upsample_rates").c_str(), id));
    GGML_ASSERT(loader.parser.arr_type(id) == GGUF_TYPE_INT32);

    auto layers = loader.parser.arr_n(id);
    auto upsample_rates = reinterpret_cast<const int*>(loader.parser.arr_data(id));

    ups.resize(layers);
    scale_factor = hop_len;
    for (size_t i = 0; i != layers; ++i)
    {
        auto& up = ups[i];
        int upsample_rate = upsample_rates[i];
        scale_factor *= upsample_rate;
        up.s = upsample_rate;
        LOAD_SUBMODULE_EX(std::format("ups.{}", i).c_str(), up);
    }
    
    source_downs.resize(layers);
    source_resblocks.resize(layers);
    for (size_t i = 0; i != layers - 1; ++i)
    {
        auto down = new CausalConv1dDownSample;
        down->s = 1;
        for (const auto rate : std::ranges::reverse_view(std::span(upsample_rates + 1 + i, layers - i - 1)))
            down->s *= rate;
        source_downs[i].reset(down);

        LOAD_SUBMODULE_EX(std::format("source_downs.{}", i).c_str(), *down);
        LOAD_SUBMODULE_EX(std::format("source_resblocks.{}", i).c_str(), source_resblocks[i]);
    }
    // Final block.
    {
        const auto layer_idx = layers - 1;
        auto conv = new CausalConv1d;
        conv->d = 1;
        conv->causal_type = CausalConv1d::causal_type_t::left;
        source_downs[layer_idx].reset(conv);

        LOAD_SUBMODULE_EX(std::format("source_downs.{}", layer_idx).c_str(), *conv);
        LOAD_SUBMODULE_EX(std::format("source_resblocks.{}", layer_idx).c_str(), source_resblocks[layer_idx]);
    }

    layers *= num_kernels;
    resblocks.resize(layers);
    for (size_t i = 0; i != layers; ++i)
        LOAD_SUBMODULE_EX(std::format("resblocks.{}", i).c_str(), resblocks[i]);

    LOAD_SUBMODULE(conv_post);
    conv_post.causal_type = CausalConv1d::causal_type_t::left;
}

void Qwen2MLP::OnLoad(gguf_loader& loader, const std::string& prefix)
{
    LOAD_SUBMODULE(gate_proj);
    LOAD_SUBMODULE(up_proj);
    LOAD_SUBMODULE(down_proj);
}

void Qwen2RMSNorm::OnLoad(gguf_loader& loader, const std::string& prefix)
{
    LOAD_TENSOR(weight);
}

void Qwen2Attention::OnLoad(gguf_loader& loader, const std::string& prefix)
{
    LOAD_SUBMODULE(q_proj);
    LOAD_SUBMODULE(k_proj);
    LOAD_SUBMODULE(v_proj);
    LOAD_SUBMODULE(o_proj);
}

void Qwen2DecoderLayer::OnLoad(gguf_loader& loader, const std::string& prefix)
{
    LOAD_SUBMODULE(self_attn);
    LOAD_SUBMODULE(mlp);
    LOAD_SUBMODULE(input_layernorm);
    LOAD_SUBMODULE(post_attention_layernorm);
}

void CosyVoice3LM::OnLoad(gguf_loader& loader, const std::string& prefix)
{
    LOAD_TENSOR_EX("embed_tokens.weight", embed_tokens_weight);
    LOAD_TENSOR_EX("speech_embedding.weight", speech_embedding_weight);
    LOAD_SUBMODULE(norm);
    LOAD_SUBMODULE(llm_decoder);

    int num_hidden_layers;
    LOAD_METADATA(num_hidden_layers);
    LOAD_METADATA(num_attention_heads);
    LOAD_METADATA(num_key_value_heads);
    LOAD_METADATA(rms_norm_eps);
    LOAD_METADATA(rope_theta);
    LOAD_METADATA(sos_token_id);
    LOAD_METADATA(task_token_id);

    GGML_ASSERT(embed_tokens_weight->ne[0] == speech_embedding_weight->ne[0]);
    layers.resize(num_hidden_layers);

    auto layers_prefix = combine_prefix(prefix, "layers");
    for (int i = 0; i != num_hidden_layers; ++i)
    {
        auto& layer = layers[i];
        auto name = std::format("{}.{}", layers_prefix, i);
        LOAD_SUBMODULE_EX(name.c_str(), layer);
    }
}

static ggml_type cosyvoice_kv_cache_type_to_ggml(cosyvoice_kv_cache_type_t t)
{
    switch (t)
    {
    case COSYVOICE_KV_CACHE_TYPE_F32: return GGML_TYPE_F32;
    case COSYVOICE_KV_CACHE_TYPE_F16: return GGML_TYPE_F16;
    case COSYVOICE_KV_CACHE_TYPE_Q8_0: return GGML_TYPE_Q8_0;
    case COSYVOICE_KV_CACHE_TYPE_Q5_1: return GGML_TYPE_Q5_1;
    case COSYVOICE_KV_CACHE_TYPE_Q5_0: return GGML_TYPE_Q5_0;
    case COSYVOICE_KV_CACHE_TYPE_Q4_1: return GGML_TYPE_Q4_1;
    case COSYVOICE_KV_CACHE_TYPE_Q4_0: return GGML_TYPE_Q4_0;
    default: GGML_ABORT("unexpected kv cache type");
    }
}

static cosyvoice_kv_cache_type_t cosyvoice_ggml_to_kv_cache_type(ggml_type t)
{
    switch (t)
    {
    case GGML_TYPE_F32: return COSYVOICE_KV_CACHE_TYPE_F32;
    case GGML_TYPE_F16: return COSYVOICE_KV_CACHE_TYPE_F16;
    case GGML_TYPE_Q8_0: return COSYVOICE_KV_CACHE_TYPE_Q8_0;
    case GGML_TYPE_Q5_1: return COSYVOICE_KV_CACHE_TYPE_Q5_1;
    case GGML_TYPE_Q5_0: return COSYVOICE_KV_CACHE_TYPE_Q5_0;
    case GGML_TYPE_Q4_1: return COSYVOICE_KV_CACHE_TYPE_Q4_1;
    case GGML_TYPE_Q4_0: return COSYVOICE_KV_CACHE_TYPE_Q4_0;
    default: GGML_ABORT("unexpected ggml type for kv cache");
    }
}

static ggml_type cosyvoice_get_kv_fallback_type(ggml_type t)
{
    switch (t)
    {
    case GGML_TYPE_Q4_0: return GGML_TYPE_Q4_1;
    case GGML_TYPE_Q4_1: return GGML_TYPE_Q5_0;
    case GGML_TYPE_Q5_0: return GGML_TYPE_Q5_1;
    case GGML_TYPE_Q5_1: return GGML_TYPE_Q8_0;
    case GGML_TYPE_Q8_0: return GGML_TYPE_F16;
    case GGML_TYPE_F16:  return GGML_TYPE_F32;
    default: GGML_ABORT("fatal error");
    }
}

union kv_cache_type_union
{
    struct
    {
#if defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__) || defined(_BYTE_ORDER) && (_BYTE_ORDER == _BIG_ENDIAN) || defined(__BIG_ENDIAN__) && !defined(__LITTLE_ENDIAN__) || defined(__ARMEB__) || defined(__MIPSEB__) || defined(__sparc__)
        uint32_t                    kv_cache_separate_buffers : 1;
        cosyvoice_kv_cache_type_t : 16;
        cosyvoice_kv_cache_type_t   kv_cache_fallback : 5;
        cosyvoice_kv_cache_type_t   v_cache_type : 5;
        cosyvoice_kv_cache_type_t   k_cache_type : 5;
#else
        cosyvoice_kv_cache_type_t   k_cache_type : 5;
        cosyvoice_kv_cache_type_t   v_cache_type : 5;
        cosyvoice_kv_cache_type_t   kv_cache_fallback : 5;
        cosyvoice_kv_cache_type_t : 16;
        uint32_t                    kv_cache_separate_buffers : 1;
#endif
    };
    cosyvoice_kv_cache_type_t       kv_cache_type;
};

static std::tuple<ggml_type, ggml_type> cosyvoice_check_kv_cache_types(
    ggml_context* ctx, ggml_backend_t backend,
    bool& fattn, int num_attn_heads, int num_kv_heads,
    const Linear& q_proj, const Linear& k_proj, const Linear& v_proj,
    bool kv_fallback, kv_cache_type_union& kv_type_union)
{
    ggml_type k_type, v_type;

    if (fattn)
    {
        auto fattn_check = [&](ggml_type check_k, ggml_type check_v) -> bool
        {
            const auto k_head_dim = k_proj.weight->ne[1] / num_kv_heads;
            const auto v_head_dim = v_proj.weight->ne[1] / num_kv_heads;

            auto position_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
            auto q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, q_proj.weight->ne[1] / num_attn_heads, 1, num_attn_heads, 1);
            auto k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k_head_dim * num_kv_heads, 1, 1);
            auto v = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, v_head_dim * num_kv_heads, 1, 1);
            auto cached_k = ggml_new_tensor_3d(ctx, check_k, k_head_dim * num_kv_heads, 1, 1);
            auto cached_v = ggml_new_tensor_3d(ctx, check_v, v_head_dim * num_kv_heads, 1, 1);

            cached_k = ggml_set_rows(ctx, cached_k, k, position_ids);
            cached_v = ggml_set_rows(ctx, cached_v, v, position_ids);
            if (!ggml_backend_supports_op(backend, cached_k) || !ggml_backend_supports_op(backend, cached_v))
                return false;

            cached_k = ggml_view_4d(ctx, cached_k, k_head_dim, num_kv_heads, 1, 1, cached_k->nb[1] / num_kv_heads, cached_k->nb[1], cached_k->nb[2], 0);
            cached_k = ggml_permute(ctx, cached_k, 0, 2, 1, 3);
            cached_v = ggml_view_4d(ctx, cached_v, v_head_dim, num_kv_heads, 1, 1, cached_v->nb[1] / num_kv_heads, cached_v->nb[1], cached_v->nb[2], 0);
            cached_v = ggml_permute(ctx, cached_v, 0, 2, 1, 3);

            auto o = ggml_flash_attn_ext(ctx, q, cached_k, cached_v, nullptr, 1.f / std::sqrt(static_cast<float>(k_head_dim)), 0.f, 0.f);
            return ggml_backend_supports_op(backend, o);
        };

        fattn = fattn_check(GGML_TYPE_F32, GGML_TYPE_F32);
        if (fattn)
        {
            if (kv_fallback)
            {
                ggml_type cur_type;

                do
                {
                    if (kv_type_union.kv_cache_separate_buffers)
                    {
                        if (k_type = cosyvoice_kv_cache_type_to_ggml(kv_type_union.k_cache_type),
                            v_type = cosyvoice_kv_cache_type_to_ggml(kv_type_union.v_cache_type);
                            fattn_check(k_type, v_type))
                        {
                            kv_type_union.kv_cache_type = COSYVOICE_MAKE_SEPARATE_KV_CACHE(
                                kv_type_union.k_cache_type,
                                kv_type_union.v_cache_type,
                                kv_type_union.v_cache_type);
                            break;
                        }

                        cur_type = cosyvoice_kv_cache_type_to_ggml(kv_type_union.kv_cache_fallback);
                    }
                    else
                        cur_type = cosyvoice_kv_cache_type_to_ggml(kv_type_union.kv_cache_type);

                    do
                    {
                        if (fattn_check(cur_type, cur_type))
                        {
                            k_type = cur_type;
                            v_type = cur_type;
                            kv_type_union.kv_cache_type = cosyvoice_ggml_to_kv_cache_type(cur_type);
                            break;
                        }

                        cur_type = cosyvoice_get_kv_fallback_type(cur_type);
                    } while (cur_type != GGML_TYPE_F32);

                    if (kv_type_union.kv_cache_separate_buffers)
                        kv_type_union.kv_cache_type = COSYVOICE_MAKE_SEPARATE_KV_CACHE(
                            cosyvoice_ggml_to_kv_cache_type(k_type),
                            cosyvoice_ggml_to_kv_cache_type(v_type),
                            cosyvoice_ggml_to_kv_cache_type(v_type));
                } while (false);
            }
            else if (kv_type_union.kv_cache_separate_buffers)
            {
                k_type = cosyvoice_kv_cache_type_to_ggml(kv_type_union.k_cache_type);
                v_type = cosyvoice_kv_cache_type_to_ggml(kv_type_union.v_cache_type);
                fattn = fattn_check(k_type, v_type);
                if (fattn)
                    kv_type_union.kv_cache_type = COSYVOICE_MAKE_SEPARATE_KV_CACHE(
                        kv_type_union.k_cache_type,
                        kv_type_union.v_cache_type,
                        kv_type_union.v_cache_type);
            }
            else
            {
                k_type = cosyvoice_kv_cache_type_to_ggml(kv_type_union.kv_cache_type);
                v_type = k_type;
                fattn = fattn_check(k_type, v_type);
            }
        }
    }

    if (!fattn)
    {
        ggml_type cur_type;
        auto attn_check = [&](ggml_type check_k, ggml_type check_v) -> bool
        {
            if (ggml_is_quantized(check_v)) return false;

            const auto k_head_dim = k_proj.weight->ne[1] / num_kv_heads;
            const auto v_head_dim = v_proj.weight->ne[1] / num_kv_heads;

            auto position_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
            auto q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, q_proj.weight->ne[1] / num_attn_heads, 1, num_attn_heads, 1);
            auto k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k_head_dim * num_kv_heads, 1, 1);
            auto cached_k = ggml_new_tensor_3d(ctx, check_k, k_head_dim * num_kv_heads, 1, 1);

            // K: head-merged cache rows scattered by position ids, then viewed back into
            // heads and permuted (see cosyvoice_kv_cache::update_cache)
            cached_k = ggml_set_rows(ctx, cached_k, k, position_ids);
            cached_k = ggml_view_4d(ctx, cached_k, k_head_dim, num_kv_heads, 1, 1, cached_k->nb[1] / num_kv_heads, cached_k->nb[1], cached_k->nb[2], 0);
            cached_k = ggml_permute(ctx, cached_k, 0, 2, 1, 3);

            // V: stored transposed and updated element-wise through the flattened cache
            // via v_idxs (see cosyvoice_kv_cache::update_cache)
            auto v = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, v_head_dim * num_kv_heads);
            auto cached_v = ggml_new_tensor_2d(ctx, check_v, 1, v_head_dim * num_kv_heads);
            auto v_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, v_head_dim * num_kv_heads);
            auto v_flat = ggml_reshape_2d(ctx, v, 1, ggml_nelements(v));
            auto cached_v_flat = ggml_reshape_2d(ctx, cached_v, 1, ggml_nelements(cached_v));
            cached_v = ggml_set_rows(ctx, cached_v_flat, v_flat, v_idxs);
            cached_v = ggml_view_4d(ctx, cached_v, 1, v_head_dim, num_kv_heads, 1, cached_v->nb[1], v_head_dim * cached_v->nb[1], cached_v->nb[2], 0);

            if (!ggml_backend_supports_op(backend, cached_k) || !ggml_backend_supports_op(backend, cached_v))
                return false;

            auto s = ggml_mul_mat(ctx, cached_k, q);
            auto o = ggml_mul_mat(ctx, cached_v, s);
            return ggml_backend_supports_op(backend, o) && ggml_backend_supports_op(backend, s);
        };

        do
        {
            if (kv_type_union.kv_cache_separate_buffers)
            {
                if (k_type = cosyvoice_kv_cache_type_to_ggml(kv_type_union.k_cache_type),
                    v_type = cosyvoice_kv_cache_type_to_ggml(kv_type_union.v_cache_type);
                    attn_check(k_type, v_type))
                {
                    kv_type_union.kv_cache_type = COSYVOICE_MAKE_SEPARATE_KV_CACHE(
                        kv_type_union.k_cache_type,
                        kv_type_union.v_cache_type,
                        kv_type_union.v_cache_type);
                    break;
                }
                else if (kv_fallback && attn_check(k_type, GGML_TYPE_F16))
                {
                    k_type = cosyvoice_kv_cache_type_to_ggml(kv_type_union.k_cache_type);
                    v_type = GGML_TYPE_F16;
                    kv_type_union.kv_cache_type = COSYVOICE_MAKE_SEPARATE_KV_CACHE(
                        kv_type_union.k_cache_type,
                        COSYVOICE_KV_CACHE_TYPE_F16,
                        COSYVOICE_KV_CACHE_TYPE_F16);
                    break;
                }
                else
                    cur_type = cosyvoice_kv_cache_type_to_ggml(kv_type_union.kv_cache_fallback);
            }
            else
                cur_type = cosyvoice_kv_cache_type_to_ggml(kv_type_union.kv_cache_type);

            if (kv_type_union.kv_cache_separate_buffers)
            {
                v_type = ggml_is_quantized(cur_type) ? GGML_TYPE_F16 : cur_type;
                do
                {
                    if (attn_check(cur_type, v_type))
                    {
                        k_type = cur_type;
                        break;
                    }

                    GGML_ASSERT(kv_fallback);
                    cur_type = cosyvoice_get_kv_fallback_type(cur_type);
                } while (cur_type != GGML_TYPE_F32);

                kv_type_union.kv_cache_type = COSYVOICE_MAKE_SEPARATE_KV_CACHE(
                    cosyvoice_ggml_to_kv_cache_type(cur_type),
                    cosyvoice_ggml_to_kv_cache_type(v_type),
                    cosyvoice_ggml_to_kv_cache_type(v_type));

                k_type = cur_type;
            }
            else
            {
                if (attn_check(cur_type, cur_type))
                {
                    k_type = cur_type;
                    v_type = cur_type;
                    kv_type_union.kv_cache_type = cosyvoice_ggml_to_kv_cache_type(cur_type);
                    break;
                }
                else
                {
                    GGML_ASSERT(kv_fallback);
                    cur_type = GGML_TYPE_F16;
                }
                k_type = cur_type;
                v_type = cur_type;
                kv_type_union.kv_cache_type = cosyvoice_ggml_to_kv_cache_type(cur_type);
            }
        } while (false);
    }

    return { k_type, v_type };
}

void cosyvoice_model_3::load(gguf_loader& loader)
{
    auto& flow = cv3_shared->flow;
    auto& hift = cv3_shared->hift;
    auto& llm = cv3_shared->llm;

    {
        // Resolve the effective number of diffusion steps before the flow submodule
        // is loaded (which sizes t_span) and before the DiT KV slots are clamped.
        // Precedence: params override (v4) > GGUF metadata > default 10, clamped to [1, MAX].
        int32_t md_steps = 10;
        loader.get_metadata("decoder", "diffusion_steps", md_steps);

        const int32_t req_steps = shared->params.diffusion_steps;
        int diffusion_steps = req_steps > 0 ? req_steps : md_steps;
        if (diffusion_steps < 1)
            diffusion_steps = 1;
        if (diffusion_steps > CausalConditionalCFM::MAX_DIFFUSION_STEPS)
        {
            cosyvoice_call_ggml_log_callback(GGML_LOG_LEVEL_WARN,
                std::format("decoder.diffusion_steps {} clamped to the maximum of {}.\n", diffusion_steps, CausalConditionalCFM::MAX_DIFFUSION_STEPS).c_str());
            diffusion_steps = CausalConditionalCFM::MAX_DIFFUSION_STEPS;
        }
        flow.decoder.diffusion_steps = diffusion_steps;

        auto& n_fixed_slots = shared->params.dit_kv_fixed_slots;
        auto& n_offloadable_slots = shared->params.dit_kv_offloadable_slots;
        if (n_fixed_slots > diffusion_steps)
            n_fixed_slots = diffusion_steps;
        if (n_offloadable_slots + n_fixed_slots > diffusion_steps)
            n_offloadable_slots = diffusion_steps - n_fixed_slots;
    }


    flow.OnLoad(loader, {});
    hift.OnLoad(loader, {});
    llm.OnLoad(loader, {});

    auto& tensors = loader.tensors;

    ggml_init_params params =
    {
        .mem_size = (tensors.size() + 6 + 2 * shared->params.n_workers) * ggml_tensor_overhead(),
        .mem_buffer = nullptr,
        .no_alloc = true
    };

    // init sinusodal position embedding
    constexpr int dim = 256;
    constexpr int half_dim = dim / 2;

    auto emb_buffer = std::make_unique<float[]>(half_dim);
    const auto emb = std::log(10000.f) / (half_dim - 1);

    for (int i = 0; i != half_dim; ++i)
        emb_buffer[i] = std::exp(i * -emb) * 1000;

    auto backend = worker->backend.get();
    auto cpu_backend = worker->cpu_backend.get();
    auto buft = ggml_backend_get_default_buffer_type(backend);
    auto alignment = ggml_backend_buft_get_alignment(buft);

    const int mel_dim = flow.decoder.estimator.mel_dim;
    size_t mem_size = get_aligned_size(sizeof(float) * half_dim, alignment)
        + get_aligned_size(sizeof(float) * (hift.nfft / 2 + 1), alignment)
        + get_aligned_size(sizeof(float) * (hift.nfft / 2 + 1) * hift.nfft, alignment) * 2
        + get_aligned_size(sizeof(int) * shared->params.n_batch, alignment) * shared->params.n_workers;
    for (const auto& [name, tensor] : tensors)
        if (tensor != &llm.embed_tokens_weight
            && tensor != &llm.speech_embedding_weight)
            mem_size += get_aligned_size(ggml_backend_buft_get_alloc_size(buft, *tensor), alignment);

    shared->buffer.reset(ggml_backend_buft_alloc_buffer(buft, mem_size));
    auto buffer_base = reinterpret_cast<char*>(ggml_backend_buffer_get_base(shared->buffer.get()));

    shared->backend_uma = backend_looks_uma(backend, shared->buffer.get());
    if (shared->params.inference_buffer_policy == COSYVOICE_INFERENCE_BUFFER_POLICY_BALANCED
        && shared->backend_uma)
    {
        shared->params.inference_buffer_policy = COSYVOICE_INFERENCE_BUFFER_POLICY_DEDICATED;
        cosyvoice_call_ggml_log_callback(GGML_LOG_LEVEL_INFO, "Detected UMA-like backend memory; switching balanced inference buffers to dedicated mode.\n");
    }

    shared->ctx.reset(ggml_init(params));

    mem_size = ggml_backend_buft_get_alloc_size(ggml_backend_cpu_buffer_type(), llm.embed_tokens_weight)
        + ggml_backend_buft_get_alloc_size(ggml_backend_cpu_buffer_type(), llm.speech_embedding_weight)
        + sizeof(float) * hift.nfft;
    shared->cpu_buffer.reset(ggml_backend_buft_alloc_buffer(ggml_backend_get_default_buffer_type(cpu_backend), mem_size));
    auto cpu_buffer_base = reinterpret_cast<char*>(ggml_backend_buffer_get_base(shared->cpu_buffer.get()));

    auto set_tensor = [&](ggml_tensor* tensor, const void* data, size_t size)
    {
        ggml_backend_tensor_set_async(backend, tensor, data, 0, size);
        buffer_base += get_aligned_size(ggml_backend_buft_get_alloc_size(buft, tensor), alignment);
    };
    auto set_cpu_tensor = [&](ggml_tensor* tensor, const void* data, size_t size)
    {
        ggml_backend_tensor_set(tensor, data, 0, size);
        cpu_buffer_base += ggml_backend_buft_get_alloc_size(ggml_backend_get_default_buffer_type(cpu_backend), tensor);
    };
    flow.decoder.estimator.time_embed.time_embed.emb = ggml_new_tensor_1d(shared->ctx.get(), GGML_TYPE_F32, half_dim);
    ggml_backend_tensor_alloc(shared->buffer.get(), flow.decoder.estimator.time_embed.time_embed.emb, buffer_base);
    set_tensor(flow.decoder.estimator.time_embed.time_embed.emb, emb_buffer.get(), sizeof(float) * half_dim);

    for (const auto& [name, tensor] : tensors)
    {
        size_t tensor_size = ggml_nbytes(*tensor);
        auto new_tensor = ggml_new_tensor(shared->ctx.get(), (*tensor)->type, GGML_MAX_DIMS, (*tensor)->ne);
        if (tensor == &llm.embed_tokens_weight
            || tensor == &llm.speech_embedding_weight)
        {
            ggml_backend_tensor_alloc(shared->cpu_buffer.get(), new_tensor, cpu_buffer_base);
            set_cpu_tensor(new_tensor, ggml_get_data(*tensor), tensor_size);
        }
        else
        {
            ggml_backend_tensor_alloc(shared->buffer.get(), new_tensor, buffer_base);

            const auto eps = get_tensor_eps(*tensor);
            if (eps == 0.f)
                set_tensor(new_tensor, ggml_get_data(*tensor), tensor_size);
            else
            {
                auto data = reinterpret_cast<const float*>(ggml_get_data(*tensor));
                const auto nelements = ggml_nelements(*tensor);
                std::unique_ptr<float[]> buffer;
                for (int64_t i = 0; i != nelements; ++i)
                    if (std::abs(data[i]) < eps)
                    {
                        if (!buffer)
                        {
                            buffer.reset(new float[nelements]);
                            memcpy(buffer.get(), data, sizeof(float) * nelements);
                        }

                        buffer[i] = eps;
                    }
                set_tensor(new_tensor, buffer ? buffer.get() : data, tensor_size);
            }
        }

        *tensor = new_tensor;
        ggml_set_param(*tensor);
        ggml_set_name(*tensor, name.c_str());
    }

    hift.window = ggml_new_tensor_1d(shared->ctx.get(), GGML_TYPE_F32, hift.nfft);
    ggml_set_param(hift.window);
    ggml_set_name(hift.window, "stft_window");
    ggml_backend_tensor_alloc(shared->cpu_buffer.get(), hift.window, cpu_buffer_base);
    for (int i = 0; i != hift.nfft; ++i)
        reinterpret_cast<float*>(hift.window->data)[i] = (1.0f - std::cos(2.0f * 3.14159265358979323846f * i / hift.nfft)) / 2.f;

    hift.fctx = create_fft_context(hift.nfft);
    hift.ictx = create_istft_context(hift.nfft, shared->ctx.get(),
        [&](ggml_tensor* tensor, void* data, size_t size)
        {
            ggml_backend_tensor_alloc(shared->buffer.get(), tensor, buffer_base);
            set_tensor(tensor, data, size);
            ggml_set_param(tensor);
            ggml_backend_synchronize(backend);
        });

    shared->full_position_ids.reset(new int[shared->params.n_max_seq]);
    for (int i = 0; i != shared->params.n_max_seq; ++i)
        shared->full_position_ids.get()[i] = i;

    for (uint32_t i = 0; i != shared->params.n_workers; ++i)
    {
        auto worker = workers + i;

        worker->position_ids = ggml_new_tensor_1d(shared->ctx.get(), GGML_TYPE_I32, shared->params.n_batch);
        ggml_set_name(worker->position_ids, std::format("position_ids.{}", i).c_str());
        ggml_backend_tensor_alloc(shared->buffer.get(), worker->position_ids, buffer_base);
        ggml_set_param(worker->position_ids);
        buffer_base += get_aligned_size(worker->position_ids->nb[1], alignment);

        worker->causal_mask = ggml_new_tensor_2d(shared->ctx.get(), GGML_TYPE_F16, shared->params.n_max_seq - 1, shared->params.n_batch);
        ggml_set_input(worker->causal_mask);
    }

    shared->noise_rng.seed(shared->params.seed);

    hift.m_source.l_sin_gen.rand_ini = ggml_new_tensor_1d(shared->ctx.get(), GGML_TYPE_F32, hift.nb_harmonics + 1);
    ggml_backend_tensor_alloc(shared->buffer.get(), hift.m_source.l_sin_gen.rand_ini, buffer_base);

    auto rand_ini_buffer = std::make_unique<float[]>(hift.nb_harmonics + 1);
    rand_ini_buffer[0] = 0.f;
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (auto& i : std::span(rand_ini_buffer.get() + 1, hift.nb_harmonics))
        i = dist(shared->noise_rng);
    hift.set_rand_ini(rand_ini_buffer.get());

    int64_t id = loader.parser.find_key("stop_token_ids");
    auto stop_tok_data = reinterpret_cast<const int*>(loader.parser.arr_data(id));
    id = static_cast<int64_t>(loader.parser.arr_n(id));
    cv3_shared->stop_tokens.insert(stop_tok_data, stop_tok_data + id);

    // FSQ silent and breath tokens �?loaded from GGUF if available,
    // otherwise fall back to hardcoded defaults matching CosyVoice3.
    id = loader.parser.find_key("silent_token_ids");
    if (id != -1)
    {
        auto silent_tok_data = reinterpret_cast<const int*>(loader.parser.arr_data(id));
        id = static_cast<int64_t>(loader.parser.arr_n(id));
        cv3_shared->silent_tokens.insert(silent_tok_data, silent_tok_data + id);
    }
    else
    {
        static const int kDefaultSilentTokens[] = { 1, 2, 28, 29, 55, 248, 494, 2241, 2242, 2322, 2323 };
        cv3_shared->silent_tokens.insert(std::begin(kDefaultSilentTokens), std::end(kDefaultSilentTokens));
    }

    id = loader.parser.find_key("cosyvoice.instruction_prefix");
    if (id != -1)
    {
        auto str = loader.parser.val_str(id);
        auto len = strlen(str) + 1;
        shared->instruction_prefix.reset(new char[len]);
        memcpy(shared->instruction_prefix.get(), str, len);
    }

    shared->config.temperature = 1.f;
    shared->config.max_token_text_ratio = 20.f;
    shared->config.min_token_text_ratio = 2.f;

    auto& sampling = shared->config.sampling;
    LOAD_METADATA_NOPREFIX(sampling.top_k);
    LOAD_METADATA_NOPREFIX(sampling.top_p);
    LOAD_METADATA_NOPREFIX(sampling.win_size);
    LOAD_METADATA_NOPREFIX(sampling.tau_r);

    for (uint32_t i = 0; i != shared->params.n_workers; ++i)
        workers[i].config = shared->config;

    auto [llm_k_type, llm_v_type] = cosyvoice_check_kv_cache_types(worker->ctx0.get(), worker->backend.get(),
        shared->params.llm_use_flash_attn, cv3_shared->llm.num_attention_heads, cv3_shared->llm.num_key_value_heads,
        cv3_shared->llm.layers[0].self_attn.q_proj, cv3_shared->llm.layers[0].self_attn.k_proj, cv3_shared->llm.layers[0].self_attn.v_proj,
        shared->params.llm_allow_kv_cache_fallback, reinterpret_cast<kv_cache_type_union&>(shared->params.llm_kv_cache_type));

    auto [dit_k_type, dit_v_type] = shared->params.dit_kv_fixed_slots != 0 ?
        cosyvoice_check_kv_cache_types(worker->ctx0.get(), worker->backend.get(),
            shared->params.flow_use_flash_attn, cv3_shared->flow.decoder.estimator.transformer_blocks[0].attn.heads, cv3_shared->flow.decoder.estimator.transformer_blocks[0].attn.heads,
            cv3_shared->flow.decoder.estimator.transformer_blocks[0].attn.to_q, cv3_shared->flow.decoder.estimator.transformer_blocks[0].attn.to_k, cv3_shared->flow.decoder.estimator.transformer_blocks[0].attn.to_v,
            shared->params.dit_allow_kv_cache_fallback, reinterpret_cast<kv_cache_type_union&>(shared->params.dit_kv_cache_type))
        : std::tuple<ggml_type, ggml_type>();

    for (auto& worker : std::span(workers, shared->params.n_workers))
    {
        worker.chunk_size = 25 + cv3_shared->flow.pre_lookahead_layer.pre_lookahead_len;
        worker.nucleus_probs_capacity = static_cast<uint32_t>(sampling.top_k * 2);
        worker.nucleus_probs.reset(new float[worker.nucleus_probs_capacity]);
        worker.nucleus_probs_len = 0;
        worker.probs.reset(new float[llm.llm_decoder.weight->ne[1]]);
        worker.batch_buffer.reset(new char[shared->params.n_batch * std::max(llm.embed_tokens_weight->nb[1], llm.speech_embedding_weight->nb[1])]);

        worker.llm_kv_cache.build_kv_cache(
            backend,
            worker.llm_kv_buffer,
            static_cast<int>(llm.layers.size()),
            static_cast<int>(llm.layers[0].self_attn.k_proj.weight->ne[1] / llm.num_key_value_heads),
            static_cast<int>(llm.layers[0].self_attn.v_proj.weight->ne[1] / llm.num_key_value_heads),
            llm.num_key_value_heads,
            shared->params.n_max_seq,
            llm_k_type,
            llm_v_type,
            1,
            1,
            shared->params.inference_buffer_policy == COSYVOICE_INFERENCE_BUFFER_POLICY_DEDICATED ? 0 : 1,
            shared->params.llm_use_flash_attn
        );

        if (shared->params.dit_kv_fixed_slots + shared->params.dit_kv_offloadable_slots != 0)
        {
            const auto dit_blocks = flow.decoder.estimator.transformer_blocks;
            worker.dit_kv_cache.build_kv_cache(
                backend,
                worker.dit_kv_buffer,
                static_cast<int>(dit_blocks.size()),
                static_cast<int>(dit_blocks[0].attn.to_k.weight->ne[1] / dit_blocks[0].attn.heads),
                static_cast<int>(dit_blocks[0].attn.to_v.weight->ne[1] / dit_blocks[0].attn.heads),
                dit_blocks[0].attn.heads,
                shared->params.dit_kv_cache_length,
                dit_k_type,
                dit_v_type,
                2,
                shared->params.dit_kv_fixed_slots + (shared->params.dit_kv_offloadable_slots != 0 ? 1 : 0),
                shared->params.dit_kv_offloadable_slots,
                shared->params.flow_use_flash_attn
            );
        }
    }

    for (auto& block : flow.decoder.estimator.transformer_blocks)
        block.attn.fattn = shared->params.flow_use_flash_attn;

    for (uint32_t i = 0; i < shared->params.n_workers; ++i)
    {
        auto cv3_worker = cv3_workers + i;
        auto worker = workers + i;

        switch (shared->params.inference_buffer_policy)
        {
        case COSYVOICE_INFERENCE_BUFFER_POLICY_BALANCED:
        case COSYVOICE_INFERENCE_BUFFER_POLICY_SHARED:
            cv3_worker->token2wav_buffer.reset(worker->llm_kv_buffer.get());
        case COSYVOICE_INFERENCE_BUFFER_POLICY_DEDICATED:
            break;
        default:
            throw std::invalid_argument("unexpected policy");
        }
    }

    auto arch = loader.get_string("general.architecture");
    shared->architecture.reset(new char[arch.size() + 1]);
    memcpy(shared->architecture.get(), arch.data(), arch.size() + 1);
    shared->hift_overlap = hift.overlap_length();

    ggml_backend_synchronize(backend);
}
