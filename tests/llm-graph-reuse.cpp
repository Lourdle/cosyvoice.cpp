// Model-backed regression for embedding-index lifetime in reused LLM graphs.
#include "cosyvoice.h"
#include "cosyvoice-lowlevel.h"
#include "cosyvoice-interface.h"
#include <ggml-backend.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>

static int capture_probs(cosyvoice_llm_token_prob_t*, int, float* probs, uint32_t size,
    const cosyvoice_sampling_params_t*, int*, uint32_t, void* opaque)
{
    auto& output = *static_cast<std::vector<float>*>(opaque);
    output.assign(probs, probs + size);
    return 0;
}

int main(int argc, char** argv)
{
    if (argc != 3) {
        std::fprintf(stderr, "Usage: %s MODEL.gguf BACKEND (e.g. CPU or MTL0)\n", argv[0]);
        return 2;
    }
    cosyvoice_init_backend_from_path(nullptr);
    auto backend = ggml_backend_init_by_name(argv[2], nullptr);
    if (!backend) return 2;

    cosyvoice_context_params_t params;
    cosyvoice_init_default_context_params(&params);
    params.llm_use_flash_attn = true;
    params.llm_kv_cache_type = COSYVOICE_KV_CACHE_TYPE_F32;
    params.n_max_seq = 128;
    params.n_batch = 128;
    std::vector<float> actual, expected;
    params.sampler = capture_probs;
    params.sampler_ctx = &actual;
    auto ctx = cosyvoice_load_from_file_ext(argv[1], &params, backend, 4,
        COSYVOICE_CONTEXT_PARAMS_VERSION);
    if (!ctx) return 2;
    std::unique_ptr<cosyvoice_context, decltype(&cosyvoice_free)> owner(ctx, cosyvoice_free);
    const auto* weight = ctx->get_speech_token_embed_weight();
    if (weight->type == GGML_TYPE_F32) {
        std::fprintf(stderr, "This regression needs a model with non-F32 speech embeddings.\n");
        return 2;
    }
    ctx->get_context_params(&params);
    if (!params.llm_use_flash_attn) {
        std::fprintf(stderr, "The backend must support LLM Flash Attention.\n");
        return 2;
    }
    // Fixed embedding IDs exercise the same input sequence without sampling noise.
    const int ids[] = {2, 2, 28, 29, 2215, 2};
    if (weight->ne[1] <= 2215) return 2;
    const size_t row_size = weight->nb[1];
    std::vector<unsigned char> embeddings(row_size * 6);
    for (int i = 0; i < 6; ++i)
        ggml_backend_tensor_get(weight, embeddings.data() + i * row_size,
            ids[i] * row_size, row_size);

    std::vector<std::vector<float>> cached;
    for (int i = 0; i < 6; ++i) {
        if (!ctx->llm_decode(weight->type, embeddings.data() + i * row_size)) return 1;
        ctx->llm_sample_token();
        cached.push_back(actual);
    }
    bool ok = true;
    // Rebuild from each full prefix as an independent reference. Never reuse a
    // decode graph in this path. Compare distributions rather than sampled IDs.
    ctx->set_sampler(capture_probs, &expected);
    for (int count = 2; count <= 6; ++count) {
        if (!ctx->llm_set_kv_cache_len(0) ||
            !ctx->llm_prefill_logits(weight->type, embeddings.data(), count)) return 1;
        ctx->llm_sample_token();
        double max_error = 0;
        if (expected.empty() || cached[count - 1].size() != expected.size()) return 1;
        for (size_t j = 0; j < expected.size(); ++j) {
            const float a = cached[count - 1][j], b = expected[j];
            if (!std::isfinite(a) || !std::isfinite(b)) { ok = false; break; }
            max_error = std::max(max_error, std::abs(double(a) - b));
        }
        std::printf("prefix=%d max_probability_error=%.9g\n", count, max_error);
        // Small backend rounding differences are expected across batching sizes.
        ok = ok && max_error < 1e-3;
    }
    return ok ? 0 : 1;
}
