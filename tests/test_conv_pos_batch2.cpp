// Test: grouped conv1d batch independence with batch=2.
//
// Verifies that batch[0] and batch[1] are processed independently
// in the grouped convolution used by conv_pos_embed.
// If batch data leaks, this is likely the root cause of the v_cond drift.

#include <catch2/catch_test_macros.hpp>

#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static const int D = 1024;
static const int SEQ = 20;
static const int KERNEL = 31;
static const int GROUPS = 16;

static ggml_tensor* unsqueeze_dim2(ggml_context* ctx, ggml_tensor* x) {
    return ggml_view_4d(ctx, x, x->ne[0], x->ne[1], 1, x->ne[2],
                        x->nb[1], x->nb[1], x->nb[2], 0);
}

// concat_tensors along a dimension (simplified for arrays)
static ggml_tensor* concat_tensors_dim2(ggml_context* ctx, ggml_tensor** xs, int n) {
    auto* result = xs[0];
    for (int i = 1; i < n; i++)
        result = ggml_concat(ctx, result, xs[i], 2);
    return result;
}

// Build grouped conv1d graph (replicating Conv1d::build_cgraph with g>1)
static ggml_tensor* grouped_conv1d(ggml_context* ctx, ggml_tensor* x, ggml_tensor* weight, ggml_tensor* bias, int g) {
    int64_t ch_per_group = x->ne[1] / g;

    auto* xs = (ggml_tensor**)alloca(sizeof(ggml_tensor*) * g);
    auto* ws = (ggml_tensor**)alloca(sizeof(ggml_tensor*) * g);

    // Split x along dim 1 (channels)
    for (int i = 0; i < g; i++) {
        xs[i] = ggml_view_3d(ctx, x, x->ne[0], ch_per_group, x->ne[2],
                             x->nb[1], x->nb[2], i * ch_per_group * x->nb[1]);
        ws[i] = ggml_view_3d(ctx, weight, weight->ne[0], weight->ne[1], weight->ne[2] / g,
                             weight->nb[1], weight->nb[2], i * (weight->ne[2] / g) * weight->nb[2]);
    }

    for (int i = 0; i < g; i++) {
        xs[i] = unsqueeze_dim2(ctx,
            ggml_im2col(ctx,
                ws[i],
                ggml_cont(ctx, xs[i]),
                1, 0, 0, 0, 1, 0,
                false, GGML_TYPE_F32));
    }

    auto* concat_result = concat_tensors_dim2(ctx, xs, g);
    ggml_set_output(concat_result);
    auto* im2col = ggml_cont(ctx, concat_result);

    auto* w_reshaped = ggml_reshape_3d(ctx, weight,
        weight->ne[0] * weight->ne[1], weight->ne[2] / g, g);

    if (x->ne[2] != 1)
        w_reshaped = ggml_repeat_4d(ctx, w_reshaped,
            w_reshaped->ne[0], w_reshaped->ne[1], g, x->ne[2]);

    auto* result = ggml_mul_mat(ctx, im2col, w_reshaped);
    result = ggml_reshape_3d(ctx, result, im2col->ne[1], weight->ne[2], x->ne[2]);

    if (bias) {
        // bias [out_ch] needs to broadcast to result [seq, out_ch, batch]
        // unsqueeze to [1, out_ch, 1] for broadcasting
        auto* b = ggml_reshape_3d(ctx, bias, 1, bias->ne[0], 1);
        result = ggml_add(ctx, result, b);
    }

    return result;
}

static ggml_tensor* mish(ggml_context* ctx, ggml_tensor* x) {
    return ggml_mul(ctx, x, ggml_tanh(ctx, ggml_softplus(ctx, x)));
}

SCENARIO("Given grouped conv1d with batch=2, "
         "when batch[1] changes from zeros to non-zeros, "
         "then batch[0] output is identical (batch independence)",
         "[e2e][conv_batch]") {

    const size_t ctx_size = ggml_tensor_overhead() * 500 + 1024 * 1024 * 100;
    auto* ctx = ggml_init({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
    REQUIRE(ctx);

    int ch_per_group = D / GROUPS;

    auto* conv1_w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, KERNEL, ch_per_group, D);
    auto* conv1_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, D);
    auto* conv2_w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, KERNEL, ch_per_group, D);
    auto* conv2_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, D);
    ggml_set_input(conv1_w);
    ggml_set_input(conv1_b);
    ggml_set_input(conv2_w);
    ggml_set_input(conv2_b);

    // Input: [D, SEQ, 2]
    auto* input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, SEQ, 2);
    ggml_set_input(input);

    // conv_pos_embed pipeline
    auto* x = ggml_permute(ctx, input, 1, 0, 2, 3);  // [D, SEQ, 2] → [SEQ, D, 2]
    x = ggml_cont(ctx, x);

    // Conv1 with causal padding
    x = ggml_pad_ext(ctx, x, KERNEL - 1, 0, 0, 0, 0, 0, 0, 0);
    ggml_set_output(x);
    x = grouped_conv1d(ctx, x, conv1_w, conv1_b, GROUPS);
    ggml_set_output(x);
    x = mish(ctx, x);
    ggml_set_output(x);

    // Conv2
    x = ggml_pad_ext(ctx, x, KERNEL - 1, 0, 0, 0, 0, 0, 0, 0);
    x = grouped_conv1d(ctx, x, conv2_w, conv2_b, GROUPS);
    ggml_set_output(x);
    x = mish(ctx, x);

    // Permute back
    x = ggml_permute(ctx, x, 1, 0, 2, 3);
    x = ggml_cont(ctx, x);
    ggml_set_output(x);

    auto* backend = ggml_backend_cpu_init();
    REQUIRE(backend);
    auto* buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    REQUIRE(buf);

    // Fill weights with deterministic pattern
    auto fill_det = [](ggml_tensor* t, float seed) {
        int64_t n = ggml_nelements(t);
        std::vector<float> buf(n);
        for (int64_t i = 0; i < n; i++)
            buf[i] = sinf(seed + i * 0.01f) * 0.1f;
        ggml_backend_tensor_set(t, buf.data(), 0, n * sizeof(float));
    };
    fill_det(conv1_w, 1.0f);
    fill_det(conv1_b, 2.0f);
    fill_det(conv2_w, 3.0f);
    fill_det(conv2_b, 4.0f);

    int64_t batch_floats = D * SEQ;

    auto* graph = ggml_new_graph_custom(ctx, 4096, false);
    ggml_build_forward_expand(graph, x);

    // --- Run 1: batch[1] = zeros ---
    {
        std::vector<float> data(2 * batch_floats, 0.0f);
        for (int64_t i = 0; i < batch_floats; i++)
            data[i] = sinf(5.0f + i * 0.017f) * 2.0f;
        ggml_backend_tensor_set(input, data.data(), 0, data.size() * sizeof(float));
    }

    REQUIRE(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> result1(2 * batch_floats);
    ggml_backend_tensor_get(x, result1.data(), 0, result1.size() * sizeof(float));
    std::vector<float> batch0_run1(result1.begin(), result1.begin() + batch_floats);

    printf("Run 1 output: ne=[%lld, %lld, %lld]\n", x->ne[0], x->ne[1], x->ne[2]);
    float sum0 = 0;
    for (int i = 0; i < batch_floats; i++) sum0 += batch0_run1[i];
    printf("  batch0 mean=%.6f\n", sum0 / batch_floats);

    // --- Run 2: batch[1] = non-zeros ---
    {
        std::vector<float> data(2 * batch_floats);
        for (int64_t i = 0; i < batch_floats; i++)
            data[i] = sinf(5.0f + i * 0.017f) * 2.0f;  // same batch[0]
        for (int64_t i = 0; i < batch_floats; i++)
            data[batch_floats + i] = cosf(7.0f + i * 0.013f) * 3.0f;  // different batch[1]
        ggml_backend_tensor_set(input, data.data(), 0, data.size() * sizeof(float));
    }

    REQUIRE(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> result2(2 * batch_floats);
    ggml_backend_tensor_get(x, result2.data(), 0, result2.size() * sizeof(float));
    std::vector<float> batch0_run2(result2.begin(), result2.begin() + batch_floats);

    // Compare
    double max_diff = 0, sum_diff = 0;
    int n_diffs = 0;
    for (int64_t i = 0; i < batch_floats; i++) {
        double d = std::abs((double)batch0_run1[i] - (double)batch0_run2[i]);
        max_diff = std::max(max_diff, d);
        sum_diff += d;
        if (d > 1e-6) n_diffs++;
    }

    printf("Batch independence: max_diff=%.10f, mean_diff=%.10f, n_diffs=%d/%lld\n",
           max_diff, sum_diff / batch_floats, n_diffs, (long long)batch_floats);

    if (max_diff > 1e-6) {
        printf("*** BATCH CONTAMINATION DETECTED ***\n");
        for (int64_t i = 0; i < batch_floats && n_diffs > 0; i++) {
            double d = std::abs((double)batch0_run1[i] - (double)batch0_run2[i]);
            if (d > 1e-6) {
                printf("  [%lld] (ch=%lld, t=%lld): %.8f vs %.8f (diff=%.8f)\n",
                       i, i % D, i / D, batch0_run1[i], batch0_run2[i], d);
                if (--n_diffs <= 0 || i > 20) break;
            }
        }
    }

    CHECK(max_diff < 1e-6);

    ggml_backend_buffer_free(buf);
    ggml_backend_free(backend);
    ggml_free(ctx);
}
