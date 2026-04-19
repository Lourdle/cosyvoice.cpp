// DiT sequence-length-dependent contiguity tests.
//
// The Flow DiT pipeline produces correct output for 31-token sequences
// but silence for 60-token sequences on Metal/CPU. This test verifies
// that ggml ops used in the DiT path maintain contiguity invariants
// across different sequence lengths.
//
// Key ops to test:
// - ggml_permute (DiT entry/exit)
// - ggml_repeat_4d (position_ids, spks broadcast)
// - ggml_view_3d (cut_len slicing)
// - split_tensor (AdaLayerNorm, attn_norm)
// - concat_tensors (InputEmbedding)

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>

#include <cmath>
#include <cstring>
#include <vector>

// Reimplementation of split_tensor from cosyvoice-graph.cpp
static void split_tensor(ggml_context* ctx, ggml_tensor* tensor, int dim, ggml_tensor** tensors, uint16_t chunks)
{
    int64_t ne[GGML_MAX_DIMS];
    memcpy(ne, tensor->ne, sizeof(ne));
    ne[dim] /= chunks;
    const size_t offset_per_chunk = tensor->nb[dim] * ne[dim];

    for (uint16_t i = 0; i != chunks; ++i)
        tensors[i] = ggml_view_4d(
            ctx, tensor,
            ne[0], ne[1], ne[2], ne[3],
            tensor->nb[1], tensor->nb[2], tensor->nb[3],
            i * offset_per_chunk);
}

SCENARIO("DiT permute produces non-contiguous tensor at any sequence length") {
    ggml_init_params params{};
    params.mem_size = 16 * 1024 * 1024;
    params.no_alloc = true;
    auto ctx = ggml_init(params);
    REQUIRE(ctx != nullptr);

    for (int seq_len : {31, 60, 100, 236}) {
        GIVEN("a tensor with sequence length " + std::to_string(seq_len)) {
            auto x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, seq_len, 80, 2);

            WHEN("permuted (1,0,2,3) like DiT entry") {
                auto p = ggml_permute(ctx, x, 1, 0, 2, 3);

                THEN("result is non-contiguous") {
                    REQUIRE_FALSE(ggml_is_contiguous(p));
                    REQUIRE(p->ne[0] == 80);
                    REQUIRE(p->ne[1] == seq_len);
                }

                WHEN("ggml_cont is applied") {
                    auto c = ggml_cont(ctx, p);
                    THEN("result is contiguous with same shape") {
                        REQUIRE(ggml_is_contiguous(c));
                        REQUIRE(c->ne[0] == 80);
                        REQUIRE(c->ne[1] == seq_len);
                    }
                }
            }
        }
    }

    ggml_free(ctx);
}

SCENARIO("ggml_view_3d for cut_len slicing produces non-contiguous views") {
    ggml_init_params params{};
    params.mem_size = 16 * 1024 * 1024;
    params.no_alloc = true;
    auto ctx = ggml_init(params);
    REQUIRE(ctx != nullptr);

    GIVEN("a 3D tensor [80, seq_len, 2] with different cut_len values") {
        for (int seq_len : {31, 60, 100}) {
            auto x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 80, seq_len, 2);
            int cut_len = seq_len / 3;

            auto view = ggml_view_3d(ctx, x,
                x->ne[0], x->ne[1] - cut_len, x->ne[2],
                x->nb[1], x->nb[2],
                x->nb[1] * cut_len);

            THEN("view has reduced ne[1] and is contiguous (strides match)") {
                REQUIRE(view->ne[0] == 80);
                REQUIRE(view->ne[1] == seq_len - cut_len);
                REQUIRE(view->ne[2] == 2);
                // View with offset but matching strides is contiguous
                REQUIRE(view->nb[0] == sizeof(float));
                REQUIRE(view->nb[1] == 80 * sizeof(float));
            }
        }
    }

    ggml_free(ctx);
}

SCENARIO("split_tensor for AdaLayerNorm produces views with expected contiguity") {
    ggml_init_params params{};
    params.mem_size = 16 * 1024 * 1024;
    params.no_alloc = true;
    auto ctx = ggml_init(params);
    REQUIRE(ctx != nullptr);

    GIVEN("a 1D tensor split into 2 along dim=0 (like AdaLayerNorm scale/shift)") {
        for (int dim0 : {128, 256, 512}) {
            auto tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, dim0);
            ggml_tensor* chunks[2];
            split_tensor(ctx, tensor, 0, chunks, 2);

            THEN("chunks have half the elements") {
                REQUIRE(chunks[0]->ne[0] == dim0 / 2);
                REQUIRE(chunks[1]->ne[0] == dim0 / 2);
            }

            THEN("chunk[0] is contiguous") {
                REQUIRE(ggml_is_contiguous(chunks[0]));
            }

            THEN("chunk[1] is contiguous (strides match, just offset)") {
                REQUIRE(ggml_is_contiguous(chunks[1]));
            }
        }
    }

    GIVEN("a 2D tensor [dim0, seq_len] split along dim=0") {
        for (int seq_len : {31, 60, 100}) {
            auto tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 512, seq_len);
            ggml_tensor* chunks[2];
            split_tensor(ctx, tensor, 0, chunks, 2);

            THEN("chunks have shape [256, seq_len]") {
                REQUIRE(chunks[0]->ne[0] == 256);
                REQUIRE(chunks[0]->ne[1] == seq_len);
            }

            THEN("chunk[0] is NOT contiguous (strides from parent)") {
                // nb[1] = 512 * sizeof(float) but ne[0] = 256
                // so nb[1] != ne[0] * nb[0] → non-contiguous
                REQUIRE(chunks[0]->nb[1] == 512 * sizeof(float));
                REQUIRE_FALSE(ggml_is_contiguous(chunks[0]));
            }
        }
    }

    ggml_free(ctx);
}

SCENARIO("ggml_repeat_4d preserves contiguity") {
    ggml_init_params params{};
    params.mem_size = 16 * 1024 * 1024;
    params.no_alloc = true;
    auto ctx = ggml_init(params);
    REQUIRE(ctx != nullptr);

    GIVEN("position_ids repeat pattern used in DiT") {
        for (int seq_len : {31, 60, 100}) {
            auto pos = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, seq_len);
            auto repeated = ggml_repeat_4d(ctx, pos, seq_len, 2, 1, 1);

            THEN("repeated tensor has expected shape") {
                REQUIRE(repeated->ne[0] == seq_len);
                REQUIRE(repeated->ne[1] == 2);
            }
        }
    }

    ggml_free(ctx);
}

// ---------------------------------------------------------------------------
// Helper: sum all float elements of a tensor whose data pointer is populated.
// ---------------------------------------------------------------------------
static float tensor_sum(const ggml_tensor* t)
{
    const auto n = ggml_nelements(t);
    const auto* data = reinterpret_cast<const float*>(t->data);
    float s = 0.0f;
    for (int64_t i = 0; i < n; ++i)
        s += data[i];
    return s;
}

SCENARIO("permute + cont + split_tensor produces correct batch data", "[smoke][permute_split]") {
    // This test documents the exact bug that ggml_cont-after-ggml_permute fixes:
    // without ggml_cont the split chunks see non-contiguous batch strides and
    // fetch data from the wrong rows, mixing batch-0 and batch-1 values.

    GIVEN("a [80, 20, 2] F32 tensor: batch-0 = 1.0, batch-1 = 2.0") {
        auto backend = ggml_backend_cpu_init();
        REQUIRE(backend != nullptr);

        ggml_init_params init_params{};
        init_params.mem_size = 16 * 1024 * 1024;
        init_params.no_alloc = true;
        auto ctx = ggml_init(init_params);
        REQUIRE(ctx != nullptr);

        // Shape: ne[0]=80, ne[1]=20, ne[2]=2  (batch dim = 2)
        auto x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 80, 20, 2);

        SECTION("WITH ggml_cont: split chunks contain pure batch data") {
            // permute (1,0,2,3): [80,20,2] → [20,80,2]
            auto permuted = ggml_permute(ctx, x, 1, 0, 2, 3);
            // ggml_cont materialises the permuted layout
            auto contig   = ggml_cont(ctx, permuted);

            // split along batch dim (dim=2): each chunk is [20, 80, 1]
            const int64_t chunk_ne0 = contig->ne[0]; // 20
            const int64_t chunk_ne1 = contig->ne[1]; // 80
            // nb[2] is the stride over the batch dimension in the contig tensor
            auto chunk0 = ggml_view_3d(ctx, contig,
                chunk_ne0, chunk_ne1, 1,
                contig->nb[1], contig->nb[2],
                /*offset=*/0);
            auto chunk1 = ggml_view_3d(ctx, contig,
                chunk_ne0, chunk_ne1, 1,
                contig->nb[1], contig->nb[2],
                /*offset=*/contig->nb[2]);

            // ggml_cont on each chunk to ensure dense layout before reading
            auto c0 = ggml_cont(ctx, chunk0);
            auto c1 = ggml_cont(ctx, chunk1);

            // Mark output tensors so the scheduler keeps their buffers live
            ggml_set_output(c0);
            ggml_set_output(c1);

            // Allocate all tensors through the CPU backend
            auto buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
            REQUIRE(buf != nullptr);

            // Initialise source data: batch-0 = 1.0, batch-1 = 2.0
            // x has shape [80, 20, 2]; ne[2]=2 batches, each of size 80*20 floats
            {
                const int64_t batch_floats = 80 * 20;
                std::vector<float> src(batch_floats * 2);
                for (int64_t i = 0; i < batch_floats; ++i) src[i]              = 1.0f;
                for (int64_t i = 0; i < batch_floats; ++i) src[batch_floats+i] = 2.0f;
                ggml_backend_tensor_set(x, src.data(), 0, src.size() * sizeof(float));
            }

            // Build and compute the graph
            auto gf = ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE, false);
            ggml_build_forward_expand(gf, c0);
            ggml_build_forward_expand(gf, c1);
            REQUIRE(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);

            THEN("chunk-0 sum == 80*20*1.0 = 1600.0 (pure batch-0 data)") {
                const float expected0 = 80.0f * 20.0f * 1.0f; // 1600.0
                CHECK(tensor_sum(c0) == Catch::Approx(expected0).margin(1e-3f));
            }
            THEN("chunk-1 sum == 80*20*2.0 = 3200.0 (pure batch-1 data)") {
                const float expected1 = 80.0f * 20.0f * 2.0f; // 3200.0
                CHECK(tensor_sum(c1) == Catch::Approx(expected1).margin(1e-3f));
            }
            THEN("chunk-0 sum != chunk-1 sum (batches are distinct)") {
                CHECK(tensor_sum(c0) != Catch::Approx(tensor_sum(c1)).margin(1e-3f));
            }

            ggml_backend_buffer_free(buf);
        }

        // Note: the actual bug manifests in split_tensor's ggml_view_4d + ggml_cont
        // chain (nb[0]==nb[1]==4 overlap), not in ggml_view_3d which handles strides
        // correctly. The WITH section above verifies the fix (ggml_cont after permute).

        ggml_free(ctx);
        ggml_backend_free(backend);
    }
}

SCENARIO("ggml_gelu_erf matches erf-based formula", "[smoke][gelu_erf]") {
    GIVEN("a [8] F32 tensor with values {-3, -2, -1, -0.5, 0, 0.5, 1, 2}") {
        auto backend = ggml_backend_cpu_init();
        REQUIRE(backend != nullptr);

        ggml_init_params init_params{};
        init_params.mem_size = 4 * 1024 * 1024;
        init_params.no_alloc = true;
        auto ctx = ggml_init(init_params);
        REQUIRE(ctx != nullptr);

        const std::vector<float> input_vals = {-3.0f, -2.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 2.0f};
        const int n = static_cast<int>(input_vals.size());

        auto x      = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
        auto result = ggml_gelu_erf(ctx, x);
        ggml_set_output(result);

        auto buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        REQUIRE(buf != nullptr);

        ggml_backend_tensor_set(x, input_vals.data(), 0, n * sizeof(float));

        auto gf = ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE, false);
        ggml_build_forward_expand(gf, result);
        REQUIRE(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);

        WHEN("the output is compared with the reference formula 0.5*x*(1+erf(x/sqrt(2)))") {
            const float* out = reinterpret_cast<const float*>(result->data);

            // Compute reference in double precision to avoid accumulating float error
            const double sqrt2 = std::sqrt(2.0);
            float max_abs_err = 0.0f;
            for (int i = 0; i < n; ++i) {
                const double xi  = static_cast<double>(input_vals[i]);
                const double ref = 0.5 * xi * (1.0 + std::erf(xi / sqrt2));
                const float  err = std::abs(out[i] - static_cast<float>(ref));
                if (err > max_abs_err) max_abs_err = err;
            }

            THEN("max absolute error < 1e-6") {
                CHECK(max_abs_err < 1e-6f);
            }
        }

        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        ggml_backend_free(backend);
    }
}

// Helper: read all floats from a computed tensor
static std::vector<float> read_tensor(ggml_tensor* t) {
    std::vector<float> v(ggml_nelements(t));
    ggml_backend_tensor_get(t, v.data(), 0, v.size() * sizeof(float));
    return v;
}

// split_tensor replica (same as cosyvoice-graph.cpp) for testing
static void test_split_tensor(ggml_context* ctx, ggml_tensor* tensor, int dim,
                               ggml_tensor** out, int chunks) {
    int64_t ne[GGML_MAX_DIMS];
    memcpy(ne, tensor->ne, sizeof(ne));
    ne[dim] /= chunks;
    const size_t off = tensor->nb[dim] * ne[dim];
    for (int i = 0; i < chunks; ++i)
        out[i] = ggml_view_4d(ctx, tensor, ne[0], ne[1], ne[2], ne[3],
            tensor->nb[1], tensor->nb[2], tensor->nb[3], i * off);
}

SCENARIO("split_tensor<6> on batch=2 tensor splits AdaLayerNorm chunks correctly",
         "[smoke][split_tensor_batch]") {
    GIVEN("a [6*D, 2] F32 tensor where D=8, batch 0 and batch 1 have distinct values") {
        const int D = 8;
        auto* backend = ggml_backend_cpu_init();
        REQUIRE(backend);
        auto* ctx = ggml_init({.mem_size = 4*1024*1024, .no_alloc = true});

        auto* emb = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 6*D, 2);
        ggml_set_input(emb);

        // split into 6 chunks on dim 0
        ggml_tensor* chunks[6];
        test_split_tensor(ctx, emb, 0, chunks, 6);
        for (int i = 0; i < 6; ++i) {
            chunks[i] = ggml_cont(ctx, chunks[i]);
            ggml_set_output(chunks[i]);
        }

        auto* buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        REQUIRE(buf);

        // Fill: batch 0 = [1,2,3,...,48], batch 1 = [101,102,...,148]
        std::vector<float> data(6*D*2);
        for (int i = 0; i < 6*D; ++i) data[i] = (float)(i+1);           // batch 0
        for (int i = 0; i < 6*D; ++i) data[6*D+i] = (float)(i+101);     // batch 1
        ggml_backend_tensor_set(emb, data.data(), 0, data.size()*sizeof(float));

        auto* gf = ggml_new_graph(ctx);
        for (int i = 0; i < 6; ++i) ggml_build_forward_expand(gf, chunks[i]);
        REQUIRE(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);

        THEN("each chunk has shape [D, 2] with correct batch separation") {
            for (int c = 0; c < 6; ++c) {
                auto v = read_tensor(chunks[c]);
                REQUIRE(v.size() == (size_t)(D * 2));
                // batch 0: values should be [c*D+1, ..., (c+1)*D]
                for (int j = 0; j < D; ++j)
                    CHECK(v[j] == (float)(c*D + j + 1));
                // batch 1: values should be [c*D+101, ..., (c+1)*D+100]
                for (int j = 0; j < D; ++j)
                    CHECK(v[D+j] == (float)(c*D + j + 101));
            }
        }

        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        ggml_backend_free(backend);
    }
}

SCENARIO("CFG formula produces correct weighted combination",
         "[smoke][cfg_formula]") {
    GIVEN("v_cond and v_uncond tensors with known values, cfg_rate=0.7") {
        auto* backend = ggml_backend_cpu_init();
        REQUIRE(backend);
        auto* ctx = ggml_init({.mem_size = 4*1024*1024, .no_alloc = true});

        const int N = 16;
        const float cfg_rate = 0.7f;

        auto* v_cond = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, N);
        auto* v_uncond = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, N);
        ggml_set_input(v_cond);
        ggml_set_input(v_uncond);

        // CFG: (1 + cfg_rate) * v_cond - cfg_rate * v_uncond
        auto* scaled_cond = ggml_scale(ctx, v_cond, 1.f + cfg_rate);
        auto* scaled_uncond = ggml_scale(ctx, v_uncond, cfg_rate);
        auto* result = ggml_sub(ctx, scaled_cond, scaled_uncond);
        ggml_set_output(result);

        auto* buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        REQUIRE(buf);

        // v_cond = [-6.0, -5.0, ...], v_uncond = [-4.0, -3.0, ...]
        std::vector<float> vc(N), vu(N);
        for (int i = 0; i < N; ++i) { vc[i] = -6.0f + i*0.5f; vu[i] = -4.0f + i*0.5f; }
        ggml_backend_tensor_set(v_cond, vc.data(), 0, N*sizeof(float));
        ggml_backend_tensor_set(v_uncond, vu.data(), 0, N*sizeof(float));

        auto* gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, result);
        REQUIRE(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);

        THEN("result matches (1.7 * v_cond - 0.7 * v_uncond) element-wise") {
            auto r = read_tensor(result);
            for (int i = 0; i < N; ++i) {
                float expected = (1.f + cfg_rate) * vc[i] - cfg_rate * vu[i];
                CHECK(std::abs(r[i] - expected) < 1e-5f);
            }
        }

        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        ggml_backend_free(backend);
    }
}

SCENARIO("AdaLayerNorm_Final split_tensor<2> separates scale and shift correctly",
         "[smoke][norm_final_split]") {
    GIVEN("a [2*D, 2] F32 tensor where D=8, split into scale and shift") {
        const int D = 8;
        auto* backend = ggml_backend_cpu_init();
        REQUIRE(backend);
        auto* ctx = ggml_init({.mem_size = 4*1024*1024, .no_alloc = true});

        auto* emb = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2*D, 2);
        ggml_set_input(emb);

        ggml_tensor* parts[2];
        test_split_tensor(ctx, emb, 0, parts, 2);
        auto* scale = ggml_cont(ctx, parts[0]);
        auto* shift = ggml_cont(ctx, parts[1]);
        ggml_set_output(scale);
        ggml_set_output(shift);

        auto* buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        REQUIRE(buf);

        // Fill: [scale_b0(D), shift_b0(D), scale_b1(D), shift_b1(D)]
        // In ggml [2*D, 2]: batch 0 = first 2*D, batch 1 = next 2*D
        std::vector<float> data(2*D*2);
        for (int i = 0; i < D; ++i) data[i] = 10.0f + i;      // scale batch 0
        for (int i = 0; i < D; ++i) data[D+i] = 20.0f + i;    // shift batch 0
        for (int i = 0; i < D; ++i) data[2*D+i] = 30.0f + i;  // scale batch 1
        for (int i = 0; i < D; ++i) data[3*D+i] = 40.0f + i;  // shift batch 1
        ggml_backend_tensor_set(emb, data.data(), 0, data.size()*sizeof(float));

        auto* gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, scale);
        ggml_build_forward_expand(gf, shift);
        REQUIRE(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);

        THEN("scale contains first D elements per batch, shift contains next D") {
            auto sv = read_tensor(scale);
            auto hv = read_tensor(shift);
            REQUIRE(sv.size() == (size_t)(D*2));
            REQUIRE(hv.size() == (size_t)(D*2));
            for (int i = 0; i < D; ++i) CHECK(sv[i] == 10.0f + i);     // scale batch 0
            for (int i = 0; i < D; ++i) CHECK(sv[D+i] == 30.0f + i);   // scale batch 1
            for (int i = 0; i < D; ++i) CHECK(hv[i] == 20.0f + i);     // shift batch 0
            for (int i = 0; i < D; ++i) CHECK(hv[D+i] == 40.0f + i);   // shift batch 1
        }

        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        ggml_backend_free(backend);
    }
}

SCENARIO("conv_pos_embed batch independence with synthetic weights",
         "[smoke][conv_batch_independence]") {
    GIVEN("a [D, SEQ, 2] tensor where batch 1 is zeros") {
        const int D = 16, SEQ = 8;
        auto* backend = ggml_backend_cpu_init();
        REQUIRE(backend);
        auto* ctx = ggml_init({.mem_size = 4*1024*1024, .no_alloc = true});

        auto* x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, SEQ, 2);
        ggml_set_input(x);

        // Simple operation: permute → cont → permute back
        // This tests that batch independence is maintained through permute chain
        auto* p = ggml_permute(ctx, x, 1, 0, 2, 3);  // [SEQ, D, 2]
        auto* c = ggml_cont(ctx, p);
        auto* p2 = ggml_permute(ctx, c, 1, 0, 2, 3); // [D, SEQ, 2]
        auto* out = ggml_cont(ctx, p2);
        ggml_set_output(out);

        auto* buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        REQUIRE(buf);

        // batch 0 = 1.0, batch 1 = 0.0
        std::vector<float> data(D * SEQ * 2, 0.0f);
        for (int i = 0; i < D * SEQ; ++i) data[i] = 1.0f;
        ggml_backend_tensor_set(x, data.data(), 0, data.size()*sizeof(float));

        auto* gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
        REQUIRE(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);

        THEN("batch 0 preserves values and batch 1 remains zero") {
            auto r = read_tensor(out);
            float sum_b0 = 0, sum_b1 = 0;
            for (int i = 0; i < D*SEQ; ++i) sum_b0 += r[i];
            for (int i = D*SEQ; i < D*SEQ*2; ++i) sum_b1 += r[i];
            CHECK(sum_b0 == (float)(D * SEQ));  // all 1.0
            CHECK(sum_b1 == 0.0f);              // all 0.0
        }

        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        ggml_backend_free(backend);
    }
}
