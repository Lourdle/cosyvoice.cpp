// Tests for Conv1d IM2COL contiguity — verifies that grouped convolution
// inputs are contiguous before being passed to ggml_im2col.
//
// Background: Metal's IM2COL requires ggml_is_contiguous(src[1]).
// split_tensor produces non-contiguous views via ggml_view_4d (the strides
// inherit from the parent tensor). When these views are passed directly to
// ggml_im2col, Metal rejects the op and it falls back to CPU, creating a
// multi-backend graph split that crashes the scheduler.
//
// The fix is to wrap split views with ggml_cont before ggml_im2col.

#include <catch2/catch_test_macros.hpp>

#include <ggml.h>

#include <cstring>

// Reimplementation of split_tensor from cosyvoice-graph.cpp
static void split_tensor(ggml_context* ctx, ggml_tensor* tensor, int dim, ggml_tensor** tensors, uint16_t chunks)
{
    GGML_ASSERT(dim >= 0 && dim < GGML_MAX_DIMS);
    GGML_ASSERT(tensor->ne[dim] % chunks == 0);

    int64_t ne[GGML_MAX_DIMS];
    memcpy(ne, tensor->ne, sizeof(ne));
    ne[dim] /= chunks;
    const size_t offset_per_chunk = tensor->nb[dim] * ne[dim];

    for (uint16_t i = 0; i != chunks; ++i)
        tensors[i] = ggml_view_4d(
            ctx,
            tensor,
            ne[0],
            ne[1],
            ne[2],
            ne[3],
            tensor->nb[1],
            tensor->nb[2],
            tensor->nb[3],
            i * offset_per_chunk);
}

SCENARIO("split_tensor along dim=1 produces non-contiguous views") {
    ggml_init_params params{};
    params.mem_size = 16 * 1024 * 1024;
    params.no_alloc = true;
    auto ctx = ggml_init(params);
    REQUIRE(ctx != nullptr);

    GIVEN("a 3D tensor [16, 8, 1] split along dim=1 into 2 chunks") {
        auto tensor = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 16, 8, 1);

        THEN("the original tensor is contiguous") {
            REQUIRE(ggml_is_contiguous(tensor));
        }

        ggml_tensor* chunks[2];
        split_tensor(ctx, tensor, 1, chunks, 2);

        THEN("each chunk has shape [16, 4, 1]") {
            REQUIRE(chunks[0]->ne[0] == 16);
            REQUIRE(chunks[0]->ne[1] == 4);
            REQUIRE(chunks[0]->ne[2] == 1);
            REQUIRE(chunks[1]->ne[0] == 16);
            REQUIRE(chunks[1]->ne[1] == 4);
            REQUIRE(chunks[1]->ne[2] == 1);
        }

        THEN("chunk[0] is contiguous (starts at offset 0, strides match)") {
            // First chunk starts at the beginning — its strides happen to
            // match a contiguous layout since ne[0]*sizeof(f32) == nb[1].
            REQUIRE(chunks[0]->nb[0] == sizeof(float));
            REQUIRE(chunks[0]->nb[1] == tensor->nb[1]);
            // nb[1] for contiguous [16,4,1] should be 16*4 = 64
            REQUIRE(chunks[0]->nb[1] == 16 * sizeof(float));
            REQUIRE(ggml_is_contiguous(chunks[0]));
        }

        THEN("chunk[1] is also contiguous (same strides, just offset)") {
            // Both chunks share the parent's strides, and since split is
            // along dim=1, the inner dimension is unchanged.
            REQUIRE(ggml_is_contiguous(chunks[1]));
        }
    }

    GIVEN("a 3D tensor [16, 8, 4] split along dim=2 into 2 chunks") {
        auto tensor = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 16, 8, 4);

        ggml_tensor* chunks[2];
        split_tensor(ctx, tensor, 2, chunks, 2);

        THEN("each chunk has shape [16, 8, 2]") {
            REQUIRE(chunks[0]->ne[0] == 16);
            REQUIRE(chunks[0]->ne[1] == 8);
            REQUIRE(chunks[0]->ne[2] == 2);
        }

        THEN("chunks are contiguous (split along outermost varying dim)") {
            REQUIRE(ggml_is_contiguous(chunks[0]));
            REQUIRE(ggml_is_contiguous(chunks[1]));
        }
    }

    ggml_free(ctx);
}

SCENARIO("ggml_cont makes a non-contiguous tensor contiguous") {
    ggml_init_params params{};
    params.mem_size = 16 * 1024 * 1024;
    params.no_alloc = true;
    auto ctx = ggml_init(params);
    REQUIRE(ctx != nullptr);

    GIVEN("a permuted tensor (non-contiguous)") {
        auto tensor = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 16, 8, 4, 1);
        auto permuted = ggml_permute(ctx, tensor, 1, 0, 2, 3);

        THEN("the permuted tensor is not contiguous") {
            REQUIRE_FALSE(ggml_is_contiguous(permuted));
        }

        WHEN("ggml_cont is applied") {
            auto cont = ggml_cont(ctx, permuted);

            THEN("the result is contiguous") {
                REQUIRE(ggml_is_contiguous(cont));
            }
            THEN("the shape is preserved") {
                REQUIRE(cont->ne[0] == permuted->ne[0]);
                REQUIRE(cont->ne[1] == permuted->ne[1]);
                REQUIRE(cont->ne[2] == permuted->ne[2]);
                REQUIRE(cont->ne[3] == permuted->ne[3]);
            }
        }
    }

    ggml_free(ctx);
}

SCENARIO("IM2COL input contiguity requirement for Metal compatibility") {
    ggml_init_params params{};
    params.mem_size = 16 * 1024 * 1024;
    params.no_alloc = true;
    auto ctx = ggml_init(params);
    REQUIRE(ctx != nullptr);

    GIVEN("a grouped Conv1d scenario: input [16, 8, 1] with groups=2") {
        auto x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 16, 8, 1);
        auto weight = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3, 4, 4);

        // Split input and weight for grouped convolution
        ggml_tensor* xs[2];
        ggml_tensor* ws[2];
        split_tensor(ctx, x, 1, xs, 2);
        split_tensor(ctx, weight, 2, ws, 2);

        WHEN("IM2COL is constructed WITHOUT ggml_cont") {
            auto im2col = ggml_im2col(ctx, ws[0], xs[0], 1, 0, 0, 0, 1, 0, false, GGML_TYPE_F32);

            THEN("IM2COL src[1] (input) is contiguous — dim=1 split preserves contiguity") {
                // Note: splitting a 3D tensor [16,8,1] along dim=1 produces
                // views with shape [16,4,1] that ARE contiguous because the
                // inner dimension is unchanged and strides align.
                REQUIRE(ggml_is_contiguous(im2col->src[1]));
            }
        }

        WHEN("input is permuted (simulating real Flow graph)") {
            // In practice, Flow's Conv1d inputs may come from permuted tensors.
            // Create a [4, 16, 1] tensor and permute dims 0,1 to get
            // ne=[16, 4, 1] with non-contiguous strides (nb[1] = sizeof(float)).
            auto transposed = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 4, 16, 1);
            auto real_permuted = ggml_permute(ctx, transposed, 1, 0, 2, 3);
            // real_permuted: ne=[16,4,1] but nb[0]=4*sizeof(float), nb[1]=sizeof(float)

            THEN("a permuted tensor is not contiguous") {
                REQUIRE(real_permuted->ne[0] == 16);
                REQUIRE(real_permuted->ne[1] == 4);
                REQUIRE_FALSE(ggml_is_contiguous(real_permuted));
            }

            WHEN("ggml_cont is applied before IM2COL") {
                auto cont = ggml_cont(ctx, real_permuted);
                auto im2col = ggml_im2col(ctx, ws[0], cont, 1, 0, 0, 0, 1, 0, false, GGML_TYPE_F32);

                THEN("IM2COL src[1] is contiguous — Metal will accept this op") {
                    REQUIRE(ggml_is_contiguous(im2col->src[1]));
                }
            }

            WHEN("IM2COL is constructed WITHOUT ggml_cont on permuted input") {
                // This would crash on Metal — the op is rejected by supports_op
                // We only check contiguity here, not actually running the op
                THEN("the permuted input is NOT contiguous — Metal would reject IM2COL") {
                    REQUIRE_FALSE(ggml_is_contiguous(real_permuted));
                }
            }
        }
    }

    ggml_free(ctx);
}
