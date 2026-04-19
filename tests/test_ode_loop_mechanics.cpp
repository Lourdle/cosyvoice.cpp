// Tests for ODE loop iteration mechanics on CPU.
//
// Background: All DiT modules match PyTorch individually (E2E tests pass),
// but the full ODE loop produces monotonic negative drift. These tests
// isolate the loop mechanisms to find the root cause.
//
// Test A: op_params dynamic patching — does ggml_fill_inplace and ggml_scale
//         correctly use updated op_params values on repeated graph_compute?
// Test B: (future) graph reuse vs rebuild comparison
// Test C: (future) 2-step ODE E2E vs PyTorch

#include <catch2/catch_test_macros.hpp>

#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>

#include <cmath>
#include <cstring>
#include <vector>

// ---- Test A: op_params patching ----

SCENARIO("Given a ggml graph with ggml_fill_inplace + ggml_scale, "
         "when op_params are patched and graph is re-computed, "
         "then output reflects the new parameter values",
         "[ode][op_params]") {

    ggml_init_params params = {
        .mem_size = 16 * 1024 * 1024,
        .mem_buffer = nullptr,
        .no_alloc = true,
    };
    auto* ctx = ggml_init(params);
    REQUIRE(ctx);

    // Build graph: y = x + scale(fill(t_val), dt_val)
    // This mirrors the ODE step: x_new = x_old + dt * dphi_dt
    // where t_val is injected via ggml_fill_inplace and dt via ggml_scale

    const int N = 100;
    auto* x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, N);
    ggml_set_input(x);
    ggml_set_name(x, "x");

    // ggml_fill_inplace: creates a tensor filled with a constant from op_params
    float t_val_init = 1.0f;
    auto* t_filled = ggml_fill_inplace(ctx,
        ggml_new_tensor_1d(ctx, GGML_TYPE_F32, N),
        t_val_init);
    ggml_set_name(t_filled, "t_filled");

    // ggml_scale: scales tensor by op_params[0]
    float dt_init = 0.5f;
    auto* scaled = ggml_scale(ctx, t_filled, dt_init);
    ggml_set_name(scaled, "scaled");

    auto* output = ggml_add(ctx, x, scaled);
    ggml_set_output(output);
    ggml_set_name(output, "output");

    auto* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, output);

    auto* backend = ggml_backend_cpu_init();
    REQUIRE(backend);
    auto* buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    REQUIRE(buf);

    // Fill x with 1.0
    std::vector<float> xdata(N, 1.0f);
    ggml_backend_tensor_set(x, xdata.data(), 0, N * sizeof(float));

    // --- First compute: t=1.0, dt=0.5 → output = 1.0 + 1.0*0.5 = 1.5 ---
    auto status = ggml_backend_graph_compute(backend, gf);
    REQUIRE(status == GGML_STATUS_SUCCESS);

    THEN("first compute produces expected output (x + t*dt = 1.0 + 1.0*0.5 = 1.5)") {
        std::vector<float> result(N);
        ggml_backend_tensor_get(output, result.data(), 0, N * sizeof(float));

        for (int i = 0; i < N; i++) {
            INFO("i=" << i << " got=" << result[i]);
            REQUIRE(std::abs(result[i] - 1.5f) < 1e-6f);
        }
    }

    // --- Patch op_params: t=2.0, dt=0.3 → expected output = 1.0 + 2.0*0.3 = 1.6 ---
    reinterpret_cast<float*>(t_filled->op_params)[0] = 2.0f;
    reinterpret_cast<float*>(scaled->op_params)[0] = 0.3f;

    status = ggml_backend_graph_compute(backend, gf);
    REQUIRE(status == GGML_STATUS_SUCCESS);

    THEN("second compute with patched op_params produces new output (1.0 + 2.0*0.3 = 1.6)") {
        std::vector<float> result(N);
        ggml_backend_tensor_get(output, result.data(), 0, N * sizeof(float));

        for (int i = 0; i < N; i++) {
            INFO("i=" << i << " got=" << result[i] << " expected=1.6");
            REQUIRE(std::abs(result[i] - 1.6f) < 1e-6f);
        }
    }

    // --- Patch again: t=3.0, dt=0.1 → expected = 1.0 + 3.0*0.1 = 1.3 ---
    reinterpret_cast<float*>(t_filled->op_params)[0] = 3.0f;
    reinterpret_cast<float*>(scaled->op_params)[0] = 0.1f;

    status = ggml_backend_graph_compute(backend, gf);
    REQUIRE(status == GGML_STATUS_SUCCESS);

    THEN("third compute with different op_params (1.0 + 3.0*0.1 = 1.3)") {
        std::vector<float> result(N);
        ggml_backend_tensor_get(output, result.data(), 0, N * sizeof(float));

        for (int i = 0; i < N; i++) {
            INFO("i=" << i << " got=" << result[i] << " expected=1.3");
            REQUIRE(std::abs(result[i] - 1.3f) < 1e-6f);
        }
    }

    ggml_backend_buffer_free(buf);
    ggml_backend_free(backend);
    ggml_free(ctx);
}


// ---- Test: tensor copy between iterations ----

SCENARIO("Given ggml_backend_tensor_copy_async on CPU, "
         "when output is copied back to input and graph re-computed, "
         "then the new computation uses the copied values",
         "[ode][tensor_copy]") {

    ggml_init_params params = {
        .mem_size = 16 * 1024 * 1024,
        .mem_buffer = nullptr,
        .no_alloc = true,
    };
    auto* ctx = ggml_init(params);
    REQUIRE(ctx);

    const int N = 100;
    auto* x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, N);
    ggml_set_input(x);
    ggml_set_name(x, "state");

    // y = x + 1.0 (simulates one ODE step adding a constant)
    auto* ones = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, N);
    ggml_set_input(ones);
    auto* output = ggml_add(ctx, x, ones);
    ggml_set_output(output);

    auto* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, output);

    auto* backend = ggml_backend_cpu_init();
    auto* buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    REQUIRE(buf);

    // Initialize: x = 0, ones = 1
    std::vector<float> zeros(N, 0.0f);
    std::vector<float> one_vec(N, 1.0f);
    ggml_backend_tensor_set(x, zeros.data(), 0, N * sizeof(float));
    ggml_backend_tensor_set(ones, one_vec.data(), 0, N * sizeof(float));

    // Step 1: output = 0 + 1 = 1
    REQUIRE(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);
    {
        std::vector<float> r(N);
        ggml_backend_tensor_get(output, r.data(), 0, N * sizeof(float));
        REQUIRE(std::abs(r[0] - 1.0f) < 1e-6f);
    }

    // Copy output back to input: x ← output
    ggml_backend_tensor_copy_async(backend, backend, output, x);

    // Step 2: output = 1 + 1 = 2
    REQUIRE(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);

    THEN("after copy-back and re-compute, output accumulates (0→1→2)") {
        std::vector<float> r(N);
        ggml_backend_tensor_get(output, r.data(), 0, N * sizeof(float));
        INFO("step2 output[0]=" << r[0] << " expected=2.0");
        REQUIRE(std::abs(r[0] - 2.0f) < 1e-6f);
    }

    // Copy and step 3: output = 2 + 1 = 3
    ggml_backend_tensor_copy_async(backend, backend, output, x);
    REQUIRE(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);

    THEN("step 3 accumulates correctly (2→3)") {
        std::vector<float> r(N);
        ggml_backend_tensor_get(output, r.data(), 0, N * sizeof(float));
        INFO("step3 output[0]=" << r[0] << " expected=3.0");
        REQUIRE(std::abs(r[0] - 3.0f) < 1e-6f);
    }

    ggml_backend_buffer_free(buf);
    ggml_backend_free(backend);
    ggml_free(ctx);
}


// ---- Test: graph reuse with ggml_backend_sched ----

// Bug: ggml_backend_sched aliases input tensor buffer with output tensor buffer.
// After first graph_compute, the output value overwrites the input tensor.
// On second graph_compute, the input reads the stale output value instead of
// the original input.
//
// Observed: x is set to 10.0. After compute, output=11.0 (correct).
// On second compute, x reads as 11.0 (previous output!) instead of 10.0.
// So output becomes 11 + 1 = 12 instead of 10 + 1 = 11.
//
// This does NOT occur with ggml_backend_graph_compute (direct, no sched).
// Workaround: call ggml_backend_sched_reset + realloc_graph before each compute.
SCENARIO("Given ggml_backend_sched with graph reuse, "
         "when the graph is computed twice without reset, "
         "then input tensor must retain its original value (not be aliased with output)",
         "[ode][sched_buffer_alias][!shouldfail]") {

    ggml_init_params params = {
        .mem_size = 16 * 1024 * 1024,
        .mem_buffer = nullptr,
        .no_alloc = true,
    };
    auto* ctx = ggml_init(params);
    REQUIRE(ctx);

    const int N = 100;
    auto* x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, N);
    ggml_set_input(x);
    ggml_set_name(x, "input_x");

    // Graph: output = x + scale(fill(2.0), 0.5) = x + 1.0
    auto* filled = ggml_fill_inplace(ctx,
        ggml_new_tensor_1d(ctx, GGML_TYPE_F32, N), 2.0f);
    auto* scaled = ggml_scale(ctx, filled, 0.5f);
    ggml_set_output(filled);
    ggml_set_output(scaled);
    auto* output = ggml_add(ctx, x, scaled);
    ggml_set_output(output);
    ggml_set_name(output, "output");

    auto* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, output);

    auto* backend = ggml_backend_cpu_init();
    ggml_backend_t backends[] = { backend };
    auto* sched = ggml_backend_sched_new(backends, nullptr, 1, 4096, false, false);
    ggml_backend_sched_alloc_graph(sched, gf);

    std::vector<float> xdata(N, 10.0f);
    ggml_backend_tensor_set(x, xdata.data(), 0, N * sizeof(float));

    // First compute: x=10, output = 10 + 2*0.5 = 11 (always correct)
    REQUIRE(ggml_backend_sched_graph_compute(sched, gf) == GGML_STATUS_SUCCESS);
    {
        std::vector<float> r(N);
        ggml_backend_tensor_get(output, r.data(), 0, N * sizeof(float));
        REQUIRE(std::abs(r[0] - 11.0f) < 1e-5f);
    }

    // Second compute WITHOUT resetting x — x should still be 10.0
    REQUIRE(ggml_backend_sched_graph_compute(sched, gf) == GGML_STATUS_SUCCESS);

    THEN("input tensor x retains its value after re-compute (not aliased with output)") {
        std::vector<float> x_val(N), out_val(N);
        ggml_backend_tensor_get(x, x_val.data(), 0, N * sizeof(float));
        ggml_backend_tensor_get(output, out_val.data(), 0, N * sizeof(float));

        INFO("x[0]=" << x_val[0] << " (expected 10.0, bug gives 11.0 from previous output)");
        INFO("output[0]=" << out_val[0] << " (expected 11.0, bug gives 12.0)");

        // The core assertion: input must not be corrupted
        CHECK(std::abs(x_val[0] - 10.0f) < 1e-5f);
        // Consequence: output must remain correct
        CHECK(std::abs(out_val[0] - 11.0f) < 1e-5f);
    }

    ggml_backend_sched_free(sched);
    ggml_backend_free(backend);
    ggml_free(ctx);
}


SCENARIO("Given ggml_backend_sched with reset+realloc between computes, "
         "when op_params are patched and graph is re-allocated each time, "
         "then output correctly reflects new values (workaround)",
         "[ode][sched_reset]") {

    ggml_init_params params = {
        .mem_size = 16 * 1024 * 1024,
        .mem_buffer = nullptr,
        .no_alloc = true,
    };
    auto* ctx = ggml_init(params);
    REQUIRE(ctx);

    const int N = 100;
    auto* x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, N);
    ggml_set_input(x);

    float fill_val = 2.0f;
    auto* filled = ggml_fill_inplace(ctx,
        ggml_new_tensor_1d(ctx, GGML_TYPE_F32, N), fill_val);

    float scale_val = 0.5f;
    auto* scaled = ggml_scale(ctx, filled, scale_val);
    auto* output = ggml_add(ctx, x, scaled);
    ggml_set_output(output);

    auto* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, output);

    auto* backend = ggml_backend_cpu_init();
    ggml_backend_t backends[] = { backend };
    auto* sched = ggml_backend_sched_new(backends, nullptr, 1, 4096, false, false);

    ggml_backend_sched_alloc_graph(sched, gf);

    std::vector<float> xdata(N, 10.0f);
    ggml_backend_tensor_set(x, xdata.data(), 0, N * sizeof(float));

    // First compute: 10 + 2*0.5 = 11
    REQUIRE(ggml_backend_sched_graph_compute(sched, gf) == GGML_STATUS_SUCCESS);
    {
        std::vector<float> r(N);
        ggml_backend_tensor_get(output, r.data(), 0, N * sizeof(float));
        REQUIRE(std::abs(r[0] - 11.0f) < 1e-5f);
    }

    // Patch: fill=5.0, scale=0.2 → expected 10 + 5*0.2 = 11
    reinterpret_cast<float*>(filled->op_params)[0] = 5.0f;
    reinterpret_cast<float*>(scaled->op_params)[0] = 0.2f;

    // WORKAROUND: reset + realloc before re-compute
    ggml_backend_sched_reset(sched);
    ggml_backend_sched_alloc_graph(sched, gf);
    ggml_backend_tensor_set(x, xdata.data(), 0, N * sizeof(float));

    REQUIRE(ggml_backend_sched_graph_compute(sched, gf) == GGML_STATUS_SUCCESS);

    THEN("with reset+realloc, patched op_params produce correct result (10 + 5*0.2 = 11)") {
        std::vector<float> r(N);
        ggml_backend_tensor_get(output, r.data(), 0, N * sizeof(float));
        INFO("got=" << r[0] << " expected=11.0");
        REQUIRE(std::abs(r[0] - 11.0f) < 1e-5f);
    }

    // Patch again: fill=3.0, scale=2.0 → expected 10 + 3*2 = 16
    reinterpret_cast<float*>(filled->op_params)[0] = 3.0f;
    reinterpret_cast<float*>(scaled->op_params)[0] = 2.0f;

    ggml_backend_sched_reset(sched);
    ggml_backend_sched_alloc_graph(sched, gf);
    ggml_backend_tensor_set(x, xdata.data(), 0, N * sizeof(float));

    REQUIRE(ggml_backend_sched_graph_compute(sched, gf) == GGML_STATUS_SUCCESS);

    THEN("second patch with reset+realloc (10 + 3*2 = 16)") {
        std::vector<float> r(N);
        ggml_backend_tensor_get(output, r.data(), 0, N * sizeof(float));
        INFO("got=" << r[0] << " expected=16.0");
        REQUIRE(std::abs(r[0] - 16.0f) < 1e-5f);
    }

    ggml_backend_sched_free(sched);
    ggml_backend_free(backend);
    ggml_free(ctx);
}


// ---- Test C: 2-step ODE with iterative state accumulation ----
// Simulates the actual ODE loop: x_{n+1} = x_n + dt_n * f(x_n, t_n)
// Uses sched with graph reuse (as token2wav does) and checks if accumulated
// state matches the expected values.

SCENARIO("Given a 3-step ODE simulation using sched graph reuse, "
         "when state is copied back and t/dt are patched each step, "
         "then accumulated state matches manual calculation",
         "[ode][ode_loop]") {

    ggml_init_params params = {
        .mem_size = 16 * 1024 * 1024,
        .mem_buffer = nullptr,
        .no_alloc = true,
    };
    auto* ctx = ggml_init(params);
    REQUIRE(ctx);

    const int N = 10;

    // State tensor (like ditctx.x)
    auto* state = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, N);
    ggml_set_input(state);
    ggml_set_name(state, "state");

    // Simulated "velocity field" f(x,t) = t (constant w.r.t. x for simplicity)
    // So x_{n+1} = x_n + dt_n * t_n
    float t_init = 1.0f;
    auto* t_filled = ggml_fill_inplace(ctx,
        ggml_new_tensor_1d(ctx, GGML_TYPE_F32, N), t_init);
    ggml_set_name(t_filled, "t_filled");

    float dt_init = 0.1f;
    auto* step_result = ggml_scale(ctx, t_filled, dt_init);
    ggml_set_name(step_result, "dt_scaled");

    auto* output = ggml_add(ctx, state, step_result);
    ggml_set_output(output);
    ggml_set_name(output, "next_state");

    auto* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, output);

    auto* backend = ggml_backend_cpu_init();
    ggml_backend_t backends[] = { backend };
    auto* sched = ggml_backend_sched_new(backends, nullptr, 1, 4096, false, false);
    ggml_backend_sched_alloc_graph(sched, gf);

    // Initialize state = 0
    std::vector<float> zeros(N, 0.0f);
    ggml_backend_tensor_set(state, zeros.data(), 0, N * sizeof(float));

    // Schedule: t = [1.0, 2.0, 3.0], dt = [0.1, 0.2, 0.3]
    float t_vals[]  = {1.0f, 2.0f, 3.0f};
    float dt_vals[] = {0.1f, 0.2f, 0.3f};

    // Expected accumulation:
    // step 0: state = 0 + 1.0*0.1 = 0.1
    // step 1: state = 0.1 + 2.0*0.2 = 0.5
    // step 2: state = 0.5 + 3.0*0.3 = 1.4
    float expected[] = {0.1f, 0.5f, 1.4f};

    std::vector<float> actual_per_step;

    for (int step = 0; step < 3; step++) {
        if (step > 0) {
            // Copy output back to state (like token2wav does)
            ggml_backend_tensor_copy_async(backend, backend, output, state);

            // Patch t and dt
            reinterpret_cast<float*>(t_filled->op_params)[0] = t_vals[step];
            reinterpret_cast<float*>(step_result->op_params)[0] = dt_vals[step];
        }

        REQUIRE(ggml_backend_sched_graph_compute(sched, gf) == GGML_STATUS_SUCCESS);

        std::vector<float> r(N);
        ggml_backend_tensor_get(output, r.data(), 0, N * sizeof(float));
        actual_per_step.push_back(r[0]);
    }

    THEN("accumulated state matches manual calculation at each step") {
        for (int step = 0; step < 3; step++) {
            INFO("step " << step << ": got=" << actual_per_step[step]
                 << " expected=" << expected[step]
                 << " t=" << t_vals[step] << " dt=" << dt_vals[step]);
            CHECK(std::abs(actual_per_step[step] - expected[step]) < 1e-5f);
        }
    }

    ggml_backend_sched_free(sched);
    ggml_backend_free(backend);
    ggml_free(ctx);
}

// Issue #11: sched_reset + alloc_graph with external-buffer leaf tensor
// https://github.com/Lourdle/cosyvoice.cpp/issues/11
SCENARIO("Given sched_reset, when alloc_graph references external-buffer leaf, "
         "gallocr workaround computes correctly",
         "[ode][sched_gallocr_workaround]") {
    auto* backend = ggml_backend_cpu_init();
    REQUIRE(backend);

    // ctx1: external buffer context (simulates token2wav_buffer)
    auto* ctx1 = ggml_init({.mem_size = ggml_tensor_overhead() * 4, .no_alloc = true});
    auto* X = ggml_new_tensor_2d(ctx1, GGML_TYPE_F32, 20, 10);
    ggml_set_input(X);
    auto* ext_buf = ggml_backend_alloc_ctx_tensors(ctx1, backend);
    REQUIRE(ext_buf);

    // Fill X with 1.0
    std::vector<float> ones(200, 1.0f);
    ggml_backend_tensor_set(X, ones.data(), 0, ones.size() * sizeof(float));

    // --- Graph 1: scale X by 2.0 via sched ---
    auto* ctx0 = ggml_init({.mem_size = ggml_tensor_overhead() * 64 + ggml_graph_overhead() * 2, .no_alloc = true});
    auto* sched = ggml_backend_sched_new(&backend, nullptr, 1, GGML_DEFAULT_GRAPH_SIZE, true, true);

    auto* gf1 = ggml_new_graph(ctx0);
    auto* Y1 = ggml_scale(ctx0, X, 2.0f);
    ggml_set_output(Y1);
    ggml_build_forward_expand(gf1, Y1);
    ggml_backend_sched_alloc_graph(sched, gf1);
    REQUIRE(ggml_backend_sched_graph_compute(sched, gf1) == GGML_STATUS_SUCCESS);

    // Verify graph 1 result
    std::vector<float> result1(200);
    ggml_backend_tensor_get(Y1, result1.data(), 0, result1.size() * sizeof(float));
    CHECK(result1[0] == 2.0f);

    // --- sched_reset + rebuild graph ---
    ggml_backend_sched_reset(sched);
    ggml_free(ctx0);

    ctx0 = ggml_init({.mem_size = ggml_tensor_overhead() * 64 + ggml_graph_overhead() * 2, .no_alloc = true});
    auto* gf2 = ggml_new_graph(ctx0);
    auto* Y2 = ggml_scale(ctx0, X, 3.0f);  // X is still in ext_buf
    ggml_set_output(Y2);
    ggml_build_forward_expand(gf2, Y2);

    // NOTE: ggml_backend_sched_alloc_graph(sched, gf2) may trigger
    // buffer_id=-1 for X (Issue #11). Use gallocr workaround instead.

    SECTION("gallocr workaround computes correctly after sched_reset") {
        auto* gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        REQUIRE(ggml_gallocr_alloc_graph(gallocr, gf2));
        REQUIRE(ggml_backend_graph_compute(backend, gf2) == GGML_STATUS_SUCCESS);

        std::vector<float> result2(200);
        ggml_backend_tensor_get(Y2, result2.data(), 0, result2.size() * sizeof(float));
        CHECK(result2[0] == 3.0f);
        CHECK(result2[199] == 3.0f);

        ggml_gallocr_free(gallocr);
    }

    ggml_backend_sched_free(sched);
    ggml_free(ctx0);
    ggml_backend_buffer_free(ext_buf);
    ggml_free(ctx1);
    ggml_backend_free(backend);
}
