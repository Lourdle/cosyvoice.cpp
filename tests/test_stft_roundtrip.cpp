// STFT/ISTFT roundtrip tests — verifies that STFT → ISTFT reconstructs
// the original signal within acceptable tolerance.
//
// These tests run on CPU backend and exercise the same CUSTOM op code paths
// used in HiFT vocoder synthesis. They help isolate whether signal attenuation
// in HiFT comes from ISTFT implementation issues or from upstream graph errors.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <ggml.h>
#include <ggml-cpu.h>
#include <ggml-backend.h>

#include "ggml-fft.h"

#include <cmath>
#include <memory>
#include <numeric>
#include <vector>

namespace {

struct stft_test_fixture {
    ggml_backend_t backend;
    ggml_backend_sched_t sched;
    ggml_context* ctx;
    ggml_backend_buffer_t buf;

    static constexpr int NFFT = 16;
    static constexpr int HOP_LEN = NFFT / 4;

    fft_context_ptr fft_ctx;
    istft_context_ptr istft_ctx;

    stft_test_fixture() {
        backend = ggml_backend_cpu_init();
        REQUIRE(backend != nullptr);

        sched = ggml_backend_sched_new(&backend, nullptr, 1, GGML_DEFAULT_GRAPH_SIZE, false, true);
        REQUIRE(sched != nullptr);

        ggml_init_params params{};
        params.mem_size = 64 * 1024 * 1024;
        params.no_alloc = true;
        ctx = ggml_init(params);
        REQUIRE(ctx != nullptr);

        fft_ctx = create_fft_context(NFFT);
        REQUIRE(fft_ctx != nullptr);

        // Create ISTFT context — needs tensor allocation for IDFT matrices
        auto buft = ggml_backend_get_default_buffer_type(backend);
        auto istft_model_ctx = ggml_init(ggml_init_params{ .mem_size = 4 * 1024 * 1024, .no_alloc = true });
        istft_ctx = create_istft_context(NFFT, istft_model_ctx, [&](ggml_tensor* t, void* data, size_t size) {
            auto b = ggml_backend_alloc_ctx_tensors_from_buft(istft_model_ctx, buft);
            ggml_backend_tensor_set(t, data, 0, size);
            buf = b;
        });
        REQUIRE(istft_ctx != nullptr);
    }

    ~stft_test_fixture() {
        if (buf) ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        ggml_backend_sched_free(sched);
        ggml_backend_free(backend);
    }

    // Generate a simple sine wave signal
    std::vector<float> make_sine(int length, float freq, float amplitude = 1.0f) {
        std::vector<float> signal(length);
        for (int i = 0; i < length; i++)
            signal[i] = amplitude * std::sin(2.0f * 3.14159265f * freq * i / length);
        return signal;
    }

    // Generate a Hann window
    std::vector<float> make_hann_window(int size) {
        std::vector<float> window(size);
        for (int i = 0; i < size; i++)
            window[i] = 0.5f * (1.0f - std::cos(2.0f * 3.14159265f * i / size));
        return window;
    }
};

} // anonymous namespace

SCENARIO("FFT of a known signal produces expected magnitude") {
    stft_test_fixture f;

    GIVEN("a DC signal (all ones)") {
        const int n = f.NFFT;
        std::vector<float> signal(n, 1.0f);
        std::vector<float> output(n);

        fft(signal.data(), output.data(), *f.fft_ctx);

        THEN("DC bin has magnitude equal to nfft") {
            REQUIRE(output[0] == Catch::Approx(static_cast<float>(n)).margin(0.01f));
        }
        THEN("all other bins are near zero") {
            for (int i = 1; i < n; i++)
                REQUIRE(output[i] == Catch::Approx(0.0f).margin(0.01f));
        }
    }

    GIVEN("a sine wave at bin frequency") {
        const int n = f.NFFT;
        const int target_bin = 2;
        std::vector<float> signal(n);
        for (int i = 0; i < n; i++)
            signal[i] = std::sin(2.0f * 3.14159265f * target_bin * i / n);
        std::vector<float> output(n);

        fft(signal.data(), output.data(), *f.fft_ctx);

        THEN("target bin has peak magnitude") {
            float max_val = *std::max_element(output.begin(), output.end());
            REQUIRE(output[target_bin] == Catch::Approx(max_val).margin(0.01f));
        }
        THEN("target bin magnitude is approximately n/2") {
            REQUIRE(output[target_bin] == Catch::Approx(static_cast<float>(n) / 2.0f).margin(0.1f));
        }
    }
}

SCENARIO("IM2COL F16 overflow detection for Conv1d") {
    ggml_init_params params{};
    params.mem_size = 16 * 1024 * 1024;
    params.no_alloc = true;
    auto ctx = ggml_init(params);
    REQUIRE(ctx != nullptr);

    GIVEN("IM2COL output type is F16") {
        const float f16_max = 65504.0f;

        THEN("values exceeding F16 max would produce inf") {
            // F16 can represent up to 65504. Values like 99142231654400 (10^14)
            // seen in HiFT IM2COL dumps indicate F16 overflow.
            // This is a documentation test — the actual fix requires F32 intermediate.
            REQUIRE(f16_max < 100000.0f);
            REQUIRE(std::isinf(static_cast<float>(static_cast<__fp16>(100000.0f))));
        }

        THEN("values within F16 range are preserved") {
            float val = 107.5f;
            float roundtripped = static_cast<float>(static_cast<__fp16>(val));
            REQUIRE(roundtripped == Catch::Approx(val).margin(0.5f));
        }
    }

    ggml_free(ctx);
}

SCENARIO("STFT output contiguity determines whether ggml_cont is needed") {
    stft_test_fixture f;

    GIVEN("a sine wave signal and Hann window") {
        const int sig_len = f.NFFT * 4;
        auto signal = f.make_sine(sig_len, 3.0f, 1.0f);
        auto window = f.make_hann_window(f.NFFT);

        auto sig_tensor = ggml_new_tensor_1d(f.ctx, GGML_TYPE_F32, sig_len);
        auto win_tensor = ggml_new_tensor_1d(f.ctx, GGML_TYPE_F32, f.NFFT);

        WHEN("STFT graph is built and computed with ggml_cont") {
            auto gf = ggml_new_graph_custom(f.ctx, GGML_DEFAULT_GRAPH_SIZE, false);
            auto stft_result = ggml_stft(f.ctx, sig_tensor, win_tensor, f.HOP_LEN, true, f.fft_ctx.get());

            // Always use ggml_cont to ensure contiguous output (safe for both old and new code)
            auto contiguous = ggml_cont(f.ctx, stft_result);
            auto reshaped = ggml_reshape_2d(f.ctx, contiguous,
                contiguous->ne[0], contiguous->ne[1] * 2);
            ggml_build_forward_expand(gf, reshaped);

            ggml_backend_sched_reset(f.sched);
            bool alloc_ok = ggml_backend_sched_alloc_graph(f.sched, gf);
            REQUIRE(alloc_ok);

            ggml_backend_tensor_set(sig_tensor, signal.data(), 0, sig_len * sizeof(float));
            ggml_backend_tensor_set(win_tensor, window.data(), 0, f.NFFT * sizeof(float));

            auto status = ggml_backend_sched_graph_compute(f.sched, gf);
            REQUIRE(status == GGML_STATUS_SUCCESS);

            THEN("reshaped STFT output has non-zero values on CPU") {
                auto n = ggml_nelements(reshaped);
                auto buf = std::make_unique<float[]>(n);
                ggml_backend_tensor_get(reshaped, buf.get(), 0, n * sizeof(float));
                float max_abs = 0;
                for (int64_t i = 0; i < n; i++)
                    if (std::abs(buf[i]) > max_abs) max_abs = std::abs(buf[i]);
                REQUIRE(max_abs > 0.01f);
            }

            THEN("STFT output shape is [n_frames, nfft/2+1, 2]") {
                REQUIRE(stft_result->ne[2] == 2);
                REQUIRE(stft_result->ne[1] == f.NFFT / 2 + 1);
                REQUIRE(stft_result->ne[0] > 0);
            }
        }
    }
}

SCENARIO("STFT followed by ISTFT reconstructs the original signal") {
    stft_test_fixture f;

    GIVEN("a sine wave signal and Hann window") {
        const int sig_len = f.NFFT * 8;
        auto signal = f.make_sine(sig_len, 2.0f, 0.5f);
        auto window = f.make_hann_window(f.NFFT);

        auto sig_tensor = ggml_new_tensor_1d(f.ctx, GGML_TYPE_F32, sig_len);
        auto win_tensor = ggml_new_tensor_1d(f.ctx, GGML_TYPE_F32, f.NFFT);

        WHEN("signal passes through STFT then ISTFT") {
            auto gf = ggml_new_graph_custom(f.ctx, GGML_DEFAULT_GRAPH_SIZE, false);
            auto stft_result = ggml_stft(f.ctx, sig_tensor, win_tensor, f.HOP_LEN, true, f.fft_ctx.get());

            // Ensure contiguous before splitting (required for old non-contiguous STFT)
            stft_result = ggml_cont(f.ctx, stft_result);

            // Split STFT output [n_frames, nfft/2+1, 2] into real and imaginary
            auto n_frames = stft_result->ne[0];
            auto n_freq = stft_result->ne[1];
            auto real_part = ggml_view_3d(f.ctx, stft_result,
                n_frames, n_freq, 1,
                stft_result->nb[1], stft_result->nb[2], 0);
            auto imag_part = ggml_view_3d(f.ctx, stft_result,
                n_frames, n_freq, 1,
                stft_result->nb[1], stft_result->nb[2], stft_result->nb[2]);

            auto reconstructed = ggml_istft(f.ctx, real_part, imag_part,
                win_tensor, f.HOP_LEN, true, f.istft_ctx.get());
            ggml_build_forward_expand(gf, reconstructed);

            ggml_backend_sched_reset(f.sched);
            bool alloc_ok = ggml_backend_sched_alloc_graph(f.sched, gf);
            REQUIRE(alloc_ok);

            ggml_backend_tensor_set(sig_tensor, signal.data(), 0, sig_len * sizeof(float));
            ggml_backend_tensor_set(win_tensor, window.data(), 0, f.NFFT * sizeof(float));

            auto status = ggml_backend_sched_graph_compute(f.sched, gf);
            REQUIRE(status == GGML_STATUS_SUCCESS);

            THEN("reconstructed signal has similar amplitude to original") {
                auto out_len = ggml_nelements(reconstructed);
                auto out_buf = std::make_unique<float[]>(out_len);
                ggml_backend_tensor_get(reconstructed, out_buf.get(), 0, out_len * sizeof(float));

                float out_max = 0;
                for (int64_t i = 0; i < out_len; i++)
                    if (std::abs(out_buf[i]) > out_max) out_max = std::abs(out_buf[i]);

                float in_max = *std::max_element(signal.begin(), signal.end(),
                    [](float a, float b) { return std::abs(a) < std::abs(b); });
                in_max = std::abs(in_max);

                // Reconstructed amplitude should be within 50% of original
                REQUIRE(out_max > in_max * 0.5f);
                REQUIRE(out_max < in_max * 2.0f);
            }

            THEN("reconstructed signal matches original in the center region") {
                auto out_len = ggml_nelements(reconstructed);
                auto out_buf = std::make_unique<float[]>(out_len);
                ggml_backend_tensor_get(reconstructed, out_buf.get(), 0, out_len * sizeof(float));

                // Compare center region (avoid edge effects from centering)
                int margin = f.NFFT;
                int compare_len = std::min<int>(sig_len - 2 * margin, out_len - 2 * margin);
                REQUIRE(compare_len > 0);

                float max_err = 0;
                for (int i = 0; i < compare_len; i++) {
                    float err = std::abs(signal[margin + i] - out_buf[margin + i]);
                    if (err > max_err) max_err = err;
                }
                // Allow 10% relative error for roundtrip
                float in_max = 0.5f; // amplitude of sine
                REQUIRE(max_err < in_max * 0.1f);
            }
        }
    }
}

SCENARIO("STFT produces non-zero complex output for a sine wave") {
    stft_test_fixture f;

    GIVEN("a sine wave signal and Hann window") {
        // Signal length must be > nfft for STFT framing
        const int sig_len = f.NFFT * 4;
        auto signal = f.make_sine(sig_len, 3.0f, 1.0f);
        auto window = f.make_hann_window(f.NFFT);

        auto sig_tensor = ggml_new_tensor_1d(f.ctx, GGML_TYPE_F32, sig_len);
        auto win_tensor = ggml_new_tensor_1d(f.ctx, GGML_TYPE_F32, f.NFFT);

        WHEN("STFT graph is built and computed") {
            auto gf = ggml_new_graph_custom(f.ctx, GGML_DEFAULT_GRAPH_SIZE, false);
            auto stft_result = ggml_stft(f.ctx, sig_tensor, win_tensor, f.HOP_LEN, true, f.fft_ctx.get());
            ggml_build_forward_expand(gf, stft_result);

            ggml_backend_sched_reset(f.sched);
            bool alloc_ok = ggml_backend_sched_alloc_graph(f.sched, gf);
            REQUIRE(alloc_ok);

            ggml_backend_tensor_set(sig_tensor, signal.data(), 0, sig_len * sizeof(float));
            ggml_backend_tensor_set(win_tensor, window.data(), 0, f.NFFT * sizeof(float));

            auto status = ggml_backend_sched_graph_compute(f.sched, gf);
            REQUIRE(status == GGML_STATUS_SUCCESS);

            THEN("STFT output has expected shape [n_frames, nfft/2+1, 2]") {
                // ne[2] == 2 for real/imag
                REQUIRE(stft_result->ne[2] == 2);
                REQUIRE(stft_result->ne[1] == f.NFFT / 2 + 1);
                REQUIRE(stft_result->ne[0] > 0);
            }

            THEN("STFT output contains non-zero values") {
                auto n = ggml_nbytes(stft_result);
                auto buf = std::make_unique<char[]>(n);
                ggml_backend_tensor_get(stft_result, buf.get(), 0, n);
                auto fp = reinterpret_cast<float*>(buf.get());
                float max_abs = 0;
                for (size_t i = 0; i < n / sizeof(float); i++)
                    if (std::abs(fp[i]) > max_abs) max_abs = std::abs(fp[i]);
                REQUIRE(max_abs > 0.01f);
            }
        }
    }
}
