#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "cosyvoice.h"
#include "cosyvoice-lowlevel.h"

#include <ggml.h>
#include <ggml-backend.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <vector>

// ---- Helpers ----------------------------------------------------------------

static std::vector<float> load_binary_floats(const char* path)
{
    FILE* f = fopen(path, "rb");
    if (!f) return {};
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<float> buf(size / sizeof(float));
    if (!buf.empty())
        fread(buf.data(), sizeof(float), buf.size(), f);
    fclose(f);
    return buf;
}

static std::vector<int32_t> load_binary_int32(const char* path)
{
    FILE* f = fopen(path, "rb");
    if (!f) return {};
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<int32_t> buf(size / sizeof(int32_t));
    if (!buf.empty())
        fread(buf.data(), sizeof(int32_t), buf.size(), f);
    fclose(f);
    return buf;
}

static float max_abs(const std::vector<float>& v)
{
    float m = 0;
    for (auto x : v) m = std::max(m, std::abs(x));
    return m;
}

static float mean(const std::vector<float>& v)
{
    if (v.empty()) return 0;
    return std::accumulate(v.begin(), v.end(), 0.0f) / (float)v.size();
}

static float cosine_similarity(const std::vector<float>& a, const std::vector<float>& b)
{
    if (a.size() != b.size() || a.empty()) return 0;
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += (double)a[i] * b[i];
        na += (double)a[i] * a[i];
        nb += (double)b[i] * b[i];
    }
    if (na == 0 || nb == 0) return 0;
    return (float)(dot / (std::sqrt(na) * std::sqrt(nb)));
}

// ---- Test data paths (from Lourdle's test kit at /tmp/cv/) ------------------
// These tests require the test kit to be present. They will be skipped if
// the files are not found.

static const char* TEST_KIT_DIR = "/tmp/cv";
static const char* MODEL_PATH = nullptr; // Set via env or skip

static const char* get_model_path()
{
    static const char* path = std::getenv("COSYVOICE_TEST_MODEL");
    return path;
}

static bool test_kit_available()
{
    auto tokens = load_binary_int32("/tmp/cv/flow_input_token.bin");
    return !tokens.empty();
}

// ---- Tests ------------------------------------------------------------------

SCENARIO("flow test kit data has expected shapes") {
    if (!test_kit_available()) {
        SKIP("Test kit not found at /tmp/cv/");
    }

    GIVEN("Lourdle's flow test kit files") {
        auto tokens = load_binary_int32("/tmp/cv/flow_input_token.bin");
        auto prompt_tokens = load_binary_int32("/tmp/cv/flow_input_prompt_token.bin");
        auto noise = load_binary_floats("/tmp/cv/flow_input_noise.bin");
        auto prompt_feat = load_binary_floats("/tmp/cv/flow_input_prompt_feat.bin");
        auto embedding = load_binary_floats("/tmp/cv/flow_input_embedding.bin");
        auto cuda_output = load_binary_floats("/tmp/cv/flow_output_final.bin");
        auto hift_noise = load_binary_floats("/tmp/cv/hift_input_noise.bin");

        THEN("token file has 38 tokens") {
            REQUIRE(tokens.size() == 38);
        }
        THEN("prompt token file has 87 tokens") {
            REQUIRE(prompt_tokens.size() == 87);
        }
        THEN("flow noise has 20000 floats [250*80]") {
            REQUIRE(noise.size() == 20000);
        }
        THEN("prompt feat has 13920 floats [174*80]") {
            REQUIRE(prompt_feat.size() == 13920);
        }
        THEN("embedding has 192 floats [1*192]") {
            REQUIRE(embedding.size() == 192);
        }
        THEN("CUDA flow output has 6080 floats [76*80]") {
            REQUIRE(cuda_output.size() == 6080);
        }
        THEN("HiFT noise has 328320 floats") {
            REQUIRE(hift_noise.size() == 328320);
        }
    }
}

SCENARIO("CUDA reference flow output has expected statistics") {
    if (!test_kit_available()) {
        SKIP("Test kit not found at /tmp/cv/");
    }

    GIVEN("CUDA flow_output_final.bin reference") {
        auto cuda_output = load_binary_floats("/tmp/cv/flow_output_final.bin");

        THEN("max_abs is approximately 10.706") {
            REQUIRE_THAT(max_abs(cuda_output),
                Catch::Matchers::WithinAbs(10.706, 0.01));
        }
        THEN("mean is approximately -3.991") {
            REQUIRE_THAT(mean(cuda_output),
                Catch::Matchers::WithinAbs(-3.991, 0.01));
        }
        THEN("first 4 values match expected") {
            REQUIRE_THAT(cuda_output[0], Catch::Matchers::WithinAbs(-4.5809, 0.01));
            REQUIRE_THAT(cuda_output[1], Catch::Matchers::WithinAbs(-3.4415, 0.01));
            REQUIRE_THAT(cuda_output[2], Catch::Matchers::WithinAbs(-3.8687, 0.01));
            REQUIRE_THAT(cuda_output[3], Catch::Matchers::WithinAbs(-3.9511, 0.01));
        }
    }
}

SCENARIO("CPU flow output maintains minimum cosine similarity with CUDA reference") {
    if (!test_kit_available()) {
        SKIP("Test kit not found at /tmp/cv/");
    }

    GIVEN("a CPU-generated flow_output_final dump") {
        auto cpu_output = load_binary_floats("/tmp/cv_dump/flow_output_final.bin");
        auto cuda_output = load_binary_floats("/tmp/cv/flow_output_final.bin");

        if (cpu_output.empty()) {
            SKIP("CPU dump not found at /tmp/cv_dump/flow_output_final.bin "
                 "(run with COSYVOICE_DUMP_DIR=/tmp/cv_dump first)");
        }

        THEN("output shape matches CUDA reference") {
            REQUIRE(cpu_output.size() == cuda_output.size());
        }

        WHEN("comparing CPU vs CUDA flow output") {
            float cos_sim = cosine_similarity(cpu_output, cuda_output);
            float cpu_max = max_abs(cpu_output);
            float cpu_mean_val = mean(cpu_output);

            THEN("cosine similarity is at least 0.80 (regression gate)") {
                INFO("cosine_similarity = " << cos_sim);
                REQUIRE(cos_sim >= 0.80f);
            }
            THEN("output is not silence (max_abs > 1.0)") {
                REQUIRE(cpu_max > 1.0f);
            }
            THEN("output has negative mean (speech feature characteristic)") {
                REQUIRE(cpu_mean_val < 0.0f);
            }

            // This threshold will tighten as Flow bugs are fixed
            THEN("cosine similarity is tracked for improvement") {
                INFO("Current cosine_similarity = " << cos_sim);
                INFO("Target: >= 0.99 (matching CUDA)");
                INFO("Current CPU: max_abs=" << cpu_max << " mean=" << cpu_mean_val);
                INFO("CUDA ref:    max_abs=10.706 mean=-3.991");
                // Soft check — will become REQUIRE as we fix Flow
                CHECK(cos_sim >= 0.85f);
            }
        }
    }
}

SCENARIO("flow input noise has standard normal distribution characteristics") {
    if (!test_kit_available()) {
        SKIP("Test kit not found at /tmp/cv/");
    }

    GIVEN("flow_input_noise.bin from test kit") {
        auto noise = load_binary_floats("/tmp/cv/flow_input_noise.bin");

        THEN("mean is approximately zero") {
            REQUIRE_THAT(mean(noise), Catch::Matchers::WithinAbs(0.0, 0.05));
        }
        THEN("max_abs is within reasonable range for normal distribution") {
            // For N=20000 samples from N(0,1), max_abs should be < 5 almost always
            REQUIRE(max_abs(noise) < 5.0f);
        }
    }
}

SCENARIO("DiT t_span cosine schedule is correct") {
    GIVEN("the 11-point cosine schedule t_span[i] = 1 - cos(0.05*pi*i)") {
        std::array<float, 11> t_span;
        for (int i = 0; i < 11; ++i)
            t_span[i] = 1.f - std::cos(0.1f * 0.5f * 3.14159265358979323846f * i);

        THEN("t_span[0] is 0") {
            REQUIRE_THAT(t_span[0], Catch::Matchers::WithinAbs(0.0, 1e-6));
        }
        THEN("t_span[10] is 1") {
            REQUIRE_THAT(t_span[10], Catch::Matchers::WithinAbs(1.0, 1e-6));
        }
        THEN("schedule is monotonically increasing") {
            for (int i = 1; i < 11; ++i) {
                REQUIRE(t_span[i] > t_span[i-1]);
            }
        }
        THEN("dt values are positive and increasing") {
            for (int i = 2; i < 11; ++i) {
                float dt_prev = t_span[i-1] - t_span[i-2];
                float dt_curr = t_span[i] - t_span[i-1];
                REQUIRE(dt_curr > dt_prev);
            }
        }
    }
}
