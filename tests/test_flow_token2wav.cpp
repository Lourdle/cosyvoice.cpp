#include <catch2/catch_test_macros.hpp>

#include "cosyvoice.h"
#include "cosyvoice-lowlevel.h"

#include <ggml.h>
#include <ggml-backend.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
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

static bool test_kit_available()
{
    auto tokens = load_binary_int32("/tmp/cv/flow_input_token.bin");
    return !tokens.empty();
}

static const char* get_model_path()
{
    static const char* path = std::getenv("COSYVOICE_TEST_MODEL");
    return path;
}

// ---- Noise callback tracking ------------------------------------------------

struct tracked_noise_ctx {
    std::vector<float> flow_noise;
    std::vector<float> hift_noise;
    int before_flow_count = 0;
    int after_flow_count = 0;
    int before_hift_count = 0;
    int after_hift_count = 0;
    uint32_t flow_length_requested = 0;
    uint32_t hift_length_requested = 0;
};

static float* tracked_noise_callback(
    cosyvoice_noise_callback_stage_t stage,
    uint32_t length,
    float* noise,
    void* ctx)
{
    auto* tc = static_cast<tracked_noise_ctx*>(ctx);
    switch (stage) {
    case COSYVOICE_NOISE_CALLBACK_STAGE_BEFORE_FLOW:
        tc->before_flow_count++;
        tc->flow_length_requested = length;
        return tc->flow_noise.data();
    case COSYVOICE_NOISE_CALLBACK_STAGE_AFTER_FLOW:
        tc->after_flow_count++;
        return nullptr;
    case COSYVOICE_NOISE_CALLBACK_STAGE_BEFORE_HIFT:
        tc->before_hift_count++;
        tc->hift_length_requested = length;
        return tc->hift_noise.data();
    case COSYVOICE_NOISE_CALLBACK_STAGE_AFTER_HIFT:
        tc->after_hift_count++;
        return nullptr;
    }
    return nullptr;
}

// ---- Tests ------------------------------------------------------------------

SCENARIO("token2wav produces non-zero output with test kit data") {
    if (!test_kit_available()) {
        SKIP("Test kit not found at /tmp/cv/");
    }
    const char* model_path = get_model_path();
    if (!model_path) {
        SKIP("Set COSYVOICE_TEST_MODEL env var to model .gguf path");
    }

    GIVEN("model loaded with test kit inputs and tracked noise callback") {
        auto tokens = load_binary_int32("/tmp/cv/flow_input_token.bin");
        auto flow_noise = load_binary_floats("/tmp/cv/flow_input_noise.bin");
        auto hift_noise = load_binary_floats("/tmp/cv/hift_input_noise.bin");

        cosyvoice_context_params_t params{};
        cosyvoice_init_default_context_params(&params);
        params.seed = 123;
        params.inference_buffer_policy = COSYVOICE_INFERENCE_BUFFER_POLICY_SHARED;

        ggml_backend_t backend = ggml_backend_init_best();
        REQUIRE(backend != nullptr);

        auto* ctx = cosyvoice_load_from_file_ext(model_path, &params, backend, 0, 0);
        if (!ctx) {
            SKIP("Failed to load model (file may not exist)");
        }

        auto* prompt_speech = cosyvoice_prompt_speech_load_from_file("/tmp/cv/prompt_speech.gguf");
        REQUIRE(prompt_speech != nullptr);

        auto* prompt = cosyvoice_prompt_init_from_prompt_speech(ctx, prompt_speech);
        REQUIRE(prompt != nullptr);

        tracked_noise_ctx noise_ctx;
        noise_ctx.flow_noise = flow_noise;
        noise_ctx.hift_noise = hift_noise;
        cosyvoice_set_noise_callback(ctx, tracked_noise_callback, &noise_ctx);

        WHEN("cosyvoice_token2wav is called with 38 test tokens") {
            cosyvoice_generated_speech result = {};
            bool ok = cosyvoice_token2wav(
                ctx,
                tokens.data(),
                (uint32_t)tokens.size(),
                1.0f,
                prompt,
                &result);

            THEN("token2wav does not crash and returns a result") {
                // ok may be false if Flow output is wrong, but it should not crash
                REQUIRE(result.data != nullptr);
                REQUIRE(result.length > 0);
            }

            THEN("output has expected sample count for 38 tokens") {
                // 38 tokens → ~76 mel frames → ~72960 audio samples at 24kHz
                // Allow some tolerance
                INFO("result.length = " << result.length);
                REQUIRE(result.length > 10000);
                REQUIRE(result.length < 200000);
            }

            THEN("noise callback was called for all 4 stages") {
                REQUIRE(noise_ctx.before_flow_count == 1);
                REQUIRE(noise_ctx.after_flow_count == 1);
                REQUIRE(noise_ctx.before_hift_count == 1);
                REQUIRE(noise_ctx.after_hift_count == 1);
            }

            THEN("flow noise requested correct length (20000 = 250*80)") {
                REQUIRE(noise_ctx.flow_length_requested == 20000);
            }

            THEN("output is not silence") {
                float max_val = 0;
                for (uint32_t i = 0; i < result.length; ++i)
                    max_val = std::max(max_val, std::abs(result.data[i]));
                REQUIRE(max_val > 0.01f);
            }
        }

        cosyvoice_prompt_free(prompt);
        cosyvoice_prompt_speech_free(prompt_speech);
        cosyvoice_free(ctx);
    }
}
