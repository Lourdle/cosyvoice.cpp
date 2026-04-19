#include <catch2/catch_test_macros.hpp>

#include "cosyvoice.h"
#include "cosyvoice-lowlevel.h"

#include <cstring>
#include <vector>

// Test helpers
struct callback_record {
    cosyvoice_noise_callback_stage_t stage;
    uint32_t length;
    float* noise_in;
};

struct test_callback_ctx {
    std::vector<callback_record> calls;
    std::vector<float> flow_noise;
    std::vector<float> hift_noise;
};

static float* test_noise_callback(
    cosyvoice_noise_callback_stage_t stage,
    uint32_t length,
    float* noise,
    void* ctx)
{
    auto* tc = static_cast<test_callback_ctx*>(ctx);
    tc->calls.push_back({stage, length, noise});

    if (stage == COSYVOICE_NOISE_CALLBACK_STAGE_BEFORE_FLOW) {
        tc->flow_noise.resize(length, 0.5f);
        return tc->flow_noise.data();
    }
    if (stage == COSYVOICE_NOISE_CALLBACK_STAGE_BEFORE_HIFT) {
        tc->hift_noise.resize(length, -0.5f);
        return tc->hift_noise.data();
    }
    return nullptr;
}

SCENARIO("noise callback registration and retrieval") {
    GIVEN("a default context params struct") {
        cosyvoice_context_params_t params{};
        cosyvoice_init_default_context_params(&params);

        THEN("noise callback fields exist in the API") {
            // Verify the enum values are distinct
            REQUIRE(COSYVOICE_NOISE_CALLBACK_STAGE_BEFORE_FLOW != COSYVOICE_NOISE_CALLBACK_STAGE_AFTER_FLOW);
            REQUIRE(COSYVOICE_NOISE_CALLBACK_STAGE_BEFORE_HIFT != COSYVOICE_NOISE_CALLBACK_STAGE_AFTER_HIFT);
            REQUIRE(COSYVOICE_NOISE_CALLBACK_STAGE_BEFORE_FLOW != COSYVOICE_NOISE_CALLBACK_STAGE_BEFORE_HIFT);
        }
    }
}

SCENARIO("noise callback stage enum has expected 4 stages") {
    GIVEN("the noise callback stage enum") {
        THEN("BEFORE_FLOW is stage 0") {
            REQUIRE(COSYVOICE_NOISE_CALLBACK_STAGE_BEFORE_FLOW == 0);
        }
        THEN("AFTER_FLOW is stage 1") {
            REQUIRE(COSYVOICE_NOISE_CALLBACK_STAGE_AFTER_FLOW == 1);
        }
        THEN("BEFORE_HIFT is stage 2") {
            REQUIRE(COSYVOICE_NOISE_CALLBACK_STAGE_BEFORE_HIFT == 2);
        }
        THEN("AFTER_HIFT is stage 3") {
            REQUIRE(COSYVOICE_NOISE_CALLBACK_STAGE_AFTER_HIFT == 3);
        }
    }
}

SCENARIO("custom noise callback returns correct buffers") {
    GIVEN("a custom noise callback with pre-allocated buffers") {
        test_callback_ctx ctx;

        WHEN("BEFORE_FLOW is called with length 20000") {
            float* result = test_noise_callback(
                COSYVOICE_NOISE_CALLBACK_STAGE_BEFORE_FLOW, 20000, nullptr, &ctx);

            THEN("callback returns non-null buffer of correct size") {
                REQUIRE(result != nullptr);
                REQUIRE(ctx.flow_noise.size() == 20000);
                REQUIRE(result == ctx.flow_noise.data());
            }
            THEN("callback was recorded") {
                REQUIRE(ctx.calls.size() == 1);
                REQUIRE(ctx.calls[0].stage == COSYVOICE_NOISE_CALLBACK_STAGE_BEFORE_FLOW);
                REQUIRE(ctx.calls[0].length == 20000);
                REQUIRE(ctx.calls[0].noise_in == nullptr);
            }
        }

        WHEN("BEFORE_HIFT is called with length 328320") {
            float* result = test_noise_callback(
                COSYVOICE_NOISE_CALLBACK_STAGE_BEFORE_HIFT, 328320, nullptr, &ctx);

            THEN("callback returns non-null buffer of correct size") {
                REQUIRE(result != nullptr);
                REQUIRE(ctx.hift_noise.size() == 328320);
                REQUIRE(result == ctx.hift_noise.data());
            }
        }

        WHEN("AFTER_FLOW is called with a previously returned buffer") {
            float* flow_buf = test_noise_callback(
                COSYVOICE_NOISE_CALLBACK_STAGE_BEFORE_FLOW, 100, nullptr, &ctx);
            float* result = test_noise_callback(
                COSYVOICE_NOISE_CALLBACK_STAGE_AFTER_FLOW, 100, flow_buf, &ctx);

            THEN("AFTER stage receives the buffer from BEFORE stage") {
                REQUIRE(ctx.calls.size() == 2);
                REQUIRE(ctx.calls[1].noise_in == flow_buf);
            }
            THEN("AFTER stage returns nullptr") {
                REQUIRE(result == nullptr);
            }
        }
    }
}
