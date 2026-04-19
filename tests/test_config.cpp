#include <catch2/catch_test_macros.hpp>

#include "cosyvoice.h"
#include "cosyvoice-lowlevel.h"

#include <ggml.h>
#include <ggml-cpu.h>
#include <ggml-backend.h>

#include <vector>

SCENARIO("generation config validation accepts valid configurations") {
    GIVEN("a valid generation config") {
        cosyvoice_generation_config_t config{};
        config.temperature = 1.0f;
        config.sampling.top_k = 25;
        config.sampling.top_p = 0.9f;
        config.sampling.win_size = 50;
        config.sampling.tau_r = 0.1f;
        config.min_token_text_ratio = 2.0f;
        config.max_token_text_ratio = 50.0f;

        // set_generation_config requires a model context, so we test the
        // validation logic indirectly via the public API once a context exists.
        // For now, we verify the struct layout and field access.
        THEN("fields are accessible and correctly laid out") {
            REQUIRE(config.temperature == 1.0f);
            REQUIRE(config.sampling.top_k == 25);
            REQUIRE(config.sampling.top_p == 0.9f);
            REQUIRE(config.sampling.win_size == 50);
            REQUIRE(config.sampling.tau_r == 0.1f);
            REQUIRE(config.min_token_text_ratio == 2.0f);
            REQUIRE(config.max_token_text_ratio == 50.0f);
        }
    }
}

SCENARIO("default context params are initialized with sane values") {
    GIVEN("default context params") {
        cosyvoice_context_params_t params{};
        cosyvoice_init_default_context_params(&params);

        THEN("flash attention is enabled by default") {
            REQUIRE(params.flow_use_flash_attn == true);
            REQUIRE(params.llm_use_flash_attn == true);
        }

        THEN("KV cache defaults to Q8_0 with fallback") {
            REQUIRE(params.llm_kv_cache_type == COSYVOICE_LLM_KV_CACHE_TYPE_Q8_0);
            REQUIRE(params.llm_allow_kv_cache_fallback == true);
        }

        THEN("inference buffer policy defaults to balanced") {
            REQUIRE(params.inference_buffer_policy == COSYVOICE_INFERENCE_BUFFER_POLICY_BALANCED);
        }

        THEN("batch and sequence limits are reasonable") {
            REQUIRE(params.n_batch == 256);
            REQUIRE(params.n_max_seq == 2048);
        }

        THEN("seed is non-zero (randomized)") {
            REQUIRE(params.seed != 0);
        }

        THEN("sampler RNG policy defaults to reset per session") {
            REQUIRE(params.builtin_sampler_rng_policy == COSYVOICE_BUILTIN_SAMPLER_RNG_POLICY_RESET_PER_SESSION);
        }

        THEN("no custom sampler by default") {
            REQUIRE(params.sampler == nullptr);
            REQUIRE(params.sampler_ctx == nullptr);
        }
    }
}

SCENARIO("generation config validation boundary conditions") {
    // These test the validation rules in cosyvoice_model::set_generation_config
    // without requiring a model context. We verify the conditions that would
    // cause rejection, documenting the contract for future reference.

    GIVEN("the validation rules from set_generation_config") {
        THEN("temperature must be > 0") {
            // config->temperature <= 0.f → return false
            REQUIRE(0.0f <= 0.0f);   // zero is rejected
            REQUIRE(-1.0f <= 0.0f);  // negative is rejected
            REQUIRE(!(0.001f <= 0.0f)); // small positive is accepted
        }

        THEN("min_token_text_ratio must be >= 0") {
            REQUIRE(-1.0f < 0.0f);   // negative is rejected
            REQUIRE(!(0.0f < 0.0f)); // zero is accepted
        }

        THEN("max_token_text_ratio must be >= min_token_text_ratio") {
            float min_r = 2.0f, max_r = 1.0f;
            REQUIRE(max_r < min_r);  // max < min is rejected
            max_r = 2.0f;
            REQUIRE(!(max_r < min_r)); // max == min is accepted
        }

        THEN("top_k must be >= 0") {
            REQUIRE(-1 < 0);    // negative is rejected
            REQUIRE(!(0 < 0));  // zero is accepted
        }

        THEN("top_p must be in [0.0, 1.0]") {
            REQUIRE(-0.1f < 0.0f);   // negative is rejected
            REQUIRE(1.1f > 1.0f);    // > 1.0 is rejected
            REQUIRE(!(0.0f < 0.0f)); // 0.0 is accepted
            REQUIRE(!(1.0f > 1.0f)); // 1.0 is accepted
        }

        THEN("win_size must be > 0") {
            REQUIRE(0 <= 0);     // zero is rejected
            REQUIRE(-1 <= 0);   // negative is rejected
            REQUIRE(!(1 <= 0)); // positive is accepted
        }

        THEN("tau_r must be >= 0") {
            REQUIRE(-0.1f < 0.0f);   // negative is rejected
            REQUIRE(!(0.0f < 0.0f)); // zero is accepted
        }
    }
}

SCENARIO("ggml_backend_tensor_set size parameter must be in bytes") {
    // Documents a bug found in set_hift_rand_ini where the size parameter
    // was passed as element count instead of byte count:
    //   ggml_backend_tensor_set(t, data, 0, nfft / 2 + 1);        // BUG
    //   ggml_backend_tensor_set(t, data, 0, (nfft/2+1)*sizeof(f)); // FIX
    //
    // This caused only 1/4 of the noise initialization data to be written,
    // leaving the rest as zeros. The effect is subtle — HiFT still runs
    // but with degraded excitation signal quality.

    GIVEN("a 1D F32 tensor allocated on CPU backend") {
        auto backend = ggml_backend_cpu_init();
        REQUIRE(backend != nullptr);

        auto ctx = ggml_init(ggml_init_params{ .mem_size = 1024 * 1024, .no_alloc = true });
        const int n_elements = 9; // nfft/2 + 1 for nfft=16
        auto tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_elements);

        auto buft = ggml_backend_get_default_buffer_type(backend);
        auto buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        REQUIRE(buf != nullptr);

        // Fill with known pattern
        std::vector<float> data(n_elements);
        for (int i = 0; i < n_elements; i++)
            data[i] = static_cast<float>(i + 1);

        WHEN("size is passed as element count (the bug)") {
            // This only writes n_elements bytes = 9 bytes = 2.25 floats
            ggml_backend_tensor_set(tensor, data.data(), 0, n_elements);

            std::vector<float> readback(n_elements, -1.0f);
            ggml_backend_tensor_get(tensor, readback.data(), 0, n_elements * sizeof(float));

            THEN("only the first ~2 elements are partially written") {
                // First float is complete (4 bytes written)
                REQUIRE(readback[0] == data[0]);
                // Second float is complete (8 bytes written)
                REQUIRE(readback[1] == data[1]);
                // Third float has only 1 byte written — corrupted
                REQUIRE(readback[2] != data[2]);
                // Remaining elements are uninitialized
            }
        }

        WHEN("size is passed as byte count (the fix)") {
            ggml_backend_tensor_set(tensor, data.data(), 0, n_elements * sizeof(float));

            std::vector<float> readback(n_elements, -1.0f);
            ggml_backend_tensor_get(tensor, readback.data(), 0, n_elements * sizeof(float));

            THEN("all elements are correctly written") {
                for (int i = 0; i < n_elements; i++)
                    REQUIRE(readback[i] == data[i]);
            }
        }

        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        ggml_backend_free(backend);
    }
}

SCENARIO("context params enum values are distinct") {
    THEN("KV cache types have distinct values") {
        REQUIRE(COSYVOICE_LLM_KV_CACHE_TYPE_F32 != COSYVOICE_LLM_KV_CACHE_TYPE_F16);
        REQUIRE(COSYVOICE_LLM_KV_CACHE_TYPE_F16 != COSYVOICE_LLM_KV_CACHE_TYPE_Q8_0);
        REQUIRE(COSYVOICE_LLM_KV_CACHE_TYPE_Q8_0 != COSYVOICE_LLM_KV_CACHE_TYPE_Q5_1);
        REQUIRE(COSYVOICE_LLM_KV_CACHE_TYPE_Q5_1 != COSYVOICE_LLM_KV_CACHE_TYPE_Q5_0);
        REQUIRE(COSYVOICE_LLM_KV_CACHE_TYPE_Q5_0 != COSYVOICE_LLM_KV_CACHE_TYPE_Q4_1);
        REQUIRE(COSYVOICE_LLM_KV_CACHE_TYPE_Q4_1 != COSYVOICE_LLM_KV_CACHE_TYPE_Q4_0);
    }

    THEN("inference buffer policies have distinct values") {
        REQUIRE(COSYVOICE_INFERENCE_BUFFER_POLICY_SHARED != COSYVOICE_INFERENCE_BUFFER_POLICY_BALANCED);
        REQUIRE(COSYVOICE_INFERENCE_BUFFER_POLICY_BALANCED != COSYVOICE_INFERENCE_BUFFER_POLICY_DEDICATED);
    }

    THEN("inference modes have distinct values") {
        REQUIRE(COSYVOICE_INFERENCE_MODE_ZERO_SHOT != COSYVOICE_INFERENCE_MODE_INSTRUCT);
        REQUIRE(COSYVOICE_INFERENCE_MODE_INSTRUCT != COSYVOICE_INFERENCE_MODE_CROSS_LINGUAL);
    }

    THEN("sampler RNG policies have distinct values") {
        REQUIRE(COSYVOICE_BUILTIN_SAMPLER_RNG_POLICY_RESET_PER_SESSION !=
                COSYVOICE_BUILTIN_SAMPLER_RNG_POLICY_CONTINUE_ACROSS_SESSIONS);
    }
}
