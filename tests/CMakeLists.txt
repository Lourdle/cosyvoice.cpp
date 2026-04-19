FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.7.1
)
FetchContent_MakeAvailable(Catch2)

add_executable(cosyvoice-tests
    test_config.cpp
    test_graph_routing.cpp
    test_conv1d_contiguity.cpp
    # test_stft_roundtrip.cpp  # temporarily disabled: ggml FFT API changed
    test_dit_contiguity.cpp
    test_im2col_chunking.cpp
    test_noise_callback.cpp
    test_flow_numerical.cpp
    test_flow_token2wav.cpp
    # test_grouped_conv_bias.cpp    # source missing from worktree
    # test_grouped_conv_contract.cpp
    # test_gguf_conv_weights.cpp
    # test_conv1_vs_conv2.cpp
    # test_conv_chain_contract.cpp
    # test_conv_chain_isolation.cpp
    # test_input_embed_contract.cpp
    # test_e2e_conv_vs_pytorch.cpp  # requires gold gen
    test_e2e_dit_vs_pytorch.cpp
    test_ode_loop_mechanics.cpp
    test_e2e_dit_full.cpp
    test_e2e_ode_step.cpp
    test_e2e_ode_full.cpp
    test_cond_path.cpp
    test_conv_pos_batch2.cpp
    test_cond_dit_full.cpp
)

target_link_libraries(cosyvoice-tests PRIVATE cosyvoice ggml Catch2::Catch2WithMain)
target_include_directories(cosyvoice-tests PRIVATE
    "${CMAKE_SOURCE_DIR}/include"
    "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/vendor/miniaudio"
    "${CMAKE_SOURCE_DIR}/common"
    "${GGML_SOURCE_DIR}/include"
)

include(CTest)
include(Catch)
catch_discover_tests(cosyvoice-tests)

# --- Gold data generation targets ---
# These require: Python 3, torch, gguf pip packages, and COSYVOICE_TEST_MODEL env var.
# Gold data is written to /tmp/cv_*_gold/ and is NOT checked into the repository.
# Tests that depend on gold data will SKIP (not FAIL) if the files are missing.
#
# Usage:
#   export COSYVOICE_TEST_MODEL=path/to/CosyVoice3-2512_F32.gguf
#   cmake --build <build-dir> --target gen-gold
#   ctest --test-dir <build-dir>

# Scripts that run standalone (only need GGUF model, no pipeline dumps)
set(GOLD_SCRIPTS
    gen_cond_dit_gold.py
    gen_cond_path_gold.py
    gen_e2e_dit_gold.py
    gen_e2e_dit_full_gold.py
)
# These require pipeline dump files and are run manually:
#   gen_e2e_encoder_gold.py  (needs /tmp/cv_repro3/)
#   gen_e2e_ode_step_gold.py (needs encoder gold)
#   gen_e2e_ode_full_gold.py (needs encoder gold)
#   gen_hift_gold.py         (needs flow output dump)

add_custom_target(gen-gold
    COMMENT "Generating Python gold reference data for tests (requires COSYVOICE_TEST_MODEL)"
)

foreach(SCRIPT IN LISTS GOLD_SCRIPTS)
    get_filename_component(TARGET_NAME "${SCRIPT}" NAME_WE)
    add_custom_target(${TARGET_NAME}
        COMMAND python3 "${CMAKE_CURRENT_SOURCE_DIR}/${SCRIPT}"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        COMMENT "Running ${SCRIPT} (set COSYVOICE_TEST_MODEL env var)"
        VERBATIM
    )
    add_dependencies(gen-gold ${TARGET_NAME})
endforeach()
