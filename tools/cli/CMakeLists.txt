file(GLOB CLI_SOURCES "cosyvoice-cli.cpp")

# Effective playback control:
#   If user explicitly set COSYVOICE_CLI_NO_PLAYBACK via -D, respect it.
#   Otherwise, fall back to COSYVOICE_NO_AUDIO.
if(DEFINED COSYVOICE_CLI_NO_PLAYBACK)
    set(_COSYVOICE_CLI_NO_PLAYBACK "${COSYVOICE_CLI_NO_PLAYBACK}")
else()
    set(_COSYVOICE_CLI_NO_PLAYBACK "${COSYVOICE_NO_AUDIO}")
endif()

if(NOT _COSYVOICE_CLI_NO_PLAYBACK)
    list(APPEND CLI_SOURCES "cli_audio_player.cpp" "cli_audio_player.h")
endif()

add_executable(cosyvoice-cli ${CLI_SOURCES})
target_link_libraries(cosyvoice-cli PRIVATE cosyvoice common tools_common ggml)
target_link_libraries(cosyvoice-cli PRIVATE)

if(NOT _COSYVOICE_CLI_NO_PLAYBACK)
    target_include_directories(cosyvoice-cli PRIVATE "${PROJECT_SOURCE_DIR}/vendor/miniaudio")
endif()

if(_COSYVOICE_CLI_NO_PLAYBACK)
    target_compile_definitions(cosyvoice-cli PRIVATE COSYVOICE_CLI_NO_PLAYBACK)
endif()

if(COSYVOICE_NO_AUDIO)
    target_compile_definitions(cosyvoice-cli PRIVATE COSYVOICE_NO_AUDIO)
elseif(COSYVOICE_AUDIO_BACKEND STREQUAL "FFMPEG")
    target_compile_definitions(cosyvoice-cli PRIVATE COSYVOICE_AUDIO_BACKEND_FFMPEG)
endif()

if(COSYVOICE_NO_FRONTEND)
    target_compile_definitions(cosyvoice-cli PRIVATE COSYVOICE_NO_FRONTEND)
endif()

if(COSYVOICE_NO_ICU)
    target_compile_definitions(cosyvoice-cli PRIVATE COSYVOICE_NO_ICU)
endif()

install(TARGETS cosyvoice-cli
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
