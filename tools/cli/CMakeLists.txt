file(GLOB CLI_SOURCES "*.cpp")
add_executable(cosyvoice-cli ${CLI_SOURCES})
target_link_libraries(cosyvoice-cli PRIVATE cosyvoice common ggml)
target_link_libraries(cosyvoice-cli PRIVATE tools_common)

if(COSYVOICE_NO_AUDIO)
    target_compile_definitions(cosyvoice-cli PRIVATE COSYVOICE_NO_AUDIO)
    target_link_libraries(cosyvoice-cli PRIVATE ggml-base)
endif()

if(COSYVOICE_AUDIO_BACKEND STREQUAL "FFMPEG")
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
