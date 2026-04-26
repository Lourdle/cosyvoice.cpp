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

if(WIN32)
    add_custom_command(TARGET cosyvoice-cli POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_RUNTIME_DLLS:cosyvoice-cli>
            $<TARGET_FILE_DIR:cosyvoice-cli>
        COMMAND_EXPAND_LISTS
    )

    if(COSYVOICE_AUDIO_BACKEND STREQUAL "FFMPEG" AND DEFINED FFMPEG_RUNTIME_DLLS)
        add_custom_command(TARGET cosyvoice-cli POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${FFMPEG_RUNTIME_DLLS}
                $<TARGET_FILE_DIR:cosyvoice-cli>
            COMMAND_EXPAND_LISTS
        )
    endif()
endif()

install(TARGETS cosyvoice-cli
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
