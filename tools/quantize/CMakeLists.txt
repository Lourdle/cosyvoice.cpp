file(GLOB QUANTIZE_SOURCES "*.cpp")
add_executable(quantize ${QUANTIZE_SOURCES})
target_link_libraries(quantize PRIVATE ggml)

if(WIN32)
    add_custom_command(TARGET quantize POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_RUNTIME_DLLS:quantize>
            $<TARGET_FILE_DIR:quantize>
        COMMAND_EXPAND_LISTS
    )
endif()

install(TARGETS quantize
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
