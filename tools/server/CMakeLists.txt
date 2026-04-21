set(SERVER_SOURCES
    cosyvoice-server.cpp
    "${CMAKE_SOURCE_DIR}/vendor/cpp-httplib/httplib.cpp"
)

add_executable(cosyvoice-server ${SERVER_SOURCES})
target_link_libraries(cosyvoice-server PRIVATE cosyvoice common ggml tools_common)
target_include_directories(cosyvoice-server PRIVATE
    "${CMAKE_SOURCE_DIR}/vendor"
    "${CMAKE_SOURCE_DIR}/vendor/cpp-httplib"
    "${CMAKE_SOURCE_DIR}/vendor/nlohmann"
)

if(COSYVOICE_NO_AUDIO)
    target_compile_definitions(cosyvoice-server PRIVATE COSYVOICE_NO_AUDIO)
    target_link_libraries(cosyvoice-server PRIVATE ggml-base)
endif()

if(COSYVOICE_NO_FRONTEND)
    target_compile_definitions(cosyvoice-server PRIVATE COSYVOICE_NO_FRONTEND)
endif()

if(COSYVOICE_NO_ICU)
    target_compile_definitions(cosyvoice-server PRIVATE COSYVOICE_NO_ICU)
endif()

if(WIN32)
    add_custom_command(TARGET cosyvoice-server POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_RUNTIME_DLLS:cosyvoice-server>
            $<TARGET_FILE_DIR:cosyvoice-server>
        COMMAND_EXPAND_LISTS
    )
endif()

install(TARGETS cosyvoice-server
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
