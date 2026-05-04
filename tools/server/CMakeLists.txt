set(SERVER_SOURCES
    cosyvoice-server.h
    cosyvoice-server.cpp
    cosyvoice-server-backend.cpp
)

add_executable(cosyvoice-server ${SERVER_SOURCES})
target_link_libraries(cosyvoice-server PRIVATE cosyvoice common ggml tools_common)
target_include_directories(cosyvoice-server PRIVATE
    "${CMAKE_SOURCE_DIR}/vendor"
    "${CMAKE_SOURCE_DIR}/vendor/cpp-httplib"
)

if(COSYVOICE_NO_AUDIO)
    target_compile_definitions(cosyvoice-server PRIVATE COSYVOICE_NO_AUDIO)
    target_link_libraries(cosyvoice-server PRIVATE ggml-base)
elseif(COSYVOICE_AUDIO_BACKEND STREQUAL "FFMPEG")
    target_compile_definitions(cosyvoice-server PRIVATE COSYVOICE_AUDIO_BACKEND_FFMPEG)
endif()

if(COSYVOICE_NO_FRONTEND)
    target_compile_definitions(cosyvoice-server PRIVATE COSYVOICE_NO_FRONTEND)
endif()

if(COSYVOICE_NO_ICU)
    target_compile_definitions(cosyvoice-server PRIVATE COSYVOICE_NO_ICU)
endif()

install(TARGETS cosyvoice-server
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
