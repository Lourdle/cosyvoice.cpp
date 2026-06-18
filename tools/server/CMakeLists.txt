set(SERVER_SOURCES
    nlohmann-json.ixx
    httplib.ixx
    cosyvoice-server.h
    cosyvoice-server.cpp
    server_common.h
    cosyvoice-server-backend.cpp
    cosyvoice-server-webui.cpp
    server_common.cpp
)

# Embedded resources: Windows uses RC, others use C23 #embed
if(WIN32)
    list(APPEND SERVER_SOURCES
        resource.h
        resource_win.c
        cosyvoice-server.rc
    )
else()
    set(CMAKE_C_STANDARD 23)
    list(APPEND SERVER_SOURCES
        resource.h
        resource_embed.c
    )
endif()

set(SERVER_COMMON_INCLUDES
    "${CMAKE_SOURCE_DIR}/vendor"
    "${CMAKE_SOURCE_DIR}/vendor/cpp-httplib"
)
set(SERVER_COMMON_LIBS cosyvoice common ggml tools_common)

add_executable(cosyvoice-server ${SERVER_SOURCES})
target_link_libraries(cosyvoice-server PRIVATE ${SERVER_COMMON_LIBS})
target_include_directories(cosyvoice-server PRIVATE ${SERVER_COMMON_INCLUDES})

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
