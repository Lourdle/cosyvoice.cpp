set(SERVER_SOURCES
    cosyvoice-server.h
    cosyvoice-server.cpp
)

set(SERVER_EXTRA_SOURCES
    server_common.h
    cosyvoice-server-backend.cpp
    server_common.cpp
)

set(SERVER_MODULE_INTERFACES
    nlohmann-json.ixx
    httplib.ixx
)

set(SERVER_COMMON_INCLUDES
    "${CMAKE_SOURCE_DIR}/vendor"
    "${CMAKE_SOURCE_DIR}/vendor/cpp-httplib"
)
set(SERVER_COMMON_LIBS cosyvoice common ggml tools_common)

option(COSYVOICE_SERVER_NO_WEBUI
    "Disable embedded WebUI in cosyvoice-server (removes C23 #embed dependency)" OFF)

set(COSYVOICE_SERVER_DEFAULT_MODE "WEBUI" CACHE STRING
    "Default server mode when neither --api nor --webui is given (WEBUI or API)")

if(NOT COSYVOICE_SERVER_DEFAULT_MODE STREQUAL "WEBUI" AND
   NOT COSYVOICE_SERVER_DEFAULT_MODE STREQUAL "API")
    message(FATAL_ERROR "COSYVOICE_SERVER_DEFAULT_MODE must be WEBUI or API (got: ${COSYVOICE_SERVER_DEFAULT_MODE})")
endif()

if(NOT COSYVOICE_SERVER_NO_WEBUI)
    list(APPEND SERVER_EXTRA_SOURCES
        cosyvoice-server-webui.cpp
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
endif()

# C++20 module dependency scanning is only supported with specific generator
# and compiler combinations. If unsupported, fall back to precompiled headers.
set(_use_modules TRUE)
set(_fallback_reason "")

# Makefile-based generators do not support module scanning
if(CMAKE_GENERATOR MATCHES "Makefiles$|^NMake")
    set(_use_modules FALSE)
    set(_fallback_reason "${CMAKE_GENERATOR} does not support module scanning")
endif()

# Ninja requires version 1.11+ for module scanning
if(_use_modules AND CMAKE_GENERATOR MATCHES "Ninja")
    execute_process(COMMAND "${CMAKE_MAKE_PROGRAM}" --version
        OUTPUT_VARIABLE _ninja_version
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
    if(_ninja_version VERSION_LESS "1.11")
        set(_use_modules FALSE)
        set(_fallback_reason "Ninja ${_ninja_version} does not support module scanning (requires 1.11+)")
    endif()
endif()

# Compiler must support module dependency scanning
if(_use_modules)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "14.0")
            set(_use_modules FALSE)
            set(_fallback_reason "GCC ${CMAKE_CXX_COMPILER_VERSION} does not support module scanning (requires 14+)")
        endif()
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "16.0")
            set(_use_modules FALSE)
            set(_fallback_reason "Clang ${CMAKE_CXX_COMPILER_VERSION} does not support module scanning (requires 16+)")
        endif()
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "14.34")
            set(_use_modules FALSE)
            set(_fallback_reason "MSVC ${CMAKE_CXX_COMPILER_VERSION} does not support module scanning (requires 14.34+)")
        endif()
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
        if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "16.0")
            set(_use_modules FALSE)
            set(_fallback_reason "AppleClang ${CMAKE_CXX_COMPILER_VERSION} not verified for module scanning")
        endif()
    else()
        set(_use_modules FALSE)
        set(_fallback_reason "Unknown compiler ${CMAKE_CXX_COMPILER_ID} — module scanning not verified")
    endif()
endif()

if(NOT _use_modules)
    message(WARNING "${_fallback_reason}. "
                    "Falling back to precompiled headers for cosyvoice-server. "
                    "Use a supported compiler/generator for module support.")

    # PCH object library (only for selected translation units that need httplib/nlohmann)
    add_library(cosyvoice-server-provider STATIC
        pch.h
        ${SERVER_EXTRA_SOURCES}
    )
    target_precompile_headers(cosyvoice-server-provider PRIVATE pch.h)
    target_include_directories(cosyvoice-server-provider PRIVATE ${SERVER_COMMON_INCLUDES})
    target_link_libraries(cosyvoice-server-provider PRIVATE ${SERVER_COMMON_LIBS})
    target_compile_definitions(cosyvoice-server-provider PRIVATE COSYVOICE_USE_PCH)

    # Main executable (no PCH; objects from cosyvoice-server-provider link in automatically)
    add_executable(cosyvoice-server ${SERVER_SOURCES})
    target_link_libraries(cosyvoice-server PRIVATE cosyvoice-server-provider ${SERVER_COMMON_LIBS})
    target_include_directories(cosyvoice-server PRIVATE ${SERVER_COMMON_INCLUDES})

    # Apply compile definitions to both targets
    foreach(tgt cosyvoice-server cosyvoice-server-provider)
        if(COSYVOICE_NO_AUDIO)
            target_compile_definitions(${tgt} PRIVATE COSYVOICE_NO_AUDIO)
            target_link_libraries(${tgt} PRIVATE ggml-base)
        elseif(COSYVOICE_AUDIO_BACKEND STREQUAL "FFMPEG")
            target_compile_definitions(${tgt} PRIVATE COSYVOICE_AUDIO_BACKEND_FFMPEG)
        endif()
        if(COSYVOICE_NO_FRONTEND)
            target_compile_definitions(${tgt} PRIVATE COSYVOICE_NO_FRONTEND)
        endif()
        if(COSYVOICE_NO_ICU)
            target_compile_definitions(${tgt} PRIVATE COSYVOICE_NO_ICU)
        endif()
        if(COSYVOICE_SERVER_NO_WEBUI)
            set_property(SOURCE cosyvoice-server.cpp APPEND PROPERTY COMPILE_DEFINITIONS "COSYVOICE_SERVER_NO_WEBUI")
        endif()
        if(COSYVOICE_SERVER_DEFAULT_MODE STREQUAL "API")
            set_property(SOURCE cosyvoice-server.cpp APPEND PROPERTY COMPILE_DEFINITIONS "COSYVOICE_SERVER_DEFAULT_MODE_API")
        endif()
    endforeach()
else()
    # C++20 modules (Ninja, Visual Studio, Xcode, etc.)
    add_executable(cosyvoice-server ${SERVER_SOURCES} ${SERVER_EXTRA_SOURCES})
    target_sources(cosyvoice-server PRIVATE
        FILE_SET CXX_MODULES
        FILES ${SERVER_MODULE_INTERFACES}
    )

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
    if(COSYVOICE_SERVER_NO_WEBUI)
        set_property(SOURCE cosyvoice-server.cpp APPEND PROPERTY COMPILE_DEFINITIONS "COSYVOICE_SERVER_NO_WEBUI")
    endif()
    if(COSYVOICE_SERVER_DEFAULT_MODE STREQUAL "API")
        set_property(SOURCE cosyvoice-server.cpp APPEND PROPERTY COMPILE_DEFINITIONS "COSYVOICE_SERVER_DEFAULT_MODE_API")
    endif()
endif()

install(TARGETS cosyvoice-server
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
