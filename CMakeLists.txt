cmake_minimum_required(VERSION 3.24)
project(cosyvoice LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build position independent code (PIC) by default.
# This is crucial for OBJECT libraries that will be linked into a SHARED library on Linux.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Ensure all binaries and libraries go to the same output directories for easier local testing
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

include(FetchContent)

# -----------------------------------------------------------------------------
# Options & Paths
# -----------------------------------------------------------------------------
option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(COSYVOICE_NO_AUDIO "Disable audio API" OFF)
set(COSYVOICE_AUDIO_BACKEND "MINIAUDIO" CACHE STRING "Audio backend: MINIAUDIO or FFMPEG")
set_property(CACHE COSYVOICE_AUDIO_BACKEND PROPERTY STRINGS MINIAUDIO FFMPEG)
option(COSYVOICE_NO_ICU "Disable ICU dependency" OFF)
option(COSYVOICE_NO_FRONTEND "Disable ONNX Runtime dependency" OFF)

set(GGML_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/vendor/ggml" CACHE PATH "Path to ggml source directory")
set(ICU_PREBUILT_DIR "${CMAKE_BINARY_DIR}/_deps/icu" CACHE PATH "Path to custom prebuilt ICU")
set(ORT_PREBUILT_DIR "${CMAKE_BINARY_DIR}/_deps/onnxruntime" CACHE PATH "Path to custom prebuilt ONNX Runtime")
set(FFMPEG_PREBUILT_DIR "${CMAKE_BINARY_DIR}/_deps/ffmpeg" CACHE PATH "Path to custom prebuilt FFmpeg")

# -----------------------------------------------------------------------------
# Dependencies
# -----------------------------------------------------------------------------
include(cmake/Dependencies.cmake)

# -----------------------------------------------------------------------------
# Internal Common Utilities
# -----------------------------------------------------------------------------
add_subdirectory(common)

# -----------------------------------------------------------------------------
# Core Library: cosyvoice
# -----------------------------------------------------------------------------
file(GLOB COSYVOICE_SOURCES "src/*.cpp" "src/*.h" "include/*.h")

if(COSYVOICE_NO_AUDIO)
    # Audio disabled - skip all audio files
elseif(COSYVOICE_AUDIO_BACKEND STREQUAL "FFMPEG")
    list(REMOVE_ITEM COSYVOICE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/cosyvoice-audio.cpp")
else()
    list(REMOVE_ITEM COSYVOICE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/cosyvoice-audio-ffmpeg.cpp")
endif()

# Separate frontend and text-norm into OBJECT libraries so their heavy dependencies 
# (ONNX Runtime, ICU) don't pollute the compile flags of the rest of the project.
# This prevents a full project rebuild when toggling COSYVOICE_NO_ICU or COSYVOICE_NO_FRONTEND.
list(REMOVE_ITEM COSYVOICE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/cosyvoice-frontend.cpp")
list(REMOVE_ITEM COSYVOICE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/cosyvoice-frontend-text-norm.cpp")

set(OBJECT_LIBS "")

# Text Norm (ICU)
add_library(cosyvoice-frontend-text-norm OBJECT "src/cosyvoice-frontend-text-norm.cpp")
target_include_directories(cosyvoice-frontend-text-norm PRIVATE "include" "vendor/miniaudio")
target_link_libraries(cosyvoice-frontend-text-norm PRIVATE ggml pcre2-8 pcre2-16)
if(NOT COSYVOICE_NO_ICU)
    target_link_libraries(cosyvoice-frontend-text-norm PRIVATE ICU::uc ICU::i18n)
else()
    target_compile_definitions(cosyvoice-frontend-text-norm PRIVATE COSYVOICE_NO_ICU)
endif()
list(APPEND OBJECT_LIBS cosyvoice-frontend-text-norm)

# ONNX Frontend
if(NOT COSYVOICE_NO_FRONTEND)
    add_library(cosyvoice-frontend OBJECT "src/cosyvoice-frontend.cpp")
    target_include_directories(cosyvoice-frontend PRIVATE "include" "vendor/miniaudio" "common")
    target_link_libraries(cosyvoice-frontend PRIVATE ggml common pcre2-8 pcre2-16 onnxruntime)
    if(COSYVOICE_NO_AUDIO)
        target_compile_definitions(cosyvoice-frontend PRIVATE COSYVOICE_NO_AUDIO)
    endif()
    list(APPEND OBJECT_LIBS cosyvoice-frontend)
endif()

add_library(cosyvoice ${COSYVOICE_SOURCES})
target_link_libraries(cosyvoice PRIVATE common ${OBJECT_LIBS})

target_include_directories(cosyvoice PUBLIC "include")
target_include_directories(cosyvoice PRIVATE "vendor/miniaudio" "common")
if(COSYVOICE_AUDIO_BACKEND STREQUAL "FFMPEG" AND EXISTS "${FFMPEG_PREBUILT_DIR}/include")
    target_include_directories(cosyvoice PRIVATE "${FFMPEG_PREBUILT_DIR}/include")
endif()

# -----------------------------------------------------------------------------
# Compiler Flags & SIMD Support
# -----------------------------------------------------------------------------
if(NOT MSVC)
    add_compile_definitions("$<$<CONFIG:Debug>:_DEBUG>")
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        set(SIMD_FLAGS "-mavx" "-mavx2" "-mfma")
        
        target_compile_options(cosyvoice PRIVATE ${SIMD_FLAGS})
        
        if(TARGET cosyvoice-frontend)
            target_compile_options(cosyvoice-frontend PRIVATE ${SIMD_FLAGS})
        endif()
        if(TARGET cosyvoice-frontend-text-norm)
            target_compile_options(cosyvoice-frontend-text-norm PRIVATE ${SIMD_FLAGS})
        endif()
    elseif(MSVC)
        if(TARGET cosyvoice-frontend)
            target_compile_options(cosyvoice-frontend PRIVATE "/arch:AVX2")
        endif()
        if(TARGET cosyvoice-frontend-text-norm)
            target_compile_options(cosyvoice-frontend-text-norm PRIVATE "/arch:AVX2")
        endif()
    endif()
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
    # ARM: Use SIMDe (SIMD Everywhere) for AVX2/FMA→NEON translation.
    # Install SIMDe via your package manager (e.g. `brew install simde`)
    # or place headers under vendor/simde/.
    find_path(SIMDE_INCLUDE_DIR NAMES simde/x86/avx2.h
              HINTS /opt/homebrew/include /usr/local/include /usr/include
                    "${CMAKE_CURRENT_SOURCE_DIR}/vendor/simde")
    if(SIMDE_INCLUDE_DIR)
        message(STATUS "SIMDe found: ${SIMDE_INCLUDE_DIR}")
        target_include_directories(cosyvoice SYSTEM PRIVATE ${SIMDE_INCLUDE_DIR})
        if(TARGET cosyvoice-frontend)
            target_include_directories(cosyvoice-frontend SYSTEM PRIVATE ${SIMDE_INCLUDE_DIR})
        endif()
        if(TARGET cosyvoice-frontend-text-norm)
            target_include_directories(cosyvoice-frontend-text-norm SYSTEM PRIVATE ${SIMDE_INCLUDE_DIR})
        endif()
    else()
        message(FATAL_ERROR
                "SIMDe is required on ARM64 because AVX2 intrinsics in cosyvoice-frontend and fft "
                "unconditionally include SIMDe headers. Install SIMDe or set SIMDE_INCLUDE_DIR to "
                "the directory containing the top-level 'simde/' folder.")
    endif()
endif()

if(NOT BUILD_SHARED_LIBS)
    target_compile_definitions(cosyvoice PUBLIC COSYVOICE_STATIC)
    foreach(OBJ_LIB IN LISTS OBJECT_LIBS)
        target_compile_definitions(${OBJ_LIB} PUBLIC COSYVOICE_STATIC)
    endforeach()
elseif(NOT WIN32)
    set_target_properties(cosyvoice ${OBJECT_LIBS} PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN YES
    )
endif()

# Link the main library to core dependencies.
# Note: Object library dependencies (ICU, ONNX) are implicitly forwarded as link dependencies,
# but we still need to add ICU::data manually on Windows so it gets picked up by post-build DLL copy.
set(COSYVOICE_LINK_LIBS ggml pcre2-8 pcre2-16)
if(COSYVOICE_AUDIO_BACKEND STREQUAL "FFMPEG")
    list(APPEND COSYVOICE_LINK_LIBS ffmpeg)
endif()
if(NOT COSYVOICE_NO_ICU AND WIN32)
    list(APPEND COSYVOICE_LINK_LIBS ICU::data)
endif()
target_link_libraries(cosyvoice PRIVATE ${COSYVOICE_LINK_LIBS})

# -----------------------------------------------------------------------------
# Tools (Executables)
# -----------------------------------------------------------------------------
add_subdirectory(tools)

# -----------------------------------------------------------------------------
# Post-Build & Installation Rules
# -----------------------------------------------------------------------------
include(cmake/Install.cmake)
