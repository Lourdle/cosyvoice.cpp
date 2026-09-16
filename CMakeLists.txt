cmake_minimum_required(VERSION 3.28)
project(cosyvoice VERSION 0.1.3 LANGUAGES C CXX)

set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)

# Require 64-bit architecture — this project does not support 32-bit builds.
if(CMAKE_SIZEOF_VOID_P LESS 8)
    message(FATAL_ERROR "cosyvoice requires a 64-bit build environment. "
            "Please configure with a 64-bit compiler/toolchain.")
endif()

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

include(GNUInstallDirs)

if(UNIX AND NOT APPLE)
    # Keep both the build tree and the installed tree relocatable on Linux.
    # Executables live in <prefix>/bin while shared libraries live in <prefix>/lib.
    set(CMAKE_BUILD_RPATH_USE_ORIGIN ON)
    set(CMAKE_INSTALL_RPATH "$ORIGIN;$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")
endif()

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
# Target-ISA probe: ask the *compiler* (the VS generator lies -- on an ARM64
# cross build CMAKE_SYSTEM_PROCESSOR still reports the host's "AMD64"). The
# macros below mirror exactly what the SIMD headers use to gate intrinsics.
include(CheckCXXSourceCompiles)
check_cxx_source_compiles("#if !defined(__x86_64__) && !defined(_M_X64)\n#error this compiler targets a non-x86 ABI\n#endif\nint main() { return 0; }"
                          COSYVOICE_TARGET_IS_X86)

# Build-time SIMD inclusion knobs (default: everything on).
option(COSYVOICE_NO_SIMD "Disable all SIMD; force scalar-only" OFF)
option(COSYVOICE_HAS_SCALAR "Compile the pure-scalar fallback tier" ON)
option(COSYVOICE_HAS_SSE42 "Compile the SSE4.2-class SIMD tier" ON)
option(COSYVOICE_HAS_AVX "Compile the AVX-class SIMD tier" ON)
option(COSYVOICE_HAS_AVX2 "Compile the AVX2-class SIMD tier" ON)
option(COSYVOICE_HAS_AVX512 "Compile the AVX-512 (F+BW+DQ) SIMD tier" ON)  # runtime gate additionally requires VL (MSVC dead-branch EVEX.256 encodings)
option(COSYVOICE_HAS_AVX10_1 "Compile the AVX10.1-256 tier (512-bit AVX10 parts use the AVX-512 tier)" ON)

# The SIMD classes are x86(-64). On non-x86 the SSE4.2+FMA3 class can be built
# through SIMDe; without SIMDe the build auto-enables COSYVOICE_NO_SIMD (scalar
# only). The COSYVOICE_HAS_* macros gate every intrinsic block in the sources,
# so a disabled class is not merely uninstantiated -- it is never parsed.
# Effective class set after hierarchy enforcement: disabling a lower class
# (e.g. AVX2) automatically disables everything above it (AVX-512).

file(GLOB COSYVOICE_SOURCES "src/*.cpp" "src/*.h" "include/*.h")

if(COSYVOICE_NO_AUDIO)
    list(REMOVE_ITEM COSYVOICE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/cosyvoice-audio.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/cosyvoice-audio-ffmpeg.cpp")
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
target_include_directories(cosyvoice-frontend-text-norm PRIVATE "include")
target_link_libraries(cosyvoice-frontend-text-norm PRIVATE ggml)
if(NOT COSYVOICE_NO_ICU)
    target_link_libraries(cosyvoice-frontend-text-norm PRIVATE ICU::uc ICU::i18n)
else()
    target_compile_definitions(cosyvoice-frontend-text-norm PRIVATE COSYVOICE_NO_ICU)
endif()
list(APPEND OBJECT_LIBS cosyvoice-frontend-text-norm)

# ONNX Frontend
if(NOT COSYVOICE_NO_FRONTEND)
    add_library(cosyvoice-frontend OBJECT "src/cosyvoice-frontend.cpp")
    target_include_directories(cosyvoice-frontend PRIVATE "include" "common")
    target_link_libraries(cosyvoice-frontend PRIVATE ggml common onnxruntime)
    if(COSYVOICE_NO_AUDIO)
        target_compile_definitions(cosyvoice-frontend PRIVATE COSYVOICE_NO_AUDIO)
    endif()
        list(APPEND OBJECT_LIBS cosyvoice-frontend)
endif()

# simd_detect.cpp is the only definition of g_simd_hw_caps/g_simd_caps (and
# the COSYVOICE_SIMD_LEVEL env initializer), referenced solely by x86 dispatch
# and the SIMD control API -- drop it from the sources when they can never be
# used (the old in-function REMOVE_ITEM was dead: add_library() had already
# consumed the list).
if(COSYVOICE_NO_SIMD OR NOT COSYVOICE_TARGET_IS_X86)
    list(REMOVE_ITEM COSYVOICE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/simd_detect.cpp")
endif()

# cosyvoice-simd.cpp implements the x86-only SIMD detection/control C API (the
# public header declares those symbols only on x86); on non-x86 it is an empty
# TU -- exclude it. Note: it must STAY on x86 even under COSYVOICE_NO_SIMD,
# where it exports the static scalar-only reporting picture. The
# COSYVOICE_TARGET_IS_X86 probe mirrors exactly the header's
# __x86_64__/_M_X64 gate, so declarations and exports can never disagree.
if(NOT COSYVOICE_TARGET_IS_X86)
    list(REMOVE_ITEM COSYVOICE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/cosyvoice-simd.cpp")
endif()

add_library(cosyvoice ${COSYVOICE_SOURCES})
target_link_libraries(cosyvoice PRIVATE common ${OBJECT_LIBS})

# Silence clang/GCC warnings (-Wdangling-else from -Wall, -Wimplicit-fallthrough
# from -Wextra, -Wswitch / -Wnontrivial-memcall) for the cosyvoice library sources only.
if(NOT MSVC)
    target_compile_options(cosyvoice PRIVATE -Wno-dangling-else -Wno-implicit-fallthrough -Wno-switch -Wno-nontrivial-memcall)
    target_compile_options(cosyvoice-frontend-text-norm PRIVATE -Wno-dangling-else -Wno-implicit-fallthrough -Wno-switch -Wno-nontrivial-memcall)
    if(TARGET cosyvoice-frontend)
        target_compile_options(cosyvoice-frontend PRIVATE -Wno-dangling-else -Wno-implicit-fallthrough -Wno-switch -Wno-nontrivial-memcall)
    endif()
endif()

target_include_directories(cosyvoice PUBLIC "include")
target_include_directories(cosyvoice PRIVATE "vendor/miniaudio" "common")
if(COSYVOICE_AUDIO_BACKEND STREQUAL "FFMPEG" AND EXISTS "${FFMPEG_PREBUILT_DIR}/include")
    target_include_directories(cosyvoice PRIVATE "${FFMPEG_PREBUILT_DIR}/include")
endif()

set_target_properties(cosyvoice PROPERTIES
    VERSION   ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

# -----------------------------------------------------------------------------
# Compiler Flags & SIMD Support
# -----------------------------------------------------------------------------
if(NOT MSVC)
    add_compile_definitions("$<$<CONFIG:Debug>:_DEBUG>")
endif()

# -----------------------------------------------------------------------------
# Effective SIMD classes. COSYVOICE_HAS_* cache options request a class; the
# build platform decides what actually exists:
#   x86:            user cache options, cascaded (AVX needs SSE4.2 class, ...).
#   non-x86+SIMDe:  SSE4.2+FMA3 class only (SIMDe emulation); nothing else.
#   non-x86 no SIMDe: COSYVOICE_NO_SIMD is auto-enabled -> scalar only. All the
#                   intrinsics-dependent code is #ifdef'd out in this case, so
#                   no SIMDe declarations are required anywhere.
# The _EFF_* variables below are the single source of truth for the tier
# targets and the COSYVOICE_HAS_* compile definitions.
# -----------------------------------------------------------------------------
if(NOT COSYVOICE_TARGET_IS_X86)
    find_path(SIMDE_INCLUDE_DIR NAMES simde/x86/sse4.2.h
              HINTS /opt/homebrew/include /usr/local/include /usr/include
                    "${CMAKE_CURRENT_SOURCE_DIR}/vendor/simde")
    # An auto-enabled scalar fallback is cleared as soon as SIMDe shows up.
    if(COSYVOICE_NO_SIMD AND COSYVOICE_NO_SIMD_AUTO AND SIMDE_INCLUDE_DIR)
        set(COSYVOICE_NO_SIMD OFF CACHE BOOL "" FORCE)
        set(COSYVOICE_NO_SIMD_AUTO OFF CACHE BOOL "" FORCE)
    elseif(COSYVOICE_NO_SIMD AND NOT SIMDE_INCLUDE_DIR)
        # user-requested or stale auto -- both fine, scalar only
    elseif(NOT SIMDE_INCLUDE_DIR)
        set(COSYVOICE_NO_SIMD ON CACHE BOOL "" FORCE)
        set(COSYVOICE_NO_SIMD_AUTO ON CACHE BOOL "internal: NO_SIMD auto-set (non-x86, no SIMDe)" FORCE)
        message(STATUS "cosyvoice: non-x86 target without SIMDe -> SIMD auto-disabled "
                       "(scalar-only; define SIMDE_INCLUDE_DIR to get the SSE4.2+FMA3 tier)")
    endif()
endif()

# The AVX10-256 tier additionally requires a toolchain that knows the flag
# (-mavx10.1-256; MSVC /arch:AVX10.1 /vlen=256). Probe once per config; an
# older compiler simply does not get that class. There is no AVX10-512 tier or
# probe at all: AVX10 512-bit parts run the AVX-512 tier's kernels (the same
# instruction space, compiled with plain -mavx512*/arch:AVX512 flags, so even
# toolchains without any AVX10 support serve those CPUs) -- see simd-dispatch.h.
if(COSYVOICE_TARGET_IS_X86 AND NOT COSYVOICE_NO_SIMD)
    include(CheckCXXSourceCompiles)
    if(MSVC)
        set(CMAKE_REQUIRED_FLAGS "/arch:AVX10.1 /vlen=256")
        check_cxx_source_compiles("int main() { return 0; }" _AVX10_1_256_OK)
    else()
        set(CMAKE_REQUIRED_FLAGS "-mavx10.1-256")
        check_cxx_source_compiles("int main() { return 0; }" _AVX10_1_256_OK)
    endif()
    set(CMAKE_REQUIRED_FLAGS "")
endif()

if(COSYVOICE_NO_SIMD)
    set(_EFF_SSE42 OFF)
    set(_EFF_AVX   OFF)
    set(_EFF_AVX2  OFF)
    set(_EFF_512   OFF)
    set(_EFF_10_1_256 OFF)
elseif(NOT COSYVOICE_TARGET_IS_X86)
    set(_EFF_SSE42 ON)
    set(_EFF_AVX   OFF)
    set(_EFF_AVX2  OFF)
    set(_EFF_512   OFF)
    set(_EFF_10_1_256 OFF)
else()
    set(_EFF_SSE42 ${COSYVOICE_HAS_SSE42})
    set(_EFF_AVX   ${COSYVOICE_HAS_AVX})
    set(_EFF_AVX2  ${COSYVOICE_HAS_AVX2})
    set(_EFF_512   ${COSYVOICE_HAS_AVX512})
    set(_EFF_10_1_256 OFF)
    if(COSYVOICE_HAS_AVX10_1 AND _AVX10_1_256_OK)
        set(_EFF_10_1_256 ON)
    endif()
    if(NOT _EFF_SSE42)
        set(_EFF_AVX OFF)
        set(_EFF_AVX2 OFF)
        set(_EFF_512 OFF)
    elseif(NOT _EFF_AVX)
        set(_EFF_AVX2 OFF)
        set(_EFF_512 OFF)
    elseif(NOT _EFF_AVX2)
        set(_EFF_512 OFF)
    endif()
endif()

# Apply the SIMD compile definitions (+ SIMDe on non-x86) to one library target.
function(cosyvoice_apply_simd TARGET_NAME)
    if(NOT TARGET ${TARGET_NAME})
        return()
    endif()

    # Under COSYVOICE_NO_SIMD nothing else is defined: every SIMD helper/body
    # region, including the dependent-call sites, is #ifdef'd out.
    if(COSYVOICE_NO_SIMD)
        target_compile_definitions(${TARGET_NAME} PRIVATE COSYVOICE_NO_SIMD)
        return()
    endif()

    # Presence semantics: a class is compiled in iff its COSYVOICE_HAS_* macro is
    # defined; disabled classes are simply NOT defined -- the kernel bodies and
    # dispatch branches then do not exist (preprocessor), so platforms without
    # the intrinsics (non-x86, no SIMDe) never name them.
    if(COSYVOICE_HAS_SCALAR)
        target_compile_definitions(${TARGET_NAME} PRIVATE COSYVOICE_HAS_SCALAR)
    endif()

    if(_EFF_SSE42)
        target_compile_definitions(${TARGET_NAME} PRIVATE COSYVOICE_HAS_SSE42)
    endif()
    if(_EFF_AVX)
        target_compile_definitions(${TARGET_NAME} PRIVATE COSYVOICE_HAS_AVX)
    endif()
    if(_EFF_AVX2)
        target_compile_definitions(${TARGET_NAME} PRIVATE COSYVOICE_HAS_AVX2)
    endif()
    if(_EFF_512)
        target_compile_definitions(${TARGET_NAME} PRIVATE COSYVOICE_HAS_AVX512)
    endif()
    if(_EFF_10_1_256)
        target_compile_definitions(${TARGET_NAME} PRIVATE COSYVOICE_HAS_AVX10_1_256)
    endif()

    # Non-x86 (SSE4.2+FMA3 class): SIMDe provides the intrinsic declarations
    # the compiled-in bodies reference; guaranteed present here (the scalar
    # auto-fallback above sets NO_SIMD when it is missing).
    if(NOT COSYVOICE_TARGET_IS_X86)
        target_include_directories(${TARGET_NAME} SYSTEM PRIVATE ${SIMDE_INCLUDE_DIR})
    endif()
endfunction()

foreach(_SIMD_TARGET cosyvoice cosyvoice-frontend cosyvoice-frontend-text-norm)
    cosyvoice_apply_simd(${_SIMD_TARGET})
endforeach()

# -----------------------------------------------------------------------------
# Per-tier SIMD kernel objects. Each SIMD class is ONE source file built as its
# own OBJECT library with ONLY that class's ISA flags: -m<isa>... on GNU/Clang,
# /arch:<isa> on MSVC (>=17.10 provides /arch:SSE4.2; /arch:AVX10.1 with
# /vlen=N for the AVX10 tiers -- AVX10.2 parts run the 10.1 tiers). Kernel bodies live in
# simd-kernels-impl.h, which is included solely by these TUs, so every emitted
# instruction -- intrinsics AND compiler auto-vectorization -- is bounded to the
# tier's own ISA, while caller TUs stay at the plain SSE2 baseline. The scalar
# fallback deliberately carries no flags on either toolchain.
# -----------------------------------------------------------------------------
set(COSYVOICE_TIER_LIBS "")

# Scalar fallback (simd_none): build on x86 (dispatch fallback) or when all SIMD
# is disabled (COSYVOICE_NO_SIMD).  On non-x86 + SIMDe the sse42 tier already
# provides the best path via NEON emulation; scalar is never dispatched.
if(COSYVOICE_TARGET_IS_X86 OR COSYVOICE_NO_SIMD)
    add_library(cosyvoice-tier-scalar OBJECT "${CMAKE_CURRENT_SOURCE_DIR}/src/simd/kernels_scalar.cpp")
    list(APPEND COSYVOICE_TIER_LIBS cosyvoice-tier-scalar)
endif()

function(add_cosyvoice_simd_tier TIER TIER_SRC FLAGS MSVC_ARCH EFF)
    if(NOT EFF OR COSYVOICE_NO_SIMD)
        return()
    endif()
    # Non-x86 (SIMDe-emulated tier): no native ISA flags apply.
    if(NOT COSYVOICE_TARGET_IS_X86)
        set(FLAGS "")
        set(MSVC_ARCH "")
    endif()

    add_library(cosyvoice-tier-${TIER} OBJECT "${TIER_SRC}")

    if(MSVC)
        if(MSVC_ARCH)
            target_compile_options(cosyvoice-tier-${TIER} PRIVATE ${MSVC_ARCH})
        endif()
    else()
        if(FLAGS)
            target_compile_options(cosyvoice-tier-${TIER} PRIVATE ${FLAGS})
        endif()
        target_compile_options(cosyvoice-tier-${TIER} PRIVATE
            -Wno-dangling-else -Wno-implicit-fallthrough -Wno-switch -Wno-nontrivial-memcall)
    endif()

    list(APPEND COSYVOICE_TIER_LIBS cosyvoice-tier-${TIER})
    set(COSYVOICE_TIER_LIBS ${COSYVOICE_TIER_LIBS} PARENT_SCOPE)
endfunction()

add_cosyvoice_simd_tier(sse42   "${CMAKE_CURRENT_SOURCE_DIR}/src/simd/kernels_sse42.cpp"   "-msse4.2" "/arch:SSE4.2" ${_EFF_SSE42})
add_cosyvoice_simd_tier(avx     "${CMAKE_CURRENT_SOURCE_DIR}/src/simd/kernels_avx.cpp"     "-msse4.2;-mavx" "/arch:AVX" ${_EFF_AVX})
add_cosyvoice_simd_tier(avx2    "${CMAKE_CURRENT_SOURCE_DIR}/src/simd/kernels_avx2.cpp"    "-msse4.2;-mavx;-mavx2;-mfma" "/arch:AVX2" ${_EFF_AVX2})
add_cosyvoice_simd_tier(avx512  "${CMAKE_CURRENT_SOURCE_DIR}/src/simd/kernels_avx512.cpp"  "-msse4.2;-mavx;-mavx2;-mfma;-mavx512f;-mavx512bw;-mavx512dq" "/arch:AVX512" ${_EFF_512})
add_cosyvoice_simd_tier(avx10_1_256 "${CMAKE_CURRENT_SOURCE_DIR}/src/simd/kernels_avx10_1_256.cpp" "-mavx10.1-256" "/arch:AVX10.1;/vlen=256" ${_EFF_10_1_256})

# The callers' unresolved kernel references resolve against these objects when
# libcosyvoice is linked. Scalar (simd_none) is included when built (x86 or
# COSYVOICE_NO_SIMD); on non-x86 + SIMDe it is absent by design.
if(COSYVOICE_TIER_LIBS)
    target_link_libraries(cosyvoice PRIVATE ${COSYVOICE_TIER_LIBS})
endif()

# Tier TUs include simd-kernels-impl.h / simd-math.h, whose SIMD guards
# (COSYVOICE_NO_SIMD, the SIMDe includes) key off exactly the same settings as
# the callers -- so every tier object must receive them too, above all the
# SIMDe include directory on non-x86 builds.
foreach(_TIER_TGT IN LISTS COSYVOICE_TIER_LIBS)
    cosyvoice_apply_simd(${_TIER_TGT})
endforeach()

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
set(COSYVOICE_LINK_LIBS ggml pcre2-8 pcre2-16 common)
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
