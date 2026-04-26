# -----------------------------------------------------------------------------
# Dependencies
# -----------------------------------------------------------------------------

# 1. PCRE2
# We force PCRE2 to be built as a static library regardless of the global BUILD_SHARED_LIBS
set(ORIGINAL_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build static PCRE2" FORCE)
set(PCRE2_BUILD_PCRE2_8 ON CACHE BOOL "Build 8-bit PCRE2" FORCE)
set(PCRE2_BUILD_PCRE2_16 ON CACHE BOOL "Build 16-bit PCRE2" FORCE)
set(PCRE2_BUILD_TESTS OFF CACHE BOOL "Disable PCRE2 tests" FORCE)
set(PCRE2_BUILD_GREP OFF CACHE BOOL "Disable PCRE2 grep" FORCE)
set(PCRE2_BUILD_PCRE2GREP OFF CACHE BOOL "Disable pcre2grep executable" FORCE)
add_subdirectory(vendor/pcre2)
# Restore the global BUILD_SHARED_LIBS setting for subsequent targets (like ggml and cosyvoice)
set(BUILD_SHARED_LIBS ${ORIGINAL_BUILD_SHARED_LIBS} CACHE BOOL "Build shared libraries" FORCE)

# 2. GGML
if(NOT EXISTS "${GGML_SOURCE_DIR}/CMakeLists.txt")
    message(STATUS "ggml not found in ${GGML_SOURCE_DIR}. Cloning from https://github.com/ggml-org/ggml.git...")
    execute_process(
        COMMAND git clone --depth=1 https://github.com/ggml-org/ggml.git "${GGML_SOURCE_DIR}"
    )
endif()

# Apply local patches to ggml (idempotent — skips if already applied).
set(GGML_PAD_PATCH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/patches/ggml-metal-pad-beg.patch")
if(APPLE AND EXISTS "${GGML_PAD_PATCH}")
    execute_process(
        COMMAND git apply --check "${GGML_PAD_PATCH}"
        WORKING_DIRECTORY "${GGML_SOURCE_DIR}"
        RESULT_VARIABLE PATCH_CHECK_RESULT
        OUTPUT_QUIET ERROR_QUIET
    )
    if(PATCH_CHECK_RESULT EQUAL 0)
        message(STATUS "Applying ggml-metal PAD beg-padding patch...")
        execute_process(
            COMMAND git apply "${GGML_PAD_PATCH}"
            WORKING_DIRECTORY "${GGML_SOURCE_DIR}"
        )
    else()
        message(STATUS "ggml-metal PAD beg-padding patch already applied or not applicable — skipping.")
    endif()
endif()

add_subdirectory("${GGML_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/_deps/ggml-build")

# 3. ICU
if(NOT COSYVOICE_NO_ICU)
  if(NOT EXISTS "${ICU_PREBUILT_DIR}/include/unicode/utypes.h")
    find_package(ICU COMPONENTS i18n uc data QUIET)

    if(ICU_FOUND)
      message(STATUS "Found ICU via find_package: ${ICU_INCLUDE_DIRS}")
    else()
      if(WIN32)
        message(STATUS "ICU not found. Downloading pre-built Windows binaries to ${ICU_PREBUILT_DIR}...")

        if(CMAKE_SYSTEM_PROCESSOR MATCHES "ARM64|aarch64")
          set(ICU_ZIP_URL "https://github.com/unicode-org/icu/releases/download/release-78.2/icu4c-78.2-WinARM64-MSVC2022.zip")
        else()
          set(ICU_ZIP_URL "https://github.com/unicode-org/icu/releases/download/release-78.2/icu4c-78.2-Win64-MSVC2022.zip")
        endif()

        file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/_deps")
        file(DOWNLOAD "${ICU_ZIP_URL}" "${CMAKE_BINARY_DIR}/_deps/icu.zip" STATUS DL_STATUS)
        list(GET DL_STATUS 0 DL_ERR_CODE)

        if(DL_ERR_CODE EQUAL 0)
          file(MAKE_DIRECTORY "${ICU_PREBUILT_DIR}")
          execute_process(
            COMMAND ${CMAKE_COMMAND} -E tar xf "${CMAKE_BINARY_DIR}/_deps/icu.zip"
            WORKING_DIRECTORY "${ICU_PREBUILT_DIR}"
          )

          # Auto-detect actual ICU root after extraction.
          unset(_ICU_FOUND_ROOT)
          if(EXISTS "${ICU_PREBUILT_DIR}/include/unicode/utypes.h")
            set(_ICU_FOUND_ROOT "${ICU_PREBUILT_DIR}")
          endif()

          if(NOT _ICU_FOUND_ROOT)
            file(GLOB _ICU_DIRS LIST_DIRECTORIES true "${ICU_PREBUILT_DIR}/*")
            foreach(_dir IN LISTS _ICU_DIRS)
              if(EXISTS "${_dir}/include/unicode/utypes.h")
                set(_ICU_FOUND_ROOT "${_dir}")
                break()
              endif()
            endforeach()
          endif()

          if(_ICU_FOUND_ROOT)
            set(ICU_PREBUILT_DIR "${_ICU_FOUND_ROOT}" CACHE PATH "Path to custom prebuilt ICU" FORCE)
            message(STATUS "Using detected ICU root: ${ICU_PREBUILT_DIR}")
          else()
            message(FATAL_ERROR
              "ICU downloaded but could not locate include/unicode/utypes.h under ${CMAKE_BINARY_DIR}/_deps"
            )
          endif()
        else()
          message(FATAL_ERROR "Failed to download ICU from ${ICU_ZIP_URL}")
        endif()
      else()
        message(FATAL_ERROR "ICU not found! Please install libicu-dev (Linux) or icu4c (macOS).")
      endif()
    endif()
  endif()

  if(EXISTS "${ICU_PREBUILT_DIR}/include/unicode/utypes.h")
    message(STATUS "Using prebuilt ICU from ${ICU_PREBUILT_DIR}")

    add_library(ICU::uc SHARED IMPORTED)
    add_library(ICU::i18n SHARED IMPORTED)

    if(WIN32)
      file(GLOB ICU_UC_DLL   "${ICU_PREBUILT_DIR}/bin64/icuuc*.dll" "${ICU_PREBUILT_DIR}/lib64/icuuc*.dll")
      file(GLOB ICU_I18N_DLL "${ICU_PREBUILT_DIR}/bin64/icuin*.dll" "${ICU_PREBUILT_DIR}/lib64/icuin*.dll")
      file(GLOB ICU_DT_DLL   "${ICU_PREBUILT_DIR}/bin64/icudt*.dll" "${ICU_PREBUILT_DIR}/lib64/icudt*.dll")

      list(LENGTH ICU_UC_DLL _icu_uc_len)
      list(LENGTH ICU_I18N_DLL _icu_i18n_len)
      list(LENGTH ICU_DT_DLL _icu_dt_len)

      if(_icu_uc_len EQUAL 0 OR _icu_i18n_len EQUAL 0 OR _icu_dt_len EQUAL 0)
        message(FATAL_ERROR
          "ICU root found at ${ICU_PREBUILT_DIR}, but required DLLs were not found in bin64/lib64"
        )
      endif()

      list(GET ICU_UC_DLL 0 ICU_UC_DLL_PATH)
      list(GET ICU_I18N_DLL 0 ICU_I18N_DLL_PATH)
      list(GET ICU_DT_DLL 0 ICU_DT_DLL_PATH)

      add_library(ICU::data SHARED IMPORTED)

      set_target_properties(ICU::data PROPERTIES
        IMPORTED_LOCATION "${ICU_DT_DLL_PATH}"
        IMPORTED_IMPLIB "${ICU_PREBUILT_DIR}/lib64/icudt.lib"
      )
      set_target_properties(ICU::uc PROPERTIES
        IMPORTED_LOCATION "${ICU_UC_DLL_PATH}"
        IMPORTED_IMPLIB "${ICU_PREBUILT_DIR}/lib64/icuuc.lib"
        INTERFACE_INCLUDE_DIRECTORIES "${ICU_PREBUILT_DIR}/include"
      )
      set_target_properties(ICU::i18n PROPERTIES
        IMPORTED_LOCATION "${ICU_I18N_DLL_PATH}"
        IMPORTED_IMPLIB "${ICU_PREBUILT_DIR}/lib64/icuin.lib"
        INTERFACE_INCLUDE_DIRECTORIES "${ICU_PREBUILT_DIR}/include"
      )
    else()
      set_target_properties(ICU::uc PROPERTIES
        IMPORTED_LOCATION "${ICU_PREBUILT_DIR}/lib/libicuuc.so"
        INTERFACE_INCLUDE_DIRECTORIES "${ICU_PREBUILT_DIR}/include"
      )
      set_target_properties(ICU::i18n PROPERTIES
        IMPORTED_LOCATION "${ICU_PREBUILT_DIR}/lib/libicui18n.so"
        INTERFACE_INCLUDE_DIRECTORIES "${ICU_PREBUILT_DIR}/include"
      )
    endif()
  endif()
endif()

# 4. ONNX Runtime
if(NOT COSYVOICE_NO_FRONTEND)
    if(NOT EXISTS "${ORT_PREBUILT_DIR}/include/onnxruntime_c_api.h")
        find_package(onnxruntime QUIET)
        if(onnxruntime_FOUND)
            message(STATUS "Found ONNX Runtime via find_package")
        else()
            message(STATUS "ONNX Runtime not found. Downloading pre-built binaries to ${ORT_PREBUILT_DIR}...")
            if(WIN32)
                if(CMAKE_SYSTEM_PROCESSOR MATCHES "ARM64|aarch64")
                    set(ORT_URL "https://github.com/microsoft/onnxruntime/releases/download/v1.24.4/onnxruntime-win-arm64-1.24.4.zip")
                    set(ORT_DIR "onnxruntime-win-arm64-1.24.4")
                else()
                    set(ORT_URL "https://github.com/microsoft/onnxruntime/releases/download/v1.24.4/onnxruntime-win-x64-1.24.4.zip")
                    set(ORT_DIR "onnxruntime-win-x64-1.24.4")
                endif()
                set(ORT_EXT "zip")
            elseif(APPLE)
                set(ORT_URL "https://github.com/microsoft/onnxruntime/releases/download/v1.24.4/onnxruntime-osx-arm64-1.24.4.tgz")
                set(ORT_DIR "onnxruntime-osx-arm64-1.24.4")
                set(ORT_EXT "tgz")
            else()
                if(CMAKE_SYSTEM_PROCESSOR MATCHES "ARM64|aarch64")
                    set(ORT_URL "https://github.com/microsoft/onnxruntime/releases/download/v1.24.4/onnxruntime-linux-aarch64-1.24.4.tgz")
                    set(ORT_DIR "onnxruntime-linux-aarch64-1.24.4")
                else()
                    set(ORT_URL "https://github.com/microsoft/onnxruntime/releases/download/v1.24.4/onnxruntime-linux-x64-1.24.4.tgz")
                    set(ORT_DIR "onnxruntime-linux-x64-1.24.4")
                endif()
                set(ORT_EXT "tgz")
            endif()
            
            file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/_deps")
            file(DOWNLOAD ${ORT_URL} "${CMAKE_BINARY_DIR}/_deps/ort.${ORT_EXT}" STATUS DL_STATUS)
            list(GET DL_STATUS 0 DL_ERR_CODE)
            if(DL_ERR_CODE EQUAL 0)
                execute_process(COMMAND ${CMAKE_COMMAND} -E tar xf "${CMAKE_BINARY_DIR}/_deps/ort.${ORT_EXT}" WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/_deps/")
                file(RENAME "${CMAKE_BINARY_DIR}/_deps/${ORT_DIR}" "${ORT_PREBUILT_DIR}")
            else()
                message(FATAL_ERROR "Failed to download ONNX Runtime from ${ORT_URL}")
            endif()
        endif()
    endif()

    if(EXISTS "${ORT_PREBUILT_DIR}/include/onnxruntime_c_api.h")
        message(STATUS "Using prebuilt ONNX Runtime from ${ORT_PREBUILT_DIR}")
        add_library(onnxruntime SHARED IMPORTED)
        if(WIN32)
            set_target_properties(onnxruntime PROPERTIES
                IMPORTED_LOCATION "${ORT_PREBUILT_DIR}/lib/onnxruntime.dll"
                IMPORTED_IMPLIB "${ORT_PREBUILT_DIR}/lib/onnxruntime.lib"
                INTERFACE_INCLUDE_DIRECTORIES "${ORT_PREBUILT_DIR}/include"
            )
        elseif(APPLE)
            set_target_properties(onnxruntime PROPERTIES
                IMPORTED_LOCATION "${ORT_PREBUILT_DIR}/lib/libonnxruntime.dylib"
                INTERFACE_INCLUDE_DIRECTORIES "${ORT_PREBUILT_DIR}/include"
            )
        else()
            set_target_properties(onnxruntime PROPERTIES
                IMPORTED_LOCATION "${ORT_PREBUILT_DIR}/lib/libonnxruntime.so"
                INTERFACE_INCLUDE_DIRECTORIES "${ORT_PREBUILT_DIR}/include"
            )
        endif()
    endif()
endif()

# 5. FFmpeg (optional, only needed for audio when using FFMPEG backend)
if(COSYVOICE_AUDIO_BACKEND STREQUAL "FFMPEG")
    if(NOT EXISTS "${FFMPEG_PREBUILT_DIR}/include/libavcodec/avcodec.h")
        if(WIN32)
            if(CMAKE_SYSTEM_PROCESSOR MATCHES "ARM64|aarch64")
                set(FFMPEG_URL "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n8.1-latest-winarm64-lgpl-shared-8.1.zip")
                set(FFMPEG_DIR "ffmpeg-n8.1-latest-winarm64-lgpl-shared-8.1")
            else()
                set(FFMPEG_URL "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n8.1-latest-win64-lgpl-shared-8.1.zip")
                set(FFMPEG_DIR "ffmpeg-n8.1-latest-win64-lgpl-shared-8.1")
            endif()
            message(STATUS "FFmpeg not found. Downloading pre-built binaries to ${FFMPEG_PREBUILT_DIR}...")

            file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/_deps")
            file(DOWNLOAD "${FFMPEG_URL}" "${CMAKE_BINARY_DIR}/_deps/ffmpeg.zip" STATUS DL_STATUS)
            list(GET DL_STATUS 0 DL_ERR_CODE)

            if(DL_ERR_CODE EQUAL 0)
                execute_process(
                    COMMAND ${CMAKE_COMMAND} -E tar xf "${CMAKE_BINARY_DIR}/_deps/ffmpeg.zip"
                    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/_deps"
                )
                file(RENAME "${CMAKE_BINARY_DIR}/_deps/${FFMPEG_DIR}" "${FFMPEG_PREBUILT_DIR}")
            else()
                message(FATAL_ERROR "Failed to download FFmpeg from ${FFMPEG_URL}")
            endif()
        elseif(APPLE)
            find_program(FFMPEG_EXTERNAL ffmpeg)
            if(NOT FFMPEG_EXTERNAL)
                message(FATAL_ERROR "FFmpeg not found! Please install via Homebrew: brew install ffmpeg")
            endif()
            execute_process(COMMAND which ffmpeg OUTPUT_VARIABLE FFMPEG_PATH OUTPUT_STRIP_TRAILING_WHITESPACE)
            message(STATUS "Found FFmpeg at: ${FFMPEG_PATH}")
            set(FFMPEG_PREBUILT_DIR "${FFMPEG_PATH}" CACHE PATH "FFmpeg path" FORCE)
        else()
            find_package(PkgConfig QUIET)
            if(PkgConfig_FOUND)
                pkg_check_modules(FFMPEG libavcodec libavformat libavutil libswresample QUIET)
            endif()
            if(NOT FFMPEG_FOUND)
                message(FATAL_ERROR "FFmpeg not found! Please install libavcodec-dev, libavformat-dev, libavutil-dev, libswresample-dev (Debian/Ubuntu) or equivalent.")
            endif()
            set(FFMPEG_PREBUILT_DIR "${FFMPEG_PREFIX}" CACHE PATH "FFmpeg path" FORCE)
        endif()
    endif()

    if(EXISTS "${FFMPEG_PREBUILT_DIR}/include/libavcodec/avcodec.h")
        message(STATUS "Using FFmpeg from ${FFMPEG_PREBUILT_DIR}")
        add_library(ffmpeg INTERFACE IMPORTED)

        if(WIN32)
            file(GLOB FFMPEG_DLLS "${FFMPEG_PREBUILT_DIR}/bin/*.dll")
            list(LENGTH FFMPEG_DLLS _ffmpeg_dlls_len)
            if(_ffmpeg_dlls_len EQUAL 0)
                message(FATAL_ERROR "FFmpeg DLLs not found in ${FFMPEG_PREBUILT_DIR}/bin")
            endif()
            set(FFMPEG_RUNTIME_DLLS ${FFMPEG_DLLS})

            set_target_properties(ffmpeg PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES "${FFMPEG_PREBUILT_DIR}/include"
                INTERFACE_LINK_LIBRARIES
                    "${FFMPEG_PREBUILT_DIR}/lib/avcodec.lib;${FFMPEG_PREBUILT_DIR}/lib/avformat.lib;${FFMPEG_PREBUILT_DIR}/lib/avutil.lib;${FFMPEG_PREBUILT_DIR}/lib/swresample.lib"
            )
        else()
            find_library(FFMPEG_AVCODEC_LIBRARY NAMES avcodec HINTS "${FFMPEG_PREBUILT_DIR}/lib" "/opt/homebrew/lib" "/usr/local/lib" "/usr/lib" "/usr/lib64")
            find_library(FFMPEG_AVFORMAT_LIBRARY NAMES avformat HINTS "${FFMPEG_PREBUILT_DIR}/lib" "/opt/homebrew/lib" "/usr/local/lib" "/usr/lib" "/usr/lib64")
            find_library(FFMPEG_AVUTIL_LIBRARY NAMES avutil HINTS "${FFMPEG_PREBUILT_DIR}/lib" "/opt/homebrew/lib" "/usr/local/lib" "/usr/lib" "/usr/lib64")
            find_library(FFMPEG_SWRESAMPLE_LIBRARY NAMES swresample HINTS "${FFMPEG_PREBUILT_DIR}/lib" "/opt/homebrew/lib" "/usr/local/lib" "/usr/lib" "/usr/lib64")

            if(NOT FFMPEG_AVCODEC_LIBRARY OR NOT FFMPEG_AVFORMAT_LIBRARY OR NOT FFMPEG_AVUTIL_LIBRARY OR NOT FFMPEG_SWRESAMPLE_LIBRARY)
                message(FATAL_ERROR "FFmpeg libraries not found. Install ffmpeg dev packages or Homebrew ffmpeg.")
            endif()

            set_target_properties(ffmpeg PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES "${FFMPEG_PREBUILT_DIR}/include"
                INTERFACE_LINK_LIBRARIES "${FFMPEG_AVCODEC_LIBRARY};${FFMPEG_AVFORMAT_LIBRARY};${FFMPEG_AVUTIL_LIBRARY};${FFMPEG_SWRESAMPLE_LIBRARY}"
            )
        endif()
    endif()
endif()
