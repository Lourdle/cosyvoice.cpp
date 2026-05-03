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

function(cosyvoice_fetch_github_latest_tag REPO OUT_TAG)
  string(REPLACE "/" "_" _repo_id "${REPO}")
  set(_latest_json "${CMAKE_BINARY_DIR}/_deps/${_repo_id}_latest.json")
  file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/_deps")

  file(DOWNLOAD
    "https://api.github.com/repos/${REPO}/releases/latest"
    "${_latest_json}"
    STATUS _download_status
    TLS_VERIFY ON
    HTTPHEADER "Accept: application/vnd.github+json"
    HTTPHEADER "User-Agent: cosyvoice-cmake"
  )

  list(GET _download_status 0 _download_code)
  if(NOT _download_code EQUAL 0)
    set(${OUT_TAG} "" PARENT_SCOPE)
    return()
  endif()

  file(READ "${_latest_json}" _latest_json_content)
  string(REGEX MATCH "\"tag_name\"[ \t\r\n]*:[ \t\r\n]*\"([^\"]+)\"" _tag_match "${_latest_json_content}")
  if(NOT CMAKE_MATCH_1)
    set(${OUT_TAG} "" PARENT_SCOPE)
    return()
  endif()

  set(${OUT_TAG} "${CMAKE_MATCH_1}" PARENT_SCOPE)
endfunction()

# 3. ICU
if(NOT COSYVOICE_NO_ICU)
  if(NOT EXISTS "${ICU_PREBUILT_DIR}/include/unicode/utypes.h")
    find_package(ICU COMPONENTS i18n uc data QUIET)

    if(ICU_FOUND)
      message(STATUS "Found ICU via find_package: ${ICU_INCLUDE_DIRS}")
    else()
      if(WIN32)
        message(STATUS "ICU not found. Downloading pre-built Windows binaries to ${ICU_PREBUILT_DIR}...")

        set(_ICU_FALLBACK_VERSION "78.2")
        set(_ICU_FALLBACK_TAG "release-78.2")
        set(_ICU_RELEASE_TAG "")
        cosyvoice_fetch_github_latest_tag("unicode-org/icu" _ICU_RELEASE_TAG)

        if(_ICU_RELEASE_TAG MATCHES "^release-([0-9]+)\.([0-9]+)$")
          set(_ICU_VERSION_MAJOR "${CMAKE_MATCH_1}")
          set(_ICU_VERSION_MINOR "${CMAKE_MATCH_2}")
          set(_ICU_VERSION "${_ICU_VERSION_MAJOR}.${_ICU_VERSION_MINOR}")
          message(STATUS "ICU latest release: tag=${_ICU_RELEASE_TAG}, version=${_ICU_VERSION}")
        else()
          set(_ICU_VERSION "${_ICU_FALLBACK_VERSION}")
          set(_ICU_RELEASE_TAG "${_ICU_FALLBACK_TAG}")
          message(STATUS "Failed to query latest ICU release; using configured version=${_ICU_VERSION}")
        endif()

        if(CMAKE_SYSTEM_PROCESSOR MATCHES "ARM64|aarch64")
          set(ICU_ZIP_URL "https://github.com/unicode-org/icu/releases/download/${_ICU_RELEASE_TAG}/icu4c-${_ICU_VERSION}-WinARM64-MSVC2022.zip")
        else()
          set(ICU_ZIP_URL "https://github.com/unicode-org/icu/releases/download/${_ICU_RELEASE_TAG}/icu4c-${_ICU_VERSION}-Win64-MSVC2022.zip")
        endif()

        message(STATUS "ICU download URL: ${ICU_ZIP_URL}")

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
        find_package(PkgConfig QUIET)
        if(PkgConfig_FOUND)
          pkg_check_modules(ICU QUIET IMPORTED_TARGET icu-i18n icu-uc icu-data)
        endif()

        if(ICU_FOUND)
          message(STATUS "Found ICU via pkg-config: ${ICU_INCLUDE_DIRS}")
          if(NOT TARGET ICU::uc)
            add_library(ICU::uc INTERFACE IMPORTED)
            target_link_libraries(ICU::uc INTERFACE PkgConfig::ICU)
          endif()
          if(NOT TARGET ICU::i18n)
            add_library(ICU::i18n INTERFACE IMPORTED)
            target_link_libraries(ICU::i18n INTERFACE PkgConfig::ICU)
          endif()
          if(NOT TARGET ICU::data)
            add_library(ICU::data INTERFACE IMPORTED)
            target_link_libraries(ICU::data INTERFACE PkgConfig::ICU)
          endif()
        else()
          message(FATAL_ERROR "ICU not found! Please install libicu-dev (Linux) or icu4c (macOS), or make sure pkg-config can find ICU.")
        endif()
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

            set(_ORT_FALLBACK_VERSION "1.24.4")
            set(_ORT_RELEASE_TAG "")
            cosyvoice_fetch_github_latest_tag("microsoft/onnxruntime" _ORT_RELEASE_TAG)

            if(_ORT_RELEASE_TAG MATCHES "^v([0-9]+\.[0-9]+\.[0-9]+)$")
                set(_ORT_VERSION "${CMAKE_MATCH_1}")
                message(STATUS "ONNX Runtime latest release: tag=${_ORT_RELEASE_TAG}, version=${_ORT_VERSION}")
            else()
                set(_ORT_VERSION "${_ORT_FALLBACK_VERSION}")
                set(_ORT_RELEASE_TAG "v${_ORT_FALLBACK_VERSION}")
                message(STATUS "Failed to query latest ONNX Runtime release; falling back to version=${_ORT_VERSION}")
            endif()

            if(WIN32)
                if(CMAKE_SYSTEM_PROCESSOR MATCHES "ARM64|aarch64")
                    set(ORT_URL "https://github.com/microsoft/onnxruntime/releases/download/${_ORT_RELEASE_TAG}/onnxruntime-win-arm64-${_ORT_VERSION}.zip")
                    set(ORT_DIR "onnxruntime-win-arm64-${_ORT_VERSION}")
                else()
                    set(ORT_URL "https://github.com/microsoft/onnxruntime/releases/download/${_ORT_RELEASE_TAG}/onnxruntime-win-x64-${_ORT_VERSION}.zip")
                    set(ORT_DIR "onnxruntime-win-x64-${_ORT_VERSION}")
                endif()
                set(ORT_EXT "zip")
            elseif(APPLE)
                set(ORT_URL "https://github.com/microsoft/onnxruntime/releases/download/${_ORT_RELEASE_TAG}/onnxruntime-osx-arm64-${_ORT_VERSION}.tgz")
                set(ORT_DIR "onnxruntime-osx-arm64-${_ORT_VERSION}")
                set(ORT_EXT "tgz")
            else()
                if(CMAKE_SYSTEM_PROCESSOR MATCHES "ARM64|aarch64")
                    set(ORT_URL "https://github.com/microsoft/onnxruntime/releases/download/${_ORT_RELEASE_TAG}/onnxruntime-linux-aarch64-${_ORT_VERSION}.tgz")
                    set(ORT_DIR "onnxruntime-linux-aarch64-${_ORT_VERSION}")
                else()
                    set(ORT_URL "https://github.com/microsoft/onnxruntime/releases/download/${_ORT_RELEASE_TAG}/onnxruntime-linux-x64-${_ORT_VERSION}.tgz")
                    set(ORT_DIR "onnxruntime-linux-x64-${_ORT_VERSION}")
                endif()
                set(ORT_EXT "tgz")
            endif()

            message(STATUS "ONNX Runtime download URL: ${ORT_URL}")
            
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
    set(FFMPEG_FOUND FALSE)

    if(EXISTS "${FFMPEG_PREBUILT_DIR}/include/libavcodec/avcodec.h")
        message(STATUS "Using FFmpeg from ${FFMPEG_PREBUILT_DIR}")
        add_library(ffmpeg INTERFACE IMPORTED)

        if(WIN32)
            file(GLOB FFMPEG_DLLS "${FFMPEG_PREBUILT_DIR}/bin/*.dll")
            list(LENGTH FFMPEG_DLLS _ffmpeg_dlls_len)
            if(_ffmpeg_dlls_len EQUAL 0)
                message(FATAL_ERROR "FFmpeg DLLs not found in ${FFMPEG_PREBUILT_DIR}/bin")
            endif()
            set(FFMPEG_RUNTIME_SHARED_LIBS ${FFMPEG_DLLS})

            set_target_properties(ffmpeg PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES "${FFMPEG_PREBUILT_DIR}/include"
                INTERFACE_LINK_LIBRARIES
                    "${FFMPEG_PREBUILT_DIR}/lib/avcodec.lib;${FFMPEG_PREBUILT_DIR}/lib/avformat.lib;${FFMPEG_PREBUILT_DIR}/lib/avutil.lib;${FFMPEG_PREBUILT_DIR}/lib/swresample.lib"
            )
        else()
            file(GLOB FFMPEG_SHARED_LIBS
                "${FFMPEG_PREBUILT_DIR}/lib/libavcodec.so"
                "${FFMPEG_PREBUILT_DIR}/lib/libavcodec.so.*"
                "${FFMPEG_PREBUILT_DIR}/lib/libavcodec.dylib"
                "${FFMPEG_PREBUILT_DIR}/lib/libavformat.so"
                "${FFMPEG_PREBUILT_DIR}/lib/libavformat.so.*"
                "${FFMPEG_PREBUILT_DIR}/lib/libavformat.dylib"
                "${FFMPEG_PREBUILT_DIR}/lib/libavutil.so"
                "${FFMPEG_PREBUILT_DIR}/lib/libavutil.so.*"
                "${FFMPEG_PREBUILT_DIR}/lib/libavutil.dylib"
                "${FFMPEG_PREBUILT_DIR}/lib/libswresample.so"
                "${FFMPEG_PREBUILT_DIR}/lib/libswresample.so.*"
                "${FFMPEG_PREBUILT_DIR}/lib/libswresample.dylib"
            )
            list(LENGTH FFMPEG_SHARED_LIBS _ffmpeg_shared_len)
            if(_ffmpeg_shared_len EQUAL 0)
                message(FATAL_ERROR "FFmpeg shared libraries not found in ${FFMPEG_PREBUILT_DIR}/lib")
            endif()
            set(FFMPEG_RUNTIME_SHARED_LIBS ${FFMPEG_SHARED_LIBS})

            find_library(FFMPEG_AVCODEC_LIBRARY NAMES avcodec HINTS
                "${FFMPEG_PREBUILT_DIR}/lib"
                "/opt/homebrew/lib" "/usr/local/lib"
                "/usr/lib/x86_64-linux-gnu" "/usr/lib/aarch64-linux-gnu"
                "/usr/lib/arm-linux-gnueabihf" "/usr/lib/arm-linux-gnueabi"
                "/usr/lib" "/usr/lib64"
            )
            find_library(FFMPEG_AVFORMAT_LIBRARY NAMES avformat HINTS
                "${FFMPEG_PREBUILT_DIR}/lib"
                "/opt/homebrew/lib" "/usr/local/lib"
                "/usr/lib/x86_64-linux-gnu" "/usr/lib/aarch64-linux-gnu"
                "/usr/lib/arm-linux-gnueabihf" "/usr/lib/arm-linux-gnueabi"
                "/usr/lib" "/usr/lib64"
            )
            find_library(FFMPEG_AVUTIL_LIBRARY NAMES avutil HINTS
                "${FFMPEG_PREBUILT_DIR}/lib"
                "/opt/homebrew/lib" "/usr/local/lib"
                "/usr/lib/x86_64-linux-gnu" "/usr/lib/aarch64-linux-gnu"
                "/usr/lib/arm-linux-gnueabihf" "/usr/lib/arm-linux-gnueabi"
                "/usr/lib" "/usr/lib64"
            )
            find_library(FFMPEG_SWRESAMPLE_LIBRARY NAMES swresample HINTS
                "${FFMPEG_PREBUILT_DIR}/lib"
                "/opt/homebrew/lib" "/usr/local/lib"
                "/usr/lib/x86_64-linux-gnu" "/usr/lib/aarch64-linux-gnu"
                "/usr/lib/arm-linux-gnueabihf" "/usr/lib/arm-linux-gnueabi"
                "/usr/lib" "/usr/lib64"
            )

            if(NOT FFMPEG_AVCODEC_LIBRARY OR NOT FFMPEG_AVFORMAT_LIBRARY OR NOT FFMPEG_AVUTIL_LIBRARY OR NOT FFMPEG_SWRESAMPLE_LIBRARY)
                message(FATAL_ERROR "FFmpeg libraries not found in ${FFMPEG_PREBUILT_DIR}.")
            endif()

            set_target_properties(ffmpeg PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES "${FFMPEG_PREBUILT_DIR}/include"
                INTERFACE_LINK_LIBRARIES "${FFMPEG_AVCODEC_LIBRARY};${FFMPEG_AVFORMAT_LIBRARY};${FFMPEG_AVUTIL_LIBRARY};${FFMPEG_SWRESAMPLE_LIBRARY}"
            )
        endif()
        set(FFMPEG_FOUND TRUE)
    elseif(WIN32)
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

        if(EXISTS "${FFMPEG_PREBUILT_DIR}/include/libavcodec/avcodec.h")
            message(STATUS "Using FFmpeg from ${FFMPEG_PREBUILT_DIR}")
            add_library(ffmpeg INTERFACE IMPORTED)

            file(GLOB FFMPEG_DLLS "${FFMPEG_PREBUILT_DIR}/bin/*.dll")
            list(LENGTH FFMPEG_DLLS _ffmpeg_dlls_len)
            if(_ffmpeg_dlls_len EQUAL 0)
                message(FATAL_ERROR "FFmpeg DLLs not found in ${FFMPEG_PREBUILT_DIR}/bin")
            endif()
            set(FFMPEG_RUNTIME_SHARED_LIBS ${FFMPEG_DLLS})

            set_target_properties(ffmpeg PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES "${FFMPEG_PREBUILT_DIR}/include"
                INTERFACE_LINK_LIBRARIES
                    "${FFMPEG_PREBUILT_DIR}/lib/avcodec.lib;${FFMPEG_PREBUILT_DIR}/lib/avformat.lib;${FFMPEG_PREBUILT_DIR}/lib/avutil.lib;${FFMPEG_PREBUILT_DIR}/lib/swresample.lib"
            )
            set(FFMPEG_FOUND TRUE)
        else()
            message(FATAL_ERROR "FFmpeg not found in ${FFMPEG_PREBUILT_DIR}.")
        endif()
    else()
        find_package(PkgConfig QUIET)
        if(PkgConfig_FOUND)
            pkg_check_modules(FFMPEG_PC QUIET IMPORTED_TARGET libavcodec libavformat libavutil libswresample)
        endif()

        if(FFMPEG_PC_FOUND)
            add_library(ffmpeg INTERFACE IMPORTED)
            target_link_libraries(ffmpeg INTERFACE PkgConfig::FFMPEG_PC)
            set(FFMPEG_FOUND TRUE)
        else()
            find_path(FFMPEG_INCLUDE_DIR NAMES libavcodec/avcodec.h HINTS
                "/opt/homebrew/include" "/usr/local/include"
                "/usr/include/x86_64-linux-gnu" "/usr/include/aarch64-linux-gnu"
                "/usr/include/arm-linux-gnueabihf" "/usr/include/arm-linux-gnueabi"
                "/usr/include"
            )
            find_library(FFMPEG_AVCODEC_LIBRARY NAMES avcodec HINTS
                "/opt/homebrew/lib" "/usr/local/lib"
                "/usr/lib/x86_64-linux-gnu" "/usr/lib/aarch64-linux-gnu"
                "/usr/lib/arm-linux-gnueabihf" "/usr/lib/arm-linux-gnueabi"
                "/usr/lib" "/usr/lib64"
            )
            find_library(FFMPEG_AVFORMAT_LIBRARY NAMES avformat HINTS
                "/opt/homebrew/lib" "/usr/local/lib"
                "/usr/lib/x86_64-linux-gnu" "/usr/lib/aarch64-linux-gnu"
                "/usr/lib/arm-linux-gnueabihf" "/usr/lib/arm-linux-gnueabi"
                "/usr/lib" "/usr/lib64"
            )
            find_library(FFMPEG_AVUTIL_LIBRARY NAMES avutil HINTS
                "/opt/homebrew/lib" "/usr/local/lib"
                "/usr/lib/x86_64-linux-gnu" "/usr/lib/aarch64-linux-gnu"
                "/usr/lib/arm-linux-gnueabihf" "/usr/lib/arm-linux-gnueabi"
                "/usr/lib" "/usr/lib64"
            )
            find_library(FFMPEG_SWRESAMPLE_LIBRARY NAMES swresample HINTS
                "/opt/homebrew/lib" "/usr/local/lib"
                "/usr/lib/x86_64-linux-gnu" "/usr/lib/aarch64-linux-gnu"
                "/usr/lib/arm-linux-gnueabihf" "/usr/lib/arm-linux-gnueabi"
                "/usr/lib" "/usr/lib64"
            )

            if(FFMPEG_INCLUDE_DIR AND FFMPEG_AVCODEC_LIBRARY AND FFMPEG_AVFORMAT_LIBRARY AND FFMPEG_AVUTIL_LIBRARY AND FFMPEG_SWRESAMPLE_LIBRARY)
                add_library(ffmpeg INTERFACE IMPORTED)
                set_target_properties(ffmpeg PROPERTIES
                    INTERFACE_INCLUDE_DIRECTORIES "${FFMPEG_INCLUDE_DIR}"
                    INTERFACE_LINK_LIBRARIES "${FFMPEG_AVCODEC_LIBRARY};${FFMPEG_AVFORMAT_LIBRARY};${FFMPEG_AVUTIL_LIBRARY};${FFMPEG_SWRESAMPLE_LIBRARY}"
                )
                set(FFMPEG_FOUND TRUE)
            else()
                message(FATAL_ERROR "FFmpeg not found. On Linux/macOS, install ffmpeg development packages, install pkg-config, or set FFMPEG_PREBUILT_DIR to a prebuilt FFmpeg tree.")
            endif()
        endif()
    endif()
endif()
