# -----------------------------------------------------------------------------
# Post-Build DLL Copying & Installation (Core Library)
# -----------------------------------------------------------------------------
include(GNUInstallDirs)

set(COSYVOICE_PUBLIC_HEADERS
    include/cosyvoice.h
    include/cosyvoice-interface.h
    include/cosyvoice-lowlevel.h
)

if(NOT COSYVOICE_NO_FRONTEND)
    list(APPEND COSYVOICE_PUBLIC_HEADERS include/cosyvoice-frontend.h)
endif()

if(NOT COSYVOICE_NO_AUDIO)
    list(APPEND COSYVOICE_PUBLIC_HEADERS include/cosyvoice-audio.h)
endif()

# Track prebuilt dynamic libraries that need to be copied/installed
set(PREBUILT_SHARED_LIBS "")
if(WIN32 AND TARGET onnxruntime AND NOT onnxruntime_FOUND)
    list(APPEND PREBUILT_SHARED_LIBS $<TARGET_FILE:onnxruntime>)
endif()

if(WIN32 AND TARGET ICU::uc AND NOT ICU_FOUND)
    list(APPEND PREBUILT_SHARED_LIBS $<TARGET_FILE:ICU::uc> $<TARGET_FILE:ICU::i18n>)
    if(TARGET ICU::data)
        list(APPEND PREBUILT_SHARED_LIBS $<TARGET_FILE:ICU::data>)
    endif()
endif()

if(TARGET ffmpeg AND DEFINED FFMPEG_RUNTIME_SHARED_LIBS)
    list(APPEND PREBUILT_SHARED_LIBS ${FFMPEG_RUNTIME_SHARED_LIBS})
endif()

if(UNIX AND NOT APPLE)
    if(TARGET onnxruntime AND NOT onnxruntime_FOUND)
        file(GLOB _ORT_SHARED_LIBS CONFIGURE_DEPENDS
            "${ORT_PREBUILT_DIR}/lib/libonnxruntime.so*"
        )
        list(APPEND PREBUILT_SHARED_LIBS ${_ORT_SHARED_LIBS})
    endif()

    if(TARGET ICU::uc AND NOT ICU_FOUND)
        file(GLOB _ICU_SHARED_LIBS CONFIGURE_DEPENDS
            "${ICU_PREBUILT_DIR}/lib/libicuuc.so*"
            "${ICU_PREBUILT_DIR}/lib/libicui18n.so*"
            "${ICU_PREBUILT_DIR}/lib/libicudata.so*"
        )
        list(APPEND PREBUILT_SHARED_LIBS ${_ICU_SHARED_LIBS})
    endif()

    if(TARGET ffmpeg AND DEFINED FFMPEG_RUNTIME_SHARED_LIBS)
        file(GLOB _FFMPEG_SHARED_LIBS CONFIGURE_DEPENDS
            "${FFMPEG_PREBUILT_DIR}/lib/libavcodec.so*"
            "${FFMPEG_PREBUILT_DIR}/lib/libavformat.so*"
            "${FFMPEG_PREBUILT_DIR}/lib/libavutil.so*"
            "${FFMPEG_PREBUILT_DIR}/lib/libswresample.so*"
        )
        list(APPEND PREBUILT_SHARED_LIBS ${_FFMPEG_SHARED_LIBS})
    endif()

    if(PREBUILT_SHARED_LIBS)
        list(FILTER PREBUILT_SHARED_LIBS EXCLUDE REGEX "\.a$")
        list(REMOVE_DUPLICATES PREBUILT_SHARED_LIBS)
    endif()
endif()

if(WIN32)
    # For local execution on Windows: Copy ALL required DLLs next to the binaries
    # Note: $<TARGET_RUNTIME_DLLS:...> only works for EXECUTABLE, SHARED, or MODULE targets.
    if(BUILD_SHARED_LIBS)
        add_custom_command(TARGET cosyvoice POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_RUNTIME_DLLS:cosyvoice>
                $<TARGET_FILE_DIR:cosyvoice>
            COMMAND_EXPAND_LISTS
        )
    endif()

    if(TARGET ffmpeg AND DEFINED FFMPEG_RUNTIME_SHARED_LIBS)
        add_custom_command(TARGET cosyvoice POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${FFMPEG_RUNTIME_SHARED_LIBS}
                $<TARGET_FILE_DIR:cosyvoice>
            COMMAND_EXPAND_LISTS
        )
    endif()

    # Windows installation of prebuilt DLLs
    if(PREBUILT_SHARED_LIBS)
        install(FILES ${PREBUILT_SHARED_LIBS} DESTINATION ${CMAKE_INSTALL_BINDIR})
    endif()
elseif(UNIX AND NOT APPLE)
    if(PREBUILT_SHARED_LIBS)
        install(FILES ${PREBUILT_SHARED_LIBS} DESTINATION ${CMAKE_INSTALL_LIBDIR})
    endif()
endif()

# Core Installation Rules
install(TARGETS cosyvoice
    EXPORT CosyVoiceTargets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(FILES ${COSYVOICE_PUBLIC_HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
