# -----------------------------------------------------------------------------
# Post-Build DLL Copying & Installation (Core Library)
# -----------------------------------------------------------------------------
include(GNUInstallDirs)

# Track prebuilt dynamic libraries that need to be copied/installed
set(PREBUILT_SHARED_LIBS "")
if(TARGET onnxruntime AND NOT onnxruntime_FOUND)
    list(APPEND PREBUILT_SHARED_LIBS $<TARGET_FILE:onnxruntime>)
endif()

if(TARGET ICU::uc AND NOT ICU_FOUND)
    list(APPEND PREBUILT_SHARED_LIBS $<TARGET_FILE:ICU::uc> $<TARGET_FILE:ICU::i18n>)
    if(TARGET ICU::data)
        list(APPEND PREBUILT_SHARED_LIBS $<TARGET_FILE:ICU::data>)
    endif()
endif()

if(TARGET ffmpeg AND NOT FFMPEG_FOUND)
    if(DEFINED FFMPEG_RUNTIME_DLLS)
        list(APPEND PREBUILT_SHARED_LIBS ${FFMPEG_RUNTIME_DLLS})
    else()
        file(GLOB FFMPEG_DLLS "${FFMPEG_PREBUILT_DIR}/bin/*.dll")
        list(APPEND PREBUILT_SHARED_LIBS ${FFMPEG_DLLS})
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

    # Windows installation of prebuilt DLLs
    if(PREBUILT_SHARED_LIBS)
        install(FILES ${PREBUILT_SHARED_LIBS} DESTINATION ${CMAKE_INSTALL_BINDIR})
    endif()
else()
    # Unix installation of prebuilt SO/Dylibs
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

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

if(COSYVOICE_AUDIO_BACKEND STREQUAL "FFMPEG")
    install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.FFmpeg" DESTINATION ${CMAKE_INSTALL_DOCDIR})
endif()
