add_library(tools_common STATIC
    common.cpp
    cosyvoice-text-chunk.cpp
    cosyvoice-text-chunk.h
    tool_common.h
    tool_common_cosyvoice.h)
target_include_directories(tools_common PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_SOURCE_DIR}/common")
target_link_libraries(tools_common PRIVATE common)
