add_library(tools_common STATIC common.cpp tool_common.h)
target_include_directories(tools_common PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_SOURCE_DIR}/common")
target_link_libraries(tools_common PRIVATE common)
