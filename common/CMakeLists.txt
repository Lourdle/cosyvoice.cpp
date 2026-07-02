file(GLOB COSYVOICE_COMMON_SOURCES "*.cpp" "*.h")
add_library(common STATIC ${COSYVOICE_COMMON_SOURCES})
target_include_directories(common PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(common PUBLIC ggml)
