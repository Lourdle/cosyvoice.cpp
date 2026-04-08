add_library(common STATIC common.cpp common.h)
target_include_directories(common PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
