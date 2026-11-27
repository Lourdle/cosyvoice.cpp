# Every subdirectory that has a CMakeLists.txt is discovered automatically:
# adding or removing a tool directory needs no edit here.

file(GLOB COSYVOICE_TOOL_CMAKELISTS CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*/CMakeLists.txt")

foreach(tool_cmakelists IN LISTS COSYVOICE_TOOL_CMAKELISTS)
    get_filename_component(tool_dir "${tool_cmakelists}" DIRECTORY)
    get_filename_component(tool_name "${tool_dir}" NAME)
    add_subdirectory("${tool_name}")
endforeach()
