file(GLOB COSYVOICE_IMATRIX_SOURCES "*.cpp")
add_executable(cosyvoice-imatrix ${COSYVOICE_IMATRIX_SOURCES})
target_link_libraries(cosyvoice-imatrix PRIVATE cosyvoice common tools_common ggml)

install(TARGETS cosyvoice-imatrix
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)