add_executable(test-llm-graph-reuse llm-graph-reuse.cpp)
target_link_libraries(test-llm-graph-reuse PRIVATE cosyvoice common tools_common ggml)
target_include_directories(test-llm-graph-reuse PRIVATE ../include)

add_executable(test-tts-stream tts-stream.cpp)
target_link_libraries(test-tts-stream PRIVATE cosyvoice common tools_common ggml)
target_include_directories(test-tts-stream PRIVATE ../include)