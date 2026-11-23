add_executable(test-llm-graph-reuse llm-graph-reuse.cpp)
target_link_libraries(test-llm-graph-reuse PRIVATE cosyvoice common tools_common ggml)
target_include_directories(test-llm-graph-reuse PRIVATE ../include)

add_executable(test-tts-stream tts-stream.cpp)
target_link_libraries(test-tts-stream PRIVATE cosyvoice common tools_common ggml)
target_include_directories(test-tts-stream PRIVATE ../include)

add_executable(test-kv-offload kv-cache-offload.cpp)
target_link_libraries(test-kv-offload PRIVATE cosyvoice common tools_common ggml)
target_include_directories(test-kv-offload PRIVATE ../include)

add_executable(test-llm-prefill-stop llm-prefill-stop.cpp)
target_link_libraries(test-llm-prefill-stop PRIVATE cosyvoice common tools_common ggml)
target_include_directories(test-llm-prefill-stop PRIVATE ../include)

add_executable(test-mmap-zerocopy mmap-zerocopy.cpp)
target_link_libraries(test-mmap-zerocopy PRIVATE cosyvoice common tools_common ggml)
target_include_directories(test-mmap-zerocopy PRIVATE ../include)

add_executable(test-mmap-delete mmap-delete.cpp)
target_link_libraries(test-mmap-delete PRIVATE cosyvoice common tools_common ggml)
target_include_directories(test-mmap-delete PRIVATE ../include)

add_executable(test-memory-pool memory-pool.cpp)
target_link_libraries(test-memory-pool PRIVATE cosyvoice common tools_common ggml)
target_include_directories(test-memory-pool PRIVATE ../include)
