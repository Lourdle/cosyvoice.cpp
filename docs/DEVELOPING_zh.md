# 开发者指南

本仓库的工具与测试的贡献规范。API 参考与 SIMD 内部实现见 [API_zh.md](API_zh.md) 与 [SIMD_zh.md](SIMD_zh.md)。

## 入口约定：使用 `tool_entry`，禁止写 `main`

本仓库中的每个可执行程序——工具（`tools/cli`、`tools/server`、`tools/quantize`）与测试（`tests/`）——都必须实现

```cpp
int tool_entry(int argc, char** argv);
```

并且**不得定义 `main`**。POSIX 上 `tool_common.h` 将 `tool_entry` `#define` 为 `main`——工具的入口函数本身就是 `main`；Windows 上唯一的 `main` 位于 `tools/common/common.cpp`，由它调用 `tool_entry`。

### 为什么

在 **Windows** 上，`main(int argc, char** argv)` 入口点收到的是按系统 *ANSI 代码页*（ACP）编码的参数：运行时启动时从 `GetCommandLineA` 解析出 `argv`，而 ACP 随系统区域设置而变（如 zh-CN 为 GBK、西方为 Windows-1252）——它**不是 UTF-8**。任何含非 ASCII 字符的路径、文本或参数因此在到达你的代码之前就可能被错误编码或丢失。（下方 `main` 设置的“控制台输出代码页”只影响字节的显示方式，并不会改变 `argv` 的编码。）

本项目要求所有路径与字符串在所有平台上均为 **UTF-8**（见 [API_zh.md — 编码约定](API_zh.md)）。为保证这一点，`tools/common/common.cpp` 中的 `main` 在 Windows 上调用 `tool_entry` 之前会：

1. `SetConsoleOutputCP(CP_UTF8)` / `SetConsoleCP(CP_UTF8)` —— 让控制台以 UTF-8 收发；
2. `CommandLineToArgvW` + `WideCharToMultiByte(CP_UTF8, ...)` —— 从宽字符（原生）命令行重建 UTF-8 `argv`，完全绕开 ANSI 代码页。

非 Windows 平台参数本就是 UTF-8，且 `tool_common.h` 已将 `tool_entry` `#define` 为 `main`，因此完全不存在包装层。

如果自己定义 `main`（或链接时未包含 `tools_common`），就会失去这一保证：在代码页非 UTF-8 的 Windows 系统上，UTF-8 参数与路径早在你的代码看到它们之前就被破坏了。

## 链接新的工具或测试

`tools_common` 静态库（`tools/common/`）提供 Windows 上的 `main` 与 UTF-8 argv 转换（POSIX 上 `tool_entry` 被 `#define` 为 `main`），以及共享的辅助函数。每个需要生成可执行文件的目标都要链接它：

```cmake
add_executable(my-tool my-tool.cpp)
target_link_libraries(my-tool PRIVATE cosyvoice common tools_common ggml)
target_include_directories(my-tool PRIVATE ../include)
```

- 工具放在 `tools/<name>/` 下，各带自己的 `CMakeLists.txt`（参考 `tools/cli/CMakeLists.txt`、`tools/quantize/CMakeLists.txt`、`tools/server/CMakeLists.txt`）。
- 测试放在 `tests/` 下，由 `COSYVOICE_BUILD_TESTS` 开关控制（见 `tests/CMakeLists.txt` 与 [tests/README.md](../tests/README.md)）。
- `cosyvoice` 是核心推理库。`common` 是**本仓库自己的**共享辅助库（`common/`）：包含 `gguf_parser` GGUF 解析器、`file_mmap`，以及 UTF-8 感知的 `open_ifstream_utf8()` / `open_ofstream_utf8()` / `utf8_to_wstr()`；它公开链接 `ggml`（`common/CMakeLists.txt`）。工具直接使用这些符号——`quantize` 调用 `gguf_parser` / `file_mmap` / `open_ifstream_utf8`（`tools/quantize/quantize.cpp`），CLI 调用 `open_ifstream_utf8`（`tools/cli/cosyvoice-cli.cpp`），且 `tool_common.h` 本身 `#include "common.h"`——因此新可执行目标应同时链接 `cosyvoice` 与 `common`。
- 主项目默认以 `BUILD_SHARED_LIBS=ON` 构建；工具链接共享的 `cosyvoice` 库，因此头文件来自 `include/`。

只在自己的代码里前向声明或定义 `tool_entry`；绝不能让两个 `main` 定义同时参与链接。

## 可用的辅助工具

头文件：`"tool_common.h"`（或 cosyvoice 相关的用 `"tool_common_cosyvoice.h"`）：

- **`tool_common.h`**（`tools/common/tool_common.h`）：
  - 字符串/CLI 解析：`to_lower`、`str_casecmp`、`trim_copy`、`parse_float_arg`、`parse_uint32_arg`、`parse_int_arg`、`parse_uint16_port`；
  - 其它：`elapsed_ms`、`bytes_to_mib`、`get_local_timestamp_ms`。
- **`tool_common_cosyvoice.h`**（`tools/common/tool_common_cosyvoice.h`）：
  - 每个需要释放的 cosyvoice API 类型的 RAII 句柄：`cosyvoice_context_handle`、`cosyvoice_frontend_handle`、`cosyvoice_prompt_speech_handle`、`cosyvoice_prompt_handle`、`cosyvoice_tts_context_handle`，以及（音频支持启用时的）`audio_buffer_handle` / `cosyvoice_audio_encoder_handle`；
  - CLI 与 server 共享的参数解析器：`parse_simd_level_arg` / `simd_level_to_string` / `simd_caps_to_string`、`parse_kv_cache_type_arg` 等。

## 工具代码的 UTF-8 规则

- 把从 `tool_entry`（或某个 cosyvoice API）收到的每个 `char*`/`std::string` 都当作 UTF-8。不要用 ANSI 代码页做任何转换。
- 需要把路径传给 C 运行时/操作系统函数时：
  - **POSIX**：直接把 UTF-8 `char*` 原样传递——文件 API 接收字节串，UTF-8 即为惯例。
  - **Windows**：**不要**把 UTF-8 `char*` 传给 ANSI（"A"）系列 API——它们按系统 ANSI 代码页（ACP，如 GBK）解码，而非 UTF-8，除非进程显式选用 UTF-8 代码页（Windows 10 1809+ 应用清单 `activeCodePage=UTF-8`，或系统级“使用 Unicode UTF-8”选项）——本项目**不**依赖此机制。宽字符（"W"）系列 API 接收的是 UTF-16，也非 UTF-8。请在边界处用 `utf8_to_wstr()`（`common/common.h`，基于 `MultiByteToWideChar(CP_UTF8)`）转换后再调 W API，或直接使用现成的 `open_ifstream_utf8()` / `open_ofstream_utf8()` / `file_mmap` 助手，它们已封装好（底层为 `CreateFileW`、宽字符 `std::ifstream` 构造、`ma_*_file_w` 等）。
- 不要在你的工具里自己做 Windows ANSI 转换；把路径交给 `common` 助手，由它们在边界处理。

## 优先使用现代 C++

项目以 C++20 为目标，核心库已遵循以下写法——新代码保持一致：

- **格式化**——分两档：
  - **允许**：向控制台打印一条短消息，格式串只做简单转换（`%s`、`%d`、`%u`、`%zu`、`%.Nf` 等）时，直接用 `printf` / `fprintf` 没问题——本仓库的帮助文本和错误输出大多属于这一类。
  - **必须 `std::format`**：只要是在*组装*字符串——`char buf[N]` + `snprintf`、多个片段、条件后缀、或 `std::to_string` + 拼接链——就改用 `std::format` 构建，然后再输出（`fputs`、`std::cout`、或 `printf("%s", ...)` 均可）。`std::format` 类型安全、无需手工指定缓冲大小，并能直接接受 `std::string` / `std::string_view`：
    ```cpp
    // 不要组装进缓冲区：
    char buf[160];
    snprintf(buf, sizeof(buf), "requested: %d, actual: %d (%s)", req, act, why);
    print_kv_line_string("steps", buf);
    // 先构建，再输出：
    print_kv_line_string("steps", std::format("requested: {}, actual: {} ({})", req, act, why).c_str());
    ```
- **字符串视图**：库/工具内部只读字符串参数用 `std::string_view`；C API 边界（`cosyvoice.h` 等为 C 接口）保持 `const char*`。
- **RAII**：优先使用 `std::unique_ptr`（含 `tool_common_cosyvoice.h` 中的 `*_handle` 别名）和 `std::vector` / `std::string`，避免裸 `new`/`delete` 与拥有型裸指针。C API 的 `create`/`free` 配对属于 C ABI 的一部分，属例外。
- **数字解析**：输入解析用 `std::from_chars`，避免 `strtol` / `strtof` 家族（参考 `src/cosyvoice-loader.cpp` 中 `COSYVOICE_MAX_WEIGHT_BUFFER_SIZE` 的处理）。
- **惯用法**：优先 `nullptr` 而非 `NULL`/`0`，`static_cast`/`reinterpret_cast` 而非 C 风格强转，善用 if-初始化语句、结构化绑定，非 C API 代码用 `enum class` 而非普通 `enum`。

以上针对新代码；简单的 printf 风格控制台输出不必重写。
