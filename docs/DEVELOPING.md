# Developer Guide

Guidelines for contributing tools and tests to this repository. API reference and SIMD internals are covered in [API.md](API.md) and [SIMD.md](SIMD.md).

## Entry point convention: `tool_entry`, never `main`

Every executable in this repository — tools (`tools/cli`, `tools/server`, `tools/quantize`) and tests (`tests/`) — implements

```cpp
int tool_entry(int argc, char** argv);
```

and **does not define `main`**. On POSIX, `tool_common.h` `#define`s `tool_entry` to `main` — the tool's entry function *is* `main`; on Windows, the single `main` lives in `tools/common/common.cpp` and calls `tool_entry`.

### Why

On **Windows**, the `main(int argc, char** argv)` entry point receives arguments encoded in the system *ANSI code page* (ACP): the runtime startup derives `argv` from `GetCommandLineA`, and the ACP depends on the locale (e.g. GBK on zh-CN Windows, Windows-1252 on Western) — it is **not UTF-8**. Any path, text, or argument containing non-ASCII characters would therefore arrive mis-encoded or lossy. (Setting the console output codepage, as `main` does below, only affects how bytes are displayed; it does not change how `argv` is encoded.)

This project requires every path and string to be **UTF-8** on all platforms (see [API.md — encoding convention](API.md)). To guarantee that, `tools/common/common.cpp`'s `main` does the following on Windows before calling `tool_entry`:

1. `SetConsoleOutputCP(CP_UTF8)` / `SetConsoleCP(CP_UTF8)` — the console talks UTF-8;
2. `CommandLineToArgvW` + `WideCharToMultiByte(CP_UTF8, ...)` — rebuilds `argv` as UTF-8 strings from the wide (native) command line, bypassing the ANSI codepage entirely.

On non-Windows the arguments are already UTF-8, and `tool_common.h` `#define`s `tool_entry` to `main`, so no wrapper runs at all.

If you define your own `main` (or link without `tools_common`), you lose this guarantee: on a Windows system with a non-UTF-8 codepage, UTF-8 arguments and paths will be corrupted before your code ever sees them.

## Linking a new tool or test

The `tools_common` static library (`tools/common/`) provides the Windows `main` with the UTF-8 argv conversion (on POSIX, `tool_entry` is `#define`d to `main`), plus shared helper functions. Link it everywhere you build an executable:

```cmake
add_executable(my-tool my-tool.cpp)
target_link_libraries(my-tool PRIVATE cosyvoice common tools_common ggml)
target_include_directories(my-tool PRIVATE ../include)
```

- Tools live under `tools/<name>/` each with its own `CMakeLists.txt` (see `tools/cli/CMakeLists.txt`, `tools/quantize/CMakeLists.txt`, `tools/server/CMakeLists.txt`).
- Tests live under `tests/` and are gated behind `COSYVOICE_BUILD_TESTS` (see `tests/CMakeLists.txt` and [tests/README.md](../tests/README.md)).
- `cosyvoice` is the core inference library. `common` is **this repository's** shared helper library (`common/`): the `gguf_parser` GGUF reader, `file_mmap`, and the UTF-8-aware `open_ifstream_utf8()` / `open_ofstream_utf8()` / `utf8_to_wstr()`; it links `ggml` publicly (`common/CMakeLists.txt`). Tools use these symbols directly — `quantize` calls `gguf_parser` / `file_mmap` / `open_ifstream_utf8` (`tools/quantize/quantize.cpp`), the CLI calls `open_ifstream_utf8` (`tools/cli/cosyvoice-cli.cpp`), and `tool_common.h` itself includes `common.h` — so a new executable links both `cosyvoice` and `common`.
- The main project is built with `BUILD_SHARED_LIBS=ON` by default; the tools link the shared `cosyvoice` library, so headers from `include/` are what you use.

Only forward-declare or define `tool_entry` in your own code; never link two definitions of `main`.

## Available helpers

Headers, include via `"tool_common.h"` (or `"tool_common_cosyvoice.h"` for the cosyvoice-aware ones):

- **`tool_common.h`** (`tools/common/tool_common.h`):
  - string/CLI parsing: `to_lower`, `str_casecmp`, `trim_copy`, `parse_float_arg`, `parse_uint32_arg`, `parse_int_arg`, `parse_uint16_port`;
  - misc: `elapsed_ms`, `bytes_to_mib`, `get_local_timestamp_ms`.
- **`tool_common_cosyvoice.h`** (`tools/common/tool_common_cosyvoice.h`):
  - RAII handles for every cosyvoice API type that needs freeing: `cosyvoice_context_handle`, `cosyvoice_frontend_handle`, `cosyvoice_prompt_speech_handle`, `cosyvoice_prompt_handle`, `cosyvoice_tts_context_handle`, plus `audio_buffer_handle` / `cosyvoice_audio_encoder_handle` (when audio support is enabled);
  - argument parsers shared by the CLI and server: `parse_simd_level_arg` / `simd_level_to_string` / `simd_caps_to_string`, `parse_kv_cache_type_arg`, and friends.

## UTF-8 rules for tool code

- Treat every `char*`/`std::string` you receive from `tool_entry` (or a cosyvoice API) as UTF-8. Do not convert with the ANSI codepage.
- When you need to hand a path to a C runtime/OS function:
  - **POSIX**: pass the UTF-8 `char*` as-is — file APIs take byte strings and UTF-8 is the convention.
  - **Windows**: do **not** pass UTF-8 `char*` to ANSI ("A") APIs — they decode with the system ANSI codepage (ACP, e.g. GBK), not UTF-8, unless the process opts into a UTF-8 codepage (Windows 10 1809+ app-manifest `activeCodePage=UTF-8` or the system "Use Unicode UTF-8" option), which this project does **not** rely on. The wide ("W") APIs take UTF-16, not UTF-8. Convert at the boundary with `utf8_to_wstr()` (`common/common.h`, `MultiByteToWideChar(CP_UTF8)`) and call the W API, or use the ready-made helpers `open_ifstream_utf8()` / `open_ofstream_utf8()` / `file_mmap`, which already do this (backed by `CreateFileW`, the wide `std::ifstream` constructor, `ma_*_file_w`, etc.).
- Do not add Windows ANSI conversions in your tool; pass paths through the `common` helpers and let them handle the boundary.

## Prefer modern C++

The project targets C++20, and the core library already follows these idioms — keep new code consistent with them:

- **Formatting** — two tiers:
  - **Allowed**: printing a short message to the console with `printf` / `fprintf` is fine when the format string only does simple conversions (`%s`, `%d`, `%u`, `%zu`, `%.Nf`, ...). Most help text and error prints in this repo are exactly that.
  - **Required**: when you *compose* a string — `char buf[N]` + `snprintf`, multiple pieces, conditional suffixes, or `std::to_string` + concatenation chains — build it with `std::format` instead, then output the result (`fputs`, `std::cout`, or `printf("%s", ...)`). `std::format` is type-safe, needs no manual buffer sizing, and accepts `std::string` / `std::string_view` directly:
    ```cpp
    // Instead of composing into a buffer:
    char buf[160];
    snprintf(buf, sizeof(buf), "requested: %d, actual: %d (%s)", req, act, why);
    print_kv_line_string("steps", buf);
    // Compose, then output:
    print_kv_line_string("steps", std::format("requested: {}, actual: {} ({})", req, act, why).c_str());
    ```
- **String views**: use `std::string_view` for read-only string parameters in library/tool internals; keep `const char*` at the C API boundary (`cosyvoice.h` and friends are C interfaces).
- **RAII**: prefer `std::unique_ptr` (including the `*_handle` aliases in `tool_common_cosyvoice.h`) and `std::vector` / `std::string` over raw `new`/`delete` / owning raw pointers. The C API `create`/`free` pairs are the exception, since they are part of the C ABI.
- **Numeric parsing**: use `std::from_chars` for input parsing rather than `strtol` / `strtof` families (see the `COSYVOICE_MAX_WEIGHT_BUFFER_SIZE` handling in `src/cosyvoice-loader.cpp`).
- **Idioms**: prefer `nullptr` over `NULL`/`0`, `static_cast`/`reinterpret_cast` over C-style casts, `if` with initializer, structured bindings, and `enum class` over plain `enum` in non-C-API code.

New code should follow these; simple `printf`-style console output does not need rewriting.
