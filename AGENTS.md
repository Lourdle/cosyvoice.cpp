# AGENTS.md

Guidance for AI coding agents working in this repository. `docs/DEVELOPING.md` is the authoritative reference for the conventions summarized here.

## Project

C++20/CMake port of the CosyVoice TTS inference pipeline (CosyVoice3), MIT-licensed. The core inference library is `cosyvoice` in `src/` with the public C API exposed in `include/cosyvoice*.h`. `common/` is the repository's own helper library (`gguf_parser`, `file_mmap`, UTF-8-aware file helpers) that links `ggml` publicly.

## Build & verify

- Backend/model-backed tests are opt-in and gated behind `COSYVOICE_BUILD_TESTS=ON` (`tests/`, run manually, not by plain `ctest`).
- Before finishing a task, build the touched targets to confirm they compile.

## Hard rules

- **Entry point** — every new executable (tool or test) implements `int tool_entry(int argc, char** argv)` and must **not** define `main`; link `tools_common` (plus `cosyvoice`, `common`, `ggml` as needed). On POSIX, `tools/common/tool_common.h` `#define`s `tool_entry` to `main`; on Windows, the single `main` (with the UTF-8 argv conversion) lives in `tools/common/common.cpp` and calls `tool_entry`.
- **Encoding** — every path and string is UTF-8 in memory. On Windows, do **not** pass a UTF-8 `char*` to an ANSI ("A") API (it decodes with the locale ACP, e.g. GBK, not UTF-8). Use the helpers from `common/`: `open_ifstream_utf8()`, `open_ofstream_utf8()`, `file_mmap`, or `utf8_to_wstr()` before calling a W API.
- **Modern C++** — simple `printf`/`fprintf` console prints with plain `%s`/`%d`/`%u` conversions are fine; when *composing* strings (buffers + `snprintf`, multi-part or conditional pieces, `std::to_string` concatenation), build them with `std::format`, then output. New code follows modern C++: RAII (`unique_ptr`, `vector`, `string`), `std::string_view` for read-only internal parameters, `std::from_chars` for numeric parsing, `nullptr`/`static_cast`, etc.
- **C API boundary** — `include/cosyvoice.h` and friends are C interfaces: keep `const char*` and C-compatible types there, no C++-only types.

## Docs conventions

- Documentation is bilingual: `X.md` + `X_zh.md` pairs must stay in sync (same content, links adjusted to the localized counterparts).
- Chinese docs (`*_zh.md`) carry a UTF-8 BOM; all markdown files end with a single trailing newline.
- No language-switcher / back-to-root navigation lines at the top of docs (title, then content); the root READMEs keep their existing language-switcher line.
- When editing tables or section titles, keep the English/Chinese anchor points in sync.

## Layout

- `src/` — core library implementation.
- `include/` — public C API headers.
- `common/` — repository helper library (`gguf_parser`, `file_mmap`, UTF-8 file helpers).
- `tools/` — executables: `cli/`, `server/`, `quantize/`, plus `common/` (`tools_common` static library).
- `tests/` — manual model-backed regression tests.
- `docs/` — user and developer documentation.

## Further reading

- `docs/DEVELOPING.md` — tool/test conventions, link pattern, modern-C++ guidance.
- `docs/BUILD.md`, `docs/TOOLS.md`, `docs/ENVIRONMENT.md`, `docs/STREAMING.md`, `docs/CONVERT.md`.
