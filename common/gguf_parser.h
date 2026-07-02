// Lightweight GGUF parser

#pragma once

#include <ggml.h>
#include <gguf.h>

#include <cstdint>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Parsed GGUF context
// ---------------------------------------------------------------------------

struct gguf_kv_entry {
    std::string key;
    bool is_array = false;
    enum gguf_type type = GGUF_TYPE_UINT8;
    std::vector<int8_t> data;
    std::string str_value;
    std::vector<int8_t> arr_data;
    size_t arr_len = 0;
    std::vector<std::string_view> arr_strings;
};

struct gguf_tensor_info {
    ggml_tensor t;
    uint64_t offset;
};

class gguf_parser {
public:
    size_t alignment = GGUF_DEFAULT_ALIGNMENT;
    std::vector<gguf_kv_entry> kv;
    std::vector<gguf_tensor_info> info;

    bool parse(
        const uint8_t* buffer, size_t size, ggml_context** out_ggml_ctx);

    // ---- Accessors ----

    int64_t find_key(const char* key_name) const;

    int64_t n_kv() const { return static_cast<int64_t>(kv.size()); }
    enum gguf_type arr_type(int64_t id) const;

    uint32_t val_u32(int64_t id) const;
    int32_t  val_i32(int64_t id) const;
    float    val_f32(int64_t id) const;
    const char* val_str(int64_t id) const;

    size_t arr_n(int64_t id) const;
    const void* arr_data(int64_t id) const;
    std::string_view arr_str(int64_t id, size_t i) const;
};
