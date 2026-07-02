#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <format>

#include "cosyvoice-internal.h"
#include "common.h"
#include "gguf_parser.h"

inline
std::string combine_prefix(const std::string& prefix, const char* name)
{
    if (prefix.empty())
        return name;
    else
        return std::format("{}.{}", prefix, name);
}

// ---------------------------------------------------------------------------
// Metadata loader — wraps parsed_context, provides compatible API
// ---------------------------------------------------------------------------

struct gguf_metadata_loader
{
    gguf_parser& parser;

    explicit gguf_metadata_loader(gguf_parser* c) : parser(*c) {}
    explicit gguf_metadata_loader(gguf_parser& c) : parser(c) {}

    operator bool() const { return true; }

    template<typename T>
    bool get_metadata(const std::string& prefix, const char* name, T& value) const
    {
        return get_metadata(combine_prefix(prefix, name), value);
    }

    template<typename T>
    bool get_metadata(const char* key_name, T& value) const
    {
        return get_metadata(std::string(key_name), value);
    }

    template<typename T>
    bool get_metadata(const std::string& key_name, T& value) const
    {
        int64_t id = parser.find_key(key_name.c_str());
        if (id < 0) return false;

        if constexpr (std::is_same_v<T, int32_t>)
            value = parser.val_i32(id);
        else if constexpr (std::is_same_v<T, uint32_t>)
            value = parser.val_u32(id);
        else if constexpr (std::is_same_v<T, float>)
            value = parser.val_f32(id);
        else static_assert("unsupported type");
        return true;
    }

    std::string_view get_string(const char* key_name) const
    {
        int64_t id = parser.find_key(key_name);
        GGML_ASSERT(id >= 0);
        return std::string_view(parser.val_str(id));
    }

    int64_t find_key(const char* key_name) const
    {
        return parser.find_key(key_name);
    }

    bool find_metadata_key(const char* key_name, int64_t& key_id) const
    {
        key_id = parser.find_key(key_name);
        return key_id >= 0;
    }
};

// ---------------------------------------------------------------------------
// Full GGUF loader — metadata + tensor resolution, zero-copy from mmap
// ---------------------------------------------------------------------------

struct gguf_loader : gguf_metadata_loader
{
    ggml_context* ggml_ctx = nullptr;

    gguf_loader(gguf_parser& parser, const void* data, size_t size) :
        gguf_metadata_loader(parser),
        ggml_ctx(nullptr)
    {
        parser.parse(static_cast<const uint8_t*>(data), size, &ggml_ctx);
    }
    gguf_loader(const gguf_loader&) = delete;
    gguf_loader(gguf_loader&&) = delete;

    ~gguf_loader() { close(); }

    void close()
    {
        if (ggml_ctx)
        {
            ggml_free(ggml_ctx);
            ggml_ctx = nullptr;
        }
    }

    operator ggml_context* () const { return ggml_ctx; }

    ggml_tensor* get_gguf_tensor(const std::string& prefix, const char* name, bool optional = false) const
    {
        auto tensor = ggml_get_tensor(ggml_ctx, combine_prefix(prefix, name).c_str());
        GGML_ASSERT(optional || tensor);
        return tensor;
    }
};
