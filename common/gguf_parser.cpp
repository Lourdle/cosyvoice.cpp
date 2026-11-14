// Lightweight GGUF parser implementation.

#include "gguf_parser.h"

#include <cstring>
#include <vector>
#include <cstdio>
#include <format>

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

#define LOG_ERROR(fmt, ...) std::fputs(std::format("\033[31m[GGUF] {}: " fmt "\n\033[0m", __func__ __VA_OPT__(,) __VA_ARGS__).c_str(), stderr)

static constexpr size_t pad(size_t n, size_t alignment)
{
    return (n + alignment - 1) & ~(alignment - 1);
}

static constexpr size_t gguf_type_size(gguf_type type)
{
    switch (type) {
    case GGUF_TYPE_UINT8:   return sizeof(uint8_t);
    case GGUF_TYPE_INT8:    return sizeof(int8_t);
    case GGUF_TYPE_UINT16:  return sizeof(uint16_t);
    case GGUF_TYPE_INT16:   return sizeof(int16_t);
    case GGUF_TYPE_UINT32:  return sizeof(uint32_t);
    case GGUF_TYPE_INT32:   return sizeof(int32_t);
    case GGUF_TYPE_FLOAT32: return sizeof(float);
    case GGUF_TYPE_BOOL:    return sizeof(int8_t);
    case GGUF_TYPE_STRING:  return 0;
    case GGUF_TYPE_ARRAY:   return 0;
    case GGUF_TYPE_UINT64:  return sizeof(uint64_t);
    case GGUF_TYPE_INT64:   return sizeof(int64_t);
    case GGUF_TYPE_FLOAT64: return sizeof(double);
    default:                return 0;
    }
}

template<typename T>
static inline bool read_le(const uint8_t*& p, const uint8_t* end, T& out)
{
    if (static_cast<size_t>(end - p) < sizeof(T)) return false;
    std::memcpy(&out, p, sizeof(T));
    p += sizeof(T);
    return true;
}

static inline bool read_gguf_string(const uint8_t*& p, const uint8_t* end, std::string_view& out)
{
    uint64_t len;
    if (!read_le(p, end, len)) return false;
    if (len > 1024 * 1024 * 1024)
    {
        LOG_ERROR("string length {} exceeds maximum", len);
        return false;
    }
    if (static_cast<size_t>(end - p) < len)
    {
        LOG_ERROR("string length {} exceeds remaining buffer", len);
        return false;
    }
    out = std::string_view(reinterpret_cast<const char*>(p), static_cast<size_t>(len));
    p += len;
    return true;
}

// ---------------------------------------------------------------------------
// gguf_parser methods
// ---------------------------------------------------------------------------

int64_t gguf_parser::find_key(const char* key_name) const
{
    for (int64_t i = 0; i < static_cast<int64_t>(kv.size()); ++i)
        if (kv[static_cast<size_t>(i)].key == key_name) return i;
    return -1;
}

// ---- Accessor implementations ----

gguf_type gguf_parser::arr_type(int64_t id) const 
{
    if (id < 0 || id >= static_cast<int64_t>(kv.size())) return GGUF_TYPE_COUNT;
    const auto& e = kv[static_cast<size_t>(id)];
    return e.is_array ? e.type : GGUF_TYPE_COUNT;
}

uint32_t gguf_parser::val_u32(int64_t id) const
{
    if (id < 0 || id >= static_cast<int64_t>(kv.size()) || kv[static_cast<size_t>(id)].data.size() < 4) return 0;
    uint32_t v; std::memcpy(&v, kv[static_cast<size_t>(id)].data.data(), 4); return v;
}

int32_t  gguf_parser::val_i32(int64_t id) const
{
    if (id < 0 || id >= static_cast<int64_t>(kv.size()) || kv[static_cast<size_t>(id)].data.size() < 4) return 0;
    int32_t v; std::memcpy(&v, kv[static_cast<size_t>(id)].data.data(), 4); return v;
}

float    gguf_parser::val_f32(int64_t id) const
{
    if (id < 0 || id >= static_cast<int64_t>(kv.size()) || kv[static_cast<size_t>(id)].data.size() < 4) return 0.f;
    float v; std::memcpy(&v, kv[static_cast<size_t>(id)].data.data(), 4); return v;
}

std::string_view gguf_parser::val_str(int64_t id) const
{
    if (id < 0 || id >= static_cast<int64_t>(kv.size())) return {};
    return kv[static_cast<size_t>(id)].str_value;
}

size_t gguf_parser::arr_n(int64_t id) const
{
    if (id < 0 || id >= static_cast<int64_t>(kv.size())) return 0;
    const auto& e = kv[static_cast<size_t>(id)];
    if (e.type == GGUF_TYPE_STRING && e.is_array) return e.arr_strings.size();
    return e.arr_len;
}

const void* gguf_parser::arr_data(int64_t id) const
{
    if (id < 0 || id >= static_cast<int64_t>(kv.size())) return nullptr;
    const auto& e = kv[static_cast<size_t>(id)];
    if (e.type == GGUF_TYPE_STRING) return nullptr;
    return e.arr_owned.empty() ? static_cast<const void*>(e.arr_view) : e.arr_owned.data();
}

std::string_view gguf_parser::arr_str(int64_t id, size_t i) const
{
    if (id < 0 || id >= static_cast<int64_t>(kv.size())) return "";
    const auto& e = kv[static_cast<size_t>(id)];
    if (i < e.arr_strings.size()) return e.arr_strings[i];
    return "";
}

// ---------------------------------------------------------------------------
// Parser — populates this context in-place
// ---------------------------------------------------------------------------

bool gguf_parser::parse(const uint8_t* _buffer, size_t _size, ggml_context** out_ggml_ctx)
{
    if (!_buffer || _size < 24)
    {
        LOG_ERROR("buffer too small");
        return false;
    }

    const uint8_t* end = _buffer + _size;
    const uint8_t* p = _buffer;

    // --- Magic ---
    char magic[4];
    std::memcpy(magic, p, 4);
    p += 4;
    if (std::memcmp(magic, GGUF_MAGIC, 4) != 0)
    {
        char c0 = isprint(magic[0]) ? magic[0] : '?';
        char c1 = isprint(magic[1]) ? magic[1] : '?';
        char c2 = isprint(magic[2]) ? magic[2] : '?';
        char c3 = isprint(magic[3]) ? magic[3] : '?';
        LOG_ERROR("invalid magic '{}{}{}{}', expected 'GGUF'", c0, c1, c2, c3);
        return false;
    }

    // --- Version ---
    uint32_t version;
    if (!read_le(p, end, version) || version != 3)
    {
        LOG_ERROR("unsupported GGUF version {}", version);
        return false;
    }

    // --- Counters ---
    int64_t n_tensors, n_kv;
    if (!read_le(p, end, n_tensors) || n_tensors < 0
        || !read_le(p, end, n_kv) || n_kv < 0)
    {
        LOG_ERROR("invalid header");
        return false;
    }

    // Reject counts that cannot fit in the remaining buffer before reserving:
    // every KV pair needs at least 10 bytes, every tensor record at least 24.
    if (static_cast<uint64_t>(n_kv) > _size / 10)
    {
        LOG_ERROR("kv count {} exceeds buffer size", n_kv);
        return false;
    }
    if (static_cast<uint64_t>(n_tensors) > _size / 24)
    {
        LOG_ERROR("tensor count {} exceeds buffer size", n_tensors);
        return false;
    }

    auto buffer = _buffer;
    auto size = _size;
    kv.reserve(static_cast<size_t>(n_kv));
    info.reserve(static_cast<size_t>(n_tensors));

    // --- KV pairs ---
    for (int64_t i = 0; i < n_kv; ++i)
    {
        auto& kv = this->kv.emplace_back();

        // Key
        if (!read_gguf_string(p, end, kv.key)) return false;

        // Value type
        gguf_type type;
        if (!read_le(p, end, type)) return false;
        kv.type = type;

        if (type == GGUF_TYPE_ARRAY)
        {
            kv.is_array = true;
            auto& elem_type = kv.type;
            if (!read_le(p, end, elem_type)) return false;
            uint64_t arr_len;
            if (!read_le(p, end, arr_len)) return false;
            kv.arr_len = arr_len;

            if (elem_type == GGUF_TYPE_STRING)
            {
                // Each string needs at least an 8-byte length field.
                if (arr_len > static_cast<uint64_t>(end - p) / 8)
                {
                    LOG_ERROR("string array length {} exceeds buffer", arr_len);
                    return false;
                }
                kv.arr_strings.resize(static_cast<size_t>(arr_len));
                for (uint64_t j = 0; j < arr_len; ++j)
                    if (!read_gguf_string(p, end, kv.arr_strings[static_cast<size_t>(j)])) return false;
            }
            else
            {
                const size_t elem_sz = gguf_type_size(elem_type);
                if (elem_sz == 0)
                {
                    LOG_ERROR("invalid array element type");
                    return false;
                }
                if (arr_len > SIZE_MAX / elem_sz)
                {
                    LOG_ERROR("array length {} overflows", arr_len);
                    return false;
                }
                const size_t total = elem_sz * static_cast<size_t>(arr_len);
                if (static_cast<size_t>(end - p) < total) return false;

                // Zero-copy view; copy only when the buffer view would be misaligned.
                if (total > 0 && reinterpret_cast<uintptr_t>(p) % elem_sz != 0)
                    kv.arr_owned.assign(reinterpret_cast<const int8_t*>(p),
                        reinterpret_cast<const int8_t*>(p) + total);
                else
                    kv.arr_view = reinterpret_cast<const int8_t*>(p);
                p += total;
            }
        }
        else
        {
            size_t elem_sz = gguf_type_size(type);
            if (elem_sz > 0)
            {
                if (static_cast<size_t>(end - p) < elem_sz) return false;
                kv.data.resize(elem_sz);
                std::memcpy(kv.data.data(), p, elem_sz);
                p += elem_sz;
                // String scalar
            }
            else if (!read_gguf_string(p, end, kv.str_value)) return false;
        }
    }

    // --- Tensor info ---
    for (int64_t i = 0; i < n_tensors; ++i)
    {
        auto& ti = this->info.emplace_back();
        std::memset(&ti.t, 0, sizeof(ggml_tensor));
        ti.t.ne[1] = 1;
        ti.t.ne[2] = 1;
        ti.t.ne[3] = 1;

        // Name
        {
            std::string_view name;
            if (!read_gguf_string(p, end, name)) return false;
            if (name.size() >= GGML_MAX_NAME)
            {
                LOG_ERROR("tensor '{}' name too long", name);
                return false;
            }
            std::memcpy(ti.t.name, name.data(), name.size());
        }

        // Dimensions
        {
            uint32_t n_dims;
            if (!read_le(p, end, n_dims)) return false;
            if (n_dims > GGML_MAX_DIMS)
            {
                LOG_ERROR("tensor '{}' has {} dimensions", ti.t.name, n_dims);
                return false;
            }
            for (uint32_t j = 0; j < n_dims; ++j)
            {
                if (!read_le(p, end, ti.t.ne[j])) return false;
                if (ti.t.ne[j] < 0)
                {
                    LOG_ERROR("tensor '{}' dimension {} negative", ti.t.name, j);
                    return false;
                }
            }

            // Overflow check
            if (ti.t.ne[0] > 0)
                if (INT64_MAX / ti.t.ne[0] <= ti.t.ne[1] ||
                    (ti.t.ne[2] > 0 && INT64_MAX / (ti.t.ne[0] * ti.t.ne[1]) <= ti.t.ne[2]) ||
                    (ti.t.ne[3] > 0 && INT64_MAX / (ti.t.ne[0] * ti.t.ne[1] * ti.t.ne[2]) <= ti.t.ne[3]))
                {
                    LOG_ERROR("overflow in tensor '{}'", ti.t.name);
                    return false;
                }
        }

        // Type
        {
            ggml_type ttype;
            if (!read_le(p, end, ttype)) return false;
            ti.t.type = ttype;
            if (ttype < 0 || ttype >= GGML_TYPE_COUNT)
            {
                LOG_ERROR("tensor '{}' invalid type {}", ti.t.name, static_cast<int>(ttype));
                return false;
            }
            size_t type_size = ggml_type_size(ttype);
            int64_t blck_size = ggml_blck_size(ttype);
            if (blck_size == 0 || ti.t.ne[0] % blck_size != 0)
            {
                LOG_ERROR("tensor '{}' row size not divisible by block size", ti.t.name);
                return false;
            }
            ti.t.nb[0] = type_size;
            ti.t.nb[1] = static_cast<size_t>(ti.t.ne[0] / blck_size) * type_size;
            for (int j = 2; j < GGML_MAX_DIMS; ++j) {
                ti.t.nb[j] = ti.t.nb[j - 1] * static_cast<size_t>(ti.t.ne[j - 1]);
            }
        }

        // Data offset
        if (!read_le(p, end, ti.offset)) return false;
    }

    // --- Alignment ---
    {
        uint32_t align_val = GGUF_DEFAULT_ALIGNMENT;
        for (int64_t i = 0; i < static_cast<int64_t>(kv.size()); ++i)
            if (kv[i].key == "general.alignment")
            {
                if (!kv[i].data.empty() && kv[i].data.size() >= sizeof(uint32_t))
                    std::memcpy(&align_val, kv[i].data.data(), sizeof(uint32_t));
                break;
            }

        alignment = align_val;
        if (alignment == 0 || (alignment & (alignment - 1)) != 0)
        {
            LOG_ERROR("alignment {} is not a power of 2", alignment);
            return false;
        }
    }

    // --- Data section ---
    size_t data_start = pad(static_cast<size_t>(p - buffer), alignment);
    if (data_start > size)
    {
        LOG_ERROR("data section beyond buffer");
        return false;
    }

    auto data_section = buffer + data_start;

    // Validate offsets and compute data size
    {
        size_t expected_offset = 0;
        for (size_t i = 0; i < info.size(); ++i)
        {
            if (info[i].offset != expected_offset)
            {
                LOG_ERROR("tensor '{}' offset {}, expected {}",
                    info[i].t.name, info[i].offset, expected_offset);
                return false;
            }
            size_t tensor_bytes = ggml_nbytes(&info[i].t);
            expected_offset += pad(tensor_bytes, alignment);
        }
    }

    // Set tensor data pointers into mmap region (zero-copy)
    for (size_t i = 0; i < info.size(); ++i)
        info[i].t.data = const_cast<void*>(
            reinterpret_cast<const void*>(data_section + info[i].offset));

    // --- Create ggml_context with tensor descriptors ---
    if (out_ggml_ctx)
    {
        ggml_init_params pdata = {
            .mem_size   = info.size() * ggml_tensor_overhead(),
            .mem_buffer = nullptr,
            .no_alloc   = true,
        };

        *out_ggml_ctx = ggml_init(pdata);
        if (!*out_ggml_ctx)
        {
            LOG_ERROR("failed to initialize ggml context");
            return false;
        }

        for (size_t i = 0; i < info.size(); ++i)
        {
            const auto& ti = info[i];
            ggml_tensor* cur = ggml_new_tensor(*out_ggml_ctx, ti.t.type, GGML_MAX_DIMS, ti.t.ne);
            if (!cur)
            {
                LOG_ERROR("failed to create tensor '{}'", ti.t.name);
                return false;
            }
            ggml_set_name(cur, ti.t.name);
            cur->data = ti.t.data;
        }
    }

    return true;
}
