// Importance-matrix GGUF serialization implementation.
//
// Both layouts (`native` and `llama`) are read with `gguf_parser` and written
// with the regular gguf GGUF writer. The llama.cpp layout appends the
// ".in_sum2"/".counts" suffixes to the tensor names, so it only works while the
// base names stay short enough (checked when writing).

#include "imatrix.h"
#include "common.h"
#include "gguf_parser.h"

#include <ggml.h>
#include <gguf.h>
#include <ggml-cpp.h>

#include <algorithm>
#include <cstdint>
#include <format>
#include <string_view>
#include <vector>

namespace
{

constexpr std::string_view kSum2Suffix = ".in_sum2";
constexpr std::string_view kCountsSuffix = ".counts";

bool ends_with(std::string_view value, std::string_view suffix)
{
    return value.size() > suffix.size()
        && value.substr(value.size() - suffix.size()) == suffix;
}

} // namespace

// ---------------------------------------------------------------------------
// Loader
// ---------------------------------------------------------------------------

bool load_imatrix(const char* path, imatrix_data& out, imatrix_info* info, std::string& error, imatrix_format format)
{
    out.clear();
    if (info)
        *info = imatrix_info{};

    file_mmap mmap(path);
    if (!mmap)
    {
        error = std::format("failed to memory-map the file \"{}\"", path);
        return false;
    }

    gguf_parser parser;
    if (!parser.parse(static_cast<const uint8_t*>(mmap.data()), mmap.size(), nullptr))
    {
        error = std::format("failed to parse the GGUF file \"{}\"", path);
        return false;
    }

    const auto* file_start = static_cast<const uint8_t*>(mmap.data());
    const auto* file_end = file_start + mmap.size();

    auto data_in_bounds = [&](const ggml_tensor& t, size_t needed)
    {
        const auto* data = static_cast<const uint8_t*>(t.data);
        return data >= file_start && data <= file_end
            && static_cast<size_t>(file_end - data) >= needed;
    };

    struct entry_t
    {
        std::vector<float> sum2;
        std::vector<float> counts;
    };
    std::map<std::string, entry_t> entries;

    if (format == imatrix_format::native)
    {
        for (const auto& tensor : parser.info)
        {
            const auto& t = tensor.t;
            if (t.type != GGML_TYPE_F32 || t.ne[0] <= 0 || t.ne[1] != 2 || t.ne[2] != 1 || t.ne[3] != 1)
            {
                error = std::format("tensor \"{}\" must be an F32 tensor with shape [n_columns, 2]", t.name);
                return false;
            }

            const auto ncols = static_cast<size_t>(t.ne[0]);
            if (!data_in_bounds(t, t.nb[1] + ncols * sizeof(float)))
            {
                error = std::format("tensor \"{}\" data is out of bounds in \"{}\"", t.name, path);
                return false;
            }

            auto& entry = entries[t.name];
            if (!entry.sum2.empty())
            {
                error = std::format("duplicate tensor \"{}\"", t.name);
                return false;
            }

            const auto* data = static_cast<const uint8_t*>(t.data);
            const auto* sum2 = reinterpret_cast<const float*>(data);
            const auto* counts = reinterpret_cast<const float*>(data + t.nb[1]);
            entry.sum2.assign(sum2, sum2 + ncols);
            entry.counts.assign(counts, counts + ncols);
        }
    }
    else
    {
        for (const auto& tensor : parser.info)
        {
            const auto& t = tensor.t;
            const std::string_view name(t.name);

            std::string_view base;
            bool is_sum2 = false;
            if (ends_with(name, kSum2Suffix))
            {
                base = name.substr(0, name.size() - kSum2Suffix.size());
                is_sum2 = true;
            }
            else if (ends_with(name, kCountsSuffix))
                base = name.substr(0, name.size() - kCountsSuffix.size());
            else
            {
                error = std::format("unexpected tensor \"{}\"; expected \"<name>.in_sum2\" or \"<name>.counts\"", name);
                return false;
            }

            if (t.type != GGML_TYPE_F32 || t.ne[0] <= 0 || t.ne[1] != 1 || t.ne[2] != 1 || t.ne[3] != 1)
            {
                error = std::format("tensor \"{}\" must be a 1-D F32 tensor", name);
                return false;
            }

            const auto n = static_cast<size_t>(t.ne[0]);
            if (!data_in_bounds(t, n * sizeof(float)))
            {
                error = std::format("tensor \"{}\" data is out of bounds in \"{}\"", name, path);
                return false;
            }

            auto& entry = entries[std::string(base)];
            auto& dst = is_sum2 ? entry.sum2 : entry.counts;
            if (!dst.empty())
            {
                error = std::format("duplicate tensor \"{}\"", name);
                return false;
            }
            const auto* src = reinterpret_cast<const float*>(t.data);
            dst.assign(src, src + n);
        }

        for (auto& [name, entry] : entries)
        {
            if (entry.sum2.empty() || entry.counts.empty())
            {
                error = std::format("tensor \"{}\" is missing its .in_sum2 or .counts counterpart", name);
                return false;
            }
            if (entry.sum2.size() != entry.counts.size())
            {
                error = std::format("tensor \"{}\" has {} .in_sum2 columns but {} .counts columns",
                    name, entry.sum2.size(), entry.counts.size());
                return false;
            }
        }
    }

    // Normalize the accumulated sum2/counts pairs into importance values.
    for (auto& [name, entry] : entries)
    {
        auto& values = out[name];
        values.resize(entry.sum2.size());
        for (size_t i = 0; i != entry.sum2.size(); ++i)
        {
            if (entry.counts[i] <= 0)
            {
                error = std::format("tensor \"{}\" column {} has non-positive count {}", name, i, entry.counts[i]);
                return false;
            }
            values[i] = entry.sum2[i] / entry.counts[i];
        }
    }

    if (out.empty())
    {
        error = std::format("no importance-matrix entries found in \"{}\"", path);
        return false;
    }

    if (info)
    {
        const auto id = parser.find_key("imatrix.chunk_count");
        if (id >= 0)
            info->chunk_count = parser.val_u32(id);
    }

    return true;
}

// ---------------------------------------------------------------------------
// Writer
// ---------------------------------------------------------------------------

bool write_imatrix(
    const char*                                        path,
    const std::map<std::string, imatrix_tensor_stats>& stats,
    const std::vector<std::string>&                    datasets,
    uint32_t                                           chunk_count,
    uint32_t                                           chunk_size,
    std::string&                                       error,
    imatrix_format                                     format)
{
    if (stats.empty())
    {
        error = "no importance-matrix entries to write";
        return false;
    }

    for (const auto& [name, entry] : stats)
        if (name.empty() || entry.in_sum2.empty() || entry.in_sum2.size() != entry.in_counts.size())
        {
            error = std::format("invalid importance-matrix entry \"{}\"", name);
            return false;
        }

    struct tensor_out
    {
        std::string name;
        const float* data;
        int64_t ne0;
        int64_t ne1;
    };
    std::vector<tensor_out> tensors;
    std::map<std::string, std::vector<float>> assembled; // Native layout storage.

    if (format == imatrix_format::native)
    {
        // Row 0 holds the per-column sums of squared activations, row 1 the
        // per-column row counts. The map keeps the names sorted, which matches
        // the GGUF convention.
        for (const auto& [name, entry] : stats)
        {
            if (name.size() >= GGML_MAX_NAME)
            {
                error = std::format("importance-matrix entry \"{}\" does not fit a GGUF tensor name", name);
                return false;
            }
            auto& rows = assembled[name];
            rows.reserve(entry.in_sum2.size() * 2);
            rows.insert(rows.end(), entry.in_sum2.begin(), entry.in_sum2.end());
            rows.insert(rows.end(), entry.in_counts.begin(), entry.in_counts.end());
            tensors.push_back({ name, rows.data(), static_cast<int64_t>(entry.in_sum2.size()), 2 });
        }
    }
    else
    {
        // llama.cpp layout: one <name>.in_sum2 and one <name>.counts tensor per
        // weight, sorted by full name.
        for (const auto& [name, entry] : stats)
        {
            if (name.size() + kSum2Suffix.size() >= GGML_MAX_NAME)
            {
                error = std::format(
                    "importance-matrix entry \"{}\" is too long for the llama.cpp layout (tensor names must stay below GGML_MAX_NAME)",
                    name);
                return false;
            }
            tensors.push_back({ name + std::string(kSum2Suffix), entry.in_sum2.data(),
                static_cast<int64_t>(entry.in_sum2.size()), 1 });
            tensors.push_back({ name + std::string(kCountsSuffix), entry.in_counts.data(),
                static_cast<int64_t>(entry.in_counts.size()), 1 });
        }
        std::sort(tensors.begin(), tensors.end(),
            [](const tensor_out& a, const tensor_out& b) { return a.name < b.name; });
    }

    gguf_context_ptr gguf_ctx(gguf_init_empty());
    if (!gguf_ctx)
    {
        error = "gguf_init_empty failed";
        return false;
    }

    {
        std::vector<const char*> dataset_names;
        dataset_names.reserve(datasets.size());
        for (const auto& dataset : datasets)
            dataset_names.push_back(dataset.c_str());
        gguf_set_arr_str(gguf_ctx.get(), "imatrix.datasets", dataset_names.data(), dataset_names.size());
        gguf_set_val_u32(gguf_ctx.get(), "imatrix.chunk_count", chunk_count);
        gguf_set_val_u32(gguf_ctx.get(), "imatrix.chunk_size", chunk_size);
    }

    // Register the tensors with pointers into the assembled or stats buffers.
    {
        ggml_init_params params = {
            .mem_size = ggml_tensor_overhead() * tensors.size(),
            .mem_buffer = nullptr,
            .no_alloc = true,
        };
        ggml_context_ptr ml_ctx(ggml_init(params));
        if (!ml_ctx)
        {
            error = "ggml_init failed";
            return false;
        }
        for (const auto& t : tensors)
        {
            auto* tensor = t.ne1 == 2
                ? ggml_new_tensor_2d(ml_ctx.get(), GGML_TYPE_F32, t.ne0, 2)
                : ggml_new_tensor_1d(ml_ctx.get(), GGML_TYPE_F32, t.ne0);
            tensor->data = const_cast<float*>(t.data);
            ggml_set_name(tensor, t.name.c_str());
            gguf_add_tensor(gguf_ctx.get(), tensor);
        }
    }

    if (!gguf_write_to_file(gguf_ctx.get(), path, false))
    {
        error = std::format("failed to write the file \"{}\"", path);
        return false;
    }

    return true;
}
