// Importance-matrix (imatrix) collection statistics and GGUF serialization.
//
// Two on-disk layouts are supported:
//
// - `imatrix_format::native` (default): one F32 tensor per collected weight,
//   named exactly like the model tensor, with shape [n_columns, 2]. Row 0 holds
//   the per-column sums of squared activations; row 1 holds the per-column
//   number of accumulated rows.
// - `imatrix_format::llama`: llama.cpp's layout, two 1-D F32 tensors per weight
//   named `<name>.in_sum2` (sums of squared activations) and `<name>.counts`
//   (row counts). The suffix makes names longer, so this layout cannot be used
//   when the model tensor names are close to GGML_MAX_NAME.
//
// Metadata keys in both layouts: `imatrix.datasets`, `imatrix.chunk_count`,
// `imatrix.chunk_size`.

#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

// On-disk importance-matrix layout.
enum class imatrix_format
{
    native, // One tensor per weight, named after the model tensor, shape [n_columns, 2].
    llama,  // llama.cpp style: `<name>.in_sum2` and `<name>.counts` tensors.
};

// Raw per-tensor statistics accumulated by a collector.
struct imatrix_tensor_stats
{
    std::vector<float> in_sum2;   // Per-column sums of squared activations.
    std::vector<float> in_counts; // Per-column number of accumulated rows.
};

// Per-tensor importance values (in_sum2 / in_counts), ready to be passed as the
// `imatrix` argument of ggml_quantize_chunk.
using imatrix_data = std::map<std::string, std::vector<float>>;

// Importance-matrix file metadata.
struct imatrix_info
{
    uint32_t chunk_count = 0; // Number of evaluated activation chunks, when known.
};

// Load an importance matrix from a GGUF file. Returns false and fills `error`
// on failure.
bool load_imatrix(
    const char*     path,
    imatrix_data&   out,
    imatrix_info*   info,
    std::string&    error,
    imatrix_format  format = imatrix_format::native
);

// Write an importance matrix to a GGUF file. `datasets` names the calibration
// data sets, `chunk_count` is the number of evaluated activation chunks and
// `chunk_size` is the number of tokens per chunk. Returns false and fills
// `error` on failure.
bool write_imatrix(
    const char*                                        path,
    const std::map<std::string, imatrix_tensor_stats>& stats,
    const std::vector<std::string>&                    datasets,
    uint32_t                                           chunk_count,
    uint32_t                                           chunk_size,
    std::string&                                       error,
    imatrix_format                                     format = imatrix_format::native
);
