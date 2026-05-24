#pragma once

#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace cosyvoice_internal {

// Split UTF-8 text into the smallest natural fragments — one per sentence ender
// (".", "?", "!", "\n", "。", "？", "！") and one per soft separator
// (",", ";", ":", "、", "；"). Each fragment retains its trailing separator so
// reassembly is loss-less. Whitespace-only fragments are dropped.
std::vector<std::string> split_into_fragments(std::string_view text);

// Function type for measuring a fragment in tokens. Implementations pass the
// fragment to the model tokenizer and return the count.
using token_count_fn = std::function<std::size_t(std::string_view)>;

// Re-concatenate fragments into chunks so that no chunk's tokenized length
// exceeds `max_tokens`. A single fragment that exceeds the budget on its own
// becomes a one-fragment chunk (callers must size `max_tokens` so this stays
// rare — splitting individual fragments would require sub-fragment tokenization
// loops which the current pipeline does not need).
std::vector<std::string> reassemble_by_token_budget(
    const std::vector<std::string>& fragments,
    std::size_t max_tokens,
    const token_count_fn& count);

// Fast-split overload: merge pre-tokenized fragment vectors into chunks
// directly by token count, returning token-ID vectors for each chunk.
std::vector<std::vector<int>> reassemble_by_token_budget(
    std::vector<std::vector<int>>& fragment_tokens,
    std::size_t max_tokens);

}  // namespace cosyvoice_internal
