#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace cosyvoice_common {

// Default chunk-size budget in bytes. CosyVoice's LLM has n_max_seq=2048 by default; the KV cache,
// reference-audio prompt tokens and the speech tokens decoded from the input all share that budget.
// 600 bytes leaves room for ~200 kana or ~150 mixed CJK characters per chunk while staying clear of
// the assertion in llm_prefill that previously aborted on long input.
inline constexpr std::size_t default_max_chunk_bytes = 600;

// Split UTF-8 text into chunks suitable for sequential TTS calls.
//
// - Splits at hard sentence enders (".", "?", "!", "\n", and their full-width counterparts) whenever
//   they appear, regardless of chunk size.
// - When no sentence end is reached and the running chunk exceeds `max_bytes`, falls back to the most
//   recent soft separator (",", ";", ":", "、", "；").
// - If no separator is available, force-splits at a UTF-8 code-point boundary so multi-byte sequences
//   are never cut in half.
// - Whitespace-only fragments are dropped.
std::vector<std::string> split_text_into_chunks(
    std::string_view text,
    std::size_t max_bytes = default_max_chunk_bytes);

}  // namespace cosyvoice_common
