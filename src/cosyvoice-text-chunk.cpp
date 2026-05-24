#include "cosyvoice-text-chunk.h"

#include <array>
#include <cstring>
#include <utility>

namespace cosyvoice_internal {
namespace {

constexpr std::array<std::string_view, 7> hard_enders = {
    "\n", ".", "?", "!", "\xE3\x80\x82", "\xEF\xBC\x9F", "\xEF\xBC\x81",
    // U+3002 IDEOGRAPHIC FULL STOP "。"
    // U+FF1F FULLWIDTH QUESTION MARK "？"
    // U+FF01 FULLWIDTH EXCLAMATION MARK "！"
};

constexpr std::array<std::string_view, 5> soft_seps = {
    ",", ";", ":", "\xE3\x80\x81", "\xEF\xBC\x9B",
    // U+3001 IDEOGRAPHIC COMMA "、"
    // U+FF1B FULLWIDTH SEMICOLON "；"
};

inline std::size_t utf8_char_len(unsigned char b)
{
    if ((b & 0x80) == 0x00) return 1;
    if ((b & 0xE0) == 0xC0) return 2;
    if ((b & 0xF0) == 0xE0) return 3;
    if ((b & 0xF8) == 0xF0) return 4;
    return 1;
}

inline bool match_at(std::string_view text, std::size_t pos, std::string_view pattern)
{
    return pos + pattern.size() <= text.size()
        && std::memcmp(text.data() + pos, pattern.data(), pattern.size()) == 0;
}

inline bool is_ws(char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; }

inline void emit_trimmed(std::vector<std::string>& out, std::string_view text, std::size_t lo, std::size_t hi)
{
    while (lo < hi && is_ws(text[lo])) ++lo;
    while (hi > lo && is_ws(text[hi - 1])) --hi;
    if (lo < hi)
        out.emplace_back(text.substr(lo, hi - lo));
}

}  // namespace

std::vector<std::string> split_into_fragments(std::string_view text)
{
    std::vector<std::string> fragments;
    if (text.empty())
        return fragments;

    std::size_t start = 0;
    std::size_t i = 0;
    while (i < text.size())
    {
        bool matched = false;

        for (const auto& ender : hard_enders)
        {
            if (match_at(text, i, ender))
            {
                const auto end = i + ender.size();
                emit_trimmed(fragments, text, start, end);
                start = end;
                i = end;
                matched = true;
                break;
            }
        }
        if (matched) continue;

        for (const auto& sep : soft_seps)
        {
            if (match_at(text, i, sep))
            {
                const auto end = i + sep.size();
                emit_trimmed(fragments, text, start, end);
                start = end;
                i = end;
                matched = true;
                break;
            }
        }
        if (matched) continue;

        i += utf8_char_len(static_cast<unsigned char>(text[i]));
    }

    if (start < text.size())
        emit_trimmed(fragments, text, start, text.size());

    return fragments;
}

std::vector<std::string> reassemble_by_token_budget(
    const std::vector<std::string>& fragments,
    std::size_t max_tokens,
    const token_count_fn& count)
{
    std::vector<std::string> chunks;
    if (fragments.empty() || max_tokens == 0)
        return chunks;

    std::string current;
    std::size_t current_tokens = 0;
    for (const auto& fragment : fragments)
    {
        const std::size_t fragment_tokens = count(fragment);
        if (current_tokens != 0 && current_tokens + fragment_tokens > max_tokens)
        {
            chunks.push_back(std::move(current));
            current.clear();
            current_tokens = 0;
        }
        current += fragment;
        current_tokens += fragment_tokens;
    }
    if (!current.empty())
        chunks.push_back(std::move(current));

    return chunks;
}

std::vector<std::vector<int>> reassemble_by_token_budget(
    std::vector<std::vector<int>>& fragment_tokens,
    std::size_t max_tokens)
{
    std::vector<std::vector<int>> chunks;
    if (fragment_tokens.empty() || max_tokens == 0)
        return chunks;

    std::vector<int> current;
    for (auto& tokens : fragment_tokens)
        if (current.size() != 0 && current.size() + tokens.size() > max_tokens)
        {
            current.swap(tokens);
            chunks.push_back(std::move(tokens));
        }
        else
            current.insert(current.end(), tokens.begin(), tokens.end());
    if (!current.empty())
        chunks.push_back(std::move(current));

    return chunks;
}

}  // namespace cosyvoice_internal
