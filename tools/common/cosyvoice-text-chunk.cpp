#include "cosyvoice-text-chunk.h"

#include <array>
#include <cstring>

namespace cosyvoice_common {
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
    return 1;  // invalid byte; advance by one to avoid getting stuck
}

inline bool match_at(std::string_view text, std::size_t pos, std::string_view pattern)
{
    return pos + pattern.size() <= text.size()
        && std::memcmp(text.data() + pos, pattern.data(), pattern.size()) == 0;
}

inline bool is_ws(char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; }

inline void emit_trimmed(std::vector<std::string>& chunks, std::string_view text, std::size_t lo, std::size_t hi)
{
    while (lo < hi && is_ws(text[lo])) ++lo;
    while (hi > lo && is_ws(text[hi - 1])) --hi;
    if (lo < hi)
        chunks.emplace_back(text.substr(lo, hi - lo));
}

}  // namespace

std::vector<std::string> split_text_into_chunks(std::string_view text, std::size_t max_bytes)
{
    std::vector<std::string> chunks;
    if (text.empty())
        return chunks;
    if (max_bytes == 0)
        max_bytes = default_max_chunk_bytes;

    std::size_t start = 0;
    std::size_t last_soft = 0;  // byte position right after the most recent soft separator
    std::size_t i = 0;
    while (i < text.size())
    {
        bool matched = false;

        for (const auto& ender : hard_enders)
        {
            if (match_at(text, i, ender))
            {
                const auto end = i + ender.size();
                emit_trimmed(chunks, text, start, end);
                start = end;
                last_soft = 0;
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
                i += sep.size();
                last_soft = i;
                if (i - start >= max_bytes)
                {
                    emit_trimmed(chunks, text, start, last_soft);
                    start = last_soft;
                    last_soft = 0;
                }
                matched = true;
                break;
            }
        }
        if (matched) continue;

        i += utf8_char_len(static_cast<unsigned char>(text[i]));

        if (i - start >= max_bytes)
        {
            std::size_t cut = (last_soft > start) ? last_soft : i;
            while (cut > start && cut < text.size()
                && (static_cast<unsigned char>(text[cut]) & 0xC0) == 0x80)
                --cut;
            if (cut <= start) cut = i;
            emit_trimmed(chunks, text, start, cut);
            start = cut;
            last_soft = 0;
        }
    }

    if (start < text.size())
        emit_trimmed(chunks, text, start, text.size());

    return chunks;
}

}  // namespace cosyvoice_common
