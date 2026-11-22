#if defined(_MSC_VER)
    #define _SILENCE_CXX17_CODECVT_HEADER_DEPRECATION_WARNING
#endif

#include "unicode.h"
#include "unicode-data.h"

#include <array>
#include <cassert>
#include <codecvt>
#include <cstdint>
#include <format>
#include <locale>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#define PCRE2_CODE_UNIT_WIDTH 0
#include <pcre2.h>

static size_t utf8_encode(uint32_t cpt, char* out) {
    if (/* 0x00 <= cpt && */ cpt <= 0x7f) {
        out[0] = static_cast<char>(cpt);
        return 1;
    }
    if (0x80 <= cpt && cpt <= 0x7ff) {
        out[0] = static_cast<char>(0xc0 | ((cpt >> 6) & 0x1f));
        out[1] = static_cast<char>(0x80 | (cpt & 0x3f));
        return 2;
    }
    if (0x800 <= cpt && cpt <= 0xffff) {
        out[0] = static_cast<char>(0xe0 | ((cpt >> 12) & 0x0f));
        out[1] = static_cast<char>(0x80 | ((cpt >> 6) & 0x3f));
        out[2] = static_cast<char>(0x80 | (cpt & 0x3f));
        return 3;
    }
    if (0x10000 <= cpt && cpt <= 0x10ffff) {
        out[0] = static_cast<char>(0xf0 | ((cpt >> 18) & 0x07));
        out[1] = static_cast<char>(0x80 | ((cpt >> 12) & 0x3f));
        out[2] = static_cast<char>(0x80 | ((cpt >> 6) & 0x3f));
        out[3] = static_cast<char>(0x80 | (cpt & 0x3f));
        return 4;
    }

    throw std::invalid_argument("invalid codepoint");
}

size_t unicode_len_utf8(char src) {
    const size_t lookup[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4 };
    uint8_t highbits = static_cast<uint8_t>(src) >> 4;
    return lookup[highbits];
}

uint32_t unicode_cpt_from_utf8(const std::string_view& utf8, size_t& offset) {
    assert(offset < utf8.size());
    if (!(utf8[offset + 0] & 0x80)) {
        auto result = utf8[offset + 0];
        offset += 1;
        return result;
    }
    if (!(utf8[offset + 0] & 0x40)) {
        throw std::invalid_argument("invalid character");
    }
    if (!(utf8[offset + 0] & 0x20)) {
        if (offset + 1 >= utf8.size() || !((utf8[offset + 1] & 0xc0) == 0x80)) {
            throw std::invalid_argument("invalid character");
        }
        auto result = ((utf8[offset + 0] & 0x1f) << 6) | (utf8[offset + 1] & 0x3f);
        offset += 2;
        return result;
    }
    if (!(utf8[offset + 0] & 0x10)) {
        if (offset + 2 >= utf8.size() || !((utf8[offset + 1] & 0xc0) == 0x80) || !((utf8[offset + 2] & 0xc0) == 0x80)) {
            throw std::invalid_argument("invalid character");
        }
        auto result = ((utf8[offset + 0] & 0x0f) << 12) | ((utf8[offset + 1] & 0x3f) << 6) | (utf8[offset + 2] & 0x3f);
        offset += 3;
        return result;
    }
    if (!(utf8[offset + 0] & 0x08)) {
        if (offset + 3 >= utf8.size() || !((utf8[offset + 1] & 0xc0) == 0x80) || !((utf8[offset + 2] & 0xc0) == 0x80) || !((utf8[offset + 3] & 0xc0) == 0x80)) {
            throw std::invalid_argument("invalid character");
        }
        auto result = ((utf8[offset + 0] & 0x07) << 18) | ((utf8[offset + 1] & 0x3f) << 12) | ((utf8[offset + 2] & 0x3f) << 6) | (utf8[offset + 3] & 0x3f);
        offset += 4;
        return result;
    }
    throw std::invalid_argument("failed to convert utf8 to codepoint");
}

static std::vector<unicode_cpt_flags> unicode_cpt_flags_array() {
    std::vector<unicode_cpt_flags> cpt_flags(MAX_CODEPOINTS, unicode_cpt_flags::UNDEFINED);

    assert(unicode_ranges_flags.begin()[0].first == 0);
    assert(unicode_ranges_flags.begin()[unicode_ranges_flags.size() - 1].first == MAX_CODEPOINTS);
    for (size_t i = 1; i < unicode_ranges_flags.size(); ++i) {
        const auto range_ini = unicode_ranges_flags.begin()[i - 1];  // codepoint_ini, flags
        const auto range_end = unicode_ranges_flags.begin()[i];    // codepoint_end, flags
        for (uint32_t cpt = range_ini.first; cpt < range_end.first; ++cpt) {
            cpt_flags[cpt] = range_ini.second;
        }
    }

    for (auto cpt : unicode_set_whitespace) {
        cpt_flags[cpt].is_whitespace = true;
    }

    for (auto p : unicode_map_lowercase) {
        cpt_flags[p.second].is_lowercase = true;
    }

    for (auto p : unicode_map_uppercase) {
        cpt_flags[p.second].is_uppercase = true;
    }

    for (auto& range : unicode_ranges_nfd) {  // start, last, nfd
        cpt_flags[range.nfd].is_nfd = true;
    }

    return cpt_flags;
}

struct unicode_byte_to_utf8_table {
    std::array<std::array<char, 4>, 256> data{};
    std::array<uint8_t, 256> len{};

    unicode_byte_to_utf8_table() {
        uint32_t next_cpt = 256;
        for (uint32_t ch = 0; ch < 256; ++ch) {
            uint32_t cpt;
            if ((0x21 <= ch && ch <= 0x7E) || (0xA1 <= ch && ch <= 0xAC) || (0xAE <= ch && ch <= 0xFF))
                cpt = ch;
            else
                cpt = next_cpt++;
            len[ch] = static_cast<uint8_t>(utf8_encode(cpt, data[ch].data()));
        }
    }

    std::string_view view(uint8_t byte) const { return { data[byte].data(), len[byte] }; }
};

static const unicode_byte_to_utf8_table& unicode_byte_to_utf8_table_get() {
    static const unicode_byte_to_utf8_table table;
    return table;
}

static std::unordered_map<std::string, uint8_t> unicode_utf8_to_byte_map() {
    std::unordered_map<std::string, uint8_t> map;
    for (int ch = 0x21; ch <= 0x7E; ++ch) {  // u'!' to u'~'
        assert(0 <= ch && ch < 256);
        map[unicode_cpt_to_utf8(ch)] = ch;
    }
    for (int ch = 0xA1; ch <= 0xAC; ++ch) {  // u'¡' to u'¬'
        assert(0 <= ch && ch < 256);
        map[unicode_cpt_to_utf8(ch)] = ch;
    }
    for (int ch = 0xAE; ch <= 0xFF; ++ch) {  // u'®' to u'ÿ'
        assert(0 <= ch && ch < 256);
        map[unicode_cpt_to_utf8(ch)] = ch;
    }
    auto n = 0;
    for (int ch = 0; ch < 256; ++ch) {
        if (map.find(unicode_cpt_to_utf8(ch)) == map.end()) {
            map[unicode_cpt_to_utf8(256 + n)] = ch;
            ++n;
        }
    }
    return map;
}

static inline std::wstring unicode_wstring_from_utf8(const std::string& s) {
#if defined(__clang__)
    // disable C++17 deprecation warning for std::codecvt_utf8
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wdeprecated-declarations"
#elif defined(__GNUC__)
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

    std::wstring_convert<std::codecvt_utf8<wchar_t>> conv;

#if defined(__clang__)
#    pragma clang diagnostic pop
#elif defined(__GNUC__)
#    pragma GCC diagnostic pop
#endif

    return conv.from_bytes(s);
}

static std::string pcre2_error_message(int error_code) {
    PCRE2_UCHAR8 buffer[256];
    int rc = pcre2_get_error_message_8(error_code, buffer, sizeof(buffer));
    if (rc < 0) {
        return std::string("Unknown PCRE2 error");
    }
    return std::string(reinterpret_cast<const char*>(buffer));
}

static const std::map<std::string, int> k_ucat_enum = {
    { "\\p{N}", unicode_cpt_flags::NUMBER },
    { "\\p{L}", unicode_cpt_flags::LETTER },
    { "\\p{P}", unicode_cpt_flags::PUNCTUATION },
    { "\\p{M}", unicode_cpt_flags::ACCENT_MARK },
    { "\\p{S}", unicode_cpt_flags::SYMBOL },
    { "\\p{Lu}", unicode_cpt_flags::LETTER }, // Uppercase letter
    { "\\p{Ll}", unicode_cpt_flags::LETTER }, // Lowercase letter
    { "\\p{Lt}", unicode_cpt_flags::LETTER }, // Titlecase letter
    { "\\p{Lm}", unicode_cpt_flags::LETTER }, // Modifier letter
    { "\\p{Lo}", unicode_cpt_flags::LETTER }, // Other letter
};

static const std::map<int, int> k_ucat_cpt = {
    { unicode_cpt_flags::NUMBER,      0xD1 },
    { unicode_cpt_flags::LETTER,      0xD2 },
    { unicode_cpt_flags::PUNCTUATION, 0xD3 },
    { unicode_cpt_flags::ACCENT_MARK, 0xD4 },
    { unicode_cpt_flags::SYMBOL,      0xD5 },
};

static const std::map<int, std::string> k_ucat_map = {
    { unicode_cpt_flags::NUMBER,      "\x30-\x39" }, // 0-9
    { unicode_cpt_flags::LETTER,      "\x41-\x5A\x61-\x7A" }, // A-Za-z
    { unicode_cpt_flags::PUNCTUATION, "\x21-\x23\x25-\x2A\x2C-\x2F\x3A-\x3B\x3F-\x40\\\x5B-\\\x5D\x5F\\\x7B\\\x7D" }, // !-#%-*,-/:-;?-@\[-\]_\{\}
    { unicode_cpt_flags::ACCENT_MARK, "" }, // no sub-128 codepoints
    { unicode_cpt_flags::SYMBOL,      "\\\x24\\\x2B\x3C-\x3E\x5E\x60\\\x7C" }, // $+<=>^`|
};

struct unicode_regex_splitter::impl {
    std::string regex;
    bool use_collapsed = false;
    std::string regex_expr_collapsed;
    std::wstring wregex;
    pcre2_code_8*  code8  = nullptr;
    pcre2_code_16* code16 = nullptr;

    ~impl() {
        if (code8)  pcre2_code_free_8(code8);
        if (code16) pcre2_code_free_16(code16);
    }
};

static pcre2_code_8* compile_pcre2_8(const std::string& regex_expr) {
    int error_code = 0;
    PCRE2_SIZE error_offset = 0;
    pcre2_code_8* code = pcre2_compile_8(
        reinterpret_cast<PCRE2_SPTR8>(regex_expr.c_str()),
        regex_expr.size(),
        0,
        &error_code,
        &error_offset,
        nullptr);

    if (!code) {
        throw std::runtime_error(std::format(
            "Failed to compile regex (8-bit) at offset {}: {}", error_offset, pcre2_error_message(error_code)));
    }

    return code;
}

static pcre2_code_16* compile_pcre2_16(const std::wstring& regex_expr) {
    int error_code = 0;
    PCRE2_SIZE error_offset = 0;
    pcre2_code_16* code = pcre2_compile_16(
        reinterpret_cast<PCRE2_SPTR16>(regex_expr.c_str()),
        regex_expr.size(),
        0,
        &error_code,
        &error_offset,
        nullptr);

    if (!code) {
        throw std::runtime_error(std::format(
            "Failed to compile regex (16-bit) at offset {}: {}", error_offset, pcre2_error_message(error_code)));
    }

    return code;
}

unicode_regex_splitter::unicode_regex_splitter(const std::string& regex) : impl_(new impl()) {
    impl_->regex = regex;

    for (const auto& ucat : k_ucat_enum) {
        if (regex.find(ucat.first) != std::string::npos) {
            impl_->use_collapsed = true;
            break;
        }
    }

    if (impl_->use_collapsed) {
        // sanity-check that the original regex does not contain any non-ASCII characters
        const auto cpts_regex = unicode_cpts_from_utf8(regex);
        for (auto cpt : cpts_regex) {
            if (cpt >= 128) {
                throw std::runtime_error("Regex includes both unicode categories and non-ASCII characters - not supported");
            }
        }

        // generate a collapsed representation of the regex
        bool inside = false;
        for (size_t i = 0; i < regex.size(); ++i) {
            if (regex[i] == '[' && (i == 0 || regex[i - 1] != '\\')) {
                impl_->regex_expr_collapsed += '[';
                inside = true;
                continue;
            }

            if (inside && regex[i] == ']' && regex[i - 1] != '\\') {
                impl_->regex_expr_collapsed += ']';
                inside = false;
                continue;
            }

            // Match \p{...} Unicode properties of varying lengths
            if (regex[i + 0] == '\\' && i + 3 < regex.size() &&
                regex[i + 1] == 'p' &&
                regex[i + 2] == '{') {
                // Find the closing brace
                size_t closing_brace = regex.find('}', i + 3);
                if (closing_brace != std::string::npos && closing_brace <= i + 10) { // reasonable limit
                    const std::string pat = regex.substr(i, closing_brace - i + 1);
                    if (k_ucat_enum.find(pat) != k_ucat_enum.end()) {
                        if (!inside) {
                            impl_->regex_expr_collapsed += '[';
                        }
                        impl_->regex_expr_collapsed += k_ucat_cpt.at(k_ucat_enum.at(pat));
                        impl_->regex_expr_collapsed += k_ucat_map.at(k_ucat_enum.at(pat));
                        if (!inside) {
                            impl_->regex_expr_collapsed += ']';
                        }
                        i = closing_brace;
                        continue;
                    }
                }
            }

            impl_->regex_expr_collapsed += regex[i];
        }

        impl_->code8 = compile_pcre2_8(impl_->regex_expr_collapsed);
    }
    else {
        impl_->wregex = unicode_wstring_from_utf8(regex);
        impl_->code16 = compile_pcre2_16(impl_->wregex);
    }
}

unicode_regex_splitter::~unicode_regex_splitter() = default;

// use PCRE2 16-bit to split the text
static std::vector<size_t> unicode_regex_split_pcre2_16(pcre2_code_16* code, const std::wstring& wtext, const std::vector<size_t>& offsets) {
    std::unique_ptr<pcre2_match_data_16, decltype(&pcre2_match_data_free_16)> match_data(
        pcre2_match_data_create_from_pattern_16(code, nullptr),
        pcre2_match_data_free_16);

    if (!match_data) {
        throw std::runtime_error("Failed to allocate PCRE2 match data (16-bit)");
    }

    std::vector<size_t> bpe_offsets; // store the offset of each word
    bpe_offsets.reserve(offsets.size()); // Reserve memory for the approximate size

    const PCRE2_UCHAR16* subject_base = nullptr;
    std::vector<PCRE2_UCHAR16> wtext16_storage;
    if (sizeof(wchar_t) == 2) {
        subject_base = reinterpret_cast<const PCRE2_UCHAR16*>(wtext.data());
    } else {
        wtext16_storage.resize(wtext.size());
        for (size_t i = 0; i < wtext.size(); ++i) {
            wtext16_storage[i] = static_cast<PCRE2_UCHAR16>(wtext[i]);
        }
        subject_base = wtext16_storage.data();
    }

    size_t start = 0;
    for (auto offset : offsets) {
        const PCRE2_UCHAR16* subject = subject_base + start;
        const PCRE2_SIZE length = static_cast<PCRE2_SIZE>(offset);

        PCRE2_SIZE start_offset = 0;
        int64_t start_idx = 0;

        while (start_offset <= length) {
            int rc = pcre2_match_16(
                code,
                subject,
                length,
                start_offset,
                0,
                match_data.get(),
                nullptr);

            if (rc == PCRE2_ERROR_NOMATCH) {
                break;
            }
            if (rc < 0) {
                throw std::runtime_error("PCRE2 match error (16-bit): " + pcre2_error_message(rc));
            }

            PCRE2_SIZE* ovector = pcre2_get_ovector_pointer_16(match_data.get());
            const PCRE2_SIZE match_start = ovector[0];
            const PCRE2_SIZE match_end = ovector[1];

            if (match_start > static_cast<PCRE2_SIZE>(start_idx)) {
                bpe_offsets.emplace_back(match_start - start_idx);
            }
            bpe_offsets.emplace_back(match_end - match_start);
            start_idx = static_cast<int64_t>(match_end);

            if (match_end <= start_offset) {
                start_offset = match_end + 1;
            } else {
                start_offset = match_end;
            }

            if (start_offset > length) {
                break;
            }
        }

        if (start_idx < static_cast<int64_t>(offset)) {
            bpe_offsets.emplace_back(offset - start_idx);
        }
        start += offset;
    }

    return bpe_offsets;
}

// use PCRE2 8-bit to split the text
static std::vector<size_t> unicode_regex_split_pcre2_8(pcre2_code_8* code, const std::string& text, const std::vector<size_t>& offsets) {
    std::unique_ptr<pcre2_match_data_8, decltype(&pcre2_match_data_free_8)> match_data(
        pcre2_match_data_create_from_pattern_8(code, nullptr),
        pcre2_match_data_free_8);

    if (!match_data) {
        throw std::runtime_error("Failed to allocate PCRE2 match data (8-bit)");
    }

    std::vector<size_t> bpe_offsets; // store the offset of each word
    bpe_offsets.reserve(offsets.size()); // Reserve memory for the approximate size

    size_t start = 0;
    for (auto offset : offsets) {
        const PCRE2_UCHAR8* subject = reinterpret_cast<const PCRE2_UCHAR8*>(text.data() + start);
        const PCRE2_SIZE length = static_cast<PCRE2_SIZE>(offset);

        PCRE2_SIZE start_offset = 0;
        int64_t start_idx = 0;

        while (start_offset <= length) {
            int rc = pcre2_match_8(
                code,
                subject,
                length,
                start_offset,
                0,
                match_data.get(),
                nullptr);

            if (rc == PCRE2_ERROR_NOMATCH) {
                break;
            }
            if (rc < 0) {
                throw std::runtime_error("PCRE2 match error (8-bit): " + pcre2_error_message(rc));
            }

            PCRE2_SIZE* ovector = pcre2_get_ovector_pointer_8(match_data.get());
            const PCRE2_SIZE match_start = ovector[0];
            const PCRE2_SIZE match_end = ovector[1];

            if (match_start > static_cast<PCRE2_SIZE>(start_idx)) {
                bpe_offsets.emplace_back(match_start - start_idx);
            }
            bpe_offsets.emplace_back(match_end - match_start);
            start_idx = static_cast<int64_t>(match_end);

            if (match_end <= start_offset) {
                start_offset = match_end + 1;
            } else {
                start_offset = match_end;
            }

            if (start_offset > length) {
                break;
            }
        }

        if (start_idx < static_cast<int64_t>(offset)) {
            bpe_offsets.emplace_back(offset - start_idx);
        }
        start += offset;
    }

    return bpe_offsets;
}

//
// interface
//

std::string unicode_cpt_to_utf8(uint32_t cpt) {
    char buf[4];
    return std::string(buf, utf8_encode(cpt, buf));
}

std::vector<uint32_t> unicode_cpts_from_utf8(const std::string_view& utf8) {
    std::vector<uint32_t> result;
    result.reserve(utf8.size());
    size_t offset = 0;
    while (offset < utf8.size()) {
        try {
            result.push_back(unicode_cpt_from_utf8(utf8, offset));
        }
        catch (const std::invalid_argument& /*ex*/) {
            // Silently ignore invalid UTF-8 input to avoid leaking the exception beyond llama_tokenize
            ++offset;
            result.emplace_back(0xFFFD); // replacement character
        }
    }
    return result;
}

unicode_cpt_flags unicode_cpt_flags_from_cpt(const uint32_t cpt) {
    static const unicode_cpt_flags undef(unicode_cpt_flags::UNDEFINED);
    static const auto cpt_flags = unicode_cpt_flags_array();
    return cpt < cpt_flags.size() ? cpt_flags[cpt] : undef;
}

std::string unicode_byte_to_utf8(uint8_t byte) {
    return std::string(unicode_byte_to_utf8_table_get().view(byte));
}

uint8_t unicode_utf8_to_byte(const std::string& utf8) {
    static std::unordered_map<std::string, uint8_t> map = unicode_utf8_to_byte_map();
    return map.at(utf8);
}

std::vector<std::string> unicode_regex_splitter::split(std::string_view text) const {
    const auto cpts = unicode_cpts_from_utf8(text);

    // generate a "collapsed" representation of the text, where all codepoints are replaced by a single byte
    // ref: https://github.com/ggml-org/llama.cpp/pull/6920#issuecomment-2081479935
    std::string text_collapsed;
    if (impl_->use_collapsed) {
        // collapse all unicode categories
        text_collapsed.resize(cpts.size());

        for (size_t i = 0; i < cpts.size(); ++i) {
            // keep single-byte codepoints as is
            if (cpts[i] < 128) {
                text_collapsed[i] = cpts[i];
                continue;
            }

            const auto flags = unicode_cpt_flags_from_cpt(cpts[i]);

            if (flags.is_whitespace) {
                //NOTE: std::regex \s does not mach 0x85, Rust and Python regex does.
                //text_collapsed[i] = (char) 0x85;  // <Next Line> as whitespace fallback
                text_collapsed[i] = (char)0x0B;    // <vertical tab> as whitespace fallback
            }
            else if (k_ucat_cpt.find(flags.category_flag()) != k_ucat_cpt.end()) {
                text_collapsed[i] = k_ucat_cpt.at(flags.category_flag());
            }
            else {
                text_collapsed[i] = (char)0xD0; // fallback
            }
        }
    }

    std::vector<size_t> bpe_offsets = { cpts.size() };

    // fallback to general-purpose PCRE2 (8-bit / 16-bit)
    try {
        if (impl_->use_collapsed) {
            bpe_offsets = unicode_regex_split_pcre2_8(impl_->code8, text_collapsed, bpe_offsets);
        }
        else {
            // no unicode category used, we can use PCRE2-16 directly
            // PCRE2 \s does not match non-ASCII whitespaces without UCP, using 0x0B as fallback
            std::wstring wtext(cpts.begin(), cpts.end());
            for (size_t i = 0; i < wtext.size(); ++i) {
                if (wtext[i] > 0x7F && unicode_cpt_flags_from_cpt(wtext[i]).is_whitespace) {
                    wtext[i] = 0x0B;
                }
            }

            bpe_offsets = unicode_regex_split_pcre2_16(impl_->code16, wtext, bpe_offsets);
        }
    }
    catch (const std::exception& e) {
        fprintf(stderr, "Failed to process regex: '%s'\n", impl_->regex.c_str());
        fprintf(stderr, "Regex error: %s\n", e.what());
        throw std::runtime_error("Failed to process regex");
    }

    // Encode the codepoints back to UTF-8 once and remember each codepoint's
    // byte offset, so the split words can be sliced without re-encoding.
    std::string text_utf;
    text_utf.reserve(text.size());
    std::vector<size_t> byte_offsets(cpts.size() + 1);
    for (size_t i = 0; i < cpts.size(); ++i) {
        byte_offsets[i] = text_utf.size();
        char buf[4];
        text_utf.append(buf, utf8_encode(cpts[i], buf));
    }
    byte_offsets[cpts.size()] = text_utf.size();

    const auto& byte_table = unicode_byte_to_utf8_table_get();

    std::vector<std::string> bpe_encoded_words;
    bpe_encoded_words.reserve(bpe_offsets.size());

    size_t cp_start = 0;
    for (const size_t offset : bpe_offsets) {
        assert(cp_start + offset <= cpts.size());
        const size_t byte_start = byte_offsets[cp_start];
        const size_t byte_end   = byte_offsets[cp_start + offset];

        size_t encoded_len = 0;
        for (size_t i = byte_start; i < byte_end; ++i) {
            encoded_len += byte_table.len[static_cast<uint8_t>(text_utf[i])];
        }

        std::string encoded_token;
        encoded_token.reserve(encoded_len);
        for (size_t i = byte_start; i < byte_end; ++i) {
            encoded_token.append(byte_table.view(static_cast<uint8_t>(text_utf[i])));
        }

        bpe_encoded_words.emplace_back(std::move(encoded_token));
        cp_start += offset;
    }

    return bpe_encoded_words;
}
