#pragma once
#include "cosyvoice.h"
#include "cosyvoice-lowlevel.h"
#include "cosyvoice-interface.h"
#include "cosyvoice-loader.h"

#include <string_view>
#include <string>
#include <vector>
#include <unordered_map>
#include <forward_list>

enum token_type {
    TOKEN_TYPE_NORMAL = 1,
    TOKEN_TYPE_CONTROL = 3,
    TOKEN_TYPE_UNUSED = 5
};

struct token_data {
    std::string text;
    token_type type;
};

struct llm_tokenizer {
    llm_tokenizer() = default;
    virtual ~llm_tokenizer() = default;
};

struct pair_hash {
    size_t operator()(const std::pair<std::string, std::string>& p) const {
        return std::hash<std::string>{}(p.first) ^
            (std::hash<std::string>{}(p.second) << 1);
    }
};

enum class fragment_buffer_kind {
    token,
    raw_text
};

struct fragment_buffer_variant {
    fragment_buffer_variant(int _token)
        :
        type(fragment_buffer_kind::token),
        token(_token),
        raw_text(),
        offset(0),
        length(0) {
    }

    fragment_buffer_variant(const std::string_view& _raw_text, int64_t _offset, int64_t _length)
        :
        type(fragment_buffer_kind::raw_text),
        token((int)-1),
        raw_text(_raw_text),
        offset(_offset),
        length(_length) {
        GGML_ASSERT(_offset >= 0);
        GGML_ASSERT(_length >= 1);
        GGML_ASSERT(offset + length <= raw_text.length());
    }

    const fragment_buffer_kind type;
    const int token;
    const std::string_view raw_text;
    const uint64_t offset;
    const uint64_t length;
};

class cosyvoice_vocab {
public:
    void load(gguf_metadata_loader& loader);

    void tokenize(const std::string_view& raw_text, std::vector<int>& output, bool parse_special = false) const;

    int text_to_token(const std::string& text) const;
    int find_bpe_rank(const std::string& token_left, const std::string& token_right) const;
    void partition_special_tokens(std::forward_list<fragment_buffer_variant>& buffer, bool parse_special) const;

    std::unordered_map<std::string, int> token_to_id;
    std::vector<token_data>              id_to_token;
    std::vector<int>                     cache_special_tokens;
    std::unordered_map<std::pair<std::string, std::string>, int, pair_hash> bpe_ranks;
    std::unique_ptr<llm_tokenizer>       tokenizer;
};

struct cosyvoice_tokenizer : virtual cosyvoice_tokenizer_context, cosyvoice_vocab
{
    uint32_t tokenize(const char* text, uint32_t text_len, cosyvoice_tokenization_result_t result, bool parse_special);
};
