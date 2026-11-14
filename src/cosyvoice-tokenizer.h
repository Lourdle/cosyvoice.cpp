#pragma once
#include "cosyvoice-internal.h"

#include <string_view>
#include <string>
#include <vector>

class gguf_metadata_loader;

class cosyvoice_vocab : virtual public cosyvoice_object_ref_counter
{
public:
    cosyvoice_vocab();
    cosyvoice_vocab(cosyvoice_vocab&) = default;
    ~cosyvoice_vocab();

    void load(gguf_metadata_loader& loader);
    void tokenize(const std::string_view& raw_text, std::vector<int>& output, bool parse_special = false) const;
    int text_to_token(std::string_view text) const;
    int find_bpe_rank(std::string_view token_left, std::string_view token_right) const;

private:
    struct impl;
    impl* pimpl_;
};

struct cosyvoice_tokenizer : virtual cosyvoice_tokenizer_context, cosyvoice_vocab
{
    uint32_t tokenize(const char* text, uint32_t text_len, cosyvoice_tokenization_result_t result, bool parse_special);
};
