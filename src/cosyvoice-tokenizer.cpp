#include "cosyvoice-internal.h"
#include "cosyvoice-loader.h"
#include "cosyvoice-tokenizer.h"
#include "unicode.h"

#include <algorithm>
#include <queue>

struct llm_symbol {
    using index = int;
    index prev;
    index next;
    const char* text;
    size_t n;
};

static_assert(std::is_trivially_copyable<llm_symbol>::value, "llm_symbol is not trivially copyable");

template<typename T, typename Container = std::vector<T>, typename Compare = std::less<typename Container::value_type>>
class llama_priority_queue : public std::priority_queue<T, Container, Compare> {
public:
    using std::priority_queue<T, Container, Compare>::priority_queue;

    T pop_move() {
        T item = std::move(this->c.front());
        std::pop_heap(this->c.begin(), this->c.end(), this->comp);
        this->c.pop_back();
        return item;
    }

    void pop() = delete;
};

struct llm_bigram_bpe {
    struct comparator {
        bool operator()(const llm_bigram_bpe& l, const llm_bigram_bpe& r) const {
            return l.rank > r.rank || (l.rank == r.rank && l.left > r.left);
        }
    };

    using queue_storage = std::vector<llm_bigram_bpe>;
    using queue = llama_priority_queue<llm_bigram_bpe, queue_storage, comparator>;
    llm_symbol::index left;
    llm_symbol::index right;
    std::string text;
    int rank;
    size_t size;
};

struct llm_tokenizer_bpe : llm_tokenizer {
    explicit llm_tokenizer_bpe(gguf_metadata_loader& loader)
        : regex(loader.get_string("tokenizer.pre_tokenizer.regex")) {}

    std::string regex;
};

struct llm_tokenizer_bpe_session {
    llm_tokenizer_bpe_session(const cosyvoice_vocab& vocab, const llm_tokenizer_bpe& tokenizer) : vocab(vocab), tokenizer(tokenizer) {}

    void tokenize(const std::string_view& text, std::vector<int>& output) {
        int final_prev_index = -1;
        const auto word_collection = unicode_regex_split(text, tokenizer.regex);

        symbols_final.clear();

        for (const auto& word : word_collection) {
            work_queue = llm_bigram_bpe::queue();
            symbols.clear();

            int index = 0;
            size_t offset = 0;

            while (offset < word.size()) {
                llm_symbol sym;
                size_t char_len = std::min(word.size() - offset, (size_t)unicode_len_utf8(word[offset]));
                sym.text = word.c_str() + offset;
                sym.n = char_len;
                offset += sym.n;
                sym.prev = index - 1;
                sym.next = offset == word.size() ? -1 : index + 1;
                index++;
                symbols.emplace_back(sym);
            }
            for (int i = 1; i < (int)symbols.size(); ++i) {
                add_new_bigram(i - 1, i);
            }

            // build token(s)
            while (!work_queue.empty()) {
                auto bigram = work_queue.pop_move();

                auto& left_symbol = symbols[bigram.left];
                auto& right_symbol = symbols[bigram.right];

                if (left_symbol.n == 0 || right_symbol.n == 0) {
                    continue;
                }
                std::string left_token = std::string(left_symbol.text, left_symbol.n);
                std::string right_token = std::string(right_symbol.text, right_symbol.n);
                if (left_token + right_token != bigram.text) {
                    continue;  // Skip this bigram if it's outdated
                }

                // merge the right sym into the left one
                left_symbol.n += right_symbol.n;
                right_symbol.n = 0;

                // remove the right sym from the chain
                left_symbol.next = right_symbol.next;
                if (right_symbol.next >= 0) {
                    symbols[right_symbol.next].prev = bigram.left;
                }

                add_new_bigram(left_symbol.prev, bigram.left);  // left side of current symbol
                add_new_bigram(bigram.left, left_symbol.next);  // right side of current symbol
            }

            // add the finished tokens to the final list keeping correct order for next and prev
            for (auto& sym : symbols) {
                if (sym.n > 0) {
                    sym.prev = final_prev_index;
                    sym.next = -1;
                    if (final_prev_index != -1) {
                        symbols_final[final_prev_index].next = static_cast<int>(symbols_final.size());
                    }
                    symbols_final.emplace_back(sym);
                    final_prev_index = static_cast<int>(symbols_final.size()) - 1;
                }
            }
        }

        symbols = symbols_final;

        if (!symbols.empty()) {
            for (int i = 0; i != -1; i = symbols[i].next) {
                auto& symbol = symbols[i];
                if (symbol.n == 0) {
                    continue;
                }

                const std::string str = std::string(symbol.text, symbol.n);
                const auto token = vocab.text_to_token(str);

                if (token == -1) {
                    for (auto j = str.begin(); j != str.end(); ++j) {
                        std::string byte_str(1, *j);
                        auto token_multibyte = vocab.text_to_token(byte_str);
                        if (token_multibyte != -1) {
                            output.push_back(token_multibyte);
                        }
                    }
                }
                else {
                    output.push_back(token);
                }
            }
        }
    }

private:
    void add_new_bigram(int left, int right) {
        if (left == -1 || right == -1) {
            return;
        }
        std::string left_token = std::string(symbols[left].text, symbols[left].n);
        std::string right_token = std::string(symbols[right].text, symbols[right].n);

        int rank_found = -1;

        rank_found = vocab.find_bpe_rank(left_token, right_token);

        if (rank_found < 0) {
            return;
        }

        llm_bigram_bpe bigram;

        bigram.left = left;
        bigram.right = right;
        bigram.text = left_token + right_token;
        bigram.size = left_token.size() + right_token.size();
        bigram.rank = rank_found;

        work_queue.push(bigram);
    }

    const cosyvoice_vocab&   vocab;
    const llm_tokenizer_bpe& tokenizer;

    std::vector<llm_symbol>  symbols;
    std::vector<llm_symbol>  symbols_final;
    llm_bigram_bpe::queue    work_queue;
};

void cosyvoice_vocab::load(gguf_metadata_loader& loader) {
    GGML_ASSERT(loader.get_string("tokenizer.model.type") == "BPE");

    const auto token_idx = gguf_find_key(loader, "tokenizer.vocab.tokens");

    const int* toktypes = nullptr;
    const auto toktype_idx = gguf_find_key(loader, "tokenizer.vocab.token_types");
    toktypes = (const int*)gguf_get_arr_data(loader, toktype_idx);

    auto n_tokens = gguf_get_arr_n(loader, token_idx);
    id_to_token.resize(n_tokens);

    for (int i = 0; i != n_tokens; i++) {
        std::string word = gguf_get_arr_str(loader, token_idx, i);
        if (word.empty())
            word = std::format("[EMPTY_{}]", i);

        token_to_id[word] = i;

        auto& token_data = id_to_token[i];
        token_data.text = std::move(word);
        token_data.type = static_cast<token_type>(toktypes[i]);

        if (token_data.type == TOKEN_TYPE_CONTROL)
            cache_special_tokens.push_back(i);
    }
    GGML_ASSERT(id_to_token.size() == token_to_id.size());

    const auto merges_keyidx = gguf_find_key(loader, "tokenizer.model.merges");
    const auto n_merges = gguf_get_arr_n(loader, merges_keyidx);
    for (int i = 0; i < n_merges; i++) {
        const std::string_view word = gguf_get_arr_str(loader, merges_keyidx, i);

        std::string_view first;
        std::string_view second;

        const size_t pos = word.find(' ', 1);

        if (pos != std::string_view::npos) {
            first = word.substr(0, pos);
            second = word.substr(pos + 1);
        }

        bpe_ranks.emplace(std::make_pair(first, second), i);
    }

    tokenizer = std::make_unique<llm_tokenizer_bpe>(loader);

    std::sort(cache_special_tokens.begin(), cache_special_tokens.end(),
        [&](const int a, const int b) {
            return id_to_token[a].text.size() > id_to_token[b].text.size();
        }
    );
}

void cosyvoice_vocab::partition_special_tokens(std::forward_list<fragment_buffer_variant>& buffer, bool parse_special) const {
    // Scan each cached special token.
    for (const int special_id : cache_special_tokens) {
        const auto& data = id_to_token.at(special_id);
        const auto& text = data.text;

        if (!parse_special && data.type == TOKEN_TYPE_CONTROL) continue;

        // Split every raw-text fragment around matches for this special token.
        std::forward_list<fragment_buffer_variant>::iterator it = buffer.begin();
        while (it != buffer.end()) {
            auto& fragment = (*it);

            if (fragment.type == fragment_buffer_kind::raw_text) {
                const auto raw_text = fragment.raw_text;

                auto raw_text_base_offset = fragment.offset;
                auto raw_text_base_length = fragment.length;

                // Keep partitioning the current fragment until no more matches remain.
                while (true) {
                    // The search starts at the fragment offset, but the returned index is still
                    // relative to the original string view.
                    auto match = std::string_view(raw_text.data(), raw_text_base_offset + raw_text_base_length).find(text, raw_text_base_offset);

                    // No more matches in this fragment for the current special token.
                    if (match == std::string_view::npos) break;

                    auto source = std::distance(buffer.begin(), it);

                    // Emit the text to the left of the matched token first.
                    if (match > raw_text_base_offset) {
                        const int64_t left_remainder_offset = raw_text_base_offset;
                        int64_t left_remainder_length = match - raw_text_base_offset;

                        if (left_remainder_length > 0) {
                            buffer.emplace_after(it, raw_text, left_remainder_offset, left_remainder_length);
                            it++;
                        }
                    }

                    // Emit the matched special token.
                    buffer.emplace_after(it, special_id);
                    it++;

                    // Keep the remaining text on the right for the next iteration.
                    if (match + text.length() < raw_text_base_offset + raw_text_base_length) {
                        int64_t right_remainder_offset = match + text.length();
                        int64_t right_remainder_length = raw_text_base_length - ((match - raw_text_base_offset) + text.length());

                        if (right_remainder_length > 0) {
                            buffer.emplace_after(it, raw_text, right_remainder_offset, right_remainder_length);
                            it++;
                        }

                        if (source == 0) {
                            buffer.erase_after(buffer.before_begin());
                        }
                        else {
                            buffer.erase_after(std::next(buffer.begin(), (source - 1)));
                        }

                        raw_text_base_offset = right_remainder_offset;
                        raw_text_base_length = right_remainder_length;
                    }
                    else {
                        if (source == 0) {
                            buffer.erase_after(buffer.before_begin());
                        }
                        else {
                            buffer.erase_after(std::next(buffer.begin(), (source - 1)));
                        }
                        break;
                    }
                }
            }
            it++;
        }
    }
}

void cosyvoice_vocab::tokenize(
    const std::string_view& raw_text,
    std::vector<int>& output,
    bool parse_special) const {

    output.clear();
    std::forward_list<fragment_buffer_variant> fragment_buffer;

    if (!raw_text.empty()) {
        fragment_buffer.emplace_front(raw_text, 0, raw_text.length());
        partition_special_tokens(fragment_buffer, parse_special);
    }

    llm_tokenizer_bpe_session session(*this, *static_cast<llm_tokenizer_bpe*>(tokenizer.get()));

    for (const auto& fragment : fragment_buffer) {
        if (fragment.type == fragment_buffer_kind::raw_text) {
            auto text = fragment.raw_text.substr(fragment.offset, fragment.length);

            session.tokenize(text, output);
        }
        else output.push_back(fragment.token);
    }
}

int cosyvoice_vocab::text_to_token(const std::string& text) const {
    auto it = token_to_id.find(text);
    if (it != token_to_id.end()) {
        return (*it).second;
    }
    return -1;
}

int cosyvoice_vocab::find_bpe_rank(const std::string& token_left, const std::string& token_right) const {
    GGML_ASSERT(token_left.find(' ') == std::string::npos);
    GGML_ASSERT(token_left.find('\n') == std::string::npos);
    GGML_ASSERT(token_right.find(' ') == std::string::npos);
    GGML_ASSERT(token_right.find('\n') == std::string::npos);

    auto it = bpe_ranks.find(std::make_pair(token_left, token_right));
    if (it == bpe_ranks.end()) {
        return -1;
    }

    return it->second;
}

uint32_t cosyvoice_tokenizer::tokenize(const char* text, uint32_t text_len, cosyvoice_tokenization_result_t result, bool parse_special)
{
    cosyvoice_vocab::tokenize(std::string_view(text, text_len), static_cast<cosyvoice_tokenization_result_impl*>(result)->tokens, parse_special);
    return static_cast<uint32_t>(static_cast<cosyvoice_tokenization_result_impl*>(result)->tokens.size());
}
