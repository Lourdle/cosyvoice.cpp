#include "cosyvoice-internal.h"
#include "cosyvoice-model.h"
#include "cosyvoice-loader.h"

#include <cerrno>
#include <cstring>

struct crc32_context
{
    crc32_context()
    {
        for (uint32_t i = 0, crc = 0; i < 256; i++)
        {
            crc = i;
            for (char j = 0; j < 8; j++)
            {
                if (crc & 1)
                    crc = (crc >> 1) ^ 0xEDB88320;
                else
                    crc >>= 1;
            }
            table[i] = crc;
        }
    }

    uint32_t calculate(const void* data, size_t length, uint32_t value = 0) const
    {
        for (size_t i = 0; i != length; ++i)
            value = ((value >> 8) & 0xFFFFFF) ^ table[(value ^ reinterpret_cast<const uint8_t*>(data)[i]) & 0xFF];
        return ~value;
    }

    uint32_t table[256];
};

void cosyvoice_prompt_speech::calculate_crc32()
{
    crc32_context ctx;

    crc32 = ctx.calculate(text.first.get(), sizeof(char) * text.second);
    crc32 = ctx.calculate(&text.second, sizeof(text.second), crc32);

    crc32 = ctx.calculate(feat.data, sizeof(float) * feat.shape[0] * feat.shape[1], crc32);
    crc32 = ctx.calculate(feat.shape, sizeof(matrix::shape), crc32);

    crc32 = ctx.calculate(embedding.data, sizeof(float) * embedding.shape[0] * embedding.shape[1], crc32);
    crc32 = ctx.calculate(embedding.shape, sizeof(matrix::shape), crc32);

    crc32 = ctx.calculate(tokens.first.get(), sizeof(int) * tokens.second, crc32);
    crc32 = ctx.calculate(&tokens.second, sizeof(tokens.second), crc32);
}

void cosyvoice_prompt::calculate_crc32()
{
    crc32_context ctx;

    auto size = sizeof(int) * prompt_text.size();
    prompt_crc32 = ctx.calculate(prompt_text.data(), size);
    prompt_crc32 = ctx.calculate(&size, sizeof(size), prompt_crc32);
}

cosyvoice_prompt_speech_t cosyvoice_prompt_speech_load(const void* data, size_t size)
{
    gguf_parser parser;
    gguf_loader gguf(parser, data, size);
    if (!gguf) return nullptr;

    auto feat = ggml_get_tensor(gguf, "feat");
    auto embedding = ggml_get_tensor(gguf, "embedding");
    auto tokens = ggml_get_tensor(gguf, "tokens");
    auto text = ggml_get_tensor(gguf, "text");

    if (!feat || !embedding || !tokens || !text
        || !ggml_is_matrix(feat) || !ggml_is_matrix(embedding) || !ggml_is_vector(tokens) || !ggml_is_vector(text)
        || feat->type != GGML_TYPE_F32 || embedding->type != GGML_TYPE_F32 || tokens->type != GGML_TYPE_I32 || text->type != GGML_TYPE_I8)
    {
        errno = EILSEQ;
        return nullptr;
    }

    auto prompt = new cosyvoice_prompt_speech;
    prompt->feat = matrix{ uint32_t(feat->ne[1]), uint32_t(feat->ne[0]) };
    prompt->embedding = matrix{ uint32_t(embedding->ne[1]), uint32_t(embedding->ne[0]) };
    prompt->tokens = std::make_pair(std::make_unique<int[]>(tokens->ne[0]), static_cast<uint32_t>(tokens->ne[0]));
    prompt->text = std::make_pair(std::make_unique<char[]>(text->ne[0]), static_cast<uint32_t>(text->ne[0]));

    memcpy(prompt->feat.data, feat->data, feat->nb[2]);
    memcpy(prompt->embedding.data, embedding->data, embedding->nb[2]);
    memcpy(prompt->tokens.first.get(), tokens->data, tokens->nb[1]);
    memcpy(prompt->text.first.get(), text->data, text->nb[1]);
    prompt->calculate_crc32();

    uint32_t crc32;
    if (gguf.get_metadata("crc32", crc32)
        && crc32 != prompt->crc32)
    {
        cosyvoice_prompt_speech_free(prompt);
        errno = EILSEQ;
        return nullptr;
    }

    return prompt;
}

cosyvoice_prompt_speech_t cosyvoice_prompt_speech_load_from_file(const char* filename)
{
    file_mmap file(filename);
    return file ? cosyvoice_prompt_speech_load(file.data(), file.size()) : nullptr;
}

cosyvoice_prompt_t cosyvoice_prompt_init_from_prompt_speech(cosyvoice_context_t ctx, cosyvoice_prompt_speech_t prompt_speech)
{
    auto prompt = new cosyvoice_prompt;
    cosyvoice_tokenization_result_impl tokenized_text;
    constexpr const char endofprompt[] = "<|endofprompt|>";
    ctx->tokenize(ctx->get_instruction_prefix(), &tokenized_text, true);
    prompt->prompt_text.swap(tokenized_text.tokens);
    ctx->tokenize(endofprompt, &tokenized_text, true);
    prompt->prompt_text.insert(prompt->prompt_text.end(), tokenized_text.tokens.begin(), tokenized_text.tokens.end());

    ctx->tokenize(prompt_speech->text.first.get(), prompt_speech->text.second, &tokenized_text, false);
    prompt->orig_prompt_text = std::make_pair(std::make_shared<int[]>(tokenized_text.tokens.size()), tokenized_text.get_n_tokens());
    memcpy(prompt->orig_prompt_text.first.get(), tokenized_text.get_tokens(), sizeof(int) * tokenized_text.get_n_tokens());
    prompt->prompt_text.insert(prompt->prompt_text.end(), tokenized_text.tokens.begin(), tokenized_text.tokens.end());
    prompt->calculate_crc32();

    prompt->llm_prompt_speech_tokens = prompt_speech->tokens;
    prompt->flow_prompt_speech_tokens = prompt_speech->tokens;
    prompt->prompt_speech_feat = prompt_speech->feat;
    prompt->llm_embedding = prompt_speech->embedding;
    prompt->flow_embedding = prompt_speech->embedding;
    return prompt;
}

uint32_t cosyvoice_prompt_speech_get_crc32(cosyvoice_prompt_speech_t prompt)
{
    return prompt->crc32;
}

uint32_t cosyvoice_prompt_get_crc32(cosyvoice_prompt_t prompt)
{
    return prompt->prompt_crc32;
}

void cosyvoice_prompt_speech_free(cosyvoice_prompt_speech_t prompt)
{
    delete prompt;
}

void cosyvoice_prompt_free(cosyvoice_prompt_t prompt)
{
    delete prompt;
}

cosyvoice_prompt_t cosyvoice_prompt_set_ext(cosyvoice_context_t ctx, cosyvoice_prompt_t prompt, cosyvoice_inference_mode_t mode, const int* instruction, uint32_t instruction_length, bool inplace)
{
    if (inplace)
    {
        ctx->set_prompt(prompt, mode, instruction, instruction_length);
        return prompt;
    }
    else
    {
        auto result = std::make_unique<cosyvoice_prompt>(*prompt);
        ctx->set_prompt(result.get(), mode, instruction, instruction_length);
        return result.release();
    }
}

cosyvoice_prompt_t cosyvoice_prompt_set(cosyvoice_context_t ctx, cosyvoice_prompt_t prompt, cosyvoice_inference_mode_t mode, const char* instruction, uint32_t instruction_length, bool inplace)
{
    if (instruction && instruction_length == 0xFFFFFFFFU)
        instruction_length = static_cast<uint32_t>(strlen(instruction));

    if (instruction && instruction_length != 0)
    {
        cosyvoice_tokenization_result_impl tokenized_text;
        ctx->tokenize("<|endofprompt|>", &tokenized_text, true);
        GGML_ASSERT(tokenized_text.get_n_tokens() == 1);
        int endofprompt_token = tokenized_text.tokens.back();
        ctx->tokenize(instruction, instruction_length, &tokenized_text, false);
        tokenized_text.tokens.push_back(endofprompt_token);
        return cosyvoice_prompt_set_ext(ctx, prompt, mode, tokenized_text.get_tokens(), tokenized_text.get_n_tokens(), inplace);
    }
    return cosyvoice_prompt_set_ext(ctx, prompt, mode, nullptr, 0, inplace);
}

void cosyvoice_model::set_prompt(cosyvoice_prompt_t prompt, cosyvoice_inference_mode_t mode, const int* instruction, uint32_t instruction_length)
{
    prompt->prompt_text.resize(instruction_length + prompt->orig_prompt_text.second);
    memcpy(prompt->prompt_text.data(), instruction, sizeof(int) * instruction_length);
    memcpy(prompt->prompt_text.data() + instruction_length, prompt->orig_prompt_text.first.get(), sizeof(int) * prompt->orig_prompt_text.second);

    switch (mode)
    {
    case COSYVOICE_INFERENCE_MODE_ZERO_SHOT:
        prompt->llm_prompt_speech_tokens = prompt->flow_prompt_speech_tokens;
        break;
    case COSYVOICE_INFERENCE_MODE_CROSS_LINGUAL:
    case COSYVOICE_INFERENCE_MODE_INSTRUCT:
        prompt->prompt_text.resize(instruction_length);
        prompt->llm_prompt_speech_tokens.first.reset();
        prompt->llm_prompt_speech_tokens.second = 0;
    case COSYVOICE_INFERENCE_MODE_NULL:
        break;
    default:
        throw std::invalid_argument("Invalid inference mode");
    }

    prompt->calculate_crc32();
}

bool cosyvoice_prompt_speech_save_to_file(cosyvoice_prompt_speech_t prompt, const char* filename)
{
    gguf_context_ptr gguf(gguf_init_empty());
    if (!gguf) return false;
    ggml_context_ptr ctx(ggml_init({ .mem_size = 4 * ggml_tensor_overhead(), .mem_buffer = nullptr, .no_alloc = true }));

    auto feat = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, prompt->feat.shape[1], prompt->feat.shape[0]);
    auto embedding = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, prompt->embedding.shape[1], prompt->embedding.shape[0]);
    auto tokens = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, prompt->tokens.second);
    auto text = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I8, prompt->text.second);

    feat->data = prompt->feat.data;
    embedding->data = prompt->embedding.data;
    tokens->data = prompt->tokens.first.get();
    text->data = prompt->text.first.get();

    ggml_set_name(feat, "feat");
    ggml_set_name(embedding, "embedding");
    ggml_set_name(tokens, "tokens");
    ggml_set_name(text, "text");

    gguf_add_tensor(gguf.get(), feat);
    gguf_add_tensor(gguf.get(), embedding);
    gguf_add_tensor(gguf.get(), tokens);
    gguf_add_tensor(gguf.get(), text);
    gguf_set_val_u32(gguf.get(), "crc32", prompt->crc32);

    return gguf_write_to_file(gguf.get(), filename, false);
}
