#pragma once

#include <memory>

struct fft_context;
void delete_fft_context(fft_context* ctx);
struct fft_context_deleter { void operator()(fft_context* ctx) const { delete_fft_context(ctx); } };
using fft_context_ptr = std::unique_ptr<fft_context, fft_context_deleter>;
fft_context_ptr create_fft_context(int nfft);

void fft(const float* signal, float* buffer, fft_context& ctx);
