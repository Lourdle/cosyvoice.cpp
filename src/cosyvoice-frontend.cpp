#include "cosyvoice-internal.h"
#include "cosyvoice-frontend.h"
#include "cosyvoice-audio.h"
#include "fft.h"
#include "common.h"

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#elif defined(__aarch64__) || defined(_M_ARM64)
#define SIMDE_ENABLE_NATIVE_ALIASES
#include <simde/x86/avx2.h>
#include <simde/x86/fma.h>
#else
#error "src/cosyvoice-frontend.cpp requires x86_64 or ARM64 SIMD support"
#endif

#include <onnxruntime_cxx_api.h>

#ifdef _WIN32
    #define NOMINMAX
    #include <Windows.h>
#else
    #include <sys/mman.h>
    #include <sys/stat.h>
    #include <unistd.h>
    #include <fcntl.h>
#endif

struct cosyvoice_frontend_context
{
    cosyvoice_frontend_context(const void* speech_tokenizer, size_t speech_tokenizer_length, const void* campplus, size_t campplus_length, const Ort::Env& env, const Ort::SessionOptions& session_options);

    matrix extract_speech_feat(float* speech, uint32_t len);
    tokens_t extract_speech_token(float* signal, uint32_t len);
    matrix extract_spk_embedding(float* speech, uint32_t len);

#ifndef COSYVOICE_NO_AUDIO
    cosyvoice_prompt_speech_t frontend_prompt_speech(float* speech, uint32_t len, uint32_t sample_rate);
    cosyvoice_prompt_speech_t frontend_vc_prompt_speech(float* speech, uint32_t len, uint32_t sample_rate);
#endif

    cosyvoice_prompt_speech_t frontend_prompt_speech(float* speech_24k, float* speech_16k, uint32_t speech_24k_len, uint32_t speech_16k_len);
    cosyvoice_prompt_speech_t frontend_vc_prompt_speech(float* speech_24k, float* speech_16k, uint32_t speech_24k_len, uint32_t speech_16k_len);

    matrix mel_basis, hann_window, mel_basis2, hann_window2, mel_basis_spk, povey_window;
    fft_context_ptr fft_ctx, fft_ctx2, fft_ctx_spk;
    Ort::Session speech_tokenizer_session, campplus_session;
    Ort::Env env;
};

cosyvoice_frontend_context_t cosyvoice_frontend_load_from_files(const char* speech_tokenizer, const char* campplus)
{
#ifdef _WIN32
    auto OpenFileByUTF8Name = [](PCSTR pFileName)
    {
        return CreateFileW(utf8_to_wstr(pFileName).c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    };

    struct KernelObjCloser { void operator()(HANDLE hObject) const { CloseHandle(hObject); } };
    struct ViewDataUnmapper { void operator()(PVOID pvData) const { UnmapViewOfFile(pvData); } };

    using KernelObjHdr = std::unique_ptr<VOID, KernelObjCloser>;
    KernelObjHdr hFileST(OpenFileByUTF8Name(speech_tokenizer));
    if (hFileST.get() == INVALID_HANDLE_VALUE) return nullptr;
    LARGE_INTEGER FileSizeST;
    if (!GetFileSizeEx(hFileST.get(), &FileSizeST)) return nullptr;
    KernelObjHdr hFileMappingST(CreateFileMappingA(
        hFileST.get(),
        nullptr,
        PAGE_READONLY,
        0,
        0,
        nullptr));
    if (!hFileMappingST) return nullptr;
    std::unique_ptr<VOID, ViewDataUnmapper> pvDataST(MapViewOfFile(hFileMappingST.get(), FILE_MAP_READ, 0, 0, 0));
    if (!pvDataST) return nullptr;

    KernelObjHdr hFileCP(OpenFileByUTF8Name(campplus));
    if (hFileCP.get() == INVALID_HANDLE_VALUE) return nullptr;
    LARGE_INTEGER FileSizeCP;
    if (!GetFileSizeEx(hFileCP.get(), &FileSizeCP)) return nullptr;
    std::unique_ptr<VOID, KernelObjCloser> hFileMappingCP(CreateFileMappingA(
        hFileCP.get(),
        nullptr,
        PAGE_READONLY,
        0,
        0,
        nullptr));
    if (!hFileMappingCP) return nullptr;
    std::unique_ptr<VOID, ViewDataUnmapper> pvDataCP(MapViewOfFile(hFileMappingCP.get(), FILE_MAP_READ, 0, 0, 0));
    if (!pvDataCP) return nullptr;

    return cosyvoice_frontend_load(pvDataST.get(), FileSizeST.QuadPart, pvDataCP.get(), FileSizeCP.QuadPart, nullptr, nullptr);
#else
    struct munmapper {
        off_t size;
        void operator()(void* addr) const { if (addr && addr != MAP_FAILED) munmap(addr, size); }
    };
    struct file_descriptor {
        int fd;
        file_descriptor(const char* path) : fd(open(path, O_RDONLY)) {}
        ~file_descriptor() { if (fd >= 0) close(fd); }
        operator int() const { return fd; }
    };

    file_descriptor st_fd(speech_tokenizer);
    if (st_fd < 0) return nullptr;
    file_descriptor cp_fd(campplus);
    if (cp_fd < 0) return nullptr;

    struct stat cur_stat;
    if (fstat(st_fd, &cur_stat) != 0) return nullptr;
    std::unique_ptr<void, munmapper> data_st(
        mmap(nullptr, cur_stat.st_size, PROT_READ, MAP_SHARED, st_fd, 0),
        munmapper{ cur_stat.st_size });
    if (data_st.get() == MAP_FAILED) return nullptr;

    if (fstat(cp_fd, &cur_stat) != 0) return nullptr;
    std::unique_ptr<void, munmapper> data_cp(
        mmap(nullptr, cur_stat.st_size, PROT_READ, MAP_SHARED, cp_fd, 0),
        munmapper{ cur_stat.st_size });
    if (data_cp.get() == MAP_FAILED) return nullptr;

    return cosyvoice_frontend_load(data_st.get(), data_st.get_deleter().size, data_cp.get(), data_cp.get_deleter().size, nullptr, nullptr);
#endif
}

cosyvoice_frontend_context_t cosyvoice_frontend_load(const void* speech_tokenizer_data, size_t speech_tokenizer_size, const void* campplus_data, size_t campplus_size, const OrtEnv* env, const OrtSessionOptions* session_options)
{
    auto to_cref = [](auto&& obj) -> const auto& { return obj; };

    try
    {
        if (env) return new cosyvoice_frontend_context(
            speech_tokenizer_data,
            speech_tokenizer_size,
            campplus_data,
            campplus_size,
            *reinterpret_cast<const Ort::Env*>(&env),
            session_options ? *reinterpret_cast<const Ort::SessionOptions*>(&session_options) : to_cref(Ort::SessionOptions()));
        else
        {
            Ort::Env default_env(ORT_LOGGING_LEVEL_WARNING, "cosyvoice-frontend");
            auto ctx = new cosyvoice_frontend_context(
                speech_tokenizer_data,
                speech_tokenizer_size,
                campplus_data,
                campplus_size,
                default_env,
                session_options ? *reinterpret_cast<const Ort::SessionOptions*>(&session_options) : to_cref(Ort::SessionOptions()));
            ctx->env = std::move(default_env);
            return ctx;
        }
    }
    catch (const std::exception& e)
    {
        std::string_view view = e.what();
        cosyvoice_call_ggml_log_callback(GGML_LOG_LEVEL_ERROR, view.data());
        if (view.rbegin()[0] != '\n')
            cosyvoice_call_ggml_log_callback(GGML_LOG_LEVEL_ERROR, "\n");
        return nullptr;
    }
}

static void prepare_prompt_text(cosyvoice_prompt_speech_t prompt_speech, const char* prompt_text)
{
    prompt_speech->text.second = prompt_text ? static_cast<uint32_t>(strlen(prompt_text)) : 0;
    if (prompt_speech->text.second != 0)
    {
        prompt_speech->text.first.reset(new char[prompt_speech->text.second]);
        memcpy(prompt_speech->text.first.get(), prompt_text, prompt_speech->text.second);
    }
    prompt_speech->calculate_crc32();
}

#ifndef COSYVOICE_NO_AUDIO
cosyvoice_prompt_speech_t cosyvoice_frontend_prompt_speech(cosyvoice_frontend_context_t ctx, float* speech, uint32_t len, uint32_t sample_rate, const char* prompt_text)
{
    try
    {
        auto prompt_speech = ctx->frontend_prompt_speech(speech, len, sample_rate);
        prepare_prompt_text(prompt_speech, prompt_text);
        return prompt_speech;
    }
    catch (const std::exception& e)
    {
        cosyvoice_call_ggml_log_callback(GGML_LOG_LEVEL_ERROR, e.what());
        return nullptr;
    }
}
#endif

cosyvoice_prompt_speech_t cosyvoice_frontend_prompt_speech_direct(cosyvoice_frontend_context_t ctx, float* speech_16k, uint32_t speech_16k_len, float* speech_24k, uint32_t speech_24k_len, const char* prompt_text)
{
    try
    {
        auto prompt_speech = ctx->frontend_prompt_speech(speech_24k, speech_16k, speech_24k_len, speech_16k_len);
        prepare_prompt_text(prompt_speech, prompt_text);
        return prompt_speech;
    }
    catch (const std::exception& e)
    {
        cosyvoice_call_ggml_log_callback(GGML_LOG_LEVEL_ERROR, e.what());
        return nullptr;
    }
}

void cosyvoice_frontend_free(cosyvoice_frontend_context_t ctx)
{
    delete ctx;
}

matrix cosyvoice_frontend_context::extract_speech_feat(float* speech, uint32_t len)
{
    matrix signal(1, len + 1440);
    // Replicate pad to match PyTorch stft center=False + F.pad reflect behavior
    // matcha.utils.audio.mel_spectrogram uses F.pad with mode='reflect'
    // PyTorch reflect: left[i] = y[pad-1-i] for i in 0..pad-1
    // i.e. y[pad-1], y[pad-2], ..., y[0]  (includes index 0)
    // This matches PyTorch F.pad mode='reflect' for 1D tensors
    for (size_t i = 0; i != 720; ++i)
    {
        signal.data[i] = speech[720 - 1 - i];                        // left padding (reflect, includes speech[0])
        signal.data[i + len + 720] = speech[len - 1 - i];            // right padding (reflect, includes speech[len-1])
    }
    memcpy(signal.data + 720, speech, len * sizeof(float));

    constexpr int hop_length = 480;
    const auto win_size = hann_window.shape[1];

    // Unfold
    signal.shape[0] = (signal.shape[1] - win_size) / hop_length + 1;
    signal.shape[1] = win_size;
    signal.stride = hop_length;

    // Apply window
    matrix windowed_frames(signal.shape[0], signal.shape[1]);
    for (uint32_t i = 0; i != windowed_frames.shape[0]; ++i)
    {
        float* signal_cur = signal.data + i * signal.stride;
        auto signal_end = signal_cur + windowed_frames.shape[1];
        auto window_ptr = hann_window.data;
        auto dest_ptr = windowed_frames.data + i * windowed_frames.stride;

        while (signal_cur + 7 < signal_end)
        {
            __m256 a = _mm256_loadu_ps(signal_cur);
            __m256 b = _mm256_loadu_ps(window_ptr);
            _mm256_storeu_ps(dest_ptr, _mm256_mul_ps(a, b));
            signal_cur += 8;
            window_ptr += 8;
            dest_ptr += 8;
        }
        while (signal_cur + 3 < signal_end)
        {
            __m128 a = _mm_loadu_ps(signal_cur);
            __m128 b = _mm_loadu_ps(window_ptr);
            _mm_storeu_ps(dest_ptr, _mm_mul_ps(a, b));
            signal_cur += 4;
            window_ptr += 4;
            dest_ptr += 4;
        }
        while (signal_cur != signal_end)
        {
            *dest_ptr = *signal_cur * *window_ptr;
            ++signal_cur;
            ++window_ptr;
            ++dest_ptr;
        }
    }

    matrix mel_spectrum(windowed_frames.shape[0], windowed_frames.shape[1]);
    for (size_t i = 0; i != windowed_frames.shape[0]; ++i)
        fft(windowed_frames.data + i * windowed_frames.stride, mel_spectrum.data + i * mel_spectrum.stride, *fft_ctx);
    mel_spectrum = mel_spectrum.slice(0, mel_spectrum.shape[0], 0, mel_spectrum.shape[1] / 2 + 1);

    matrix mel(mel_spectrum.shape[0], mel_basis.shape[0]);
    auto basis_len = mel_basis.shape[1];
    for (uint32_t i = 0; i != mel.shape[0]; ++i)
    {
        auto mel_spectrum_cur_row = mel_spectrum.data + i * mel_spectrum.stride;

        for (uint32_t j = 0; j != mel.shape[1]; ++j)
        {
            auto mel_basis_cur_row = mel_basis.data + j * mel_basis.stride;
            __m256 v = _mm256_setzero_ps();
            uint32_t k = 0;
            for (; k + 7 < basis_len; k += 8)
            {
                __m256 a = _mm256_loadu_ps(mel_spectrum_cur_row + k);
                __m256 b = _mm256_loadu_ps(mel_basis_cur_row + k);
                v = _mm256_fmadd_ps(a, b, v);
            }
            __m128 vlow = _mm256_castps256_ps128(v);
            __m128 vhigh = _mm256_extractf128_ps(v, 1);
            __m128 sum128 = _mm_add_ps(vlow, vhigh);

            for (; k + 3 < basis_len; k += 4)
            {
                __m128 a = _mm_loadu_ps(mel_spectrum_cur_row + k);
                __m128 b = _mm_loadu_ps(mel_basis_cur_row + k);
                sum128 = _mm_fmadd_ps(a, b, sum128);
            }

            __m128 shuf = _mm_movehdup_ps(sum128);
            __m128 sums = _mm_add_ps(sum128, shuf);
            shuf = _mm_movehl_ps(shuf, sums);
            sums = _mm_add_ss(sums, shuf);

            float value = _mm_cvtss_f32(sums);
            for (; k < basis_len; ++k)
                value += mel_spectrum_cur_row[k] * mel_basis_cur_row[k];

            mel(i, j) = value;
        }
    }

    len = mel.shape[0] * mel.shape[1];
    auto mel_cur = mel.data;
    auto mel_end = mel_cur + len;
#ifdef _MSC_VER
    for (__m256 min_level = _mm256_set1_ps(1e-5f); mel_cur + 7 < mel_end; mel_cur += 8)
    {
        __m256 values = _mm256_loadu_ps(mel_cur);
        values = _mm256_max_ps(values, min_level);
        values = _mm256_log_ps(values);
        _mm256_storeu_ps(mel_cur, values);
    }
    for (__m128 min_level = _mm_set_ps1(1e-5f); mel_cur + 7 < mel_end; mel_cur += 4)
    {
        __m128 values = _mm_loadu_ps(mel_cur);
        values = _mm_max_ps(values, min_level);
        values = _mm_log_ps(values);
        _mm_storeu_ps(mel_cur, values);
    }
#endif
    for (mel_end = mel.data + len; mel_cur != mel_end; ++mel_cur)
    {
        auto value = std::log(std::max(1e-5f, *mel_cur));
        *mel_cur = value;
    }

    return mel;
}

tokens_t cosyvoice_frontend_context::extract_speech_token(float* signal, uint32_t len)
{
    constexpr int hop_length = 160;
    constexpr int nfft = 400;
    constexpr int win_size = nfft;

    // Center replicate pad (match PyTorch stft center=True default)
    // PyTorch stft with center=True uses replicate (repeat) padding, not reflect
    // pad = n_fft // 2 = 200 on each side
    // Total padded length = T + 400
    matrix padded_signal(1, len + nfft);
    for (uint32_t i = 0; i != nfft / 2; ++i)
    {
        padded_signal.data[i] = signal[0];                          // replicate left edge
        padded_signal.data[i + len + nfft / 2] = signal[len - 1];   // replicate right edge
    }
    memcpy(padded_signal.data + nfft / 2, signal, len * sizeof(float));

    // Unfold: match PyTorch stft center=True behavior, then [..., :-1] trim
    // PyTorch: stft center=True -> 592 frames, [..., :-1] -> 591 frames
    // C++: replicate pad -> T+400, unfold -> T/160 = 591 frames (matching [..., :-1])
    padded_signal.shape[0] = (len - win_size + nfft) / hop_length;
    padded_signal.shape[1] = win_size;
    padded_signal.stride = hop_length;

    // Apply window
    matrix mel_spectrum(padded_signal.shape[0], padded_signal.shape[1]);
    for (uint32_t i = 0; i != mel_spectrum.shape[0]; ++i)
    {
        float* signal_dataptr = padded_signal.data + i * padded_signal.stride;
        auto signal_dataptr_end = signal_dataptr + mel_spectrum.shape[1];
        auto window_ptr = hann_window2.data;
        auto windowed_ptr = mel_spectrum.data + i * mel_spectrum.stride;
        while (signal_dataptr + 7 < signal_dataptr_end)
        {
            __m256 a = _mm256_loadu_ps(signal_dataptr);
            __m256 b = _mm256_loadu_ps(window_ptr);
            _mm256_storeu_ps(windowed_ptr, _mm256_mul_ps(a, b));
            signal_dataptr += 8;
            window_ptr += 8;
            windowed_ptr += 8;
        }
        while (signal_dataptr + 3 < signal_dataptr_end)
        {
            __m128 a = _mm_loadu_ps(signal_dataptr);
            __m128 b = _mm_loadu_ps(window_ptr);
            _mm_storeu_ps(windowed_ptr, _mm_mul_ps(a, b));
            signal_dataptr += 4;
            window_ptr += 4;
            windowed_ptr += 4;
        }
        while (signal_dataptr != signal_dataptr_end)
        {
            *windowed_ptr = *signal_dataptr * *window_ptr;
            ++signal_dataptr;
            ++window_ptr;
            ++windowed_ptr;
        }
    }

    for (uint32_t i = 0; i != mel_spectrum.shape[0]; ++i)
        fft(mel_spectrum.data + i * mel_spectrum.stride, mel_spectrum.data + i * mel_spectrum.stride, *fft_ctx2);
    mel_spectrum = mel_spectrum.slice(0, mel_spectrum.shape[0], 0, mel_spectrum.shape[1] / 2 + 1);

    matrix mel(mel_basis2.shape[0], mel_spectrum.shape[0]);
    auto basis_len = mel_basis2.shape[1];
    for (uint32_t i = 0; i != mel.shape[0]; ++i)
    {
        auto mel_dataptr = mel.data + i * mel.stride;
        auto mel_basis_dataptr = mel_basis2.data + i * mel_basis2.stride;

        for (uint32_t j = 0; j != mel.shape[1]; ++j)
        {
            auto mel_spectrum_dataptr = mel_spectrum.data + j * mel_spectrum.stride;
            __m256 v = _mm256_setzero_ps();
            uint32_t k = 0;
            for (; k + 7 < basis_len; k += 8)
            {
                __m256 a = _mm256_loadu_ps(mel_spectrum_dataptr + k);
                a = _mm256_mul_ps(a, a);
                __m256 b = _mm256_loadu_ps(mel_basis_dataptr + k);
                v = _mm256_fmadd_ps(a, b, v);
            }
            __m128 vlow = _mm256_castps256_ps128(v);
            __m128 vhigh = _mm256_extractf128_ps(v, 1);
            __m128 sum128 = _mm_add_ps(vlow, vhigh);

            for (; k + 3 < basis_len; k += 4)
            {
                __m128 a = _mm_loadu_ps(mel_spectrum_dataptr + k);
                a = _mm_mul_ps(a, a);
                __m128 b = _mm_loadu_ps(mel_basis_dataptr + k);
                sum128 = _mm_fmadd_ps(a, b, sum128);
            }

            __m128 shuf = _mm_movehdup_ps(sum128);
            __m128 sums = _mm_add_ps(sum128, shuf);
            shuf = _mm_movehl_ps(shuf, sums);
            sums = _mm_add_ss(sums, shuf);

            float value = _mm_cvtss_f32(sums);
            for (; k != basis_len; ++k)
                value += mel_spectrum_dataptr[k] * mel_spectrum_dataptr[k] * mel_basis_dataptr[k];

            mel_dataptr[j] = value;
        }
    }

    len = mel.shape[0] * mel.shape[1];
    auto mel_dataptr = mel.data;
    auto mel_dataptr_end = mel_dataptr + len;
    float max_value = 1e-10f;

#ifdef _MSC_VER
    {
        __m256 maximum_256 = _mm256_set1_ps(1e-10f);
        for (__m256 min_level_vec = _mm256_set1_ps(1e-10f); mel_dataptr + 7 < mel_dataptr_end; mel_dataptr += 8)
        {
            __m256 values = _mm256_loadu_ps(mel_dataptr);
            values = _mm256_max_ps(values, min_level_vec);
            values = _mm256_log10_ps(values);
            _mm256_storeu_ps(mel_dataptr, values);
            maximum_256 = _mm256_max_ps(maximum_256, values);
        }

        __m128 maximum_128 = _mm_max_ps(_mm256_castps256_ps128(maximum_256), _mm256_extractf128_ps(maximum_256, 1));
        for (__m128 min_level_vec = _mm_set_ps1(1e-10f); mel_dataptr + 3 < mel_dataptr_end; mel_dataptr += 4)
        {
            __m128 values = _mm_loadu_ps(mel_dataptr);
            values = _mm_max_ps(values, min_level_vec);
            values = _mm_log10_ps(values);
            _mm_storeu_ps(mel_dataptr, values);
            maximum_128 = _mm_max_ps(maximum_128, values);
        }
        alignas(16) float temp[4];
        _mm_store_ps(temp, maximum_128);
        max_value = std::max(temp[0], temp[1]);
        max_value = std::max(max_value, temp[2]);
        max_value = std::max(temp[3], max_value);
    }
#endif
    for (; mel_dataptr != mel_dataptr_end; ++mel_dataptr)
    {
        auto value = std::log10(std::max(1e-10f, *mel_dataptr));
        *mel_dataptr = value;
        if (value > max_value)
            max_value = value;
    }

    max_value -= 8.f;
    __m256 maximum_256 = _mm256_set1_ps(max_value);
    mel_dataptr = mel.data;
    for (const __m256 _4 = _mm256_set1_ps(4.f); mel_dataptr + 7 < mel_dataptr_end; mel_dataptr += 8)
    {
        __m256 values = _mm256_loadu_ps(mel_dataptr);
        values = _mm256_max_ps(values, maximum_256);
        values = _mm256_add_ps(values, _4);
        values = _mm256_div_ps(values, _4);
        _mm256_storeu_ps(mel_dataptr, values);
    }
    for (const __m128 _4 = _mm_set_ps1(4.f), maximum_128 = _mm_set_ps1(max_value); mel_dataptr + 3 < mel_dataptr_end; mel_dataptr += 4)
    {
        __m128 values = _mm_loadu_ps(mel_dataptr);
        values = _mm_max_ps(values, maximum_128);
        values = _mm_add_ps(values, _4);
        values = _mm_div_ps(values, _4);
        _mm_storeu_ps(mel_dataptr, values);
    }
    for (mel_dataptr_end = mel.data + len; mel_dataptr != mel_dataptr_end; ++mel_dataptr)
    {
        auto value = *mel_dataptr;
        if (max_value > value)
            value = (max_value + 4.f) / 4.f;
        else
            value = (value + 4.f) / 4.f;
        *mel_dataptr = value;
    }

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    int64_t input1_shape[3] = { 1, static_cast<int64_t>(mel.shape[0]), static_cast<int64_t>(mel.shape[1]) };
    int64_t input2_shape[1] = { 1 };
    int32_t data = static_cast<int32_t>(mel.shape[1]);
    Ort::Value inputs[] =
    {
        Ort::Value::CreateTensor(
            memory_info,
            mel.data,
            mel.shape[0] * mel.shape[1] * sizeof(float),
            input1_shape,
            3,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT),
        Ort::Value::CreateTensor(
            memory_info,
            &data,
            sizeof(int32_t),
            input2_shape,
            1,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32)
    };

    auto input_names = speech_tokenizer_session.GetInputNames();
    auto output_names = speech_tokenizer_session.GetOutputNames();

    const char* input_names_ptrs[] = { input_names[0].c_str(), input_names[1].c_str() };
    const char* output_names_ptrs[] = { output_names[0].c_str() };
    auto output = speech_tokenizer_session.Run(
        Ort::RunOptions{ nullptr },
        input_names_ptrs,
        inputs,
        2,
        output_names_ptrs,
        1
    );
    auto& output_tensor = output[0];
    auto info = output_tensor.GetTensorTypeAndShapeInfo();
    size_t count = 1;
    for (auto i : info.GetShape())
        count *= i;
    auto tokens = std::make_shared<int[]>(count);
    memcpy(tokens.get(), output_tensor.GetTensorData<void*>(), count * sizeof(int));
    return std::make_pair(tokens, static_cast<uint32_t>(count));
}

matrix cosyvoice_frontend_context::extract_spk_embedding(float* speech, uint32_t len)
{
    constexpr int win_size = 400, win_shift = 160, padded_win_size = 512;

    // Unfold
    matrix frames((len - win_size) / win_shift + 1, win_size);
    for (uint32_t i = 0; i != frames.shape[0]; ++i)
    {
        auto speech_frame = speech + i * win_shift;
        __m256 sum256 = _mm256_setzero_ps();
        uint32_t j = 0;
        for (; j + 7 < win_size; j += 8)
        {
            __m256 v = _mm256_loadu_ps(speech_frame + j);
            sum256 = _mm256_add_ps(sum256, v);
        }

        __m128 vlow = _mm256_castps256_ps128(sum256);
        __m128 vhigh = _mm256_extractf128_ps(sum256, 1);
        __m128 sum128 = _mm_add_ps(vlow, vhigh);

        for (; j + 3 < win_size; j += 4)
    {
            __m128 v = _mm_loadu_ps(speech_frame + j);
            sum128 = _mm_add_ps(sum128, v);
        }

        __m128 shuf = _mm_movehdup_ps(sum128);
        __m128 sums = _mm_add_ps(sum128, shuf);
        shuf = _mm_movehl_ps(shuf, sums);
        sums = _mm_add_ss(sums, shuf);
        float sum = _mm_cvtss_f32(sums);

        for (; j != win_size; ++j)
            sum += speech_frame[j];

        float mean = sum / win_size;

        __m256 vmean = _mm256_set1_ps(mean);
        auto frame_dataptr = frames.data + i * frames.stride;
        for (j = 0; j + 7 < win_size; j += 8)
        {
            __m256 v = _mm256_loadu_ps(speech_frame + j);
            v = _mm256_sub_ps(v, vmean);
            _mm256_storeu_ps(frame_dataptr + j, v);
        }

        for (; j + 3 < win_size; j += 4)
        {
            __m128 v = _mm_loadu_ps(speech_frame + j);
            v = _mm_sub_ps(v, _mm256_castps256_ps128(vmean));
            _mm_storeu_ps(frame_dataptr + j, v);
        }

        for (; j < win_size; ++j)
            frame_dataptr[j] = speech_frame[j] - mean;
    }

    matrix prev(frames.shape[0], frames.shape[1]);
    for (uint32_t i = 0; i != frames.shape[0]; ++i)
    {
        prev(i, 0) = frames(i, 0);
        memcpy(prev.data + i * prev.stride + 1, frames.data + i * frames.stride, (frames.shape[1] - 1) * sizeof(float));
    }

    uint32_t numal = frames.shape[0] * frames.shape[1];
    uint32_t i = 0;
    for (__m256 preemph = _mm256_set1_ps(0.97f); i + 7 < numal; i += 8)
    {
        __m256 vframes = _mm256_loadu_ps(frames.data + i);
        __m256 vprev = _mm256_loadu_ps(prev.data + i);
        vframes = _mm256_sub_ps(vframes, _mm256_mul_ps(vprev, preemph));
        __m256 window = _mm256_loadu_ps(povey_window.data + (i % win_size));
        vframes = _mm256_mul_ps(vframes, window);
        _mm256_storeu_ps(frames.data + i, vframes);
    }

    for (__m128 preemph = _mm_set_ps1(0.97f); i + 3 < numal; i += 4)
    {
        __m128 vframes = _mm_loadu_ps(frames.data + i);
        __m128 vprev = _mm_loadu_ps(prev.data + i);
        vframes = _mm_sub_ps(vframes, _mm_mul_ps(vprev, preemph));
        __m128 window = _mm_loadu_ps(povey_window.data + (i % win_size));
        vframes = _mm_mul_ps(vframes, window);
        _mm_storeu_ps(frames.data + i, vframes);
    }

    for (; i < numal; ++i)
        frames.data[i] = (frames.data[i] - 0.97f * prev.data[i]) * povey_window.data[i % win_size];

    matrix spectrum(frames.shape[0], padded_win_size / 2 + 1);
    auto buffer = std::make_unique<float[]>(padded_win_size);
    // Zero-pad frame (win_size=400) to padded_win_size (512) for FFT
    auto fft_input = std::make_unique<float[]>(padded_win_size);
    memset(fft_input.get() + win_size, 0, (padded_win_size - win_size) * sizeof(float));
    for (uint32_t i = 0; i != frames.shape[0]; ++i)
    {
        memcpy(fft_input.get(), frames.data + i * frames.stride, win_size * sizeof(float));
        fft(fft_input.get(), buffer.get(), *fft_ctx_spk);
        uint32_t j = 0;
        for (; j + 7 < padded_win_size / 2 + 1; j += 8)
        {
            __m256 v = _mm256_loadu_ps(buffer.get() + j);
            v = _mm256_mul_ps(v, v);
            _mm256_storeu_ps(spectrum.data + i * spectrum.stride + j, v);
        }
        for (; j + 3 < padded_win_size / 2 + 1; j += 4)
        {
            __m128 v = _mm_loadu_ps(buffer.get() + j);
            v = _mm_mul_ps(v, v);
            _mm_storeu_ps(spectrum.data + i * spectrum.stride + j, v);
        }
        for (; j != padded_win_size / 2 + 1; ++j)
            spectrum(i, j) = buffer[j] * buffer[j];
    }

    matrix feat(spectrum.shape[0], mel_basis_spk.shape[0]);
    for (uint32_t i = 0; i != feat.shape[1]; ++i)
    {
        auto mel_basis_spk_dataptr = mel_basis_spk.data + i * mel_basis_spk.stride;
        for (uint32_t j = 0; j != feat.shape[0]; ++j)
        {
            auto spectrum_dataptr = spectrum.data + j * spectrum.stride;
            __m256 sum256 = _mm256_setzero_ps();
            uint32_t k = 0;
            for (; k + 7 < spectrum.shape[1]; k += 8)
            {
                __m256 a = _mm256_loadu_ps(spectrum_dataptr + k);
                __m256 b = _mm256_loadu_ps(mel_basis_spk_dataptr + k);
                sum256 = _mm256_fmadd_ps(a, b, sum256);
            }
            __m128 vlow = _mm256_castps256_ps128(sum256);
            __m128 vhigh = _mm256_extractf128_ps(sum256, 1);
            __m128 sum128 = _mm_add_ps(vlow, vhigh);

            for (; k + 3 < spectrum.shape[1]; k += 4)
            {
                __m128 a = _mm_loadu_ps(spectrum_dataptr + k);
                __m128 b = _mm_loadu_ps(mel_basis_spk_dataptr + k);
                sum128 = _mm_fmadd_ps(a, b, sum128);
            }

            __m128 shuf = _mm_movehdup_ps(sum128);
            __m128 sums = _mm_add_ps(sum128, shuf);
            shuf = _mm_movehl_ps(shuf, sums);
            sums = _mm_add_ss(sums, shuf);

            float sum = _mm_cvtss_f32(sums);
            for (; k != spectrum.shape[1]; ++k)
                sum += spectrum_dataptr[k] * mel_basis_spk_dataptr[k];
            feat(j, i) = std::log(std::max(1e-10f, sum));
        }
    }

    for (uint32_t i = 0; i != feat.shape[1]; ++i)
    {
        uint32_t j = 0;
        __m256 sum256 = _mm256_setzero_ps();
        __m256i idx256 = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        __m256i stride256 = _mm256_set1_epi32(static_cast<int>(feat.stride));
        __m256i _8stridev = _mm256_set1_epi32(static_cast<int>(8 * feat.stride));
        idx256 = _mm256_mullo_epi32(idx256, stride256);
        auto feat_start = feat.data + i;
        for (; j + 7 < feat.shape[0]; j += 8)
        {
            __m256 v = _mm256_i32gather_ps(feat_start, idx256, 4);
            sum256 = _mm256_add_ps(sum256, v);
            idx256 = _mm256_add_epi32(idx256, _8stridev);
        }

        __m128 vlow = _mm256_castps256_ps128(sum256);
        __m128 vhigh = _mm256_extractf128_ps(sum256, 1);
        __m128 sum128 = _mm_add_ps(vlow, vhigh);

        __m128i stride128 = _mm_set1_epi32(static_cast<int>(feat.stride));
        __m128i _4stridev = _mm_set1_epi32(static_cast<int>(4 * feat.stride));
        __m128i idx128 = _mm_setr_epi32(j, j + 1, j + 2, j + 3);
        idx128 = _mm_mullo_epi32(idx128, stride128);
        for (; j + 3 < feat.shape[0]; j += 4)
        {
            __m128 v = _mm_i32gather_ps(feat_start, idx128, 4);
            sum128 = _mm_add_ps(sum128, v);
            idx128 = _mm_add_epi32(idx128, _4stridev);
        }

        __m128 shuf = _mm_movehdup_ps(sum128);
        __m128 sums = _mm_add_ps(sum128, shuf);
        shuf = _mm_movehl_ps(shuf, sums);
        sums = _mm_add_ss(sums, shuf);
        float sum = _mm_cvtss_f32(sums);

        while (j != feat.shape[0])
            sum += feat(j++, i);

        float mean = sum / feat.shape[0];

        __m256 mean256 = _mm256_set1_ps(mean);
        idx256 = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        idx256 = _mm256_mullo_epi32(idx256, stride256);
        for (j = 0; j + 7 < feat.shape[0]; j += 8)
        {
            __m256 v = _mm256_i32gather_ps(feat_start, idx256, 4);
            v = _mm256_sub_ps(v, mean256);
            alignas(32) float values[8];
            _mm256_store_ps(values, v);

            auto mem = feat_start + j * feat.stride;
            for (int k = 0, l = 8; k != l; ++k)
                mem[k * feat.stride] = values[k];
            idx256 = _mm256_add_epi32(idx256, _8stridev);
    }

        __m128 mean128 = _mm_set_ps1(mean);
        idx128 = _mm_setr_epi32(j, j + 1, j + 2, j + 3);
        idx128 = _mm_mullo_epi32(idx128, stride128);
        for (; j + 3 < feat.shape[0]; j += 4)
    {
            __m128 v = _mm_i32gather_ps(feat_start, idx128, 4);
            v = _mm_sub_ps(v, mean128);
            alignas(16) float values[4];
            _mm_store_ps(values, v);
            auto mem = feat_start + j * feat.stride;
            for (int k = 0, l = 4; k != l; ++k)
                mem[k * feat.stride] = values[k];
            idx128 = _mm_add_epi32(idx128, _4stridev);
        }

        while (j != feat.shape[0])
            feat(j++, i) -= mean;
    }

    auto input_names = campplus_session.GetInputNames();
    auto output_names = campplus_session.GetOutputNames();

    const char* input_names_ptrs[] = { input_names[0].c_str() };
    const char* output_names_ptrs[] = { output_names[0].c_str() };
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    int64_t input_shape[3] = { 1, static_cast<int64_t>(feat.shape[0]), static_cast<int64_t>(feat.shape[1]) };
    Ort::Value input_tensor = Ort::Value::CreateTensor(
        memory_info,
        feat.data,
        feat.shape[0] * feat.shape[1] * sizeof(float),
        input_shape,
        3,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    auto output = campplus_session.Run(
        Ort::RunOptions{ nullptr },
        input_names_ptrs,
        &input_tensor,
        1,
        output_names_ptrs,
        1
    );
    auto& output_tensor = output[0];
    auto info = output_tensor.GetTensorTypeAndShapeInfo();
    uint32_t count = 1;
    for (auto i : info.GetShape())
        count *= static_cast<uint32_t>(i);
    matrix embedding(1, count);
    memcpy(embedding.data, output_tensor.GetTensorData<void*>(), count * sizeof(float));
    return embedding;
}

static cosyvoice_prompt_speech_t cosyvoice_frontend_prompt_speech_finalize(cosyvoice_prompt_speech_t prompt_speech)
{
    auto token_len = std::min(prompt_speech->feat.shape[0] / 2, prompt_speech->tokens.second);
    if (token_len * 2 != prompt_speech->feat.shape[0])
    {
        matrix new_speech_feat(2 * token_len, prompt_speech->feat.shape[1]);
        for (uint32_t i = 0; i != new_speech_feat.shape[0]; ++i)
            memcpy(new_speech_feat.data + i * new_speech_feat.stride, prompt_speech->feat.data + i * prompt_speech->feat.stride, new_speech_feat.stride * sizeof(float));
        prompt_speech->feat = new_speech_feat;
        prompt_speech->tokens.second = token_len;
    }
    return prompt_speech;
}

#ifndef COSYVOICE_NO_AUDIO
cosyvoice_prompt_speech_t cosyvoice_frontend_context::frontend_prompt_speech(float* speech, uint32_t len, uint32_t sample_rate)
{
    return cosyvoice_frontend_prompt_speech_finalize(
        frontend_vc_prompt_speech(speech, len, sample_rate)
    );
}

cosyvoice_prompt_speech_t cosyvoice_frontend_context::frontend_vc_prompt_speech(float* speech, uint32_t len, uint32_t sample_rate)
{
    float* speech_ptr = speech;
    uint32_t speech_len = len;
    if (sample_rate != 24000)
        cosyvoice_audio_resample(speech, len, sample_rate, &speech_ptr, &speech_len, 24000);

    matrix speech_feat = extract_speech_feat(speech_ptr, speech_len);
    if (speech != speech_ptr)
        cosyvoice_audio_free(speech_ptr);

    if (sample_rate != 16000)
        cosyvoice_audio_resample(speech, len, sample_rate, &speech_ptr, &speech_len, 16000);
    else
    {
        speech_ptr = speech;
        speech_len = len;
    }

    tokens_t speech_tokens = extract_speech_token(speech_ptr, speech_len);
    matrix embedding = extract_spk_embedding(speech_ptr, speech_len);
    if (speech != speech_ptr)
        cosyvoice_audio_free(speech_ptr);

    auto prompt = new cosyvoice_prompt_speech;
    prompt->feat = std::move(speech_feat);
    prompt->tokens = std::move(speech_tokens);
    prompt->embedding = std::move(embedding);
    return prompt;
}
#endif

cosyvoice_prompt_speech_t cosyvoice_frontend_context::frontend_prompt_speech(float* speech_24k, float* speech_16k, uint32_t speech_24k_len, uint32_t speech_16k_len)
{
    return cosyvoice_frontend_prompt_speech_finalize(
        frontend_vc_prompt_speech(speech_24k, speech_16k, speech_24k_len, speech_16k_len)
    );
}

cosyvoice_prompt_speech_t cosyvoice_frontend_context::frontend_vc_prompt_speech(float* speech_24k, float* speech_16k, uint32_t speech_24k_len, uint32_t speech_16k_len)
{
    matrix speech_feat = extract_speech_feat(speech_24k, speech_24k_len);

    tokens_t speech_tokens = extract_speech_token(speech_16k, speech_16k_len);
    matrix embedding = extract_spk_embedding(speech_16k, speech_16k_len);

    auto prompt = new cosyvoice_prompt_speech;
    prompt->feat = std::move(speech_feat);
    prompt->tokens = std::move(speech_tokens);
    prompt->embedding = std::move(embedding);
    return prompt;
}

inline
static float hz_to_mel(float frequencies) {
    constexpr auto f_min = 0.0f;
    constexpr auto f_sp = 200.0f / 3.0f;
    constexpr auto min_log_hz = 1000.0f;

    if (frequencies >= min_log_hz)
    {
        constexpr auto min_log_mel = (min_log_hz - f_min) / f_sp;
        const auto logstep = std::log(6.4f) / 27;
        return min_log_mel + std::log(frequencies / min_log_hz) / logstep;
    }
    else
        return (frequencies - f_min) / f_sp;
}

inline
static matrix operator-(const matrix& t1, const matrix& t2)
{
    matrix result(t1.shape[0], t2.shape[1]);
    for (uint32_t i = 0; i != result.shape[0]; ++i)
        for (uint32_t j = 0; j != result.shape[1]; ++j)
            result(i, j) = t1(i, j) - t2(i, j);
    return result;
}

inline
static matrix& scale_inplace(matrix& t, float scale)
{
    for (uint32_t i = 0; i != t.shape[0]; ++i)
        for (uint32_t j = 0; j != t.shape[1]; ++j)
            t(i, j) *= scale;
    return t;
}

inline
static matrix& right_divide_inplace(matrix& t, float numerator)
{
    for (uint32_t i = 0; i != t.shape[0]; ++i)
        for (uint32_t j = 0; j != t.shape[1]; ++j)
        {
            auto& value = t(i, j);
            value = numerator / value;
        }
    return t;
}

inline
static matrix divide(const matrix& t, float factor)
{
    matrix result(t.shape[0], t.shape[1]);
    for (uint32_t i = 0; i != result.shape[0]; ++i)
        for (uint32_t j = 0; j != result.shape[1]; ++j)
            result(i, j) = t(i, j) / factor;
    return result;
}

static matrix build_mel_basis(float sr, int n_fft, int n_mels, float fmin, float fmax)
{
    matrix mel_f(1, n_mels + 2);
    {
        auto mel_fmin = hz_to_mel(fmin);
        auto mel_fmax = hz_to_mel(fmax);
        auto step = (mel_fmax - mel_fmin) / (n_mels + 1);
        mel_f.data[0] = mel_fmin;
        for (int i = 1, end = n_mels + 2; i != end; ++i)
            mel_f.data[i] = mel_f.data[i - 1] + step;

        constexpr auto f_min = 0.0f;
        constexpr auto f_sp = 200.0f / 3;
        constexpr auto min_log_hz = 1000.0f;
        constexpr auto min_log_mel = (min_log_hz - f_min) / f_sp;

        const auto logstep = std::log(6.4f) / 27;

        for (int i = 0; i != mel_f.shape[0]; ++i)
            for (int j = 0; j != mel_f.shape[1]; ++j)
            {
                auto& value = mel_f(i, j);
                if (value >= min_log_mel)
                    value = min_log_hz * std::exp(logstep * (value - min_log_mel));
                else
                    value = value * f_sp + f_min;
            }
    }

    matrix fdiff(1, mel_f.shape[1] - 1);
    for (uint32_t i = 0; i != fdiff.shape[1]; ++i)
        fdiff.data[i] = mel_f.data[i + 1] - mel_f.data[i];

    matrix fftfreqs(1, n_fft / 2 + 1);
    for (int i = 0, end = n_fft / 2 + 1; i != end; ++i)
        fftfreqs.data[i] = i * sr / n_fft;

    matrix ramps(mel_f.shape[1], fftfreqs.shape[1]);
    for (uint32_t i = 0; i != ramps.shape[0]; ++i)
        for (uint32_t j = 0; j != ramps.shape[1]; ++j)
            ramps(i, j) = mel_f.data[i] - fftfreqs.data[j];

    matrix basis(n_mels, n_fft / 2 + 1);
    for (uint32_t i = 0; i != n_mels; ++i)
    {
        auto lower = divide(ramps[i], fdiff.data[i]);
        scale_inplace(lower, -1.0f);
        auto upper = divide(ramps[i + 2], fdiff.data[i + 1]);

        for (uint32_t j = 0; j != basis.shape[1]; ++j)
            basis(i, j) = std::max(0.0f, std::min(lower.data[j], upper.data[j]));
    }

    auto enorm = mel_f.slice(0, 1, 2, n_mels + 2) - mel_f.slice(0, 1, 0, n_mels);
    right_divide_inplace(enorm, 2.0f);
    for (uint32_t i = 0; i != basis.shape[0]; ++i)
        for (uint32_t j = 0; j != basis.shape[1]; ++j)
            basis(i, j) *= enorm.data[i];
    return basis;
}

static matrix build_hann_window(int win_length, int n_fft, bool periodic = true)
{
    matrix window(1, std::max(win_length, n_fft));
    int left = 0;
    if (win_length < n_fft)
    {
        left = (n_fft - win_length) / 2;
        for (int i = 0; i != left; ++i)
            window.data[i] = 0.f;
        for (int i = left + win_length; i != n_fft; ++i)
            window.data[i] = 0.f;
    }
    int N = win_length;
    if (!periodic)
        N -= 1;
    for (int i = 0; i != win_length; ++i)
        window.data[i + left] = (1.0f - std::cos(2.0f * 3.14159265358979323846f * i / N)) / 2.f;
    return window;
}

cosyvoice_frontend_context::cosyvoice_frontend_context(const void* speech_tokenizer, size_t speech_tokenizer_length, const void* campplus, size_t campplus_length, const Ort::Env& env, const Ort::SessionOptions& session_options)
    : speech_tokenizer_session(env, speech_tokenizer, speech_tokenizer_length, session_options), campplus_session(env, campplus, campplus_length, session_options)
{
    hann_window = build_hann_window(1920, 1920);
    hann_window2 = build_hann_window(400, 400);
    povey_window = build_hann_window(400, 400, false);

    fft_ctx = create_fft_context(1920);
    fft_ctx2 = create_fft_context(400);
    fft_ctx_spk = create_fft_context(512);

    mel_basis = build_mel_basis(24000.f, 1920, 80, 0.0f, 12000.0f);
    mel_basis2 = build_mel_basis(16000.f, 400, 128, 0.0f, 8000.0f);
    mel_basis_spk = matrix(80, 257);

    for (int i = 0; i != 400; ++i)
        povey_window.data[i] = std::pow(povey_window.data[i], 0.85f);

    constexpr float nyquist = 8000.f;
    constexpr auto high = nyquist;
    constexpr int num_fft_bins = 256;
    constexpr int num_mel_bins = 80;
    constexpr float fft_bin_width = 16000.f / 512;
   
    const auto mel_low = 1127.0f * std::log(1.0f + 20.f / 700.f);
    const auto mel_high = 1127.0f * std::log(1.0f + high / 700.0f);
    const auto mel_delta = (mel_high - mel_low) / (num_mel_bins + 1);

    float left_mel[num_mel_bins];
    float center_mel[num_mel_bins];
    float right_mel[num_mel_bins];
    for (int i = 0; i != num_mel_bins; ++i)
    {
        left_mel[i] = mel_low + mel_delta * i;
        center_mel[i] = mel_low + mel_delta * (i + 1);
        right_mel[i] = mel_low + mel_delta * (i + 2);
    }

    float mel[num_fft_bins];
    for (int i = 0; i != num_fft_bins; ++i)
        mel[i] = 1127.f * std::log(1.0f + i * fft_bin_width / 700.f);
    for (uint32_t i = 0; i != mel_basis_spk.shape[0]; ++i)
    {
        const float up_scaling = center_mel[i] - left_mel[i];
        const float down_scaling = right_mel[i] - center_mel[i];
        for (uint32_t j = 0; j != 256; ++j)
        {
            const auto up = (mel[j] - left_mel[i]) / up_scaling;
            const auto down = (right_mel[i] - mel[j]) / down_scaling;
            mel_basis_spk(i, j) = std::max(0.f, std::min(up, down));
        }
        mel_basis_spk(i, 256) = 0.f;
    }
}
