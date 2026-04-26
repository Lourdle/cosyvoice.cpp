#include "cosyvoice-internal.h"
#include "cosyvoice-audio.h"
#include "common.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/opt.h>
#include <libswresample/swresample.h>
}

#include <format>
#include <algorithm>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
    #define strcasecmp _stricmp
#endif

namespace {

struct utf8_input_source
{
    std::ifstream file;
};

static int read_packet(void* opaque, uint8_t* buf, int buf_size)
{
    auto* src = reinterpret_cast<utf8_input_source*>(opaque);
    if (!src || !src->file.good())
        return AVERROR_EOF;

    src->file.read(reinterpret_cast<char*>(buf), buf_size);
    const std::streamsize got = src->file.gcount();
    if (got <= 0)
        return src->file.eof() ? AVERROR_EOF : AVERROR(EIO);
    return static_cast<int>(got);
}

static int64_t seek_packet(void* opaque, int64_t offset, int whence)
{
    auto* src = reinterpret_cast<utf8_input_source*>(opaque);
    if (!src)
        return AVERROR(EINVAL);

    if (whence == AVSEEK_SIZE)
    {
        const auto pos = src->file.tellg();
        src->file.seekg(0, std::ios::end);
        const auto end = src->file.tellg();
        src->file.seekg(pos);
        return static_cast<int64_t>(end);
    }

    std::ios_base::seekdir dir = std::ios::beg;
    if (whence == SEEK_CUR) dir = std::ios::cur;
    else if (whence == SEEK_END) dir = std::ios::end;
    src->file.clear();
    src->file.seekg(offset, dir);
    if (!src->file.good())
        return AVERROR(EIO);
    return static_cast<int64_t>(src->file.tellg());
}

} // namespace

static AVSampleFormat pick_sample_format(const AVCodec* codec, AVSampleFormat preferred)
{
    if (codec && codec->sample_fmts)
    {
        for (const AVSampleFormat* fmt = codec->sample_fmts; *fmt != AV_SAMPLE_FMT_NONE; ++fmt)
        {
            if (*fmt == preferred) return *fmt;
        }
        for (const AVSampleFormat* fmt = codec->sample_fmts; *fmt != AV_SAMPLE_FMT_NONE; ++fmt)
        {
            if (*fmt == AV_SAMPLE_FMT_FLTP) return *fmt;
        }
        for (const AVSampleFormat* fmt = codec->sample_fmts; *fmt != AV_SAMPLE_FMT_NONE; ++fmt)
        {
            if (*fmt == AV_SAMPLE_FMT_FLT) return *fmt;
        }
        return codec->sample_fmts[0];
    }
    return preferred;
}

static const AVCodec* pick_encoder(const char* filename, cosyvoice_audio_encoding_format_t format, const char** mux_name)
{
    switch (format)
    {
    case COSYVOICE_AUDIO_ENCODING_FORMAT_WAV:
        *mux_name = "wav";
        return avcodec_find_encoder(AV_CODEC_ID_PCM_S16LE);
    case COSYVOICE_AUDIO_ENCODING_FORMAT_MP3:
        *mux_name = "mp3";
        return avcodec_find_encoder_by_name("libmp3lame");
    case COSYVOICE_AUDIO_ENCODING_FORMAT_AAC:
        *mux_name = "adts";
        return avcodec_find_encoder(AV_CODEC_ID_AAC);
    case COSYVOICE_AUDIO_ENCODING_FORMAT_FLAC:
        *mux_name = "flac";
        return avcodec_find_encoder(AV_CODEC_ID_FLAC);
    case COSYVOICE_AUDIO_ENCODING_FORMAT_M4A:
        *mux_name = "ipod";
        return avcodec_find_encoder(AV_CODEC_ID_AAC);
    case COSYVOICE_AUDIO_ENCODING_FORMAT_OPUS:
        *mux_name = "opus";
        return avcodec_find_encoder(AV_CODEC_ID_OPUS);
    default:
        *mux_name = "wav";
        return avcodec_find_encoder(AV_CODEC_ID_PCM_S16LE);
    }
}

static bool format_is_supported_by_backend(cosyvoice_audio_encoding_format_t format)
{
    const char* mux_name = nullptr;
    const AVCodec* codec = pick_encoder(nullptr, format, &mux_name);
    if (!codec || !mux_name)
        return false;

    if (format == COSYVOICE_AUDIO_ENCODING_FORMAT_WAV)
        return avcodec_find_encoder(AV_CODEC_ID_PCM_S16LE) != nullptr;
    if (format == COSYVOICE_AUDIO_ENCODING_FORMAT_MP3)
        return avcodec_find_encoder_by_name("libmp3lame") != nullptr;
    if (format == COSYVOICE_AUDIO_ENCODING_FORMAT_AAC)
        return avcodec_find_encoder(AV_CODEC_ID_AAC) != nullptr;
    if (format == COSYVOICE_AUDIO_ENCODING_FORMAT_FLAC)
        return avcodec_find_encoder(AV_CODEC_ID_FLAC) != nullptr;
    if (format == COSYVOICE_AUDIO_ENCODING_FORMAT_M4A)
        return avcodec_find_encoder(AV_CODEC_ID_AAC) != nullptr;
    if (format == COSYVOICE_AUDIO_ENCODING_FORMAT_OPUS)
        return avcodec_find_encoder(AV_CODEC_ID_OPUS) != nullptr;
    return false;
}

static bool ensure_path_matches_ext(const char* filename, cosyvoice_audio_encoding_format_t format)
{
    const char* ext = strrchr(filename, '.');
    if (!ext) return true;
    switch (format)
    {
    case COSYVOICE_AUDIO_ENCODING_FORMAT_MP3:  return strcasecmp(ext, ".mp3") == 0;
    case COSYVOICE_AUDIO_ENCODING_FORMAT_AAC:  return strcasecmp(ext, ".aac") == 0;
    case COSYVOICE_AUDIO_ENCODING_FORMAT_FLAC: return strcasecmp(ext, ".flac") == 0;
    case COSYVOICE_AUDIO_ENCODING_FORMAT_M4A:  return strcasecmp(ext, ".m4a") == 0;
    case COSYVOICE_AUDIO_ENCODING_FORMAT_OPUS: return strcasecmp(ext, ".opus") == 0;
    case COSYVOICE_AUDIO_ENCODING_FORMAT_WAV:
    default: return strcasecmp(ext, ".wav") == 0;
    }
}

struct ffmpeg_encode_session
{
    AVFormatContext* fmt = nullptr;
    AVCodecContext* codec = nullptr;
    AVFrame* frame = nullptr;
    AVPacket* pkt = nullptr;
    SwrContext* swr = nullptr;
    AVStream* stream = nullptr;
    AVSampleFormat in_fmt = AV_SAMPLE_FMT_FLT;
    AVSampleFormat out_fmt = AV_SAMPLE_FMT_FLT;
    int frame_size = 0;
    int64_t pts = 0;
    std::vector<uint8_t> buffer;
    uint8_t* dyn_buf = nullptr;
    int dyn_buf_size = 0;
    bool finished = false;
    bool to_memory = false;

    ~ffmpeg_encode_session() { reset(); }

    void reset()
    {
        if (swr) { swr_free(&swr); }
        if (frame) { av_frame_free(&frame); }
        if (pkt) { av_packet_free(&pkt); }
        if (codec) { avcodec_free_context(&codec); }
        if (fmt)
        {
            if (fmt->pb && (fmt->oformat->flags & AVFMT_NOFILE) == 0)
                avio_closep(&fmt->pb);
            avformat_free_context(fmt);
        }
        if (dyn_buf)
        {
            av_free(dyn_buf);
            dyn_buf = nullptr;
            dyn_buf_size = 0;
        }
        fmt = nullptr;
        stream = nullptr;
        frame_size = 0;
        pts = 0;
        finished = false;
        to_memory = false;
        buffer.clear();
    }

    bool init_output(const char* filename, cosyvoice_audio_encoding_format_t format, uint32_t sample_rate, bool want_memory)
    {
        reset();

        const char* mux_name = nullptr;
        const AVCodec* enc = pick_encoder(filename, format, &mux_name);
        if (!enc) return false;

        if (avformat_alloc_output_context2(&fmt, nullptr, mux_name, filename) < 0 || !fmt)
            return false;

        this->to_memory = want_memory;

        stream = avformat_new_stream(fmt, enc);
        if (!stream)
            return false;

        codec = avcodec_alloc_context3(enc);
        if (!codec)
            return false;

        codec->sample_rate = static_cast<int>(sample_rate);
        av_channel_layout_default(&codec->ch_layout, 1);
        codec->sample_fmt = pick_sample_format(enc, AV_SAMPLE_FMT_FLT);
        out_fmt = codec->sample_fmt;
        in_fmt = AV_SAMPLE_FMT_FLT;
        codec->bit_rate = (format == COSYVOICE_AUDIO_ENCODING_FORMAT_OPUS) ? 64000 : 128000;
        codec->time_base = AVRational{1, static_cast<int>(sample_rate)};
        stream->time_base = codec->time_base;

        if (fmt->oformat->flags & AVFMT_GLOBALHEADER)
            codec->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

        if (avcodec_open2(codec, enc, nullptr) < 0)
            return false;

        frame_size = codec->frame_size > 0 ? codec->frame_size : 1024;

        if (avcodec_parameters_from_context(stream->codecpar, codec) < 0)
            return false;

        if (want_memory)
        {
            if (avio_open_dyn_buf(&fmt->pb) < 0)
                return false;
        }
        else
        {
            if (avio_open(&fmt->pb, filename, AVIO_FLAG_WRITE) < 0)
                return false;
        }

        if (avformat_write_header(fmt, nullptr) < 0)
            return false;

        frame = av_frame_alloc();
        pkt = av_packet_alloc();
        if (!frame || !pkt)
            return false;

        frame->sample_rate = static_cast<int>(sample_rate);
        frame->format = out_fmt;
        av_channel_layout_default(&frame->ch_layout, 1);
        frame->nb_samples = frame_size;
        if (av_frame_get_buffer(frame, 0) < 0)
            return false;

        if (codec->sample_fmt != AV_SAMPLE_FMT_FLT)
        {
            swr = swr_alloc();
            if (!swr) return false;
            AVChannelLayout mono = AV_CHANNEL_LAYOUT_MONO;
            av_opt_set_chlayout(swr, "in_chlayout", &mono, 0);
            av_opt_set_chlayout(swr, "out_chlayout", &codec->ch_layout, 0);
            av_opt_set_int(swr, "in_sample_rate", sample_rate, 0);
            av_opt_set_int(swr, "out_sample_rate", sample_rate, 0);
            av_opt_set_sample_fmt(swr, "in_sample_fmt", AV_SAMPLE_FMT_FLT, 0);
            av_opt_set_sample_fmt(swr, "out_sample_fmt", codec->sample_fmt, 0);
            if (swr_init(swr) < 0)
                return false;
        }

        return true;
    }

    bool encode_pcm(const float* input, uint32_t length)
    {
        if (!fmt || !codec || !frame || !pkt || !input) return false;

        auto drain = [&]() -> bool
        {
            while (true)
            {
                const int ret = avcodec_receive_packet(codec, pkt);
                if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) return true;
                if (ret < 0) return false;
                pkt->stream_index = stream->index;
                av_packet_rescale_ts(pkt, codec->time_base, stream->time_base);
                if (av_interleaved_write_frame(fmt, pkt) < 0) return false;
                av_packet_unref(pkt);
            }
        };

        uint32_t offset = 0;
        while (offset < length)
        {
            const uint32_t chunk = std::min<uint32_t>(static_cast<uint32_t>(frame_size), length - offset);
            if (av_frame_make_writable(frame) < 0) return false;

            frame->nb_samples = static_cast<int>(chunk);
            frame->pts = pts;

            if (swr)
            {
                const uint8_t* in = reinterpret_cast<const uint8_t*>(input + offset);
                if (swr_convert(swr, frame->extended_data, chunk, &in, chunk) < 0)
                    return false;
            }
            else
            {
                std::memcpy(frame->data[0], input + offset, chunk * sizeof(float));
            }

            if (avcodec_send_frame(codec, frame) < 0) return false;
            if (!drain()) return false;

            offset += chunk;
            pts += chunk;
        }

        if (avcodec_send_frame(codec, nullptr) < 0) return false;
        return drain();
    }

    bool finish()
    {
        if (!fmt) return false;
        if (finished) return true;

        av_write_trailer(fmt);
        if (fmt->pb)
        {
            if (to_memory)
            {
                dyn_buf_size = avio_close_dyn_buf(fmt->pb, &dyn_buf);
                fmt->pb = nullptr;
                if (dyn_buf_size > 0 && dyn_buf)
                {
                    buffer.assign(dyn_buf, dyn_buf + dyn_buf_size);
                    av_free(dyn_buf);
                    dyn_buf = nullptr;
                    dyn_buf_size = 0;
                }
            }
            else
            {
                avio_closep(&fmt->pb);
            }
        }
        finished = true;
        return true;
    }
};

struct cosyvoice_audio_encoder
{
    explicit cosyvoice_audio_encoder(uint32_t sample_rate) : sample_rate(sample_rate) {}
    ~cosyvoice_audio_encoder() = default;

    bool encode(const float* input, uint32_t length, cosyvoice_audio_encoding_format_t format)
    {
        if (!input || length == 0) return false;
        if (!session) session = std::make_unique<ffmpeg_encode_session>();

        session->reset();
        const bool ok = session->init_output(nullptr, format, sample_rate, true) && session->encode_pcm(input, length) && session->finish();
        if (!ok) session->reset();
        return ok;
    }

    bool save(const char* filename, const float* input, uint32_t length, cosyvoice_audio_encoding_format_t format)
    {
        if (!filename || !input || length == 0) return false;
        if (!ensure_path_matches_ext(filename, format))
            cosyvoice_call_ggml_log_callback(GGML_LOG_LEVEL_WARN, "Output file extension does not match requested audio format.\n");

        if (!session) session = std::make_unique<ffmpeg_encode_session>();
        session->reset();
        const bool ok = session->init_output(nullptr, format, sample_rate, true) && session->encode_pcm(input, length) && session->finish();
        if (ok)
        {
            auto out = open_ofstream_utf8(filename, std::ios::binary | std::ios::trunc);
            if (!out.is_open())
                return false;
            out.write(reinterpret_cast<const char*>(session->buffer.data()), static_cast<std::streamsize>(session->buffer.size()));
            out.flush();
            if (!out.good())
                return false;
        }
        if (!ok) session->reset();
        return ok;
    }

    const uint8_t* data() const { return session ? session->buffer.data() : nullptr; }
    uint32_t size() const { return session ? static_cast<uint32_t>(session->buffer.size()) : 0; }

    uint32_t sample_rate;
    std::unique_ptr<ffmpeg_encode_session> session;
};

bool cosyvoice_audio_encoding_format_supported(cosyvoice_audio_encoding_format_t format)
{
    return format_is_supported_by_backend(format);
}

const char* cosyvoice_audio_supported_encoding_formats(void)
{
    static std::string cached;
    if (!cached.empty())
        return cached.c_str();

    const cosyvoice_audio_encoding_format_t formats[] = {
        COSYVOICE_AUDIO_ENCODING_FORMAT_WAV,
        COSYVOICE_AUDIO_ENCODING_FORMAT_MP3,
        COSYVOICE_AUDIO_ENCODING_FORMAT_AAC,
        COSYVOICE_AUDIO_ENCODING_FORMAT_FLAC,
        COSYVOICE_AUDIO_ENCODING_FORMAT_M4A,
        COSYVOICE_AUDIO_ENCODING_FORMAT_OPUS,
    };

    for (auto format : formats)
    {
        if (!format_is_supported_by_backend(format))
            continue;

        if (!cached.empty())
            cached += ", ";

        switch (format)
        {
        case COSYVOICE_AUDIO_ENCODING_FORMAT_WAV: cached += "wav"; break;
        case COSYVOICE_AUDIO_ENCODING_FORMAT_MP3: cached += "mp3"; break;
        case COSYVOICE_AUDIO_ENCODING_FORMAT_AAC: cached += "aac"; break;
        case COSYVOICE_AUDIO_ENCODING_FORMAT_FLAC: cached += "flac"; break;
        case COSYVOICE_AUDIO_ENCODING_FORMAT_OPUS: cached += "opus"; break;
        default: break;
        }
    }

    if (cached.empty())
        cached = "(none)";
    return cached.c_str();
}

cosyvoice_audio_encoder_t cosyvoice_audio_encoder_create(uint32_t sample_rate)
{
    return new cosyvoice_audio_encoder(sample_rate);
}

void cosyvoice_audio_encoder_destroy(cosyvoice_audio_encoder_t encoder)
{
    delete encoder;
}

bool cosyvoice_audio_encoder_encode(cosyvoice_audio_encoder_t encoder, const float* input, uint32_t length, cosyvoice_audio_encoding_format_t format)
{
    if (!encoder || !input) return false;
    return encoder->encode(input, length, format);
}

void cosyvoice_audio_encoder_get_encoded_data(cosyvoice_audio_encoder_t encoder, const uint8_t** data, uint32_t* length)
{
    if (!encoder || !data || !length)
    {
        if (data) *data = nullptr;
        if (length) *length = 0;
        return;
    }
    *data = encoder->data();
    *length = encoder->size();
}

bool cosyvoice_audio_load_from_file(const char* filename, float** data, uint32_t* length, uint32_t* sample_rate)
{
    if (!filename || !data || !length || !sample_rate) return false;

    AVFormatContext* format_ctx = nullptr;
    AVCodecContext* codec_ctx = nullptr;
    AVFrame* frame = nullptr;
    AVPacket* pkt = nullptr;
    SwrContext* swr_ctx = nullptr;
    AVIOContext* io_ctx = nullptr;
    utf8_input_source io_source;

    try
    {
        io_source.file = open_ifstream_utf8(filename, std::ios::binary);
        if (!io_source.file.is_open())
            throw std::runtime_error("Failed to open audio file");

        const int buffer_size = 4096;
        unsigned char* buffer = static_cast<unsigned char*>(av_malloc(buffer_size));
        if (!buffer)
            throw std::runtime_error("Failed to allocate IO buffer");

        io_ctx = avio_alloc_context(buffer, buffer_size, 0, &io_source, &read_packet, nullptr, &seek_packet);
        if (!io_ctx)
            throw std::runtime_error("Failed to allocate AVIO context");

        format_ctx = avformat_alloc_context();
        if (!format_ctx)
            throw std::runtime_error("Failed to allocate format context");

        format_ctx->pb = io_ctx;
        format_ctx->flags |= AVFMT_FLAG_CUSTOM_IO;

        if (avformat_open_input(&format_ctx, nullptr, nullptr, nullptr) < 0)
            throw std::runtime_error("Failed to open audio file");

        if (avformat_find_stream_info(format_ctx, nullptr) < 0)
            throw std::runtime_error("Failed to find stream info");

        int audio_stream_idx = -1;
        for (unsigned int i = 0; i < format_ctx->nb_streams; i++)
        {
            if (format_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO)
            {
                audio_stream_idx = static_cast<int>(i);
                break;
            }
        }
        if (audio_stream_idx < 0)
            throw std::runtime_error("No audio stream found");

        AVStream* audio_stream = format_ctx->streams[static_cast<unsigned int>(audio_stream_idx)];
        const AVCodec* codec = avcodec_find_decoder(audio_stream->codecpar->codec_id);
        if (!codec)
            throw std::runtime_error("Unsupported codec");

        codec_ctx = avcodec_alloc_context3(codec);
        if (!codec_ctx)
            throw std::runtime_error("Failed to allocate codec context");

        if (avcodec_parameters_to_context(codec_ctx, audio_stream->codecpar) < 0)
            throw std::runtime_error("Failed to copy codec parameters");

        if (avcodec_open2(codec_ctx, codec, nullptr) < 0)
            throw std::runtime_error("Failed to open codec");

        *sample_rate = codec_ctx->sample_rate;

        swr_ctx = swr_alloc();
        if (!swr_ctx)
            throw std::runtime_error("Failed to allocate resampler");

        AVChannelLayout mono = AV_CHANNEL_LAYOUT_MONO;
        av_opt_set_chlayout(swr_ctx, "in_chlayout", &codec_ctx->ch_layout, 0);
        av_opt_set_chlayout(swr_ctx, "out_chlayout", &mono, 0);
        av_opt_set_int(swr_ctx, "in_sample_rate", codec_ctx->sample_rate, 0);
        av_opt_set_int(swr_ctx, "out_sample_rate", codec_ctx->sample_rate, 0);
        av_opt_set_sample_fmt(swr_ctx, "in_sample_fmt", codec_ctx->sample_fmt, 0);
        av_opt_set_sample_fmt(swr_ctx, "out_sample_fmt", AV_SAMPLE_FMT_FLT, 0);
        if (swr_init(swr_ctx) < 0)
            throw std::runtime_error("Failed to initialize resampler");

        frame = av_frame_alloc();
        pkt = av_packet_alloc();
        if (!frame || !pkt)
            throw std::runtime_error("Failed to allocate decode buffers");

        std::vector<float> output_buffer;

        while (av_read_frame(format_ctx, pkt) >= 0)
        {
            if (pkt->stream_index == audio_stream_idx)
            {
                if (avcodec_send_packet(codec_ctx, pkt) < 0)
                {
                    av_packet_unref(pkt);
                    continue;
                }

                while (avcodec_receive_frame(codec_ctx, frame) >= 0)
                {
                    const uint8_t* in_data = frame->data[0];
                    int in_samples = frame->nb_samples;
                    int out_samples = av_rescale_rnd(in_samples, codec_ctx->sample_rate, codec_ctx->sample_rate, AV_ROUND_UP);
                    std::vector<float> out_buffer(static_cast<size_t>(out_samples));
                    int converted = swr_convert(swr_ctx, reinterpret_cast<uint8_t**>(out_buffer.data()), out_samples, &in_data, in_samples);
                    if (converted > 0)
                        output_buffer.insert(output_buffer.end(), out_buffer.begin(), out_buffer.begin() + converted);
                }
            }
            av_packet_unref(pkt);
        }

        avcodec_send_packet(codec_ctx, nullptr);
        while (avcodec_receive_frame(codec_ctx, frame) >= 0)
        {
            const uint8_t* in_data = frame->data[0];
            int in_samples = frame->nb_samples;
            int out_samples = av_rescale_rnd(in_samples, codec_ctx->sample_rate, codec_ctx->sample_rate, AV_ROUND_UP);
            std::vector<float> out_buffer(static_cast<size_t>(out_samples));
            int converted = swr_convert(swr_ctx, reinterpret_cast<uint8_t**>(out_buffer.data()), out_samples, &in_data, in_samples);
            if (converted > 0)
                output_buffer.insert(output_buffer.end(), out_buffer.begin(), out_buffer.begin() + converted);
        }

        *length = static_cast<uint32_t>(output_buffer.size());
        *data = new float[*length];
        std::memcpy(*data, output_buffer.data(), *length * sizeof(float));
    }
    catch (const std::exception& e)
    {
        cosyvoice_call_ggml_log_callback(GGML_LOG_LEVEL_ERROR, e.what());
        *data = nullptr;
        *length = 0;
        *sample_rate = 0;
    }

    if (swr_ctx) swr_free(&swr_ctx);
    if (frame) av_frame_free(&frame);
    if (pkt) av_packet_free(&pkt);
    if (codec_ctx) avcodec_free_context(&codec_ctx);
    if (format_ctx) avformat_close_input(&format_ctx);
    if (io_ctx)
    {
        av_freep(&io_ctx->buffer);
        avio_context_free(&io_ctx);
    }

    return *data != nullptr;
}

bool cosyvoice_audio_resample(const float* input, uint32_t input_length, uint32_t input_sample_rate,
                               float** output, uint32_t* output_length, uint32_t output_sample_rate)
{
    if (!input || !output || !output_length) return false;

    SwrContext* swr_ctx = swr_alloc();
    if (!swr_ctx) return false;

    AVChannelLayout mono = AV_CHANNEL_LAYOUT_MONO;
    av_opt_set_chlayout(swr_ctx, "in_chlayout", &mono, 0);
    av_opt_set_chlayout(swr_ctx, "out_chlayout", &mono, 0);
    av_opt_set_int(swr_ctx, "in_sample_rate", input_sample_rate, 0);
    av_opt_set_int(swr_ctx, "out_sample_rate", output_sample_rate, 0);
    av_opt_set_sample_fmt(swr_ctx, "in_sample_fmt", AV_SAMPLE_FMT_FLT, 0);
    av_opt_set_sample_fmt(swr_ctx, "out_sample_fmt", AV_SAMPLE_FMT_FLT, 0);

    if (swr_init(swr_ctx) < 0)
    {
        swr_free(&swr_ctx);
        return false;
    }

    int64_t delay = swr_get_delay(swr_ctx, input_sample_rate);
    int64_t out_samples = av_rescale_rnd(input_length + delay, output_sample_rate, input_sample_rate, AV_ROUND_UP);
    std::unique_ptr<float[]> out_buffer(new float[out_samples]);
    const uint8_t* in_data = reinterpret_cast<const uint8_t*>(input);
    int converted = swr_convert(swr_ctx, reinterpret_cast<uint8_t**>(out_buffer.get()), out_samples, &in_data, input_length);
    swr_free(&swr_ctx);

    if (converted < 0) return false;
    *output = out_buffer.release();
    *output_length = static_cast<uint32_t>(converted);
    return true;
}

bool cosyvoice_audio_save_to_file(const char* filename, const float* data, uint32_t length, uint32_t sample_rate)
{
    if (!filename || !data || length == 0 || sample_rate == 0) return false;

    const char* ext = strrchr(filename, '.');
    cosyvoice_audio_encoding_format_t format = COSYVOICE_AUDIO_ENCODING_FORMAT_WAV;
    if (ext)
        if (strcasecmp(ext, ".mp3") == 0) format = COSYVOICE_AUDIO_ENCODING_FORMAT_MP3;
        else if (strcasecmp(ext, ".aac") == 0) format = COSYVOICE_AUDIO_ENCODING_FORMAT_AAC;
        else if (strcasecmp(ext, ".flac") == 0) format = COSYVOICE_AUDIO_ENCODING_FORMAT_FLAC;
        else if (strcasecmp(ext, ".m4a") == 0) format = COSYVOICE_AUDIO_ENCODING_FORMAT_M4A;
        else if (strcasecmp(ext, ".opus") == 0) format = COSYVOICE_AUDIO_ENCODING_FORMAT_OPUS;
        else cosyvoice_call_ggml_log_callback(GGML_LOG_LEVEL_WARN, std::format("Unrecognized file extension ({}), defaulting to WAV format.\n", ext).c_str());

    cosyvoice_audio_encoder encoder(sample_rate);
    if (!encoder.save(filename, data, length, format))
    {
        if (format == COSYVOICE_AUDIO_ENCODING_FORMAT_WAV)
            return cosyvoice_save_wav(filename, data, length, sample_rate);
        cosyvoice_call_ggml_log_callback(GGML_LOG_LEVEL_ERROR, "Failed to save output audio file.\n");
        return false;
    }
    return true;
}

void cosyvoice_audio_free(float* data)
{
    delete[] data;
}
