#pragma once

#include "cosyvoice-internal.h"

#include <cstdint>
#include <cstdlib>

struct stage_timing_round
{
    double llm_prefill_ms = 0;
    double llm_decode_ms = 0;
    double llm_epilogue_ms = 0;
    double flow_encode_ms = 0;
    double dit_ms = 0;
    double mel_ms = 0;
    double hift_ms = 0;
    uint32_t llm_prefill_builds = 0;
    uint32_t llm_decode_calls = 0;
    uint32_t llm_decode_builds = 0;
    uint32_t dit_steps = 0;
    uint32_t dit_builds = 0;
    uint32_t hift_frames = 0;
    uint32_t sched_rereserves = 0;
    uint32_t token2wav_reallocs = 0;
    // Only accumulated into the streaming summary: `round_ms` is the wall time
    // of the round, `first_ms` the wall time of the first streaming chunk.
    double round_ms = 0;
    double first_ms = 0;
    bool active = false;
};

inline bool stage_timing_enabled()
{
    static const bool enabled = std::getenv("COSYVOICE_STAGE_TIMING") != nullptr;
    return enabled;
}

inline void stage_timing_log(const char* msg)
{
    cosyvoice_call_ggml_log_callback(GGML_LOG_LEVEL_INFO, msg);
}

inline double stage_timing_ms(int64_t begin_us, int64_t end_us)
{
    return static_cast<double>(end_us - begin_us) / 1000.0;
}

class stage_mark
{
public:
    stage_mark() : active_(stage_timing_enabled()), begin_us_(active_ ? ggml_time_us() : 0) {}

    double elapsed_ms() const
    {
        return active_ ? stage_timing_ms(begin_us_, ggml_time_us()) : 0.0;
    }

    stage_mark(const stage_mark&) = delete;
    stage_mark& operator=(const stage_mark&) = delete;

private:
    bool active_;
    int64_t begin_us_;
};
