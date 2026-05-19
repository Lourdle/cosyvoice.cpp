#pragma once

#include <cstdint>
#include <string>

bool cli_audio_play_pcm_blocking(const float* data, uint32_t length, uint32_t sample_rate, std::string* error);
