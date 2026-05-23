#pragma once

#include <fstream>
#include <string>

#ifdef _WIN32
std::wstring utf8_to_wstr(const char* utf8);
#endif

std::ifstream open_ifstream_utf8(const char* path, std::ios::openmode mode = std::ios::binary);
std::ofstream open_ofstream_utf8(const char* path, std::ios::openmode mode = std::ios::binary | std::ios::trunc);

#define COSYVOICE_DEFAULT_LLM_MAX_SEQ_LEN 4096
#define STRINGIFY(x) #x
#define EXPAND_AND_STRINGIFY(x) STRINGIFY(x)
#define COSYVOICE_DEFAULT_LLM_MAX_SEQ_LEN_STR EXPAND_AND_STRINGIFY(COSYVOICE_DEFAULT_LLM_MAX_SEQ_LEN)
