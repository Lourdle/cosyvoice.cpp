#pragma once

#include "common.h"

#include <cstdint>

#ifdef _WIN32
    #define COSYVOICE_TEXT(x) L##x
using tchar = wchar_t;
std::wstring utf8_to_wstr(const char* utf8);
void setup_console_utf8();
#else
    #define COSYVOICE_TEXT(x) x
using tchar = char;
#endif

std::string tchar_to_utf8(const tchar* value);
std::string to_lower(std::string value);
int tchar_casecmp(const tchar* lhs, const tchar* rhs);
bool parse_float_arg(const std::string& value, float* result);
bool parse_uint32_arg(const std::string& value, uint32_t* result);
bool parse_int_arg(const std::string& value, int* result);

std::ifstream open_ifstream_utf8(const char* path, std::ios::openmode mode = std::ios::binary);
