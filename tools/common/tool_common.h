#pragma once

#include "common.h"

#include <chrono>
#include <cstdint>
#include <cctype>
#include <cstdio>
#include <ctime>
#include <limits>

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
std::string trim_copy(const std::string& value);
double elapsed_ms(std::chrono::steady_clock::time_point from, std::chrono::steady_clock::time_point to);
double bytes_to_mib(size_t bytes);
bool parse_float_arg(const std::string& value, float* result);
bool parse_uint32_arg(const std::string& value, uint32_t* result);
bool parse_int_arg(const std::string& value, int* result);
bool parse_uint16_port(const std::string& value, uint16_t* result);

std::string get_local_timestamp_ms();
