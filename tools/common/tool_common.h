#pragma once

#include "common.h"

#include <vector>
#include <chrono>
#include <cstdint>
#include <cctype>
#include <cstdio>
#include <ctime>
#include <limits>

std::string to_lower(std::string value);
int str_casecmp(const char* lhs, const char* rhs);
std::string trim_copy(const std::string& value);
double elapsed_ms(std::chrono::steady_clock::time_point from, std::chrono::steady_clock::time_point to);
double bytes_to_mib(size_t bytes);
bool parse_float_arg(const std::string& value, float* result);
bool parse_uint32_arg(const char* value, uint32_t* result);
bool parse_int_arg(const std::string& value, int* result);
bool parse_uint16_port(const std::string& value, uint16_t* result);

inline
bool parse_uint32_arg(const std::string& value, uint32_t* result)
{
    return parse_uint32_arg(value.c_str(), result);
}

std::string get_local_timestamp_ms();

int tool_entry(int argc, char** argv);
