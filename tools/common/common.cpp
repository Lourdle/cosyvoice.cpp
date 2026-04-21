#include "tool_common.h"

#ifdef _WIN32
    #define NOMINMAX
    #include <Windows.h>
    #include <cwchar>
#else
    #include <strings.h>
#endif

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <limits>

std::string to_lower(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return value;
}

std::string tchar_to_utf8(const tchar* value)
{
    if (!value) return {};

#ifdef _WIN32
    const int wide_len = static_cast<int>(wcslen(value));
    const int utf8_len = WideCharToMultiByte(CP_UTF8, 0, value, wide_len, nullptr, 0, nullptr, nullptr);
    if (utf8_len <= 0) return {};

    std::string result(static_cast<size_t>(utf8_len), '\0');
    if (WideCharToMultiByte(CP_UTF8, 0, value, wide_len, &result[0], utf8_len, nullptr, nullptr) <= 0)
        return {};
    return result;
#else
    return value;
#endif
}

std::string trim_copy(const std::string& value)
{
    size_t start = 0;
    size_t end = value.size();
    while (start < end && std::isspace(static_cast<unsigned char>(value[start])))
        ++start;
    while (end > start && std::isspace(static_cast<unsigned char>(value[end - 1])))
        --end;
    return value.substr(start, end - start);
}

double elapsed_ms(std::chrono::steady_clock::time_point from, std::chrono::steady_clock::time_point to)
{
    return std::chrono::duration<double, std::milli>(to - from).count();
}

double bytes_to_mib(size_t bytes)
{
    return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

int tchar_casecmp(const tchar* lhs, const tchar* rhs)
{
#ifdef _WIN32
    return _wcsicmp(lhs, rhs);
#else
    return strcasecmp(lhs, rhs);
#endif
}

bool parse_float_arg(const std::string& value, float* result)
{
    char* end = nullptr;
    *result = strtof(value.c_str(), &end);
    return end != value.c_str() && end && *end == '\0';
}

bool parse_uint32_arg(const std::string& value, uint32_t* result)
{
    char* end = nullptr;
    unsigned long long parsed = strtoull(value.c_str(), &end, 10);
    if (end == value.c_str() || !end || *end != '\0' || parsed > UINT32_MAX)
        return false;
    *result = static_cast<uint32_t>(parsed);
    return true;
}

bool parse_int_arg(const std::string& value, int* result)
{
    char* end = nullptr;
    long long parsed = strtoll(value.c_str(), &end, 10);
    if (end == value.c_str() || !end || *end != '\0')
        return false;
    if (parsed < static_cast<long long>(std::numeric_limits<int>::min())
        || parsed > static_cast<long long>(std::numeric_limits<int>::max()))
        return false;
    *result = static_cast<int>(parsed);
    return true;
}

bool parse_uint16_port(const std::string& value, uint16_t* result)
{
    uint32_t parsed = 0;
    if (!parse_uint32_arg(value, &parsed) || parsed == 0 || parsed > 65535)
        return false;
    *result = static_cast<uint16_t>(parsed);
    return true;
}

std::string get_local_timestamp_ms()
{
    const auto now = std::chrono::system_clock::now();
    const auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    const auto t = std::chrono::system_clock::to_time_t(now);

    std::tm tm_local = {};
#ifdef _WIN32
    localtime_s(&tm_local, &t);
#else
    localtime_r(&t, &tm_local);
#endif

    char date_buf[32] = {};
    strftime(date_buf, sizeof(date_buf), "%Y-%m-%d %H:%M:%S", &tm_local);

    char final_buf[48] = {};
    snprintf(final_buf, sizeof(final_buf), "%s.%03d", date_buf, static_cast<int>(millis.count()));
    return final_buf;
}

#ifdef _WIN32
void setup_console_utf8()
{
    SetConsoleOutputCP(CP_UTF8);
}
#endif

std::ifstream open_ifstream_utf8(const char* path, std::ios::openmode mode)
{
#ifdef _WIN32
    return std::ifstream(utf8_to_wstr(path).c_str(), mode);
#else
    return std::ifstream(path, mode);
#endif
}
