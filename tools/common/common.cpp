#include "tool_common.h"

#ifdef _WIN32
    #define NOMINMAX
    #include <Windows.h>
    #include <cwchar>
#else
    #include <strings.h>
#endif

#include <algorithm>
#include <cctype>
#include <cstdlib>
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
