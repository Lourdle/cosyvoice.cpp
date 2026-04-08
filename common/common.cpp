#include "common.h"

#ifdef _WIN32
    #define NOMINMAX
    #include <Windows.h>
    #include <cwchar>
#else
    #include <strings.h>
#endif

int tchar_casecmp(const tchar* lhs, const tchar* rhs)
{
#ifdef _WIN32
    return _wcsicmp(lhs, rhs);
#else
    return strcasecmp(lhs, rhs);
#endif
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

#ifdef _WIN32
void setup_console_utf8()
{
    SetConsoleOutputCP(CP_UTF8);
}

std::wstring utf8_to_wstr(const char* utf8)
{
    int len = static_cast<int>(strlen(utf8));
    const int wide_len = MultiByteToWideChar(CP_UTF8, 0, utf8, len, nullptr, 0);
    if (wide_len <= 0)
        return std::wstring(utf8, utf8 + len);

    std::wstring wide(static_cast<size_t>(wide_len), L'\0');
    if (MultiByteToWideChar(CP_UTF8, 0, utf8, len, &wide[0], wide_len) <= 0)
        return std::wstring(utf8, utf8 + len);

    return wide;
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

std::ofstream open_ofstream_utf8(const char* path, std::ios::openmode mode)
{
#ifdef _WIN32
    return std::ofstream(utf8_to_wstr(path).c_str(), mode);
#else
    return std::ofstream(path, mode);
#endif
}
