#include "common.h"

#ifdef _WIN32
    #define NOMINMAX
    #include <Windows.h>
    #include <cwchar>
    #include <cstring>
#endif

#ifdef _WIN32
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

std::ofstream open_ofstream_utf8(const char* path, std::ios::openmode mode)
{
#ifdef _WIN32
    return std::ofstream(utf8_to_wstr(path).c_str(), mode);
#else
    return std::ofstream(path, mode);
#endif
}
