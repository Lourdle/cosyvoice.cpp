#include "tool_common.h"

#ifdef _WIN32
    #define NOMINMAX
    #include <Windows.h>
    #include <shellapi.h>
#else
    #include <strings.h>
#endif

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cctype>
#include <ctime>
#include <format>
#include <memory>

std::string to_lower(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return value;
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

int str_casecmp(const char* lhs, const char* rhs)
{
#ifdef _WIN32
    return _stricmp(lhs, rhs);
#else
    return strcasecmp(lhs, rhs);
#endif
}

bool parse_float_arg(std::string_view value, float* result)
{
    const auto [ptr, ec] = std::from_chars(value.data(), value.data() + value.size(), *result);
    return ec == std::errc{} && ptr == value.data() + value.size();
}

bool parse_uint32_arg(std::string_view value, uint32_t* result)
{
    const auto [ptr, ec] = std::from_chars(value.data(), value.data() + value.size(), *result);
    return ec == std::errc{} && ptr == value.data() + value.size();
}

bool parse_int_arg(std::string_view value, int* result)
{
    const auto [ptr, ec] = std::from_chars(value.data(), value.data() + value.size(), *result);
    return ec == std::errc{} && ptr == value.data() + value.size();
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

    return std::format("{:04}-{:02}-{:02} {:02}:{:02}:{:02}.{:03}",
        tm_local.tm_year + 1900, tm_local.tm_mon + 1, tm_local.tm_mday,
        tm_local.tm_hour, tm_local.tm_min, tm_local.tm_sec,
        static_cast<int>(millis.count()));
}

#ifdef _WIN32
int main(int argc, char**)
{
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);

    std::unique_ptr<char[]> block;
    {
        struct local_deleter_t { void operator()(wchar_t** p) { LocalFree(p); } };
        std::unique_ptr<wchar_t*, local_deleter_t> wargv(CommandLineToArgvW(GetCommandLineW(), &argc));
        if (!wargv)
        {
            fprintf(stderr, "Error: CommandLineToArgvW failed (%lu).\n", GetLastError());
            return 1;
        }

        const size_t argv_bytes = (static_cast<size_t>(argc) + 1) * sizeof(char*);
        size_t text_bytes = 0;
        for (int i = 0; i < argc; ++i)
        {
            const int len = WideCharToMultiByte(CP_UTF8, 0, wargv.get()[i], -1, nullptr, 0, nullptr, nullptr);
            if (len <= 0)
            {
                fprintf(stderr, "Error: WideCharToMultiByte failed measuring argument %d (%lu).\n", i, GetLastError());
                return 1;
            }
            text_bytes += static_cast<size_t>(len);
        }

        block.reset(new char[argv_bytes + text_bytes]);
        char** utf8_argv = reinterpret_cast<char**>(block.get());
        char* text = block.get() + argv_bytes;
        char* const text_end = block.get() + argv_bytes + text_bytes;
        for (int i = 0; i < argc; ++i)
        {
            utf8_argv[i] = text;
            const int len = WideCharToMultiByte(CP_UTF8, 0, wargv.get()[i], -1, text, static_cast<int>(text_end - text), nullptr, nullptr);
            if (len <= 0)
            {
                fprintf(stderr, "Error: WideCharToMultiByte failed converting argument %d (%lu).\n", i, GetLastError());
                return 1;
            }
            text += len;
        }
        utf8_argv[argc] = nullptr;
    }

    return tool_entry(argc, reinterpret_cast<char**>(block.get()));
}
#endif
