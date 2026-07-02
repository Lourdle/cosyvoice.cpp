#include "common.h"

#ifdef _WIN32
    #define NOMINMAX
    #include <Windows.h>
    #include <cstring>
#else
    #include <sys/mman.h>
    #include <sys/stat.h>
    #include <unistd.h>
    #include <fcntl.h>
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

#ifdef _WIN32
file_mmap::file_mmap(const char* pFileName) : hFileMapping(INVALID_HANDLE_VALUE), data_(nullptr),
hFile(CreateFileW(utf8_to_wstr(pFileName).c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr))
{
    if (hFile != INVALID_HANDLE_VALUE)
    {
        LARGE_INTEGER FileSize;
        if (GetFileSizeEx(hFile, &FileSize))
        {
            size_ = static_cast<size_t>(FileSize.QuadPart);
            hFileMapping = CreateFileMappingW(hFile, nullptr, PAGE_READONLY, 0, 0, nullptr);
            if (hFileMapping)
                data_ = MapViewOfFile(hFileMapping, FILE_MAP_READ, 0, 0, 0);
        }
    }
}

file_mmap::~file_mmap()
{
    if (data_)
        UnmapViewOfFile(data_);
    if (hFileMapping)
        CloseHandle(hFileMapping);
    if (hFile != INVALID_HANDLE_VALUE)
        CloseHandle(hFile);
}

file_mmap::operator bool() const
{
    return data_;
}

#else

file_mmap::file_mmap(const char* path) : fd(open(path, O_RDONLY)), data_(MAP_FAILED)
{
    if (fd >= 0)
    {
        struct stat st;
        if (fstat(fd, &st) == 0)
        {
            size_ = static_cast<size_t>(st.st_size);
            data_ = mmap(nullptr, size_, PROT_READ, MAP_SHARED, fd, 0);
        }
    }
}

file_mmap::~file_mmap()
{
    if (data_ != MAP_FAILED)
        munmap(data_, size_);
    if (fd >= 0)
        close(fd);
}

file_mmap::operator bool() const
{
    return data_ != MAP_FAILED;
}

#endif
