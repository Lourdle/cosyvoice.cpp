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

class file_mmap
{
public:
    file_mmap(const char* filename);
    ~file_mmap();

    operator bool() const;
    const void* data() const { return data_; }
    size_t size() const { return size_; }
private:
    void* data_;
    size_t size_;

#ifdef _WIN32
    void* hFile;
    void* hFileMapping;
#else
    int fd;
#endif
};
