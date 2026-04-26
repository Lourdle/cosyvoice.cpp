#pragma once

#include <fstream>
#include <string>

#ifdef _WIN32
std::wstring utf8_to_wstr(const char* utf8);
#endif

std::ifstream open_ifstream_utf8(const char* path, std::ios::openmode mode = std::ios::binary);
std::ofstream open_ofstream_utf8(const char* path, std::ios::openmode mode = std::ios::binary | std::ios::trunc);
