#pragma once

#include <fstream>
#include <string>

#ifdef _WIN32
	#define COSYVOICE_TEXT(x) L##x
using tchar = wchar_t;
std::wstring utf8_to_wstr(const char* utf8);
void setup_console_utf8();
#else
	#define COSYVOICE_TEXT(x) x
using tchar = char;
inline void setup_console_utf8() {}
#endif

int tchar_casecmp(const tchar* lhs, const tchar* rhs);
std::string tchar_to_utf8(const tchar* value);

std::ifstream open_ifstream_utf8(const char* path, std::ios::openmode mode = std::ios::binary);
std::ofstream open_ofstream_utf8(const char* path, std::ios::openmode mode = std::ios::binary | std::ios::trunc);

