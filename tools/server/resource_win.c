#define WIN32_LEAN_AND_MEAN
#include <Windows.h>

const void* server_resource_load(unsigned int id, size_t* out_size)
{
    HMODULE mod = GetModuleHandleW(NULL);
    if (!mod)
    {
        *out_size = 0;
        return NULL;
    }
    HRSRC res = FindResource(mod, MAKEINTRESOURCE(id), RT_RCDATA);
    if (!res)
    {
        *out_size = 0;
        return NULL;
    }
    HGLOBAL loaded = LoadResource(mod, res);
    if (!loaded)
    {
        *out_size = 0;
        return NULL;
    }
    *out_size = SizeofResource(mod, res);
    return LockResource(loaded);
}
