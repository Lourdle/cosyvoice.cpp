#include "resource.h"
#include <windows.h>

const void* server_resource_load(unsigned int id, size_t* out_size)
{
    HMODULE mod = GetModuleHandleW(NULL);
    if (!mod)
    {
        *out_size = 0;
        return NULL;
    }
    HRSRC res = FindResourceW(mod, MAKEINTRESOURCEW(id), RT_RCDATA);
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
