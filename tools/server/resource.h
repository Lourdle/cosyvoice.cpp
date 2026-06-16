#pragma once
#include <stddef.h>

/* Resource identifiers */
#define IDR_WEBUI_HTML  100

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Load an embedded resource by integer ID.
 * Returns a read-only pointer to the resource data and sets out_size.
 * Returns NULL on failure.
 */
const void* server_resource_load(unsigned int id, size_t* out_size);

#ifdef __cplusplus
}
#endif
