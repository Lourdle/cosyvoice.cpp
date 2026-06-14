#include "resource.h"

static const unsigned char webui_html[] = {
#embed "webui/cosyvoice-webui.html"
};
static const size_t webui_html_size = sizeof(webui_html);

const void* server_resource_load(unsigned int id, size_t* out_size)
{
    if (id == IDR_WEBUI_HTML)
    {
        *out_size = webui_html_size;
        return webui_html;
    }
    *out_size = 0;
    return NULL;
}
