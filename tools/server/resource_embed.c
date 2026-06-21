#include "resource.h"

static const unsigned char webui_html[] = {
#embed "webui/cosyvoice-webui.html"
};
static const size_t webui_html_size = sizeof(webui_html);

static const unsigned char webui_css[] = {
#embed "webui/cosyvoice-webui.css"
};
static const size_t webui_css_size = sizeof(webui_css);

static const unsigned char webui_js[] = {
#embed "webui/cosyvoice-webui.js"
};
static const size_t webui_js_size = sizeof(webui_js);

static const unsigned char webui_login_html[] = {
#embed "webui/cosyvoice-login.html"
};
static const size_t webui_login_html_size = sizeof(webui_login_html);

const void* server_resource_load(unsigned int id, size_t* out_size)
{
    switch (id)
    {
    case IDR_WEBUI_HTML:
        *out_size = webui_html_size;
        return webui_html;
    case IDR_WEBUI_CSS:
        *out_size = webui_css_size;
        return webui_css;
    case IDR_WEBUI_JS:
        *out_size = webui_js_size;
        return webui_js;
    case IDR_WEBUI_LOGIN_HTML:
        *out_size = webui_login_html_size;
        return webui_login_html;
    default:
        *out_size = 0;
        return NULL;
    }
}
