// Precompiled header for cosyvoice-server (fallback when Make generator is used,
// since Make does not support C++20 module dependency scanning).
//
// The module interfaces (httplib.ixx, nlohmann-json.ixx) export types into the
// global namespace (e.g. Request, Response, Server without httplib:: prefix),
// so pch.h must replicate that via using-directives.
#pragma once

#include <httplib.h>
#include <nlohmann/json.hpp>

using namespace httplib;
