module;
#include <nlohmann/json.hpp>

export module nlohmann_json;

namespace nlohmann
{
export using json = basic_json<>;
}
