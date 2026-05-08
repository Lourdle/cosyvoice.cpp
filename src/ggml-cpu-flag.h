#include <ggml.h>

constexpr uint32_t GGML_TENSOR_FLAG_BACKEND_CPU = uint32_t(1) << 24;

inline
void ggml_set_cpu(ggml_tensor* tensor)
{
    tensor->flags |= GGML_TENSOR_FLAG_BACKEND_CPU;
}

inline
bool ggml_cpu_fallback(const ggml_tensor* tensor)
{
    return (tensor->flags & GGML_TENSOR_FLAG_BACKEND_CPU) != 0;
}
