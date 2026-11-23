#include "cosyvoice-internal.h"
#include "cosyvoice-memory.h"

#include <atomic>
#include <cctype>
#include <charconv>
#include <cstdint>
#include <cstdlib>
#include <format>
#include <limits>
#include <memory_resource>
#include <mutex>
#include <new>
#include <string>
#include <string_view>
#include <system_error>

#include <ggml.h>

namespace
{
    constexpr std::size_t kMaxPoolBlock = 32 * 1024;
    constexpr std::size_t kArenaAlignment = 64 * 1024;
    constexpr std::size_t kPoolChunkBlocks = 8;
    constexpr std::size_t kMaxPoolAlignment = alignof(std::max_align_t);
    constexpr const char* kPoolBytesEnv = "COSYVOICE_MEMORY_POOL_BYTES";

    std::atomic<std::pmr::memory_resource*> g_pool{nullptr};
    std::atomic<bool> g_small_allocated{false};
    std::mutex g_init_mutex;

    inline
    void log_line(ggml_log_level level, const std::string& line)
    {
        cosyvoice_call_ggml_log_callback(level, line.c_str());
    }

    bool parse_pool_bytes(const char* text, uint64_t& out)
    {
        if (!text)
            return false;

        std::string_view value(text);
        while (!value.empty() && std::isspace(static_cast<unsigned char>(value.front())))
            value.remove_prefix(1);
        while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back())))
            value.remove_suffix(1);
        if (value.empty())
            return false;

        uint64_t parsed = 0;
        const auto [ptr, ec] = std::from_chars(value.data(), value.data() + value.size(), parsed);
        if (ec != std::errc() || ptr != value.data() + value.size())
            return false;

        out = parsed;
        return true;
    }

    constexpr std::size_t kRuntimeParamsSizes[] = { sizeof(cosyvoice_runtime_params_t) };

    bool known_runtime_params_size(std::size_t size)
    {
        for (const auto known : kRuntimeParamsSizes)
            if (size == known)
                return true;
        return false;
    }

    void* system_allocate(std::size_t bytes, std::size_t alignment)
    {
        if (alignment > kMaxPoolAlignment)
            return ::operator new(bytes, std::align_val_t(alignment));
        return ::operator new(bytes);
    }

    void system_deallocate(void* ptr, std::size_t alignment) noexcept
    {
        if (alignment > kMaxPoolAlignment)
            ::operator delete(ptr, std::align_val_t(alignment));
        else
            ::operator delete(ptr);
    }

    bool enable_pool(uint64_t requested_bytes)
    {
        if (requested_bytes > (std::numeric_limits<std::size_t>::max)() - (kArenaAlignment - 1))
        {
            log_line(GGML_LOG_LEVEL_WARN, std::format(
                "memory pool: ignoring request for {} bytes: value is too large.\n", requested_bytes));
            return false;
        }

        const std::size_t arena_bytes = (static_cast<std::size_t>(requested_bytes) + kArenaAlignment - 1)
            & ~(kArenaAlignment - 1);

        void* storage = nullptr;
        std::pmr::monotonic_buffer_resource* arena = nullptr;
        std::pmr::synchronized_pool_resource* pool = nullptr;
        try
        {
            storage = ::operator new(arena_bytes, std::align_val_t(kArenaAlignment));

            arena = new std::pmr::monotonic_buffer_resource(
                storage, arena_bytes, std::pmr::new_delete_resource());

            std::pmr::pool_options options{};
            options.max_blocks_per_chunk = kPoolChunkBlocks;
            options.largest_required_pool_block = kMaxPoolBlock;
            pool = new std::pmr::synchronized_pool_resource(options, arena);

            log_line(GGML_LOG_LEVEL_INFO, std::format(
                "memory pool: enabled with a {:.1f} MiB arena (pooled blocks up to {} KiB).\n",
                static_cast<double>(arena_bytes) / 1048576.0, kMaxPoolBlock / 1024));

            g_pool.store(pool, std::memory_order_release);
            return true;
        }
        catch (const std::bad_alloc&)
        {
            delete pool;
            delete arena;
            if (storage)
                ::operator delete(storage, std::align_val_t(kArenaAlignment));
            log_line(GGML_LOG_LEVEL_WARN, std::format(
                "memory pool: failed to reserve {:.1f} MiB.\n",
                static_cast<double>(arena_bytes) / 1048576.0));
            return false;
        }
    }

    bool init_runtime_from_env()
    {
        std::lock_guard lock(g_init_mutex);

        const char* env = std::getenv(kPoolBytesEnv);
        if (!env) return false;

        if (g_small_allocated.load(std::memory_order_relaxed))
            return false;

        uint64_t bytes = 0;
        if (!parse_pool_bytes(env, bytes))
        {
            log_line(GGML_LOG_LEVEL_WARN,
                std::format("ignoring invalid {}=\"{}\".\n", kPoolBytesEnv, env));
            return false;
        }

        if (bytes == 0)
        {
            log_line(GGML_LOG_LEVEL_INFO, "memory pool: disabled by COSYVOICE_MEMORY_POOL_BYTES=0.\n");
            return true;
        }

        return enable_pool(bytes);
    }

    bool g_initialized = init_runtime_from_env();
}

void* cosyvoice::memory_allocate(std::size_t bytes, std::size_t alignment)
{
    if (bytes == 0) bytes = 1;

    if (bytes > kMaxPoolBlock || alignment > kMaxPoolAlignment)
        return system_allocate(bytes, alignment);

    auto* pool = g_pool.load(std::memory_order_acquire);
    if (!pool)
    {
        g_small_allocated.store(true, std::memory_order_relaxed);
        return system_allocate(bytes, alignment);
    }

    return pool->allocate(bytes, alignment);
}

void cosyvoice::memory_deallocate(void* ptr, std::size_t bytes, std::size_t alignment) noexcept
{
    if (!ptr) return;

    if (bytes == 0) bytes = 1;

    if (bytes > kMaxPoolBlock || alignment > kMaxPoolAlignment)
    {
        system_deallocate(ptr, alignment);
        return;
    }

    auto* pool = g_pool.load(std::memory_order_acquire);
    if (pool)
        pool->deallocate(ptr, bytes, alignment);
    else
        system_deallocate(ptr, alignment);
}

bool cosyvoice_init_runtime(const cosyvoice_runtime_params_t* params, size_t params_size)
{
    std::lock_guard lock(g_init_mutex);

    if (params && !known_runtime_params_size(params_size))
    {
        log_line(GGML_LOG_LEVEL_WARN, std::format(
            "memory pool: ignoring cosyvoice_init_runtime: unknown params struct size {}.\n",
            params_size));
        return false;
    }

    if (g_initialized)
    {
        log_line(GGML_LOG_LEVEL_WARN, "memory pool: runtime is already initialized.\n");
        return false;
    }

    if (g_small_allocated.load(std::memory_order_relaxed))
    {
        log_line(GGML_LOG_LEVEL_WARN,
            "memory pool: cannot enable after pooled containers have already allocated; "
            "call cosyvoice_init_runtime() before loading models or tokenizers.\n");
        g_initialized = true;
        return false;
    }

    uint64_t bytes = 0;
    if (params)
        bytes = params->memory_pool_bytes;
    else if (const char* env = std::getenv(kPoolBytesEnv))
        if (!parse_pool_bytes(env, bytes))
        {
            log_line(GGML_LOG_LEVEL_WARN,
                std::format("ignoring invalid {}=\"{}\".\n", kPoolBytesEnv, env));
            g_initialized = true;
            return false;
        }

    g_initialized = true;

    if (bytes == 0)
    {
        log_line(GGML_LOG_LEVEL_INFO, "memory pool: disabled.\n");
        return true;
    }

    return enable_pool(bytes);
}
