// Runtime SIMD detection & control C API (cosyvoice.h "SIMD Detection &
// Control API" section). x86-64 only: on non-x86 targets the SIMD tier is
// fixed at compile time (SIMDe/NEON emulation above the scalar fallback, or
// scalar-only), so CMake drops this TU from the library and the
// COSYVOICE_SIMD_CONTROL_SUPPORTED feature macro is undefined there; the
// internal arch guard below is kept as defense-in-depth. Consumers must guard
// calls with the feature macro. On x86 this TU is compiled in every
// configuration: under COSYVOICE_NO_SIMD simd_detect.cpp is skipped (no
// g_simd_hw_caps/g_simd_caps to consult), the dispatch is scalar-only, and
// the API reports the static picture (set only accepts AUTO/SCALAR, both of
// which already mean scalar).

#include "cosyvoice-internal.h"
#include "simd-dispatch.h"

// Keep the two enums frozen together; the dispatch chains the internal one.
static_assert(static_cast<uint32_t>(simd_level::auto_)       == COSYVOICE_SIMD_LEVEL_AUTO);
static_assert(static_cast<uint32_t>(simd_level::scalar)      == COSYVOICE_SIMD_LEVEL_SCALAR);
static_assert(static_cast<uint32_t>(simd_level::sse42)       == COSYVOICE_SIMD_LEVEL_SSE42);
static_assert(static_cast<uint32_t>(simd_level::avx)         == COSYVOICE_SIMD_LEVEL_AVX);
static_assert(static_cast<uint32_t>(simd_level::avx2)        == COSYVOICE_SIMD_LEVEL_AVX2);
static_assert(static_cast<uint32_t>(simd_level::avx10_1_256) == COSYVOICE_SIMD_LEVEL_AVX10_1_256);
static_assert(static_cast<uint32_t>(simd_level::avx512)      == COSYVOICE_SIMD_LEVEL_AVX512);

namespace
{
    // Tiers compiled into this build, as COSYVOICE_SIMD_CAP_* bits. The
    // presence macros follow CMake's effective-class semantics: a disabled
    // lower class cascades the higher ones off before the definitions are
    // applied, so a macro defined here means the tier object really exists.
    uint32_t simd_built_caps()
    {
        uint32_t caps = 0;
#if defined(COSYVOICE_NO_SIMD)
        return caps;
#else
        #if defined(COSYVOICE_HAS_SSE42)
        caps |= COSYVOICE_SIMD_CAP_SSE42;
        #endif
        #if defined(COSYVOICE_HAS_AVX)
        caps |= COSYVOICE_SIMD_CAP_AVX;
        #endif
        #if defined(COSYVOICE_HAS_AVX2)
        caps |= COSYVOICE_SIMD_CAP_AVX2;
        caps |= COSYVOICE_SIMD_CAP_FMA3;
        #endif
        #if defined(COSYVOICE_HAS_AVX512)
        caps |= COSYVOICE_SIMD_CAP_AVX512;
        caps |= COSYVOICE_SIMD_CAP_AVX10_1_512;
        #endif
        #if defined(COSYVOICE_HAS_AVX10_1_256)
        caps |= COSYVOICE_SIMD_CAP_AVX10_1_256;
        caps |= COSYVOICE_SIMD_CAP_FMA3;
        #endif
        return caps;
#endif
    }

    uint32_t to_mask(const simd_caps& caps)
    {
        uint32_t mask = 0;
        if (caps.sse42)       mask |= COSYVOICE_SIMD_CAP_SSE42;
        if (caps.avx)         mask |= COSYVOICE_SIMD_CAP_AVX;
        if (caps.fma3)        mask |= COSYVOICE_SIMD_CAP_FMA3;
        if (caps.avx2)        mask |= COSYVOICE_SIMD_CAP_AVX2;
        if (caps.avx512)      mask |= COSYVOICE_SIMD_CAP_AVX512;
        if (caps.avx10_1_256) mask |= COSYVOICE_SIMD_CAP_AVX10_1_256;
        if (caps.avx10_1_512) mask |= COSYVOICE_SIMD_CAP_AVX10_1_512;
        return mask;
    }
}

void cosyvoice_get_simd_info(cosyvoice_simd_info_t* info)
{
    if (!info)
        return;

    info->supported = 0;
    info->current   = 0;
    info->scalar_only = false;
#if !defined(COSYVOICE_NO_SIMD)
    info->supported = to_mask(g_simd_hw_caps);
    info->built = simd_built_caps();
    // AND with built so a hardware capability whose tier was compiled out is
    // not reported as "current" (dispatch could never select it).
    info->current = to_mask(g_simd_caps.load(std::memory_order_relaxed)) & info->built;
#else
    info->scalar_only = true;
    info->built = simd_built_caps();
#endif
    info->level = cosyvoice_get_simd_level();
}

cosyvoice_simd_level_t cosyvoice_get_simd_level(void)
{
#if !defined(COSYVOICE_NO_SIMD)
    return static_cast<cosyvoice_simd_level_t>(
        g_simd_level.load(std::memory_order_relaxed));
#else
    // Scalar-only build: the level cannot change anything, report the default.
    return COSYVOICE_SIMD_LEVEL_AUTO;
#endif
}

bool cosyvoice_set_simd_level(cosyvoice_simd_level_t level)
{
    if (level >= COSYVOICE_SIMD_LEVEL_COUNT)
        return false;

#if !defined(COSYVOICE_NO_SIMD)
    // Capped-at-best semantics: a level above what the CPU supports or the
    // build includes is harmless -- g_simd_caps simply keeps the best
    // available tier at or below it (simd_caps_for_level & hardware). The
    // relaxed store is paired with the relaxed load in simd_dispatch();
    // only the selected caps value is communicated, no ordering with other
    // memory is required (it is a process-global knob).
    g_simd_caps.store(
        simd_caps_for_level(g_simd_hw_caps, static_cast<simd_level>(level)),
        std::memory_order_relaxed);
    g_simd_level.store(static_cast<uint32_t>(level), std::memory_order_relaxed);
    return true;
#else
    // Scalar-only build: every tier is compiled out, so only the two settings
    // that mean "scalar" can be honored.
    return level == COSYVOICE_SIMD_LEVEL_AUTO || level == COSYVOICE_SIMD_LEVEL_SCALAR;
#endif
}
