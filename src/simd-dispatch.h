#pragma once

// ---------------------------------------------------------------------------
// Pure SIMD dispatch: capability struct + presets + runtime tier selection.
//
// This header is intentionally free of intrinsic types and SIMD helpers (no
// <immintrin.h>, no __m*/*, no simd-math.h): caller TUs only need the dispatch
// logic and must stay unpolluted. All SIMD implementations live in tier-only
// headers -- simd-math.h (log/sincos) and simd-kernels-impl.h (the kernel
// bodies + their support helpers) -- included solely by the tier objects.
//
// Build-time SIMD inclusion knobs. On x86 each class is optional; on non-x86
// the SSE4.2+FMA class is emulated via SIMDe if present, and the build
// auto-falls back to COSYVOICE_NO_SIMD (scalar-only) when SIMDe is missing.
//   COSYVOICE_NO_SIMD        force scalar-only: no SIMD code at all.
//   COSYVOICE_HAS_SCALAR     compile the pure-scalar fallback tier (default ON).
//   COSYVOICE_HAS_SSE42/AVX/AVX2/AVX512  include that class (default ON).
//   COSYVOICE_HAS_AVX10_1_256  include the AVX10 256-bit class (default ON,
//     gated by a compiler-feature probe; it shares the AVX 8-wide bodies and
//     replaces the 128-bit vector tail with a k-masked 256-bit one, so it is
//     the only AVX10 class that needs its own tier object). The AVX10 512-bit
//     capability is NOT a separate class: AVX10.1-512 parts run the AVX-512
//     tier's kernels -- AVX10 subsumes the AVX-512F/BW/DQ/VL instruction
//     space, and the dispatch ORs the two enumeration sources (legacy CPUID
//     leaf 7 bits or AVX10 leaf 0x24 bit 17) onto Kernel<simd_avx512>
//     (verified: on MSVC the tier objects compiled with /arch:AVX512 and
//     /arch:AVX10.1 /vlen=512 are byte-identical; clang's differ only in
//     VEX-vs-EVEX width choices inside the same ISA subset). AVX10.2 parts
//     need no extra tier either: 10.2 is a strict superset of 10.1 and no
//     kernel uses a 10.2-only instruction. Re-add a dedicated AVX10 tier only
//     once some kernel actually exploits such an instruction (FP16 arithmetic,
//     VFPCLASS, ...) -- for 512, widen the AVX-512 branch's preset instead.
// Disabling a lower class automatically disables every class that requires it.
// These macros are load-bearing: the kernel bodies gate every AVX/AVX-512
// intrinsic block with `#if defined(COSYVOICE_HAS_*)`, so a platform without
// the intrinsics never parses them (no <immintrin.h>/SIMDe declarations
// needed). Macro gates and the `if constexpr` conditions inside them share one
// consumer set and are written out explicitly: a shared lower-class block also
// IS the main loop / vector tail of the classes that reuse it, so both layers
// read e.g. `HAS_SSE42||HAS_AVX||HAS_AVX2` with
// `if constexpr (C.avx || C.avx10_1_256)`, the 512 block `HAS_AVX512` with
// `if constexpr (C.avx512)`. Every 512-class branch masks its own tail and returns
// (nothing below it is ever reached); the AVX10-256 class runs the 8-wide body,
// then its k-masked 256 tail, and the separate `if constexpr (C.sse42)`
// 128-bit tail is skipped by the preset's own false bit (the chain uses
// independent if-constexprs, not `else` -- an `else` would dangle when the
// sse42 block's macro gate is off).
// The AVX-512 tier requires F+BW+DQ+VL together (16-bit tail masks need BW's
// kmovw; sincos' _mm512_test_epi32_mask needs DQ's vptestmd; the compilers
// themselves emit EVEX xmm16+/ymm16+ spill forms that need VL). A part with
// only some of them (e.g. Knights Landing: F alone) falls through to AVX2.
// The dispatch chain tries the merged 512 case first (either enumeration
// source -> Kernel<simd_avx512>), then AVX10.1-256 (EVEX tail masks), then
// the legacy widths.
// ---------------------------------------------------------------------------

#include <atomic>
#include <cstdint>
#include <stdexcept>
#include <utility>

struct simd_caps
{
    bool sse42    :1 = false;
    bool avx      :1 = false;
    bool fma3     :1 = false;
    bool avx2     :1 = false;
    bool avx512   :1 = false;   // every AVX-512 sub-set this build requires
                                //  (F+BW+DQ + OS state saves) is usable; that
                                //  and only that enables the 512 tier.
    bool avx10_1_256 :1 = false; // AVX10.1 256-bit class (k-masked 256 tail);
                                 //  AVX10.2-256 parts dispatch here too
    bool avx10_1_512 :1 = false; // AVX10 512-bit capability (leaf 0x24 bit17,
                                 //  any version): a detection-side observation
                                 //  bit only -- the dispatch ORs it onto the
                                 //  AVX-512 tier's kernels; no preset/tier.
};

constexpr simd_caps simd_none       { .sse42=false, .avx=false, .fma3=false, .avx2=false, .avx512=false, .avx10_1_256=false, .avx10_1_512=false };
constexpr simd_caps simd_sse42      { .sse42=true,  .avx=false, .fma3=false, .avx2=false, .avx512=false, .avx10_1_256=false, .avx10_1_512=false };
constexpr simd_caps simd_sse42_fma  { .sse42=true,  .avx=false, .fma3=true,  .avx2=false, .avx512=false, .avx10_1_256=false, .avx10_1_512=false };
constexpr simd_caps simd_avx        { .sse42=true,  .avx=true,  .fma3=false, .avx2=false, .avx512=false, .avx10_1_256=false, .avx10_1_512=false };
constexpr simd_caps simd_avx2       { .sse42=true,  .avx=true,  .fma3=true,  .avx2=true,  .avx512=false, .avx10_1_256=false, .avx10_1_512=false };
constexpr simd_caps simd_avx512     { .sse42=true,  .avx=true,  .fma3=true,  .avx2=true,  .avx512=true,  .avx10_1_256=false, .avx10_1_512=false };
// The AVX10-256 preset: it shares the AVX 8-wide bodies but NOT by inheriting
// the legacy class' caps bits -- the value enumerates exactly its own class.
// The shared code paths widen their conditions explicitly, e.g.
// `if constexpr (C.avx || C.avx10_1_256)`; the class additionally replaces
// the 128-bit vector tail with a k-masked 256-bit one (then the independent
// `if constexpr (C.sse42)` tail is skipped by its own false bit). fma3 is the
// one legacy bit it carries: the parts really have FMA and the shared mul/add
// helpers key their codegen on it. An AVX10.2 part selects the same preset as
// 10.1 -- 10.2 adds no instruction any kernel uses.
constexpr simd_caps simd_avx10_1_256 { .sse42=false, .avx=false, .fma3=true,  .avx2=false, .avx512=false, .avx10_1_256=true, .avx10_1_512=false };

// Detection and runtime state are split: g_simd_hw_caps is the immutable
// CPUID/XCR0 detection result; g_simd_caps is the mutable tier selection the
// dispatch actually reads (initialized to the hardware truth; the C API
// setter and the COSYVOICE_SIMD_LEVEL env-var initializer store capped copies
// into it). Both are defined by simd_detect.cpp -- x86-64 only, and skipped
// entirely under COSYVOICE_NO_SIMD -- and are extern-declared under the same
// conditions so non-x86 callers never reference a missing symbol.
#if !defined(COSYVOICE_NO_SIMD) && (defined(__x86_64__) || defined(_M_X64))
extern const simd_caps g_simd_hw_caps;
extern std::atomic<simd_caps> g_simd_caps;
#endif

// Runtime tier capping. The level enum is shared everywhere (the API TU
// static_asserts it against cosyvoice_simd_level_t); the mutable state below
// exists only where the SIMD dispatch itself exists -- under COSYVOICE_NO_SIMD
// and on non-x86 targets the dispatch is static and nothing reads it. The cap
// is applied ONCE at set time (simd_caps_for_level, called only by the C API
// setter and the env-var initializer), so the dispatch hot path is a single
// relaxed atomic load of g_simd_caps with no masking overhead. g_simd_level
// is read-back-only metadata (a caps value alone cannot unambiguously report
// which level was requested) and never feeds the dispatch. The inline
// per-process instance is shared by all caller-ish TUs; the writers live in
// simd_detect.cpp and cosyvoice-simd.cpp.
// WARNING: the numeric values are part of the public C ABI -- they must stay
// identical to the cosyvoice_simd_level_t enum in include/cosyvoice.h
// (AUTO=0, SCALAR=1, SSE42=2, AVX=3, AVX2=4, AVX10_1_256=5, AVX512=6).
enum class simd_level : uint32_t
{
    auto_       = 0,
    scalar      = 1,
    sse42       = 2,
    avx         = 3,
    avx2        = 4,
    avx10_1_256 = 5,
    avx512      = 6,
};

// Apply a tier cap to a capability set: keeps the classes at or below the
// requested level, mirroring what each preset enumerates -- sse42 clears FMA
// too (the SSE4.2 tier does not use it), avx2 keeps FMA, and the AVX10-256
// level only removes the two 512-bit enumeration sources (the AVX-512 tier
// serves both). Called once per level change, never per dispatch.
inline simd_caps simd_caps_for_level(simd_caps caps, simd_level lvl)
{
    if (lvl == simd_level::auto_ || lvl >= simd_level::avx512)
        return caps;
    caps.avx512      = false;
    caps.avx10_1_512 = false;
    if (lvl < simd_level::avx10_1_256) caps.avx10_1_256 = false;
    if (lvl < simd_level::avx2)        { caps.avx2 = false; caps.fma3 = false; }
    if (lvl < simd_level::avx)         caps.avx = false;
    if (lvl < simd_level::sse42)       caps.sse42 = false;
    return caps;
}

#if !defined(COSYVOICE_NO_SIMD) && (defined(__x86_64__) || defined(_M_X64))
inline std::atomic<uint32_t> g_simd_level{ static_cast<uint32_t>(simd_level::auto_) };
#endif

template<template<simd_caps> class Kernel, typename... Args>
inline
auto simd_dispatch(Args&&... args)
{
#ifdef COSYVOICE_NO_SIMD
    return Kernel<simd_none>::run(std::forward<Args>(args)...);
#elif !defined(__x86_64__) && !defined(_M_X64)
    // non-x86: SSE4.2+FMA3 emulated via SIMDe/NEON (in the sse42 tier object).
    return Kernel<simd_sse42_fma>::run(std::forward<Args>(args)...);
#else
    // x86: dispatch by the current tier selection (detected hardware, capped
    // by the runtime SIMD level -- see simd_caps_for_level), most capable
    // first. No exact-equality matching: a class the build disabled is simply
    // never selected and the CPU falls through to the next lower tier that the
    // build did include. The 512 tier answers BOTH enumeration sources --
    // legacy AVX-512 (leaf 7 F+BW+DQ+VL)
    // and AVX10 512-bit (leaf 0x24 bit 17, any version) -- because AVX10.1+
    // subsumes that instruction space and the tier objects are equivalent
    // (MSVC: byte-identical). Checked before AVX10.1-256 so a 512-capable part
    // is never downgraded to the 256 tier.
    const simd_caps caps = g_simd_caps.load(std::memory_order_relaxed);
#if defined(COSYVOICE_HAS_AVX512)
    if (caps.avx512 || caps.avx10_1_512)
        return Kernel<simd_avx512>::run(std::forward<Args>(args)...);
#endif
#if defined(COSYVOICE_HAS_AVX10_1_256)
    if (caps.avx10_1_256)
        return Kernel<simd_avx10_1_256>::run(std::forward<Args>(args)...);
#endif
#if defined(COSYVOICE_HAS_AVX2)
    if (caps.avx2 && caps.fma3 && caps.sse42)
        return Kernel<simd_avx2>::run(std::forward<Args>(args)...);
#endif
#if defined(COSYVOICE_HAS_AVX)
    if (caps.avx && caps.sse42)
        return Kernel<simd_avx>::run(std::forward<Args>(args)...);
#endif
#if defined(COSYVOICE_HAS_SSE42)
    if (caps.sse42)
        return Kernel<simd_sse42>::run(std::forward<Args>(args)...);
#endif
#if defined(COSYVOICE_HAS_SCALAR)
    return Kernel<simd_none>::run(std::forward<Args>(args)...);
#endif
    throw std::runtime_error("no matching SIMD tier for this CPU");
#endif
}
