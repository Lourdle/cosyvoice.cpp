#include "simd-dispatch.h"

#include <cctype>
#include <cstdlib>

#ifdef _DEBUG
    #include <format>
    #include "cosyvoice-internal.h"
#endif

#if defined(_MSC_VER)
    #include <intrin.h>
#else
    #include <cpuid.h>
#endif

inline
static simd_caps simd_detect()
{
    simd_caps caps;

#ifdef _DEBUG
    // Print the detected SIMD capabilities
    struct simd_caps_printer
    {
        ~simd_caps_printer()
        {
            cosyvoice_call_ggml_log_callback(
                GGML_LOG_LEVEL_INFO,
                std::format(
                    "Detected SIMD capabilities: SSE4.2={}, AVX={}, FMA3={}, AVX2={}, AVX-512={}, AVX10.1-256={}, AVX10.1-512={}\n",
                    bool(caps.sse42),
                    bool(caps.avx),
                    bool(caps.fma3),
                    bool(caps.avx2),
                    bool(caps.avx512),
                    bool(caps.avx10_1_256),
                    bool(caps.avx10_1_512)
                ).c_str()
            );
        }

        simd_caps& caps;
    };
    simd_caps_printer printer{ caps };
#endif

#if defined(_MSC_VER)
    int regs[4] = { 0, 0, 0, 0 };
    __cpuidex(regs, 1, 0);
    const unsigned ecx1 = static_cast<unsigned>(regs[2]);
#else
    unsigned eax = 0, ebx = 0, ecx = 0, edx = 0;
    if (!__get_cpuid(1, &eax, &ebx, &ecx, &edx))
        return caps;
    const unsigned ecx1 = ecx;
#endif

    caps.sse42 = (ecx1 & (1u << 19)) != 0;

    const bool osxsave = (ecx1 & (1u << 27)) != 0;
    const bool avx_bit = (ecx1 & (1u << 28)) != 0;
    if (!osxsave || !avx_bit)
        return caps;

#if defined(_MSC_VER)
    const unsigned xcr0 = static_cast<unsigned>(_xgetbv(0));
#else
    unsigned xcr0_lo = 0, xcr0_hi = 0;
    __asm__ volatile("xgetbv" : "=a"(xcr0_lo), "=d"(xcr0_hi) : "c"(0));
    const unsigned xcr0 = xcr0_lo;
 #endif
    if ((xcr0 & 0x6) != 0x6)
        return caps;

    caps.avx  = true;
    caps.fma3 = (ecx1 & (1u << 12)) != 0;

#if defined(_MSC_VER)
    __cpuidex(regs, 7, 0);
    const unsigned ebx7 = static_cast<unsigned>(regs[1]);
#else
    unsigned ebx7 = 0;
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx))
        ebx7 = ebx;
#endif
    const bool avx2_bit = (ebx7 & (1u << 5)) != 0;
    if (avx2_bit)
    {
        caps.avx2  = true;
        caps.fma3  = true; // no AVX2 hardware lacks FMA3; keeps the avx2 preset self-consistent
    }
    const bool avx512f_bit  = (ebx7 & (1u << 16)) != 0;
    const bool avx512dq_bit = (ebx7 & (1u << 17)) != 0;
    const bool avx512bw_bit = (ebx7 & (1u << 30)) != 0;
    const bool avx512vl_bit = (ebx7 & (1u << 31)) != 0;
    // The 512 tier requires F+DQ+BW (16-bit tail masks lower to BW's kmovw;
    // sincos' _mm512_test_epi32_mask to DQ's vptestmd). VL is required too:
    // both MSVC and gcc/clang encode register spills/prologues in these
    // objects as EVEX vmovq/vmovdqa to xmm16+/ymm16+ (AVX512VL territory).
    // Parts with only some of these sub-sets (e.g. Knights Landing: F alone)
    // fall through to AVX2.
    if (avx512f_bit && avx512dq_bit && avx512bw_bit && avx512vl_bit && (xcr0 & 0xe6) == 0xe6) // OS must also enable opmask(bit5), ZMM_Hi256(bit6), Hi16_ZMM(bit7)
    {
        caps.avx512 = true;
        caps.fma3   = true;                    // AVX-512F includes FMA
    }

    // --- AVX10 (CPUID leaf 0x24) ---
    // Enumerated independently of the legacy AVX-512 bits above: AVX10-only
    // parts (Panther Lake onward) do not set them. Layout: EAX[7:0] =
    // version (1 = AVX10.1, 2 = AVX10.2), EBX bit16 = 256-bit support,
    // bit17 = 512-bit support. Opmask (XCR0 bit 5) is required for BOTH
    // widths (k-masked tails); the 512 width additionally needs the ZMM
    // state, exactly like legacy AVX-512. Version >= 1 gates both widths
    // (10.2 is a strict superset of 10.1 and no kernel uses a 10.2-only
    // instruction). avx10_1_256 drives its own tier; the 512 bit is an
    // observation flag that the dispatch ORs onto the AVX-512 tier kernels
    // (same instruction space -- see simd-dispatch.h).
    unsigned max_leaf = 0;
#if defined(_MSC_VER)
    {
        int r0[4];
        __cpuidex(r0, 0, 0);
        max_leaf = static_cast<unsigned>(r0[0]);
    }
#else
    {
        unsigned b = 0, c = 0, d = 0;
        __get_cpuid(0, &max_leaf, &b, &c, &d);
    }
#endif
    if (max_leaf >= 0x24)
    {
        unsigned e24a = 0;
        unsigned e24b = 0;
#if defined(_MSC_VER)
        {
            int r24[4];
            __cpuidex(r24, 0x24, 0);
            e24a = static_cast<unsigned>(r24[0]);
            e24b = static_cast<unsigned>(r24[1]);
        }
#else
        {
            unsigned a = 0, b = 0, c = 0, d = 0;
            if (__get_cpuid_count(0x24, 0, &a, &b, &c, &d))
            {
                e24a = a;
                e24b = b;
            }
        }
#endif
        const unsigned avx10_ver = e24a & 0xffu;
        if (avx10_ver >= 1)
        {
            if ((e24b & (1u << 16)) != 0 && (xcr0 & 0x26) == 0x26) // opmask+YMM(+XMM)
                caps.avx10_1_256 = true;
            if ((e24b & (1u << 17)) != 0 && (xcr0 & 0xe6) == 0xe6) // + ZMM state
                caps.avx10_1_512 = true;
        }
    }

    return caps;
}

const simd_caps g_simd_hw_caps = simd_detect();
std::atomic<simd_caps> g_simd_caps{ g_simd_hw_caps };

// --- COSYVOICE_SIMD_LEVEL environment variable ---
// Read once, at library load. Everything below is defined in the SAME TU as
// the detection result, so dynamic-initialization order is definitional:
// g_simd_hw_caps is fully initialized before the env override is applied
// (no cross-TU static-init hazard). The canonical names/aliases must stay in
// sync with tools/common/tool_common_cosyvoice.h (parse_simd_level_arg).
namespace
{
    bool simd_streq_ci(const char* lhs, const char* rhs)
    {
        for (; *lhs && *rhs; ++lhs, ++rhs)
        {
            if (std::tolower(static_cast<unsigned char>(*lhs)) !=
                std::tolower(static_cast<unsigned char>(*rhs)))
                return false;
        }
        return *lhs == *rhs;
    }

    bool simd_level_from_name(const char* name, simd_level* level)
    {
        if (!name || !*name)
            return false;
        if (simd_streq_ci(name, "auto"))
            *level = simd_level::auto_;
        else if (simd_streq_ci(name, "scalar") || simd_streq_ci(name, "none"))
            *level = simd_level::scalar;
        else if (simd_streq_ci(name, "sse42") || simd_streq_ci(name, "sse4.2"))
            *level = simd_level::sse42;
        else if (simd_streq_ci(name, "avx"))
            *level = simd_level::avx;
        else if (simd_streq_ci(name, "avx2"))
            *level = simd_level::avx2;
        else if (simd_streq_ci(name, "avx10-256") || simd_streq_ci(name, "avx10_256") ||
                 simd_streq_ci(name, "avx10.1-256") || simd_streq_ci(name, "avx10_1_256"))
            *level = simd_level::avx10_1_256;
        else if (simd_streq_ci(name, "avx512") || simd_streq_ci(name, "avx-512"))
            *level = simd_level::avx512;
        else
            return false;
        return true;
    }

    // Invalid values are ignored silently: the ggml log callback's global
    // state may not be usable yet at library-load time.
    struct simd_env_initializer
    {
        simd_env_initializer()
        {
            const char* env = std::getenv("COSYVOICE_SIMD_LEVEL");
            simd_level level;
            if (env && simd_level_from_name(env, &level))
            {
                g_simd_caps.store(simd_caps_for_level(g_simd_hw_caps, level), std::memory_order_relaxed);
                g_simd_level.store(static_cast<uint32_t>(level), std::memory_order_relaxed);
            }
        }
    };

    simd_env_initializer g_simd_env_initializer;
}
