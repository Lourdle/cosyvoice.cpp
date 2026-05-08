#include "fft.h"
#include "ggml-fft.h"
#include "ggml-cpu-flag.h"

#include <math.h>
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#elif defined(__aarch64__) || defined(_M_ARM64)
#define SIMDE_ENABLE_NATIVE_ALIASES
#include <simde/x86/avx2.h>
#include <simde/x86/fma.h>
#else
#error "src/fft.cpp requires x86_64 SIMD intrinsics or SIMDe on ARM64; unsupported architecture"
#endif

#include <cstring>
#include <atomic>

struct fft_context
{
    fft_context(int nfft);
    ~fft_context() = default;

    int factors[63];
    int max_factor_value;
    int nfft;
    std::atomic_uint workers;
    std::atomic_int64_t idx;
    std::unique_ptr<float[]> twiddles_r;
    float* twiddles_i;
};

static void kf_bfly2(
    float* Foutr,
    float* Fouti,
    int          fstride,
    const float* tw_r,
    const float* tw_i,
    int          m)
{
    int i = 0;
    if (fstride == 1)
        for (; i + 7 < m; i += 8)
        {
            __m256 F2r = _mm256_loadu_ps(Foutr + m + i);
            __m256 F2i = _mm256_loadu_ps(Fouti + m + i);
            __m256 Twr = _mm256_loadu_ps(tw_r + i);
            __m256 Twi = _mm256_loadu_ps(tw_i + i);

            __m256 tr = _mm256_fmsub_ps(F2r, Twr, _mm256_mul_ps(F2i, Twi));
            __m256 ti = _mm256_fmadd_ps(F2r, Twi, _mm256_mul_ps(F2i, Twr));

            __m256 Fr = _mm256_loadu_ps(Foutr + i);
            __m256 Fi = _mm256_loadu_ps(Fouti + i);

            _mm256_storeu_ps(Foutr + m + i, _mm256_sub_ps(Fr, tr));
            _mm256_storeu_ps(Fouti + m + i, _mm256_sub_ps(Fi, ti));
            _mm256_storeu_ps(Foutr + i, _mm256_add_ps(Fr, tr));
            _mm256_storeu_ps(Fouti + i, _mm256_add_ps(Fi, ti));
        }
    else for (const auto vidx0 = _mm256_setr_epi32(
        0 * fstride, 1 * fstride, 2 * fstride, 3 * fstride,
        4 * fstride, 5 * fstride, 6 * fstride, 7 * fstride); i + 7 < m; i += 8)
    {
        __m256 F2r = _mm256_loadu_ps(Foutr + m + i);
        __m256 F2i = _mm256_loadu_ps(Fouti + m + i);
        __m256 Twr = _mm256_i32gather_ps(tw_r + i * fstride, vidx0, sizeof(float));
        __m256 Twi = _mm256_i32gather_ps(tw_i + i * fstride, vidx0, sizeof(float));

        __m256 tr = _mm256_fmsub_ps(F2r, Twr, _mm256_mul_ps(F2i, Twi));
        __m256 ti = _mm256_fmadd_ps(F2r, Twi, _mm256_mul_ps(F2i, Twr));

        __m256 Fr = _mm256_loadu_ps(Foutr + i);
        __m256 Fi = _mm256_loadu_ps(Fouti + i);

        _mm256_storeu_ps(Foutr + m + i, _mm256_sub_ps(Fr, tr));
        _mm256_storeu_ps(Fouti + m + i, _mm256_sub_ps(Fi, ti));
        _mm256_storeu_ps(Foutr + i, _mm256_add_ps(Fr, tr));
        _mm256_storeu_ps(Fouti + i, _mm256_add_ps(Fi, ti));
    }

    for (; i + 3 < m; i += 4)
    {
        __m128 F2r = _mm_loadu_ps(Foutr + m + i);
        __m128 F2i = _mm_loadu_ps(Fouti + m + i);
        __m128 Twr;
        __m128 Twi;
        if (fstride == 1)
        {
            Twr = _mm_loadu_ps(tw_r + i);
            Twi = _mm_loadu_ps(tw_i + i);
        }
        else
        {
            const __m128i vidx0_128 = _mm_setr_epi32(
                0 * fstride, 1 * fstride, 2 * fstride, 3 * fstride);
            Twr = _mm_i32gather_ps(tw_r + i * fstride, vidx0_128, sizeof(float));
            Twi = _mm_i32gather_ps(tw_i + i * fstride, vidx0_128, sizeof(float));
        }

        __m128 tr = _mm_fmsub_ps(F2r, Twr, _mm_mul_ps(F2i, Twi));
        __m128 ti = _mm_fmadd_ps(F2r, Twi, _mm_mul_ps(F2i, Twr));

        __m128 Fr = _mm_loadu_ps(Foutr + i);
        __m128 Fi = _mm_loadu_ps(Fouti + i);

        _mm_storeu_ps(Foutr + m + i, _mm_sub_ps(Fr, tr));
        _mm_storeu_ps(Fouti + m + i, _mm_sub_ps(Fi, ti));
        _mm_storeu_ps(Foutr + i, _mm_add_ps(Fr, tr));
        _mm_storeu_ps(Fouti + i, _mm_add_ps(Fi, ti));
    }

    for (; i < m; ++i)
    {
        const float twr = tw_r[i * fstride];
        const float twi = tw_i[i * fstride];

        float F2r = Foutr[m + i];
        float F2i = Fouti[m + i];

        float tr = F2r * twr - F2i * twi;
        float ti = F2r * twi + F2i * twr;

        float Fr = Foutr[i];
        float Fi = Fouti[i];

        Foutr[m + i] = Fr - tr;
        Fouti[m + i] = Fi - ti;
        Foutr[i] = Fr + tr;
        Fouti[i] = Fi + ti;
    }
}

static void kf_bfly4(
    float* Foutr,
    float* Fouti,
    int fstride,
    float* twr,
    float* twi,
    int m)
{
    const int m2 = 2 * m;
    const int m3 = 3 * m;

    int k = 0;
    for (const __m256i vidx0 = _mm256_setr_epi32(
        0, fstride, 2 * fstride, 3 * fstride,
        4 * fstride, 5 * fstride, 6 * fstride, 7 * fstride),
        vidx1 = _mm256_mullo_epi32(
            vidx0, _mm256_set1_epi32(2)),  /* 2 * fstride */
        vidx2 = _mm256_mullo_epi32(
            vidx0, _mm256_set1_epi32(3)); k + 7 < m; k += 8)
    {
        float* p0r = Foutr + k;
        float* p0i = Fouti + k;

        float* p1r = Foutr + k + m;
        float* p1i = Fouti + k + m;

        float* p2r = Foutr + k + m2;
        float* p2i = Fouti + k + m2;

        float* p3r = Foutr + k + m3;
        float* p3i = Fouti + k + m3;

        __m256 r0 = _mm256_loadu_ps(p0r);
        __m256 i0 = _mm256_loadu_ps(p0i);

        __m256 r1 = _mm256_loadu_ps(p1r);
        __m256 i1 = _mm256_loadu_ps(p1i);

        __m256 r2 = _mm256_loadu_ps(p2r);
        __m256 i2 = _mm256_loadu_ps(p2i);

        __m256 r3 = _mm256_loadu_ps(p3r);
        __m256 i3 = _mm256_loadu_ps(p3i);

        __m256 t1r, t1i, t2r, t2i, t3r, t3i;

        t1r = _mm256_i32gather_ps(twr + k * fstride, vidx0, 4);
        t1i = _mm256_i32gather_ps(twi + k * fstride, vidx0, 4);

        t2r = _mm256_i32gather_ps(twr + 2 * k * fstride,
            vidx1, 4);
        t2i = _mm256_i32gather_ps(twi + 2 * k * fstride,
            vidx1, 4);

        t3r = _mm256_i32gather_ps(twr + 3 * k * fstride,
            vidx2, 4);
        t3i = _mm256_i32gather_ps(twi + 3 * k * fstride,
            vidx2, 4);

        __m256 s0r = _mm256_fmsub_ps(r1, t1r, _mm256_mul_ps(i1, t1i));
        __m256 s0i = _mm256_fmadd_ps(r1, t1i, _mm256_mul_ps(i1, t1r));

        __m256 s1r = _mm256_fmsub_ps(r2, t2r, _mm256_mul_ps(i2, t2i));
        __m256 s1i = _mm256_fmadd_ps(r2, t2i, _mm256_mul_ps(i2, t2r));

        __m256 s2r = _mm256_fmsub_ps(r3, t3r, _mm256_mul_ps(i3, t3i));
        __m256 s2i = _mm256_fmadd_ps(r3, t3i, _mm256_mul_ps(i3, t3r));

        __m256 tmp_r = _mm256_sub_ps(r0, s1r);
        __m256 tmp_i = _mm256_sub_ps(i0, s1i);

        r0 = _mm256_add_ps(r0, s1r);
        i0 = _mm256_add_ps(i0, s1i);

        __m256 s3r = _mm256_add_ps(s0r, s2r);
        __m256 s3i = _mm256_add_ps(s0i, s2i);

        __m256 s4r = _mm256_sub_ps(s0r, s2r);
        __m256 s4i = _mm256_sub_ps(s0i, s2i);

        _mm256_storeu_ps(p2r, _mm256_sub_ps(r0, s3r));
        _mm256_storeu_ps(p2i, _mm256_sub_ps(i0, s3i));

        _mm256_storeu_ps(p0r, _mm256_add_ps(r0, s3r));
        _mm256_storeu_ps(p0i, _mm256_add_ps(i0, s3i));

        _mm256_storeu_ps(p1r, _mm256_add_ps(tmp_r, s4i));
        _mm256_storeu_ps(p1i, _mm256_sub_ps(tmp_i, s4r));

        _mm256_storeu_ps(p3r, _mm256_sub_ps(tmp_r, s4i));
        _mm256_storeu_ps(p3i, _mm256_add_ps(tmp_i, s4r));
    }

    for (const __m128i vidx0 = _mm_setr_epi32(
        0 * fstride, 1 * fstride, 2 * fstride, 3 * fstride),
        vidx1 = _mm_mullo_epi32(
            vidx0, _mm_set1_epi32(2)),  /* 2 * fstride */
        vidx2 = _mm_mullo_epi32(
            vidx0, _mm_set1_epi32(3))  /* 3 * fstride */; k + 3 < m; k += 4)
    {
        float* p0r = Foutr + k;
        float* p0i = Fouti + k;

        float* p1r = Foutr + k + m;
        float* p1i = Fouti + k + m;

        float* p2r = Foutr + k + m2;
        float* p2i = Fouti + k + m2;

        float* p3r = Foutr + k + m3;
        float* p3i = Fouti + k + m3;

        __m128 r0 = _mm_loadu_ps(p0r);
        __m128 i0 = _mm_loadu_ps(p0i);

        __m128 r1 = _mm_loadu_ps(p1r);
        __m128 i1 = _mm_loadu_ps(p1i);

        __m128 r2 = _mm_loadu_ps(p2r);
        __m128 i2 = _mm_loadu_ps(p2i);

        __m128 r3 = _mm_loadu_ps(p3r);
        __m128 i3 = _mm_loadu_ps(p3i);

        __m128 t1r, t1i, t2r, t2i, t3r, t3i;

        t1r = _mm_i32gather_ps(twr + k * fstride, vidx0, 4);
        t1i = _mm_i32gather_ps(twi + k * fstride, vidx0, 4);

        t2r = _mm_i32gather_ps(twr + 2 * k * fstride,
            vidx1, 4);
        t2i = _mm_i32gather_ps(twi + 2 * k * fstride,
            vidx1, 4);

        t3r = _mm_i32gather_ps(twr + 3 * k * fstride,
            vidx2, 4);
        t3i = _mm_i32gather_ps(twi + 3 * k * fstride,
            vidx2, 4);

        __m128 s0r = _mm_fmsub_ps(r1, t1r, _mm_mul_ps(i1, t1i));
        __m128 s0i = _mm_fmadd_ps(r1, t1i, _mm_mul_ps(i1, t1r));

        __m128 s1r = _mm_fmsub_ps(r2, t2r, _mm_mul_ps(i2, t2i));
        __m128 s1i = _mm_fmadd_ps(r2, t2i, _mm_mul_ps(i2, t2r));

        __m128 s2r = _mm_fmsub_ps(r3, t3r, _mm_mul_ps(i3, t3i));
        __m128 s2i = _mm_fmadd_ps(r3, t3i, _mm_mul_ps(i3, t3r));

        __m128 tmp_r = _mm_sub_ps(r0, s1r);
        __m128 tmp_i = _mm_sub_ps(i0, s1i);

        r0 = _mm_add_ps(r0, s1r);
        i0 = _mm_add_ps(i0, s1i);

        __m128 s3r = _mm_add_ps(s0r, s2r);
        __m128 s3i = _mm_add_ps(s0i, s2i);

        __m128 s4r = _mm_sub_ps(s0r, s2r);
        __m128 s4i = _mm_sub_ps(s0i, s2i);

        _mm_storeu_ps(p2r, _mm_sub_ps(r0, s3r));
        _mm_storeu_ps(p2i, _mm_sub_ps(i0, s3i));

        _mm_storeu_ps(p0r, _mm_add_ps(r0, s3r));
        _mm_storeu_ps(p0i, _mm_add_ps(i0, s3i));

        _mm_storeu_ps(p1r, _mm_add_ps(tmp_r, s4i));
        _mm_storeu_ps(p1i, _mm_sub_ps(tmp_i, s4r));

        _mm_storeu_ps(p3r, _mm_sub_ps(tmp_r, s4i));
        _mm_storeu_ps(p3i, _mm_add_ps(tmp_i, s4r));
    }

    for (Foutr += k, Fouti += k; k != m; ++k)
    {
        float scratchr[6];
        float scratchi[6];

        scratchr[0] = Foutr[m] * twr[k * fstride] - Fouti[m] * twi[k * fstride];
        scratchi[0] = Foutr[m] * twi[k * fstride] + Fouti[m] * twr[k * fstride];
        scratchr[1] = Foutr[m2] * twr[k * 2 * fstride] - Fouti[m2] * twi[k * 2 * fstride];
        scratchi[1] = Foutr[m2] * twi[k * 2 * fstride] + Fouti[m2] * twr[k * 2 * fstride];
        scratchr[2] = Foutr[m3] * twr[k * 3 * fstride] - Fouti[m3] * twi[k * 3 * fstride];
        scratchi[2] = Foutr[m3] * twi[k * 3 * fstride] + Fouti[m3] * twr[k * 3 * fstride];

        scratchr[5] = *Foutr - scratchr[1];
        scratchi[5] = *Fouti - scratchi[1];
        *Foutr += scratchr[1];
        *Fouti += scratchi[1];

        scratchr[3] = scratchr[0] + scratchr[2];
        scratchi[3] = scratchi[0] + scratchi[2];
        scratchr[4] = scratchr[0] - scratchr[2];
        scratchi[4] = scratchi[0] - scratchi[2];

        Foutr[m2] = *Foutr - scratchr[3];
        Fouti[m2] = *Fouti - scratchi[3];

        *Foutr += scratchr[3];
        *Fouti += scratchi[3];

        Foutr[m] = scratchr[5] + scratchi[4];
        Fouti[m] = scratchi[5] - scratchr[4];
        Foutr[m3] = scratchr[5] - scratchi[4];
        Fouti[m3] = scratchi[5] + scratchr[4];

        ++Foutr;
        ++Fouti;
    }
}

static void kf_bfly3(
    float* Foutr,
    float* Fouti,
    int    fstride,
    float* tw1_r,
    float* tw1_i,
    int    m)
{
    const int m2 = 2 * m;
    const float epi3_i = tw1_i[fstride * m];

    int k = 0;

    for (const __m256 v_half = _mm256_set1_ps(0.5f), v_epi3 = _mm256_set1_ps(epi3_i); k + 7 < m; k += 8)
    {
        __m256 fr0 = _mm256_loadu_ps(Foutr + k);
        __m256 fi0 = _mm256_loadu_ps(Fouti + k);
        __m256 fr1 = _mm256_loadu_ps(Foutr + m + k);
        __m256 fi1 = _mm256_loadu_ps(Fouti + m + k);
        __m256 fr2 = _mm256_loadu_ps(Foutr + m2 + k);
        __m256 fi2 = _mm256_loadu_ps(Fouti + m2 + k);

        __m256i idx1 = _mm256_setr_epi32(
            k * fstride, (k + 1) * fstride, (k + 2) * fstride, (k + 3) * fstride,
            (k + 4) * fstride, (k + 5) * fstride, (k + 6) * fstride, (k + 7) * fstride);
        __m256i idx2 = _mm256_mullo_epi32(idx1, _mm256_set1_epi32(2));

        __m256 tw1r = _mm256_i32gather_ps(tw1_r, idx1, 4);
        __m256 tw1i = _mm256_i32gather_ps(tw1_i, idx1, 4);
        __m256 tw2r = _mm256_i32gather_ps(tw1_r, idx2, 4);
        __m256 tw2i = _mm256_i32gather_ps(tw1_i, idx2, 4);

        __m256 sr1 = _mm256_fmsub_ps(fr1, tw1r, _mm256_mul_ps(fi1, tw1i));
        __m256 si1 = _mm256_fmadd_ps(fr1, tw1i, _mm256_mul_ps(fi1, tw1r));
        __m256 sr2 = _mm256_fmsub_ps(fr2, tw2r, _mm256_mul_ps(fi2, tw2i));
        __m256 si2 = _mm256_fmadd_ps(fr2, tw2i, _mm256_mul_ps(fi2, tw2r));

        __m256 sr3 = _mm256_add_ps(sr1, sr2);
        __m256 si3 = _mm256_add_ps(si1, si2);
        __m256 sr0 = _mm256_sub_ps(sr1, sr2);
        __m256 si0 = _mm256_sub_ps(si1, si2);

        __m256 fr1o = _mm256_sub_ps(fr0, _mm256_mul_ps(sr3, v_half));
        __m256 fi1o = _mm256_sub_ps(fi0, _mm256_mul_ps(si3, v_half));

        sr0 = _mm256_mul_ps(sr0, v_epi3);
        si0 = _mm256_mul_ps(si0, v_epi3);

        __m256 fr0o = _mm256_add_ps(fr0, sr3);
        __m256 fi0o = _mm256_add_ps(fi0, si3);

        __m256 fr2o = _mm256_add_ps(fr1o, si0);
        __m256 fi2o = _mm256_sub_ps(fi1o, sr0);

        fr1o = _mm256_sub_ps(fr1o, si0);
        fi1o = _mm256_add_ps(fi1o, sr0);

        _mm256_storeu_ps(Foutr + k, fr0o);
        _mm256_storeu_ps(Fouti + k, fi0o);
        _mm256_storeu_ps(Foutr + m + k, fr1o);
        _mm256_storeu_ps(Fouti + m + k, fi1o);
        _mm256_storeu_ps(Foutr + m2 + k, fr2o);
        _mm256_storeu_ps(Fouti + m2 + k, fi2o);
    }

    for (const __m128 v_half_128 = _mm_set_ps1(0.5f), v_epi3_128 = _mm_set_ps1(epi3_i);
        k + 3 < m; k += 4)
    {
        __m128 fr0 = _mm_loadu_ps(Foutr + k);
        __m128 fi0 = _mm_loadu_ps(Fouti + k);
        __m128 fr1 = _mm_loadu_ps(Foutr + m + k);
        __m128 fi1 = _mm_loadu_ps(Fouti + m + k);
        __m128 fr2 = _mm_loadu_ps(Foutr + m2 + k);
        __m128 fi2 = _mm_loadu_ps(Fouti + m2 + k);

        __m128 tw1r;
        __m128 tw1i;
        __m128 tw2r;
        __m128 tw2i;
        if (fstride == 1)
        {
            tw1r = _mm_loadu_ps(tw1_r + k);
            tw1i = _mm_loadu_ps(tw1_i + k);

            const int s2 = fstride * 2;
            __m128i idx2 = _mm_setr_epi32(
                k * s2, (k + 1) * s2, (k + 2) * s2, (k + 3) * s2);
            tw2r = _mm_i32gather_ps(tw1_r, idx2, 4);
            tw2i = _mm_i32gather_ps(tw1_i, idx2, 4);
        }
        else
        {
            __m128i idx1 = _mm_setr_epi32(
                k * fstride, (k + 1) * fstride, (k + 2) * fstride, (k + 3) * fstride);
            __m128i idx2 = _mm_mullo_epi32(idx1, _mm_set1_epi32(2));

            tw1r = _mm_i32gather_ps(tw1_r, idx1, 4);
            tw1i = _mm_i32gather_ps(tw1_i, idx1, 4);
            tw2r = _mm_i32gather_ps(tw1_r, idx2, 4);
            tw2i = _mm_i32gather_ps(tw1_i, idx2, 4);
        }

        __m128 sr1 = _mm_fmsub_ps(fr1, tw1r, _mm_mul_ps(fi1, tw1i));
        __m128 si1 = _mm_fmadd_ps(fr1, tw1i, _mm_mul_ps(fi1, tw1r));
        __m128 sr2 = _mm_fmsub_ps(fr2, tw2r, _mm_mul_ps(fi2, tw2i));
        __m128 si2 = _mm_fmadd_ps(fr2, tw2i, _mm_mul_ps(fi2, tw2r));

        __m128 sr3 = _mm_add_ps(sr1, sr2);
        __m128 si3 = _mm_add_ps(si1, si2);
        __m128 sr0 = _mm_sub_ps(sr1, sr2);
        __m128 si0 = _mm_sub_ps(si1, si2);

        __m128 fr1o = _mm_sub_ps(fr0, _mm_mul_ps(sr3, v_half_128));
        __m128 fi1o = _mm_sub_ps(fi0, _mm_mul_ps(si3, v_half_128));

        sr0 = _mm_mul_ps(sr0, v_epi3_128);
        si0 = _mm_mul_ps(si0, v_epi3_128);

        __m128 fr0o = _mm_add_ps(fr0, sr3);
        __m128 fi0o = _mm_add_ps(fi0, si3);

        __m128 fr2o = _mm_add_ps(fr1o, si0);
        __m128 fi2o = _mm_sub_ps(fi1o, sr0);

        fr1o = _mm_sub_ps(fr1o, si0);
        fi1o = _mm_add_ps(fi1o, sr0);

        _mm_storeu_ps(Foutr + k, fr0o);
        _mm_storeu_ps(Fouti + k, fi0o);
        _mm_storeu_ps(Foutr + m + k, fr1o);
        _mm_storeu_ps(Fouti + m + k, fi1o);
        _mm_storeu_ps(Foutr + m2 + k, fr2o);
        _mm_storeu_ps(Fouti + m2 + k, fi2o);
    }

    for (; k != m; ++k)
    {
        float tw1r = tw1_r[k * fstride];
        float tw1i = tw1_i[k * fstride];
        float tw2r = tw1_r[k * fstride * 2];
        float tw2i = tw1_i[k * fstride * 2];

        float fr0 = Foutr[k];
        float fi0 = Fouti[k];
        float fr1 = Foutr[m + k];
        float fi1 = Fouti[m + k];
        float fr2 = Foutr[m2 + k];
        float fi2 = Fouti[m2 + k];

        float sr1 = fr1 * tw1r - fi1 * tw1i;
        float si1 = fr1 * tw1i + fi1 * tw1r;
        float sr2 = fr2 * tw2r - fi2 * tw2i;
        float si2 = fr2 * tw2i + fi2 * tw2r;

        float sr3 = sr1 + sr2;
        float si3 = si1 + si2;
        float sr0 = sr1 - sr2;
        float si0 = si1 - si2;

        float fr1o = fr0 - sr3 * 0.5f;
        float fi1o = fi0 - si3 * 0.5f;

        sr0 *= epi3_i;
        si0 *= epi3_i;

        Foutr[k] = fr0 + sr3;
        Fouti[k] = fi0 + si3;
        Foutr[m + k] = fr1o - si0;
        Fouti[m + k] = fi1o + sr0;
        Foutr[m2 + k] = fr1o + si0;
        Fouti[m2 + k] = fi1o - sr0;
    }
}

inline
static void CMUL(__m256& ar, __m256& ai, __m256 br, __m256 bi)
{
    __m256 _tr = _mm256_fmsub_ps(ar, br, _mm256_mul_ps(ai, bi));
    __m256 _ti = _mm256_fmadd_ps(ar, bi, _mm256_mul_ps(ai, br));
    ar = _tr; ai = _ti;
}

inline
static void CMUL(__m128& ar, __m128& ai, __m128 br, __m128 bi)
{
    __m128 _tr = _mm_fmsub_ps(ar, br, _mm_mul_ps(ai, bi));
    __m128 _ti = _mm_fmadd_ps(ar, bi, _mm_mul_ps(ai, br));
    ar = _tr; ai = _ti;
}

static void kf_bfly5(
    float* Foutr, float* Fouti,
    int fstride,
    const float* tw_r, const float* tw_i,
    int m)
{
    int u = 0;

    float yar = tw_r[fstride * m];
    float yai = tw_i[fstride * m];
    float ybr = tw_r[fstride * 2 * m];
    float ybi = tw_i[fstride * 2 * m];

    float* F0r = Foutr, * F1r = Foutr + m, * F2r = Foutr + 2 * m,
        * F3r = Foutr + 3 * m, * F4r = Foutr + 4 * m;
    float* F0i = Fouti, * F1i = Fouti + m, * F2i = Fouti + 2 * m,
        * F3i = Fouti + 3 * m, * F4i = Fouti + 4 * m;

    for (const __m256 v_yar = _mm256_set1_ps(yar),
        v_yai = _mm256_set1_ps(yai),
        v_ybr = _mm256_set1_ps(ybr),
        v_ybi = _mm256_set1_ps(ybi); u + 7 < m; u += 8) {
        __m256i idx1 = _mm256_setr_epi32(
            u * fstride, (u + 1) * fstride, (u + 2) * fstride, (u + 3) * fstride,
            (u + 4) * fstride, (u + 5) * fstride, (u + 6) * fstride, (u + 7) * fstride
        );
        __m256 tw1r = _mm256_i32gather_ps(tw_r, idx1, 4);
        __m256 tw1i = _mm256_i32gather_ps(tw_i, idx1, 4);

        __m256i idx2 = _mm256_mullo_epi32(idx1, _mm256_set1_epi32(2));
        __m256 tw2r = _mm256_i32gather_ps(tw_r, idx2, 4);
        __m256 tw2i = _mm256_i32gather_ps(tw_i, idx2, 4);

        __m256i idx3 = _mm256_mullo_epi32(idx1, _mm256_set1_epi32(3));
        __m256 tw3r = _mm256_i32gather_ps(tw_r, idx3, 4);
        __m256 tw3i = _mm256_i32gather_ps(tw_i, idx3, 4);

        __m256i idx4 = _mm256_mullo_epi32(idx1, _mm256_set1_epi32(4));
        __m256 tw4r = _mm256_i32gather_ps(tw_r, idx4, 4);
        __m256 tw4i = _mm256_i32gather_ps(tw_i, idx4, 4);

        __m256 f0r = _mm256_loadu_ps(F0r + u);
        __m256 f0i = _mm256_loadu_ps(F0i + u);
        __m256 f1r = _mm256_loadu_ps(F1r + u);
        __m256 f1i = _mm256_loadu_ps(F1i + u);
        __m256 f2r = _mm256_loadu_ps(F2r + u);
        __m256 f2i = _mm256_loadu_ps(F2i + u);
        __m256 f3r = _mm256_loadu_ps(F3r + u);
        __m256 f3i = _mm256_loadu_ps(F3i + u);
        __m256 f4r = _mm256_loadu_ps(F4r + u);
        __m256 f4i = _mm256_loadu_ps(F4i + u);

        CMUL(f1r, f1i, tw1r, tw1i);
        CMUL(f2r, f2i, tw2r, tw2i);
        CMUL(f3r, f3i, tw3r, tw3i);
        CMUL(f4r, f4i, tw4r, tw4i);

        __m256 s7r = _mm256_add_ps(f1r, f4r);
        __m256 s7i = _mm256_add_ps(f1i, f4i);
        __m256 s10r = _mm256_sub_ps(f1r, f4r);
        __m256 s10i = _mm256_sub_ps(f1i, f4i);
        __m256 s8r = _mm256_add_ps(f2r, f3r);
        __m256 s8i = _mm256_add_ps(f2i, f3i);
        __m256 s9r = _mm256_sub_ps(f2r, f3r);
        __m256 s9i = _mm256_sub_ps(f2i, f3i);

        __m256 s5r = _mm256_fmadd_ps(s7r, v_yar, f0r);
        s5r = _mm256_fmadd_ps(s8r, v_ybr, s5r);
        __m256 s5i = _mm256_fmadd_ps(s7i, v_yar, f0i);
        s5i = _mm256_fmadd_ps(s8i, v_ybr, s5i);

        __m256 s6r = _mm256_fmadd_ps(s10i, v_yai, _mm256_mul_ps(s9i, v_ybi));
        __m256 s6i = _mm256_fnmsub_ps(s10r, v_yai, _mm256_mul_ps(s9r, v_ybi));

        __m256 t1r = _mm256_sub_ps(s5r, s6r);
        __m256 t1i = _mm256_sub_ps(s5i, s6i);
        __m256 t4r = _mm256_add_ps(s5r, s6r);
        __m256 t4i = _mm256_add_ps(s5i, s6i);

        __m256 s11r = _mm256_fmadd_ps(s7r, v_ybr, f0r);
        s11r = _mm256_fmadd_ps(s8r, v_yar, s11r);
        __m256 s11i = _mm256_fmadd_ps(s7i, v_ybr, f0i);
        s11i = _mm256_fmadd_ps(s8i, v_yar, s11i);

        __m256 tmp = _mm256_add_ps(s7r, s8r);
        f0r = _mm256_add_ps(f0r, tmp);
        tmp = _mm256_add_ps(s7i, s8i);
        f0i = _mm256_add_ps(f0i, tmp);

        __m256 s12r = _mm256_fmsub_ps(s9i, v_yai, _mm256_mul_ps(s10i, v_ybi));
        __m256 s12i = _mm256_fmsub_ps(s10r, v_ybi, _mm256_mul_ps(s9r, v_yai));

        __m256 t2r = _mm256_add_ps(s11r, s12r);
        __m256 t2i = _mm256_add_ps(s11i, s12i);
        __m256 t3r = _mm256_sub_ps(s11r, s12r);
        __m256 t3i = _mm256_sub_ps(s11i, s12i);

        _mm256_storeu_ps(F0r + u, f0r);
        _mm256_storeu_ps(F0i + u, f0i);
        _mm256_storeu_ps(F1r + u, t1r);
        _mm256_storeu_ps(F1i + u, t1i);
        _mm256_storeu_ps(F2r + u, t2r);
        _mm256_storeu_ps(F2i + u, t2i);
        _mm256_storeu_ps(F3r + u, t3r);
        _mm256_storeu_ps(F3i + u, t3i);
        _mm256_storeu_ps(F4r + u, t4r);
        _mm256_storeu_ps(F4i + u, t4i);
    }

    for (const __m128 v_yar = _mm_set_ps1(yar),
        v_yai = _mm_set_ps1(yai),
        v_ybr = _mm_set_ps1(ybr),
        v_ybi = _mm_set_ps1(ybi); u + 3 < m; u += 4)
    {
        __m128i idx1 = _mm_setr_epi32(
            u * fstride, (u + 1) * fstride, (u + 2) * fstride, (u + 3) * fstride);
        __m128 tw1r = _mm_i32gather_ps(tw_r, idx1, 4);
        __m128 tw1i = _mm_i32gather_ps(tw_i, idx1, 4);

        __m128i idx2 = _mm_mullo_epi32(idx1, _mm_set1_epi32(2));
        __m128 tw2r = _mm_i32gather_ps(tw_r, idx2, 4);
        __m128 tw2i = _mm_i32gather_ps(tw_i, idx2, 4);

        __m128i idx3 = _mm_mullo_epi32(idx1, _mm_set1_epi32(3));
        __m128 tw3r = _mm_i32gather_ps(tw_r, idx3, 4);
        __m128 tw3i = _mm_i32gather_ps(tw_i, idx3, 4);

        __m128i idx4 = _mm_mullo_epi32(idx1, _mm_set1_epi32(4));
        __m128 tw4r = _mm_i32gather_ps(tw_r, idx4, 4);
        __m128 tw4i = _mm_i32gather_ps(tw_i, idx4, 4);

        __m128 f0r = _mm_loadu_ps(F0r + u);
        __m128 f0i = _mm_loadu_ps(F0i + u);
        __m128 f1r = _mm_loadu_ps(F1r + u);
        __m128 f1i = _mm_loadu_ps(F1i + u);
        __m128 f2r = _mm_loadu_ps(F2r + u);
        __m128 f2i = _mm_loadu_ps(F2i + u);
        __m128 f3r = _mm_loadu_ps(F3r + u);
        __m128 f3i = _mm_loadu_ps(F3i + u);
        __m128 f4r = _mm_loadu_ps(F4r + u);
        __m128 f4i = _mm_loadu_ps(F4i + u);

        CMUL(f1r, f1i, tw1r, tw1i);
        CMUL(f2r, f2i, tw2r, tw2i);
        CMUL(f3r, f3i, tw3r, tw3i);
        CMUL(f4r, f4i, tw4r, tw4i);

        __m128 s7r = _mm_add_ps(f1r, f4r);
        __m128 s7i = _mm_add_ps(f1i, f4i);
        __m128 s10r = _mm_sub_ps(f1r, f4r);
        __m128 s10i = _mm_sub_ps(f1i, f4i);
        __m128 s8r = _mm_add_ps(f2r, f3r);
        __m128 s8i = _mm_add_ps(f2i, f3i);
        __m128 s9r = _mm_sub_ps(f2r, f3r);
        __m128 s9i = _mm_sub_ps(f2i, f3i);

        __m128 s5r = _mm_fmadd_ps(s7r, v_yar, f0r);
        s5r = _mm_fmadd_ps(s8r, v_ybr, s5r);
        __m128 s5i = _mm_fmadd_ps(s7i, v_yar, f0i);
        s5i = _mm_fmadd_ps(s8i, v_ybr, s5i);

        __m128 s6r = _mm_fmadd_ps(s10i, v_yai, _mm_mul_ps(s9i, v_ybi));
        __m128 s6i = _mm_fnmsub_ps(s10r, v_yai, _mm_mul_ps(s9r, v_ybi));

        __m128 t1r = _mm_sub_ps(s5r, s6r);
        __m128 t1i = _mm_sub_ps(s5i, s6i);
        __m128 t4r = _mm_add_ps(s5r, s6r);
        __m128 t4i = _mm_add_ps(s5i, s6i);

        __m128 s11r = _mm_fmadd_ps(s7r, v_ybr, f0r);
        s11r = _mm_fmadd_ps(s8r, v_yar, s11r);
        __m128 s11i = _mm_fmadd_ps(s7i, v_ybr, f0i);
        s11i = _mm_fmadd_ps(s8i, v_yar, s11i);

        __m128 tmp = _mm_add_ps(s7r, s8r);
        f0r = _mm_add_ps(f0r, tmp);
        tmp = _mm_add_ps(s7i, s8i);
        f0i = _mm_add_ps(f0i, tmp);

        __m128 s12r = _mm_fmsub_ps(s9i, v_yai, _mm_mul_ps(s10i, v_ybi));
        __m128 s12i = _mm_fmsub_ps(s10r, v_ybi, _mm_mul_ps(s9r, v_yai));

        __m128 t2r = _mm_add_ps(s11r, s12r);
        __m128 t2i = _mm_add_ps(s11i, s12i);
        __m128 t3r = _mm_sub_ps(s11r, s12r);
        __m128 t3i = _mm_sub_ps(s11i, s12i);

        _mm_storeu_ps(F0r + u, f0r);
        _mm_storeu_ps(F0i + u, f0i);
        _mm_storeu_ps(F1r + u, t1r);
        _mm_storeu_ps(F1i + u, t1i);
        _mm_storeu_ps(F2r + u, t2r);
        _mm_storeu_ps(F2i + u, t2i);
        _mm_storeu_ps(F3r + u, t3r);
        _mm_storeu_ps(F3i + u, t3i);
        _mm_storeu_ps(F4r + u, t4r);
        _mm_storeu_ps(F4i + u, t4i);
    }

    for (; u < m; ++u) {
        float* Fout0r = Foutr + u,
            * Fout1r = Foutr + m + u,
            * Fout2r = Foutr + 2 * m + u,
            * Fout3r = Foutr + 3 * m + u,
            * Fout4r = Foutr + 4 * m + u;
        float* Fout0i = Fouti + u,
            * Fout1i = Fouti + m + u,
            * Fout2i = Fouti + 2 * m + u,
            * Fout3i = Fouti + 3 * m + u,
            * Fout4i = Fouti + 4 * m + u;

        float scratchr[13], scratchi[13];

        scratchr[0] = *Fout0r;
        scratchi[0] = *Fout0i;

        scratchr[1] = *Fout1r * tw_r[u * fstride] - *Fout1i * tw_i[u * fstride];
        scratchi[1] = *Fout1r * tw_i[u * fstride] + *Fout1i * tw_r[u * fstride];
        scratchr[2] = *Fout2r * tw_r[2 * u * fstride] - *Fout2i * tw_i[2 * u * fstride];
        scratchi[2] = *Fout2r * tw_i[2 * u * fstride] + *Fout2i * tw_r[2 * u * fstride];
        scratchr[3] = *Fout3r * tw_r[3 * u * fstride] - *Fout3i * tw_i[3 * u * fstride];
        scratchi[3] = *Fout3r * tw_i[3 * u * fstride] + *Fout3i * tw_r[3 * u * fstride];
        scratchr[4] = *Fout4r * tw_r[4 * u * fstride] - *Fout4i * tw_i[4 * u * fstride];
        scratchi[4] = *Fout4r * tw_i[4 * u * fstride] + *Fout4i * tw_r[4 * u * fstride];

        scratchr[7] = scratchr[1] + scratchr[4];
        scratchi[7] = scratchi[1] + scratchi[4];
        scratchr[10] = scratchr[1] - scratchr[4];
        scratchi[10] = scratchi[1] - scratchi[4];
        scratchr[8] = scratchr[2] + scratchr[3];
        scratchi[8] = scratchi[2] + scratchi[3];
        scratchr[9] = scratchr[2] - scratchr[3];
        scratchi[9] = scratchi[2] - scratchi[3];

        *Fout0r += scratchr[7] + scratchr[8];
        *Fout0i += scratchi[7] + scratchi[8];

        scratchr[5] = scratchr[0] + scratchr[7] * yar + scratchr[8] * ybr;
        scratchi[5] = scratchi[0] + scratchi[7] * yar + scratchi[8] * ybr;

        scratchr[6] = scratchi[10] * yai + scratchi[9] * ybi;
        scratchi[6] = -(scratchr[10] * yai) - scratchr[9] * ybi;

        *Fout1r = scratchr[5] - scratchr[6];
        *Fout1i = scratchi[5] - scratchi[6];
        *Fout4r = scratchr[5] + scratchr[6];
        *Fout4i = scratchi[5] + scratchi[6];

        scratchr[11] = scratchr[0] + scratchr[7] * ybr + scratchr[8] * yar;
        scratchi[11] = scratchi[0] + scratchi[7] * ybr + scratchi[8] * yar;
        scratchr[12] = -(scratchi[10] * ybi) + scratchi[9] * yai;
        scratchi[12] = scratchr[10] * ybi - scratchr[9] * yai;

        *Fout2r = scratchr[11] + scratchr[12];
        *Fout2i = scratchi[11] + scratchi[12];
        *Fout3r = scratchr[11] - scratchr[12];
        *Fout3i = scratchi[11] - scratchi[12];

        ++Fout0r; ++Fout1r; ++Fout2r; ++Fout3r; ++Fout4r;
        ++Fout0i; ++Fout1i; ++Fout2i; ++Fout3i; ++Fout4i;
    }
}


inline
static float hsum128_ps(__m128 v)
{
    v = _mm_hadd_ps(v, v);
    v = _mm_hadd_ps(v, v);
    return _mm_cvtss_f32(v);
}

inline
static float hsum256_ps(__m256 v)
{
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    return hsum128_ps(sum);
}

static void kf_bfly_generic(
    float* Foutr,
    float* Fouti,
    int                    fstride,
    const float* twiddles_r,
    const float* twiddles_i,
    int                    Norig,
    int                    m,
    int                    p,
    float* scratchr,
    float* scratchi)
{
    for (int u = 0; u != m; ++u) {
        int k = u;
        for (int q = 0; q != p; ++q) {
            scratchr[q] = Foutr[k];
            scratchi[q] = Fouti[k];
            k += m;
        }

        k = u;
        for (int q1 = 0; q1 != p; ++q1) {

            const int step = k * fstride % Norig;

            float acc_r = scratchr[0];
            float acc_i = scratchi[0];

            alignas(32) int idx[8];
            int q = 1;
            for (; q + 7 < p; q += 8) {
                int qlocal = q;
                for (int i = 0; i != 8; ++i, ++qlocal) {
                    int idxv = qlocal * step;
                    if (idxv >= Norig)
                        idxv %= Norig;
                    idx[i] = idxv;
                }
                __m256i vindex = _mm256_loadu_si256(reinterpret_cast<__m256i*>(idx));

                __m256 twr = _mm256_i32gather_ps(twiddles_r, vindex, 4);
                __m256 twi = _mm256_i32gather_ps(twiddles_i, vindex, 4);

                __m256 sr = _mm256_loadu_ps(scratchr + q);
                __m256 si = _mm256_loadu_ps(scratchi + q);

                __m256 tr = _mm256_fmsub_ps(sr, twr, _mm256_mul_ps(si, twi));
                __m256 ti = _mm256_fmadd_ps(sr, twi, _mm256_mul_ps(si, twr));

                acc_r += hsum256_ps(tr);
                acc_i += hsum256_ps(ti);
            }

            for (; q + 3 < p; q += 4)
            {
                int qlocal = q;
                for (int i = 0; i != 8; ++i, ++qlocal) {
                    int idxv = qlocal * step;
                    if (idxv >= Norig)
                        idxv %= Norig;
                    idx[i] = idxv;
                }
                __m128i vindex = _mm_loadu_si128(reinterpret_cast<__m128i*>(idx));

                __m128 twr = _mm_i32gather_ps(twiddles_r, vindex, 4);
                __m128 twi = _mm_i32gather_ps(twiddles_i, vindex, 4);

                __m128 sr = _mm_loadu_ps(scratchr + q);
                __m128 si = _mm_loadu_ps(scratchi + q);

                __m128 tr = _mm_fmsub_ps(sr, twr, _mm_mul_ps(si, twi));
                __m128 ti = _mm_fmadd_ps(sr, twi, _mm_mul_ps(si, twr));

                acc_r += hsum128_ps(tr);
                acc_i += hsum128_ps(ti);
            }

            for (; q != p; ++q) {
                int twidx = (q * step) % Norig;
                float tr = scratchr[q] * twiddles_r[twidx] - scratchi[q] * twiddles_i[twidx];
                float ti = scratchr[q] * twiddles_i[twidx] + scratchi[q] * twiddles_r[twidx];
                acc_r += tr;
                acc_i += ti;
            }

            Foutr[k] = acc_r;
            Fouti[k] = acc_i;

            k += m;
        }
    }
}

static
void kf_work(
    float* Foutr,
    float* Fouti,
    const float* f,
    int fstride,
    int in_stride,
    int* factors,
    float* buffer_scratchr,
    float* buffer_scratchi,
    fft_context& ctx
)
{
    int Fout_beg = 0;
    const int p = *factors++; /* the radix  */
    const int m = *factors++; /* stage's fft length/p */
    const int Fout_end = p * m;

    if (m == 1) {
        do {
            Foutr[Fout_beg] = *f;
            f += fstride * in_stride;
        } while (++Fout_beg != Fout_end);
    }
    else {
        do {
            // recursive call:
            // DFT of size m*p performed by doing
            // p instances of smaller DFTs of size m,
            // each one takes a decimated version of the input
            kf_work(Foutr + Fout_beg, Fouti + Fout_beg, f, fstride * p, in_stride, factors, buffer_scratchr, buffer_scratchi, ctx);
            f += fstride * in_stride;
        } while ((Fout_beg += m) != Fout_end);
    }

    // recombine the p smaller DFTs
    switch (p) {
    case 2: kf_bfly2(Foutr, Fouti, fstride, ctx.twiddles_r.get(), ctx.twiddles_i, m); break;
    case 3: kf_bfly3(Foutr, Fouti, fstride, ctx.twiddles_r.get(), ctx.twiddles_i, m); break;
    case 4: kf_bfly4(Foutr, Fouti, fstride, ctx.twiddles_r.get(), ctx.twiddles_i, m); break;
    case 5: kf_bfly5(Foutr, Fouti, fstride, ctx.twiddles_r.get(), ctx.twiddles_i, m); break;
    default: kf_bfly_generic(Foutr, Fouti, fstride, ctx.twiddles_r.get(), ctx.twiddles_i, ctx.nfft, m, p, buffer_scratchr, buffer_scratchi);
    }
}

fft_context::fft_context(int nfft) : nfft(nfft), idx(0), workers(0), twiddles_r(std::make_unique<float[]>(nfft * 2))
{
    twiddles_i = twiddles_r.get() + nfft;

    int i = 0;
    constexpr float pi = 3.1415926535897932f;
    const float factor = -2 * pi / nfft;
#ifdef _MSC_VER
    {
        const __m256 factor256 = _mm256_set1_ps(factor);
        __m256 _8 = _mm256_set1_ps(8.f);
        __m256 base = _mm256_setr_ps(0, 1, 2, 3, 4, 5, 6, 7);
        for (; i + 7 < nfft; i += 8)
        {
            __m256 phase = _mm256_mul_ps(base, factor256);
            base = _mm256_add_ps(base, _8);

            __m256 v = _mm256_cos_ps(phase);
            _mm256_storeu_ps(twiddles_r.get() + i, v);

            v = _mm256_sin_ps(phase);
            _mm256_storeu_ps(twiddles_i + i, v);
        }
    }
    {
        const __m128 factor128 = _mm_set_ps1(factor);
        __m128 _4 = _mm_set_ps1(4.f);
        __m128 base = _mm_setr_ps(i, i + 1, i + 2, i + 3);
        for (; i + 3 < nfft; i += 4)
        {
            __m128 phase = _mm_mul_ps(base, factor128);
            base = _mm_add_ps(base, _4);

            __m128 v = _mm_cos_ps(phase);
            _mm_storeu_ps(twiddles_r.get() + i, v);

            v = _mm_sin_ps(phase);
            _mm_storeu_ps(twiddles_i + i, v);
        }
    }
#endif
    for (; i < nfft; ++i)
    {
        float phase = factor * i;
        twiddles_r[i] = cosf(phase);
        twiddles_i[i] = sinf(phase);
    }

    int p = 4;
    auto n = nfft;
    double floor_sqrt = floor(sqrt(n));
    auto facbuf = factors;
    max_factor_value = 0;

    /*factor out powers of 4, powers of 2, then any remaining primes */
    do {
        while (n % p) {
            switch (p) {
            case 4: p = 2; break;
            case 2: p = 3; break;
            default: p += 2; break;
            }
            if (p > floor_sqrt)
                p = n;          /* no more factors, skip to end */
        }
        n /= p;
        *facbuf++ = p;
        *facbuf++ = n;
        if (p > max_factor_value)
            max_factor_value = p;
        if (n > max_factor_value)
            max_factor_value = n;
    } while (n > 1);
}

static void ggml_fft_op(struct ggml_tensor* dst, int ith, int nth, fft_context* ctx)
{
    auto buffer_scratchr = reinterpret_cast<float*>(alloca(sizeof(float) * ctx->max_factor_value * 2));
    auto buffer_scratchi = buffer_scratchr + ctx->max_factor_value;
    auto const signal = dst->src[0];
    auto const data = reinterpret_cast<char*>(signal->data);
    const auto foutr = reinterpret_cast<char*>(dst->data);
    const auto fouti = foutr + dst->nb[2];
    const auto dst_nb1 = dst->nb[1];
    const auto signal_nb1 = signal->nb[1];
    const auto signal_ne1 = signal->ne[1];

    ++ctx->workers;

    for (int64_t i = ctx->idx++; i < signal_ne1; i = ctx->idx++)
    {
        const auto cur_offset = dst_nb1 * i;
        const auto cur_fouti = reinterpret_cast<float*>(fouti + cur_offset);
        memset(cur_fouti, 0, dst_nb1);
        kf_work(reinterpret_cast<float*>(foutr + cur_offset),
            cur_fouti,
            reinterpret_cast<float*>(data + signal_nb1 * i),
            1, 1, ctx->factors,
            buffer_scratchr, buffer_scratchi, *ctx);
    }

    if (--ctx->workers == 0) ctx->idx = 0;
}

void delete_fft_context(fft_context* ctx)
{
    delete ctx;
}

fft_context_ptr create_fft_context(int nfft)
{
    return fft_context_ptr(new fft_context(nfft));
}

void fft(const float* signal, float* buffer, fft_context& ctx)
{
    auto nfft = ctx.nfft;
    auto foutr = reinterpret_cast<float*>(alloca(sizeof(float) * nfft * 2 + sizeof(float) * ctx.max_factor_value * 2));
    auto fouti = foutr + nfft;
    auto buffer_scratchr = fouti + nfft;
    auto buffer_scratchi = buffer_scratchr + ctx.max_factor_value;
    memset(fouti, 0, sizeof(float) * nfft);

    kf_work(foutr, fouti, signal, 1, 1, ctx.factors, buffer_scratchr, buffer_scratchi, ctx);
    int i = 0;
    while (i + 7 < nfft)
    {
        __m256 tr = _mm256_loadu_ps(foutr + i);
        __m256 ti = _mm256_loadu_ps(fouti + i);
        auto abs = _mm256_sqrt_ps(_mm256_fmadd_ps(tr, tr, _mm256_mul_ps(ti, ti)));
        _mm256_storeu_ps(buffer + i, abs);
        i += 8;
    }
    while (i + 3 < nfft)
    {
        __m128 tr = _mm_loadu_ps(foutr + i);
        __m128 ti = _mm_loadu_ps(fouti + i);
        auto abs = _mm_sqrt_ps(_mm_fmadd_ps(tr, tr, _mm_mul_ps(ti, ti)));
        _mm_storeu_ps(buffer + i, abs);
        i += 4;
    }
    for (; i != nfft; ++i)
        buffer[i] = sqrtf(foutr[i] * foutr[i] + fouti[i] * fouti[i]);
}

ggml_tensor* ggml_fft(ggml_context* ctx, ggml_tensor* a, fft_context* fctx)
{
    GGML_ASSERT(ggml_is_matrix(a));
    GGML_ASSERT(a->ne[0] == fctx->nfft);
    GGML_ASSERT(ggml_is_contiguous(a));

    return ggml_custom_4d(ctx, a->type, fctx->nfft, a->ne[1], 2, 1, &a, 1, reinterpret_cast<ggml_custom_op_t>(ggml_fft_op), GGML_N_TASKS_MAX, fctx);
}

ggml_tensor* ggml_stft(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b, int hop_len, bool center, fft_context* fctx)
{
    GGML_ASSERT(ggml_is_vector(a));
    GGML_ASSERT(ggml_is_vector(b));
    GGML_ASSERT(ggml_is_contiguous(a));

    if (center)
    {
        const auto pad_amount = fctx->nfft / 2;
        a = ggml_pad_reflect_1d(ctx, a, pad_amount, pad_amount);
    }
    else
        a = ggml_dup(ctx, a);

    ggml_set_cpu(a);

    const auto win_size = b->ne[0];
    const auto n_frames = (a->ne[0] - win_size) / hop_len + 1;

    a = ggml_dup_inplace(ctx, a);
    a->ne[0] = win_size;
    a->ne[1] = n_frames;
    a->nb[1] = hop_len * a->nb[0];
    a->op = GGML_OP_VIEW;
    ggml_set_cpu(a);

    b = ggml_repeat(ctx, b, a);
    ggml_set_cpu(b);
    a = ggml_mul(ctx, a, b);
    ggml_set_cpu(a);

    auto result = ggml_fft(ctx, a, fctx);
    result = ggml_view_3d(ctx, result, fctx->nfft / 2 + 1, n_frames, 2, result->nb[1], result->nb[2], 0);
    return ggml_permute(ctx, result, 1, 0, 2, 3);
}

struct istft_context
{
    int nfft;
    ggml_tensor* idft_real;
    ggml_tensor* idft_imag;
};

void delete_istft_context(istft_context* ctx)
{
    delete ctx;
}

istft_context_ptr create_istft_context(int nfft, ggml_context* ctx, std::function<void(ggml_tensor*, void*, size_t)> set_data)
{
    const int f = nfft / 2 + 1;
    const int stride = f * nfft;
    auto idft_real = std::make_unique<float[]>(stride * 2);
    auto idft_imag = idft_real.get() + stride;
    for (int y = 0; y != f; ++y)
        for (int x = 0; x != nfft; ++x)
        {
            const float phase = 2.0f * 3.14159265358979323846f * x * y / nfft;
            const int offset = x * f + y;
            idft_real[offset] = std::cos(phase);
            idft_imag[offset] = phase;
        }

    for (int x = 0, y = f - 1; x != nfft; ++x)
    {
        const auto offset1 = x * f;
        const auto offset2 = offset1 + y;
        idft_real[offset1] /= nfft;
        idft_real[offset2] /= nfft;
        idft_imag[offset2] = 0.0f;
    }

    for (int y = 1, endy = f - 1; y != endy; ++y)
        for (int x = 0; x != nfft; ++x)
        {
            const int idx = x * f + y;
            auto& real = idft_real[idx];
            auto& imag = idft_imag[idx];

            real = 2.0f * real / nfft;
            imag = -2.0f * std::sin(imag) / nfft;
        }

    auto ictx = new istft_context;
    ictx->nfft = nfft;
    ictx->idft_real = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, f, nfft);
    ictx->idft_imag = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, f, nfft);
    set_data(ictx->idft_real, idft_real.get(), sizeof(float) * stride);
    set_data(ictx->idft_imag, idft_imag, sizeof(float) * stride);
    return istft_context_ptr(ictx);
}

static void ggml_istft_finalize_op(struct ggml_tensor* dst, int ith, int nth, void* userdata)
{
    auto idx = dst->src[0];
    auto idx_data = reinterpret_cast<int32_t*>(idx->data);
    auto frames = reinterpret_cast<float*>(dst->src[1]->data);
    auto w_sq = reinterpret_cast<float*>(dst->src[2]->data);
    auto dst_data = reinterpret_cast<float*>(dst->data);
    auto denom_data = reinterpret_cast<float*>(dst->src[3]->data);

    const auto offset = userdata ? reinterpret_cast<intptr_t>(userdata) : 0;
    const auto out_len = dst->ne[0];
    auto win_len = dst->src[2]->ne[0];

    const auto begin = (out_len * ith) / nth;
    const auto end = (out_len * (ith + 1)) / nth;

    memset(dst_data + begin, 0, (end - begin) * sizeof(float));
    memset(denom_data + begin, 0, (end - begin) * sizeof(float));
    const auto n = ggml_nelements(idx);
    for (int64_t i = 0; i != n; ++i) {
        const auto cur = idx_data[i] - offset;
        if (cur < begin || cur >= end)
            continue;

        dst_data[cur] += frames[i];
        denom_data[cur] += w_sq[i % win_len];
    }

    constexpr float eps = 1e-8f;
    int64_t i = begin;
    for (const __m256 eps_v = _mm256_set1_ps(eps); i + 7 < end; i += 8) {
        auto y_ptr = dst_data + i;
        __m256 denom = _mm256_loadu_ps(denom_data + i);
        __m256 y = _mm256_loadu_ps(y_ptr);
        denom = _mm256_max_ps(denom, eps_v);
        y = _mm256_div_ps(y, denom);
        _mm256_storeu_ps(y_ptr, y);
    }

    for (const __m128 eps_v = _mm_set_ps1(eps); i + 3 < end; i += 4) {
        auto y_ptr = dst_data + i;
        __m128 denom = _mm_loadu_ps(denom_data + i);
        __m128 y = _mm_loadu_ps(y_ptr);
        denom = _mm_max_ps(denom, eps_v);
        y = _mm_div_ps(y, denom);
        _mm_storeu_ps(y_ptr, y);
    }

    for (; i < end; ++i) {
        auto denom = denom_data[i];
        denom = std::max(denom, eps);
        dst_data[i] /= denom;
    }
}

ggml_tensor* ggml_istft(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b, ggml_tensor* c, int hop_len, bool center, istft_context* ictx)
{
    GGML_ASSERT(ggml_are_same_shape(a, b));
    GGML_ASSERT(a->ne[1] == ictx->nfft / 2 + 1);

    a = ggml_cont(ctx, ggml_permute(ctx, a, 1, 0, 2, 3));
    b = ggml_cont(ctx, ggml_permute(ctx, b, 1, 0, 2, 3));

    auto frames = ggml_add(ctx,
        ggml_mul_mat(ctx, ictx->idft_real, a),
        ggml_mul_mat(ctx, ictx->idft_imag, b));
    frames = ggml_mul(ctx, frames, c);

    auto t_idx = ggml_arange(ctx, 0, static_cast<float>(b->ne[1] * hop_len), static_cast<float>(hop_len));
    auto n_idx = ggml_arange(ctx, 0, static_cast<float>(ictx->nfft), 1.f);
    t_idx = ggml_reshape_2d(ctx, t_idx, 1, t_idx->ne[0]);
    t_idx = ggml_repeat_4d(ctx, t_idx, ictx->nfft, t_idx->ne[1], 1, 1);
    n_idx = ggml_repeat_4d(ctx, n_idx, n_idx->ne[0], a->ne[1], 1, 1);
    auto idx = ggml_add(ctx, t_idx, n_idx);
    idx = ggml_cast(ctx, idx, GGML_TYPE_I32);

    auto w_sq = ggml_mul(ctx, c, c);
    auto out_len_full = (b->ne[1] - 1) * hop_len + (center ? 0 : ictx->nfft);
    ggml_tensor* args[] = { idx, frames, w_sq, ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_len_full) };
    return ggml_custom_4d(ctx, GGML_TYPE_F32, out_len_full, 1, 1, 1, args, 4, ggml_istft_finalize_op, GGML_N_TASKS_MAX, reinterpret_cast<void*>(intptr_t(center ? ictx->nfft / 2 : 0)));
}
