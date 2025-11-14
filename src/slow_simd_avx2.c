// Compile with -mavx2 -O2
#include "slow_simd.h"
#include <immintrin.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef __GNUC__
#include <cpuid.h>
#endif

static inline int runtime_has_avx2(void) {
#if defined(__GNUC__)
    unsigned int a,b,c,d;
    if (!__get_cpuid_max(0,0)) return 0;
    __get_cpuid(0, &a, &b, &c, &d);
    // check extended features
    unsigned int eax, ebx, ecx, edx;
    __get_cpuid(7, &eax, &ebx, &ecx, &edx);
    return (ebx & (1 << 5)) != 0; // AVX2 bit
#else
    return 0; // unknown; fall back to compile-time checks
#endif
}

int slow_simd_runtime_has_avx2(void) {
#if defined(__AVX2__)
    return runtime_has_avx2();
#else
    return 0;
#endif
}

void slow_vector_work_avx2(float *dst, const float *src, const int *idx, size_t n, int slow_factor) {
#ifndef __AVX2__
    // fallback to scalar
    (void)dst;(void)src;(void)idx;(void)n;(void)slow_factor;
    return;
#else
    if (!slow_factor) slow_factor = 1;
    size_t i = 0;
    const size_t stride = 13; // tunable: odd stride to cause scattering
    // process blocks of 8 (256-bit)
    for (; i + 8 <= n; i += 8) {
        // create an intentionally unaligned/offset load
        const float *srcptr = &src[(i>0) ? i-1 : 0];
        __m256 a = _mm256_loadu_ps(srcptr); // may cross cacheline

        // base index chosen from idx vector (coarse-grain randomness)
        int coarse = idx ? (idx[(i/8) % (n/8)] % (int)n) : (int)(i % n);

        // build offsets to gather (likely to cause cache misses if slow_factor large)
        int32_t offsets[8];
        for (int k = 0; k < 8; ++k) offsets[k] = (coarse + k*stride) % (int)n;
        __m256i off = _mm256_loadu_si256((const __m256i*)offsets);
        __m256 g = _mm256_i32gather_ps(src, off, sizeof(float));

        // wasteful dependent chain repeated slow_factor times
        __m256 r = _mm256_permute2f128_ps(a, g, 0x01);
        for (int s = 0; s < slow_factor; ++s) {
            r = _mm256_mul_ps(r, _mm256_set1_ps(1.000001f + 1e-6f * s));
            r = _mm256_fmadd_ps(r, _mm256_set1_ps(0.999999f - 1e-7f * s), _mm256_set1_ps(1e-6f * s));
            r = _mm256_permute_ps(r, _MM_SHUFFLE(1,3,0,2));
            r = _mm256_add_ps(r, _mm256_permute_ps(r, _MM_SHUFFLE(2,1,3,0)));
        }

        // extract to scalar slowly
        float tmp[8];
        _mm256_storeu_ps(tmp, r);
        float acc = 0.0f;
        for (int k = 0; k < 8; ++k) {
            acc = acc * (1.0f - 1e-5f) + tmp[k];
        }

        // write back to the first lane - intentionally partial use
        dst[i] = acc;
    }

    // tail
    for (; i < n; ++i) {
        float v = src[i] * 0.9999f + 0.0001f;
        for (int s = 0; s < slow_factor; ++s) v = v * (1.0f - 1e-6f * s) + 1e-6f;
        dst[i] = v;
    }
#endif
}
