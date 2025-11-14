#ifndef SLOW_SIMD_H
#define SLOW_SIMD_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// slow_factor: higher => more wasted work per element (>=1)
void slow_vector_work_avx2(float *dst, const float *src, const int *idx, size_t n, int slow_factor);

// scalar fallback with same signature
void slow_vector_work_scalar(float *dst, const float *src, const int *idx, size_t n, int slow_factor);

// helper: returns 1 if AVX2 is available at compile/runtime (best effort)
int slow_simd_runtime_has_avx2(void);

#ifdef __cplusplus
}
#endif

#endif // SLOW_SIMD_H
