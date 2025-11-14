#include "slow_simd.h"
#include <stddef.h>

int slow_simd_runtime_has_avx2(void) {
    return 0;
}

void slow_vector_work_scalar(float *dst, const float *src, const int *idx, size_t n, int slow_factor) {
    if (!slow_factor) slow_factor = 1;
    const int stride = 13;
    for (size_t i = 0; i < n; ++i) {
        // waste CPU scalar work
        float v = src[i] * 0.9999f + 0.0001f;
        int base = idx ? idx[i % (n/8 ? n/8 : 1)] % (int)n : (int)i;
        for (int s = 0; s < slow_factor; ++s) {
            int j = (base + (s % 8) * stride) % (int)n;
            v = v * 0.99999f + src[j] * 1e-6f;
        }
        dst[i] = v;
    }
}
