The library provides a Python-accessible function that takes a large array of floats and an optional index array,
then performs intentionally wasteful computations on each element to slow down execution. Internally, it either uses an AVX2 SIMD implementation
or a scalar fallback to process the data in blocks, repeatedly applying multiplications, fused multiply-adds, and permutations while gathering values
from pseudo-random memory locations based on the index array. This combination of dependent calculations and scattered memory accesses artificially increases
CPU cycles and cache misses, with a tunable slow_factor controlling how much extra work is done, while writing the results back in-place so Python can access the updated array immediately.
