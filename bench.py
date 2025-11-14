import numpy as np
import time
from slowapi import slow_vector

def bench(n=1<<20, slow_factor=1):
    src = np.arange(n, dtype=np.float32)
    dst = np.zeros_like(src)
    idx = np.random.randint(0, n, size=n//8, dtype=np.int32)
    t0 = time.perf_counter()
    slow_vector(dst, src, idx, slow_factor)
    t1 = time.perf_counter()
    print(f"n={n} slow_factor={slow_factor} time={(t1-t0):.4f}s")
    print("checksum", dst[::n//16].sum())

if __name__ == "__main__":
    for s in [1,2,4,8]:
        bench(1<<20, slow_factor=s)
