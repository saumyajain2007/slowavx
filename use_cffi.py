from cffi import FFI
import numpy as np
from pathlib import Path

ffi = FFI()
ffi.cdef("void slow_vector_work_avx2(float *dst, const float *src, const int *idx, size_t n, int slow_factor);")
lib = ffi.dlopen(str(Path(__file__).resolve().parents[1] / "build" / "libslowavx.so"))

def slow_with_cffi(dst: np.ndarray, src: np.ndarray, idx: np.ndarray, slow_factor: int = 1):
    assert dst.dtype == np.float32 and src.dtype == np.float32 and idx.dtype == np.int32
    n = src.size
    lib.slow_vector_work_avx2(
        ffi.cast("float *", dst.ctypes.data),
        ffi.cast("float *", src.ctypes.data),
        ffi.cast("int *", idx.ctypes.data),
        n,
        slow_factor
    )
