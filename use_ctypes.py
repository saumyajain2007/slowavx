import ctypes
import numpy as np
from pathlib import Path

lib = ctypes.CDLL(str(Path(__file__).resolve().parents[1] / "build" / "libslowavx.so"))

lib.slow_vector_work_avx2.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_size_t,
    ctypes.c_int
]
lib.slow_vector_work_avx2.restype = None

def slow_with_ctypes(dst: np.ndarray, src: np.ndarray, idx: np.ndarray, slow_factor: int = 1):
    assert dst.dtype == np.float32 and src.dtype == np.float32 and idx.dtype == np.int32
    n = src.size
    lib.slow_vector_work_avx2(
        dst.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        src.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        idx.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        ctypes.c_size_t(n),
        ctypes.c_int(slow_factor)
    )
