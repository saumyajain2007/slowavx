import numpy as np
import slowavx   # module built via setup script

def slow_with_extension(dst: np.ndarray, src: np.ndarray, idx: np.ndarray, slow_factor: int = 1):
    # ensure contiguous and types
    assert dst.dtype == np.float32 and src.dtype == np.float32 and idx.dtype == np.int32
    n = src.size
    slowavx.slow_vector(dst.tobytes(), src.tobytes(), idx.tobytes(), n, slow_factor)  # if using buffer protocol wrapper change accordingly
