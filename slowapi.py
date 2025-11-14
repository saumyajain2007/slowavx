import numpy as np
import importlib
from pathlib import Path

# Try extension first, then cffi, then ctypes, then scalar Python fallback
_impl = None

try:
    import slowavx as _ext
    _impl = "extension"
except Exception:
    try:
        from .use_cffi import slow_with_cffi
        _impl = "cffi"
    except Exception:
        try:
            from .use_ctypes import slow_with_ctypes
            _impl = "ctypes"
        except Exception:
            _impl = "python"

def slow_vector(dst: np.ndarray, src: np.ndarray, idx: np.ndarray = None, slow_factor: int = 1):
    if idx is None:
        idx = np.random.randint(0, max(1, src.size), size=max(1, src.size//8), dtype=np.int32)
    if _impl == "extension":
        # assume extension exposes same signature; if not adapt accordingly
        _ext.slow_vector(dst, src, idx, src.size, slow_factor)
    elif _impl == "cffi":
        slow_with_cffi(dst, src, idx, slow_factor)
    elif _impl == "ctypes":
        slow_with_ctypes(dst, src, idx, slow_factor)
    else:
        # pure-python fallback: simple but slow
        n = src.size
        for i in range(n):
            v = float(src[i]) * 0.9999 + 0.0001
            base = int(idx[i % len(idx)]) % n
            for s in range(slow_factor):
                j = (base + (s % 8) * 13) % n
                v = v * 0.99999 + float(src[j]) * 1e-6
            dst[i] = v