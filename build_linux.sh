#!/usr/bin/env bash
set -e
mkdir -p build
cd build

# shared lib (ctypes/cffi)
gcc -O2 -mavx2 -fPIC -shared ../src/slow_simd_avx2.c ../src/slow_simd_scalar.c -o libslowavx.so

# python extension (in-place)
python3 - <<'PY'
from setuptools import setup, Extension
ext = Extension("slowavx",
                sources=["../src/slowmodule.c", "../src/slow_simd_avx2.c", "../src/slow_simd_scalar.c"],
                extra_compile_args=["-O2","-mavx2"])
setup(name="slowavx", ext_modules=[ext])
PY

echo "Built libslowavx.so and Python extension (may be in build/ or project root depending on setuptools)."
