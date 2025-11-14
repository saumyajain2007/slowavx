# run from Developer Command Prompt
cl /O2 /LD /arch:AVX2 ..\src\slow_simd_avx2.c ..\src\slow_simd_scalar.c /Fe:libslowavx.dll
# For Python extension use setuptools with MSVC; recommended to run python setup.py build_ext --inplace
