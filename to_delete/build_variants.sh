#!/bin/bash
set -e

echo "Building WASM matmul variants..."

# Common flags
COMMON="-O3 -flto --no-entry -Wl,--export-dynamic"

# 1. Naive (no optimizations beyond -O3)
echo "Building naive..."
emcc matmul_naive.c -o matmul_naive.wasm $COMMON \
    -s INITIAL_MEMORY=256MB \
    -s ALLOW_MEMORY_GROWTH=1 \
    -s EXPORTED_FUNCTIONS='["_matmul_naive","_malloc_f64","_free_f64"]'

# 2. Cache-optimized (blocking, no SIMD)
echo "Building cache-optimized..."
emcc matmul_cache.c -o matmul_cache.wasm $COMMON \
    -s INITIAL_MEMORY=256MB \
    -s ALLOW_MEMORY_GROWTH=1 \
    -s EXPORTED_FUNCTIONS='["_matmul_cache","_malloc_f64","_free_f64"]'

# 3. SIMD only (no cache blocking)
echo "Building SIMD-only..."
emcc matmul_simd.c -o matmul_simd.wasm $COMMON \
    -msimd128 \
    -s INITIAL_MEMORY=256MB \
    -s ALLOW_MEMORY_GROWTH=1 \
    -s EXPORTED_FUNCTIONS='["_matmul_simd","_malloc_f64","_free_f64"]'

# 4. Full optimized (cache + SIMD) with 4x4 micro-kernel
echo "Building full optimized 4x4 (cache + SIMD)..."
emcc matmul.c -o matmul_full_4x4.wasm $COMMON \
    -msimd128 \
    -s INITIAL_MEMORY=256MB \
    -s ALLOW_MEMORY_GROWTH=1 \
    -s EXPORTED_FUNCTIONS='["_matmul_f64","_dgemm","_malloc_f64","_free_f64","_cleanup"]'

# 5. Full optimized (cache + SIMD) with 2x2 micro-kernel
echo "Building full optimized 2x2 (cache + SIMD)..."
emcc matmul_full_2x2.c -o matmul_full_2x2.wasm $COMMON \
    -msimd128 \
    -s INITIAL_MEMORY=256MB \
    -s ALLOW_MEMORY_GROWTH=1 \
    -s EXPORTED_FUNCTIONS='["_matmul_f64_2x2","_dgemm_2x2","_malloc_f64","_free_f64","_cleanup"]'

# 6. Full optimized (cache + SIMD) with 6x8 micro-kernel
echo "Building full optimized 6x8 (cache + SIMD)..."
emcc matmul_full_6x8.c -o matmul_full_6x8.wasm $COMMON \
    -msimd128 \
    -s INITIAL_MEMORY=256MB \
    -s ALLOW_MEMORY_GROWTH=1 \
    -s EXPORTED_FUNCTIONS='["_matmul_f64_6x8","_dgemm_6x8","_malloc_f64","_free_f64","_cleanup"]'

# 7. Full optimized (cache + SIMD) with 8x8 micro-kernel
echo "Building full optimized 8x8 (cache + SIMD)..."
emcc matmul_full_8x8.c -o matmul_full_8x8.wasm $COMMON \
    -msimd128 \
    -s INITIAL_MEMORY=256MB \
    -s ALLOW_MEMORY_GROWTH=1 \
    -s EXPORTED_FUNCTIONS='["_matmul_f64_8x8","_dgemm_8x8","_malloc_f64","_free_f64","_cleanup"]'

# 8. OpenBLAS-style (full micro-kernel set)
echo "Building OpenBLAS-style (full micro-kernels)..."
emcc matmul_openblas.c -o matmul_openblas.wasm $COMMON \
    -msimd128 \
    -s INITIAL_MEMORY=256MB \
    -s ALLOW_MEMORY_GROWTH=1 \
    -s EXPORTED_FUNCTIONS='["_matmul_openblas","_dgemm_openblas","_malloc_f64","_free_f64","_cleanup"]'

# 9. SSE-style (pre-duplicated B packing, no runtime broadcasts)
echo "Building SSE-style (pre-duplicated B)..."
emcc matmul_sse.c -o matmul_sse.wasm $COMMON \
    -msimd128 \
    -s INITIAL_MEMORY=256MB \
    -s ALLOW_MEMORY_GROWTH=1 \
    -s EXPORTED_FUNCTIONS='["_matmul_sse","_dgemm_sse","_malloc_f64","_free_f64","_cleanup"]'

echo "Done! Built:"
ls -la *.wasm
