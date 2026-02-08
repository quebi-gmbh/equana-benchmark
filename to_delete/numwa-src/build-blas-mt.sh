#!/bin/bash
#
# Build Multithreaded BLAS WebAssembly module using Emscripten
#
# This builds the OpenBLAS-style DGEMM implementation with:
# - WASM SIMD128 optimized micro-kernels
# - GotoBLAS 3-level cache blocking algorithm
# - pthreads for multithreading (parallelizes over N dimension)
#
# IMPORTANT: Multithreaded WASM requires special server headers:
#   Cross-Origin-Embedder-Policy: require-corp
#   Cross-Origin-Opener-Policy: same-origin
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$SCRIPT_DIR"
KERNEL_DIR="$SRC_DIR/kernel"
OUT_DIR="$SCRIPT_DIR/dist"

echo "NW BLAS Multithreaded WebAssembly Build"
echo "========================================"
echo "Source:  $SRC_DIR"
echo "Kernels: $KERNEL_DIR"
echo "Output:  $OUT_DIR"
echo ""

# Create output directory
mkdir -p "$OUT_DIR"

# Check for Emscripten
if ! command -v emcc &> /dev/null; then
    echo "Error: emcc (Emscripten) not found in PATH"
    echo "Please install Emscripten or activate the emsdk environment"
    exit 1
fi

echo "Using Emscripten: $(emcc --version | head -1)"
echo ""

# Common compiler flags
COMMON_FLAGS="-O3 -flto -ffast-math -msimd128"

# Exported functions for multithreaded DGEMM module
EXPORTED_FUNCS='[
    "_dgemm_mt",
    "_dgemm_sse_mt",
    "_matmul_f64_mt",
    "_matmul_f64_sse_mt",
    "_set_num_threads",
    "_get_num_threads",
    "_malloc_f64",
    "_free_f64",
    "_cleanup",
    "_malloc",
    "_free"
]'

EXPORTED_RUNTIME='["ccall", "cwrap", "getValue", "setValue", "HEAPF64"]'

# Build multithreaded module
echo "Building multithreaded BLAS..."
emcc \
    "$SRC_DIR/level3_mt.c" \
    "$KERNEL_DIR/dgemm_kernel_4x4_wasm.c" \
    "$KERNEL_DIR/dgemm_kernel_sse_4x4_wasm.c" \
    -o "$OUT_DIR/blas_mt.js" \
    $COMMON_FLAGS \
    -pthread \
    -sPTHREAD_POOL_SIZE='navigator.hardwareConcurrency' \
    -sMALLOC=mimalloc \
    -sWASM=1 \
    -sMODULARIZE=1 \
    -sEXPORT_NAME="createBLASMTModule" \
    -sENVIRONMENT='web,worker' \
    -sEXPORTED_FUNCTIONS="$EXPORTED_FUNCS" \
    -sEXPORTED_RUNTIME_METHODS="$EXPORTED_RUNTIME" \
    -sINITIAL_MEMORY=134217728 \
    -sALLOW_MEMORY_GROWTH=1 \
    -sSTACK_SIZE=1048576

echo ""
echo "Build complete!"
echo ""
echo "Files generated:"
ls -lh "$OUT_DIR/blas_mt.js" "$OUT_DIR/blas_mt.wasm" "$OUT_DIR/blas_mt.worker.js" 2>/dev/null || ls -lh "$OUT_DIR/blas_mt."*

echo ""
echo "=== IMPORTANT ==="
echo "To use multithreaded WASM, your server MUST send these headers:"
echo "  Cross-Origin-Embedder-Policy: require-corp"
echo "  Cross-Origin-Opener-Policy: same-origin"
echo ""
echo "Usage in JavaScript:"
echo "  const Module = await createBLASMTModule();"
echo "  Module._set_num_threads(4);"
echo "  Module._matmul_f64_mt(M, N, K, ptrA, ptrB, ptrC);"
echo ""
