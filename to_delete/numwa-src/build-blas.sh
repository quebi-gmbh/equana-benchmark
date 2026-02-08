#!/bin/bash
#
# Build BLAS WebAssembly module using Emscripten
#
# This builds the OpenBLAS-style DGEMM implementation with:
# - WASM SIMD128 optimized micro-kernels
# - GotoBLAS 3-level cache blocking algorithm
# - Single-threaded operation
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$SCRIPT_DIR"
KERNEL_DIR="$SRC_DIR/kernel"
OUT_DIR="$SCRIPT_DIR/dist"

echo "NW BLAS WebAssembly Build"
echo "========================="
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

# Exported functions for DGEMM module
EXPORTED_FUNCS='[
    "_dgemm",
    "_dgemm_sse",
    "_dgemm_native",
    "_matmul_f64",
    "_matmul_f64_sse",
    "_matmul_f64_native",
    "_malloc_f64",
    "_free_f64",
    "_cleanup",
    "_malloc",
    "_free"
]'

EXPORTED_RUNTIME='["ccall", "cwrap", "getValue", "setValue", "HEAPF64"]'

# Build single-threaded CJS module
echo "Building single-threaded BLAS (CJS)..."
emcc \
    "$SRC_DIR/level3.c" \
    "$KERNEL_DIR/dgemm_kernel_4x4_wasm.c" \
    "$KERNEL_DIR/dgemm_kernel_sse_4x4_wasm.c" \
    "$KERNEL_DIR/dgemm_kernel_native_4x4_wasm.c" \
    -o "$OUT_DIR/blas.cjs" \
    $COMMON_FLAGS \
    -s WASM=1 \
    -s MODULARIZE=1 \
    -s ENVIRONMENT=web,worker,node \
    -s EXPORT_NAME="createBLASModule" \
    -s EXPORTED_FUNCTIONS="$EXPORTED_FUNCS" \
    -s EXPORTED_RUNTIME_METHODS="$EXPORTED_RUNTIME" \
    -s ALLOW_MEMORY_GROWTH=1 \
    -s INITIAL_MEMORY=16777216 \
    -s STACK_SIZE=1048576

echo "CJS build complete!"
echo ""

# Build single-threaded ESM module
echo "Building single-threaded BLAS (ESM)..."
emcc \
    "$SRC_DIR/level3.c" \
    "$KERNEL_DIR/dgemm_kernel_4x4_wasm.c" \
    "$KERNEL_DIR/dgemm_kernel_sse_4x4_wasm.c" \
    "$KERNEL_DIR/dgemm_kernel_native_4x4_wasm.c" \
    -o "$OUT_DIR/blas.mjs" \
    $COMMON_FLAGS \
    -s WASM=1 \
    -s MODULARIZE=1 \
    -s EXPORT_ES6=1 \
    -s ENVIRONMENT=web,worker \
    -s EXPORT_NAME="createBLASModule" \
    -s EXPORTED_FUNCTIONS="$EXPORTED_FUNCS" \
    -s EXPORTED_RUNTIME_METHODS="$EXPORTED_RUNTIME" \
    -s ALLOW_MEMORY_GROWTH=1 \
    -s INITIAL_MEMORY=16777216 \
    -s STACK_SIZE=1048576

echo "ESM build complete!"
echo ""

echo "Build complete!"
echo "  CJS Module:  $OUT_DIR/blas.cjs (Node.js)"
echo "  ESM Module:  $OUT_DIR/blas.mjs (Browser)"
echo "  WASM:        $OUT_DIR/blas.wasm"
echo ""

# Show file sizes
echo "File sizes:"
ls -lh "$OUT_DIR/blas.cjs" "$OUT_DIR/blas.mjs" "$OUT_DIR/blas.wasm" 2>/dev/null || true
