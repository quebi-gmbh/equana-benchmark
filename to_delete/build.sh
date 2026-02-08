#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Building matmul WASM module..."

emcc matmul.c \
    -o matmul.mjs \
    -s WASM=1 \
    -s MODULARIZE=1 \
    -s EXPORT_ES6=1 \
    -s EXPORT_NAME="createMatmulModule" \
    -s EXPORTED_FUNCTIONS='["_dgemm", "_matmul_f64", "_malloc_f64", "_free_f64", "_cleanup", "_malloc", "_free"]' \
    -s EXPORTED_RUNTIME_METHODS='["ccall", "cwrap", "HEAPF64", "getValue", "setValue"]' \
    -s ALLOW_MEMORY_GROWTH=1 \
    -s INITIAL_MEMORY=512MB \
    -s STACK_SIZE=1048576 \
    -O3 \
    -flto \
    -ffast-math \
    -msimd128 \
    -mavx2 \
    -g2 \
    --profiling-funcs

echo "Build complete: matmul.mjs + matmul.wasm"
ls -lh matmul.mjs matmul.wasm
