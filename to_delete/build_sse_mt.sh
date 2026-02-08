#!/bin/bash
set -e

echo "Building multithreaded SSE-style WASM matmul..."

# SSE-style MT build:
# - Pre-duplicated B packing (no runtime broadcasts)
# - Pthread parallelization
# - Same settings as regular MT build

emcc matmul_sse_mt.c -o matmul_sse_mt.js \
    -O3 -flto \
    -msimd128 \
    -pthread \
    -sPTHREAD_POOL_SIZE='navigator.hardwareConcurrency' \
    -sMALLOC=mimalloc \
    -sINITIAL_MEMORY=512MB \
    -sALLOW_MEMORY_GROWTH=1 \
    -sEXPORTED_FUNCTIONS='["_matmul_sse_mt","_dgemm_sse_mt","_malloc_f64","_free_f64","_cleanup","_set_num_threads","_get_num_threads"]' \
    -sEXPORTED_RUNTIME_METHODS='["ccall","cwrap","setValue","getValue","HEAPF64","wasmMemory"]' \
    -sENVIRONMENT='web,worker' \
    -sEXPORT_ALL=1 \
    -sEXPORT_NAME='createSSEModule' \
    -sMODULARIZE=1

echo ""
echo "Built:"
ls -la matmul_sse_mt.js matmul_sse_mt.wasm matmul_sse_mt.worker.js 2>/dev/null || ls -la matmul_sse_mt.*

echo ""
echo "=== IMPORTANT ==="
echo "To use multithreaded WASM, your server MUST send these headers:"
echo "  Cross-Origin-Embedder-Policy: require-corp"
echo "  Cross-Origin-Opener-Policy: same-origin"
echo ""
echo "Example with Python:"
echo "  python3 server_mt.py"
echo ""
