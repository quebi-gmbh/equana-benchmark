#!/bin/bash
set -e

echo "Building multithreaded WASM matmul..."

# Multithreaded build requires:
# - pthread flag for both compile and link
# - PTHREAD_POOL_SIZE to pre-create workers
# - PROXY_TO_PTHREAD to move main() off browser thread
# - mimalloc for better MT memory allocation
# - SharedArrayBuffer support (requires COOP/COEP headers)

emcc matmul_mt.c -o matmul_mt.js \
    -O3 -flto \
    -msimd128 \
    -pthread \
    -sPTHREAD_POOL_SIZE='navigator.hardwareConcurrency' \
    -sMALLOC=mimalloc \
    -sINITIAL_MEMORY=512MB \
    -sALLOW_MEMORY_GROWTH=1 \
    -sEXPORTED_FUNCTIONS='["_matmul_f64_mt","_dgemm_mt","_malloc_f64","_free_f64","_cleanup","_set_num_threads","_get_num_threads"]' \
    -sEXPORTED_RUNTIME_METHODS='["ccall","cwrap","setValue","getValue","HEAPF64","wasmMemory"]' \
    -sENVIRONMENT='web,worker' \
    -sEXPORT_ALL=1 \
    -sMODULARIZE=0

echo ""
echo "Built:"
ls -la matmul_mt.js matmul_mt.wasm matmul_mt.worker.js 2>/dev/null || ls -la matmul_mt.*

echo ""
echo "=== IMPORTANT ==="
echo "To use multithreaded WASM, your server MUST send these headers:"
echo "  Cross-Origin-Embedder-Policy: require-corp"
echo "  Cross-Origin-Opener-Policy: same-origin"
echo ""
echo "Example with Python:"
echo "  python3 server_mt.py"
echo ""
