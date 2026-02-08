#!/bin/bash
# Run OpenBLAS DGEMM benchmarks for all architecture variants

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Matrix sizes to benchmark
SIZES="64 128 256 512 1024 2048 4096"

# Benchmark executables
SCALAR="$SCRIPT_DIR/bin/dgemm_scalar.goto"
SSE="$SCRIPT_DIR/bin/dgemm_sse.goto"
AVX2="$SCRIPT_DIR/bin/dgemm_avx2.goto"
AVX512="$SCRIPT_DIR/bin/dgemm_avx512.goto"

# Function to extract GFLOPS from benchmark output
run_benchmark() {
    local EXE=$1
    local SIZE=$2
    local THREADS=$3

    if [ ! -f "$EXE" ]; then
        echo "N/A"
        return
    fi

    # 2 untimed warmup runs (output discarded)
    OPENBLAS_NUM_THREADS=$THREADS OPENBLAS_LOOPS=2 "$EXE" $SIZE $SIZE 1 >/dev/null 2>&1

    # 5 runs timed together (OPENBLAS_LOOPS), mean reported by gemm.c
    # Output format: " M=1024, N=1024, K=1024 :   105966.60 MFlops   0.020266 sec"
    OPENBLAS_NUM_THREADS=$THREADS OPENBLAS_LOOPS=5 "$EXE" $SIZE $SIZE 1 2>&1 | \
        grep "MFlops" | \
        sed 's/.*: *\([0-9.]*\) MFlops.*/\1/' | \
        awk '{printf "%.1f", $1/1000}' | \
        head -1
}

# Print header
print_header() {
    echo "| Size | Scalar (GFLOPS) | SSE (GFLOPS) | AVX2 (GFLOPS) | AVX-512 (GFLOPS) |"
    echo "|------|-----------------|--------------|---------------|------------------|"
}

echo "============================================================"
echo "    OpenBLAS DGEMM Benchmark - Architecture Comparison"
echo "============================================================"
echo ""

# Single-threaded benchmarks
echo "### Single-threaded (OPENBLAS_NUM_THREADS=1)"
echo ""
print_header

for size in $SIZES; do
    scalar_gflops=$(run_benchmark "$SCALAR" $size 1)
    sse_gflops=$(run_benchmark "$SSE" $size 1)
    avx2_gflops=$(run_benchmark "$AVX2" $size 1)
    avx512_gflops=$(run_benchmark "$AVX512" $size 1)

    printf "| %4d | %15s | %12s | %13s | %16s |\n" \
        $size "$scalar_gflops" "$sse_gflops" "$avx2_gflops" "$avx512_gflops"
done

echo ""

# Run multi-threaded benchmarks for various thread counts
for threads in 2 4 8 16; do
    echo ""
    echo "### Multi-threaded (OPENBLAS_NUM_THREADS=$threads)"
    echo ""
    print_header

    for size in $SIZES; do
        scalar_gflops=$(run_benchmark "$SCALAR" $size $threads)
        sse_gflops=$(run_benchmark "$SSE" $size $threads)
        avx2_gflops=$(run_benchmark "$AVX2" $size $threads)
        avx512_gflops=$(run_benchmark "$AVX512" $size $threads)

        printf "| %4d | %15s | %12s | %13s | %16s |\n" \
            $size "$scalar_gflops" "$sse_gflops" "$avx2_gflops" "$avx512_gflops"
    done
done

echo ""
echo "============================================================"
echo "Notes:"
echo "- Scalar: PRESCOTT target with -fno-tree-vectorize"
echo "- SSE: NEHALEM target (128-bit XMM registers)"
echo "- AVX2: HASWELL target (256-bit YMM registers + FMA)"
echo "- AVX-512: COOPERLAKE target (512-bit ZMM registers)"
echo "============================================================"
