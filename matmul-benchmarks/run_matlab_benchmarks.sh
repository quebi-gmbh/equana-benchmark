#!/bin/bash
# MATLAB/MKL DGEMM Benchmark
#
# Matches the methodology of the NumPy and native OpenBLAS benchmarks:
# - Sizes: 64, 128, 256, 512, 1024, 2048, 4096
# - Thread counts: 1, 2, 4, 8, 16
# - 2 untimed warmup runs + 5 timed runs measured together
#
# MKL auto-selects the best SIMD instruction set for the CPU.
# MKL_ENABLE_INSTRUCTIONS only sets an upper bound and cannot force a
# specific lower target, so only one GFLOPS column is reported.
#
# Each thread count is run in a separate MATLAB process to ensure
# MKL_NUM_THREADS takes effect at library load time.
#
# Usage:
#   bash run_matlab_benchmarks.sh

SIZES="64 128 256 512 1024 2048 4096"
THREAD_COUNTS="1 2 4 8 16"

TMPSCRIPT=$(mktemp /tmp/matlab_bench_XXXXXX.m)

cleanup() {
    rm -f "$TMPSCRIPT"
}
trap cleanup EXIT

# Run a single benchmark: one size, one thread count
run_single() {
    local SIZE=$1
    local THREADS=$2

    cat > "$TMPSCRIPT" <<'MEOF'
maxNumCompThreads(BENCH_THREADS);
N = BENCH_SIZE;
A = rand(N, N);
B = rand(N, N);
C = A * B;
C = A * B;
tic;
for r = 1:5
    C = A * B;
end
elapsed = toc;
avg_time = elapsed / 5;
gflops = 2 * N^3 / avg_time / 1e9;
fprintf('RESULT|%.6f|%.2f\n', avg_time, gflops);
MEOF

    # Replace placeholders with actual values
    sed -i "s/BENCH_THREADS/$THREADS/g" "$TMPSCRIPT"
    sed -i "s/BENCH_SIZE/$SIZE/g" "$TMPSCRIPT"

    local OUTPUT
    OUTPUT=$(MKL_NUM_THREADS="$THREADS" \
        matlab -batch "run('$TMPSCRIPT')" 2>/dev/null)

    echo "$OUTPUT" | grep "^RESULT|" | head -1 | cut -d'|' -f3
}

# Print table header
print_header() {
    echo "| Size | GFLOPS (MKL) |"
    echo "|------|--------------|"
}

echo "======================================================================"
echo "    MATLAB/MKL DGEMM Benchmark"
echo "======================================================================"
echo ""

# Print MATLAB version
matlab -batch "fprintf('MATLAB version: %s\n', version)" 2>/dev/null
echo ""

for threads in $THREAD_COUNTS; do
    if [ "$threads" -eq 1 ]; then
        label="Single-threaded"
    else
        label="Multi-threaded"
    fi

    echo "### $label (MKL_NUM_THREADS=$threads)"
    echo ""
    print_header

    for size in $SIZES; do
        >&2 printf "  %sT / %sx%s...\r" "$threads" "$size" "$size"

        gflops=$(run_single "$size" "$threads")

        printf "| %4d | %12s |\n" "$size" "${gflops:-N/A}"
    done

    echo ""
done

echo "======================================================================"
echo "Notes:"
echo "- MATLAB uses Intel MKL which auto-selects the best SIMD target"
echo "- MKL_ENABLE_INSTRUCTIONS only sets an upper bound, cannot force lower targets"
echo "- Thread count controlled via MKL_NUM_THREADS + maxNumCompThreads()"
echo "======================================================================"
