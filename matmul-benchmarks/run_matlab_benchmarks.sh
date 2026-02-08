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
# specific lower target, so only one MKL GFLOPS column is reported.
#
# If mex_avx512_dgemm.c is present, it is compiled and benchmarked
# alongside MKL. OMP_NUM_THREADS controls the MEX kernel's thread count.
#
# Each thread count is run in a separate MATLAB process to ensure
# MKL_NUM_THREADS and OMP_NUM_THREADS take effect at library load time.
#
# Usage:
#   bash run_matlab_benchmarks.sh

SIZES="64 128 256 512 1024 2048 4096"
THREAD_COUNTS="1 2 4 8 16"
MAX_RETRIES=2
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOGFILE="$SCRIPT_DIR/matlab_bench.log"

TMPSCRIPT=$(mktemp /tmp/matlab_bench_XXXXXX.m)

cleanup() {
    rm -f "$TMPSCRIPT"
}
trap cleanup EXIT

# Start fresh log
echo "=== MATLAB Benchmark Log — $(date) ===" > "$LOGFILE"

# Compile AVX-512 MEX kernel if source exists and .mexa64 is missing/stale
compile_mex() {
    local SRC="$SCRIPT_DIR/mex_avx512_dgemm.c"
    local MEX_OUT="$SCRIPT_DIR/mex_avx512_dgemm.mexa64"

    if [ ! -f "$SRC" ]; then
        echo "AVX-512 MEX source not found — skipping MEX compilation"
        return 1
    fi

    if [ -f "$MEX_OUT" ] && [ "$MEX_OUT" -nt "$SRC" ]; then
        echo "AVX-512 MEX kernel already compiled (up to date)"
        return 0
    fi

    echo "Compiling AVX-512 MEX kernel..."
    matlab -batch "cd('$SCRIPT_DIR'); compile_mex_avx512" 2>/dev/null
    if [ -f "$MEX_OUT" ]; then
        echo "AVX-512 MEX kernel compiled successfully"
        return 0
    else
        echo "AVX-512 MEX compilation failed — will benchmark MKL only"
        return 1
    fi
}

# Run a single benchmark: one size, one thread count
# Outputs: mkl_gflops|mex_gflops (mex_gflops is N/A if MEX unavailable)
run_single() {
    local SIZE=$1
    local THREADS=$2
    local HAS_MEX=$3

    cat > "$TMPSCRIPT" <<'MEOF'
try
    maxNumCompThreads(BENCH_THREADS);
    N = BENCH_SIZE;
    A = rand(N, N);
    B = rand(N, N);

    % --- MKL benchmark ---
    C = A * B;
    C = A * B;
    tic;
    for r = 1:5
        C = A * B;
    end
    elapsed = toc;
    mkl_avg = elapsed / 5;
    mkl_gflops = 2 * N^3 / mkl_avg / 1e9;

    % --- AVX-512 MEX benchmark ---
    if exist('mex_avx512_dgemm', 'file') == 3
        C = mex_avx512_dgemm(A, B);
        C = mex_avx512_dgemm(A, B);
        tic;
        for r = 1:5
            C = mex_avx512_dgemm(A, B);
        end
        elapsed = toc;
        mex_avg = elapsed / 5;
        mex_gflops = 2 * N^3 / mex_avg / 1e9;
        fprintf('RESULT|%.6f|%.2f|%.6f|%.2f\n', mkl_avg, mkl_gflops, mex_avg, mex_gflops);
    else
        fprintf('RESULT|%.6f|%.2f|N/A|N/A\n', mkl_avg, mkl_gflops);
    end
catch e
    fprintf('RESULT|ERROR|N/A|ERROR|N/A\n');
    fprintf(2, 'MATLAB error: %s\n', e.message);
end
MEOF

    # Replace placeholders with actual values
    sed -i "s/BENCH_THREADS/$THREADS/g" "$TMPSCRIPT"
    sed -i "s/BENCH_SIZE/$SIZE/g" "$TMPSCRIPT"

    local OUTPUT LINE MKL_GF MEX_GF

    echo "--- run_single SIZE=$SIZE THREADS=$THREADS $(date +%H:%M:%S) ---" >> "$LOGFILE"

    OUTPUT=$(MKL_NUM_THREADS="$THREADS" OMP_NUM_THREADS="$THREADS" \
        matlab -batch "addpath('$SCRIPT_DIR'); run('$TMPSCRIPT')" 2>&1)
    local EXIT_CODE=$?

    # Log everything
    echo "$OUTPUT" >> "$LOGFILE"
    echo "exit_code=$EXIT_CODE" >> "$LOGFILE"

    LINE=$(echo "$OUTPUT" | grep "^RESULT|" | head -1)
    MKL_GF=$(echo "$LINE" | cut -d'|' -f3)
    MEX_GF=$(echo "$LINE" | cut -d'|' -f5)

    echo "${MKL_GF:-N/A}|${MEX_GF:-N/A}"
}

# Print table header
print_header() {
    local HAS_MEX=$1
    if [ "$HAS_MEX" = "1" ]; then
        echo "| Size | GFLOPS (MKL) | GFLOPS (AVX-512 MEX) |"
        echo "|------|--------------|----------------------|"
    else
        echo "| Size | GFLOPS (MKL) |"
        echo "|------|--------------|"
    fi
}

echo "======================================================================"
echo "    MATLAB/MKL DGEMM Benchmark"
echo "======================================================================"
echo ""

# Print MATLAB version
matlab -batch "fprintf('MATLAB version: %s\n', version)" 2>/dev/null
echo ""

# Compile MEX
HAS_MEX=0
if compile_mex; then
    HAS_MEX=1
fi
echo ""

for threads in $THREAD_COUNTS; do
    if [ "$threads" -eq 1 ]; then
        label="Single-threaded"
    else
        label="Multi-threaded"
    fi

    echo "### $label (threads=$threads)"
    echo ""
    print_header "$HAS_MEX"

    for size in $SIZES; do
        mkl_gflops="N/A"
        mex_gflops="N/A"

        for attempt in $(seq 1 $((MAX_RETRIES + 1))); do
            >&2 printf "  %sT / %sx%s... (attempt %d)\r" "$threads" "$size" "$size" "$attempt"

            result=$(run_single "$size" "$threads" "$HAS_MEX")
            mkl_gflops=$(echo "$result" | cut -d'|' -f1)
            mex_gflops=$(echo "$result" | cut -d'|' -f2)

            # If we got at least the MKL result, accept it
            if [ "$mkl_gflops" != "N/A" ] && [ -n "$mkl_gflops" ]; then
                break
            fi

            if [ "$attempt" -le "$MAX_RETRIES" ]; then
                echo "  [retry] ${threads}T / ${size}x${size} — attempt $attempt failed, retrying..." >> "$LOGFILE"
                sleep 1
            fi
        done

        if [ "$HAS_MEX" = "1" ]; then
            printf "| %4d | %12s | %20s |\n" "$size" "${mkl_gflops:-N/A}" "${mex_gflops:-N/A}"
        else
            printf "| %4d | %12s |\n" "$size" "${mkl_gflops:-N/A}"
        fi
    done

    echo ""
done

echo "======================================================================"
echo "Notes:"
echo "- MATLAB uses Intel MKL which auto-selects the best SIMD target"
echo "- MKL_ENABLE_INSTRUCTIONS only sets an upper bound, cannot force lower targets"
echo "- MKL thread count: MKL_NUM_THREADS + maxNumCompThreads()"
if [ "$HAS_MEX" = "1" ]; then
    echo "- AVX-512 MEX thread count: OMP_NUM_THREADS (set per MATLAB process)"
fi
echo "- Full MATLAB output logged to: $LOGFILE"
echo "======================================================================"
