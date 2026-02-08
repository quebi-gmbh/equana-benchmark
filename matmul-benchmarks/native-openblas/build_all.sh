#!/bin/bash
# Build OpenBLAS DGEMM benchmarks for multiple instruction set levels
# AVX-512 (CooperLake), AVX2 (Haswell), SSE (Nehalem), Scalar (no SIMD)
#
# Uses a single shared OpenBLAS source directory (symlinked to reference)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OPENBLAS_SRC="$SCRIPT_DIR/OpenBLAS"

# Create output directories
mkdir -p "$SCRIPT_DIR/bin"
mkdir -p "$SCRIPT_DIR/lib"

echo "=== OpenBLAS Multi-Architecture Build ==="
echo "Using shared source: $OPENBLAS_SRC"
echo ""

# Function to build OpenBLAS variant
build_variant() {
    local NAME=$1
    local TARGET=$2
    local EXTRA_FLAGS=$3

    echo "----------------------------------------"
    echo "Building $NAME (TARGET=$TARGET)"
    echo "----------------------------------------"

    cd "$OPENBLAS_SRC"

    # Clean previous build
    make clean 2>/dev/null || true
    rm -f Makefile.conf config.h getarch getarch_2nd

    # Build
    echo "Building with: make TARGET=$TARGET $EXTRA_FLAGS -j$(nproc)"
    make TARGET=$TARGET $EXTRA_FLAGS -j$(nproc)

    # Get the library name
    LIB_NAME=$(ls libopenblas_*.a 2>/dev/null | grep -v "^libopenblas.a$" | head -1)
    if [ -z "$LIB_NAME" ]; then
        LIB_NAME="libopenblas.a"
    fi

    # Copy library
    cp "$LIB_NAME" "$SCRIPT_DIR/lib/libopenblas_${NAME}.a"

    # Build benchmark
    cd benchmark
    cc -O2 -DMAX_STACK_ALLOC=2048 -Wall -m64 -I.. -c -UCOMPLEX -DDOUBLE -o dgemm.o gemm.c
    cc -o dgemm.goto dgemm.o "../$LIB_NAME" -lm -lpthread

    # Copy benchmark executable
    cp dgemm.goto "$SCRIPT_DIR/bin/dgemm_${NAME}.goto"
    echo "Built: $SCRIPT_DIR/bin/dgemm_${NAME}.goto"
    echo ""

    cd "$SCRIPT_DIR"
}

# 1. AVX-512 (CooperLake)
build_variant "avx512" "COOPERLAKE" ""

# 2. AVX2 (Haswell)
build_variant "avx2" "HASWELL" "NO_AVX512=1"

# 3. SSE (Nehalem)
build_variant "sse" "NEHALEM" "NO_AVX=1 NO_AVX2=1 NO_AVX512=1"

# 4. Scalar (PRESCOTT - oldest x86_64 target)
build_variant "scalar" "PRESCOTT" "NO_AVX=1 NO_AVX2=1 NO_AVX512=1"

echo "=== Build Complete ==="
echo ""
echo "Executables in $SCRIPT_DIR/bin/:"
ls -la "$SCRIPT_DIR/bin/"
echo ""
echo "Libraries in $SCRIPT_DIR/lib/:"
ls -la "$SCRIPT_DIR/lib/"
