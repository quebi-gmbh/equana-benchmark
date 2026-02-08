# Native DGEMM Benchmarks

Reference benchmarks for double-precision matrix multiplication (DGEMM) using NumPy/OpenBLAS, Intel MKL, native C/OpenBLAS, MATLAB/MKL, and a custom AVX-512 MEX kernel. Results are displayed on the [Downloads page](https://benchmark.equana.dev/#/downloads) for comparison with the in-browser WASM benchmarks.

## Methodology

All benchmarks use identical methodology:
- **Matrix sizes:** 64, 128, 256, 512, 1024, 2048, 4096
- **Thread counts:** 1, 2, 4, 8, 16
- **Warmup:** 2 untimed DGEMM calls
- **Measurement:** 5 DGEMM calls timed together as one block, mean reported

## Benchmarks

### 1. NumPy / OpenBLAS

Benchmarks NumPy's `@` operator (backed by OpenBLAS) across four SIMD architecture targets.

```bash
# Prerequisites: Python 3.8+, NumPy with OpenBLAS
pip install numpy

# Run
python run_numpy_benchmarks.py
```

Each combination of size/threads/architecture runs in a separate subprocess with `OPENBLAS_CORETYPE` set to force the SIMD target:

| Column | OPENBLAS_CORETYPE | SIMD Level |
|--------|-------------------|------------|
| Scalar | PRESCOTT | No vectorization |
| SSE | NEHALEM | 128-bit XMM |
| AVX2 | HASWELL | 256-bit YMM + FMA |
| AVX-512 | COOPERLAKE | 512-bit ZMM |

### 2. Native C / OpenBLAS

Compiles OpenBLAS from source with four architecture-specific targets, then runs the bundled `gemm.c` benchmark harness.

```bash
# Prerequisites: build-essential, gfortran
sudo apt install -y build-essential gfortran

# Build all four variants (compiles OpenBLAS 4x with different TARGET=)
cd native-openblas
bash build_all.sh

# Run benchmarks
bash run_benchmarks.sh
```

Produces four statically-linked binaries in `native-openblas/bin/`:

| Binary | OpenBLAS TARGET | SIMD Level |
|--------|----------------|------------|
| `dgemm_scalar.goto` | PRESCOTT | No vectorization |
| `dgemm_sse.goto` | NEHALEM | 128-bit XMM |
| `dgemm_avx2.goto` | HASWELL | 256-bit YMM + FMA |
| `dgemm_avx512.goto` | COOPERLAKE | 512-bit ZMM |

### 3. MATLAB / MKL

Benchmarks MATLAB's `A * B` operator (backed by Intel MKL). MKL auto-selects the best SIMD instruction set — `MKL_ENABLE_INSTRUCTIONS` only sets an upper bound and cannot force a lower target, so a single GFLOPS column is reported.

```bash
# Prerequisites: MATLAB R2020a+ with a valid license

# Shell script (separate process per thread count, recommended)
bash run_matlab_benchmarks.sh

# Or single-session quick test
matlab -batch "run_matlab_benchmarks"
```

### 4. Intel MKL (Direct ctypes)

Calls Intel MKL's `cblas_dgemm` directly via Python `ctypes` — no NumPy dependency, no backend ambiguity.

```bash
# Prerequisites: Python 3.8+, Intel MKL runtime
pip install mkl

# Run
python run_mkl_benchmarks.py
```

On Intel CPUs, sweeps SIMD upper bounds via `MKL_ENABLE_INSTRUCTIONS`:

| Column | MKL_ENABLE_INSTRUCTIONS | SIMD Level |
|--------|------------------------|------------|
| SSE4.2 | SSE4_2 | At most SSE4.2 |
| AVX2 | AVX2 | At most AVX2 + FMA |
| AVX-512 | AVX512 | At most AVX-512 |

On AMD CPUs, `MKL_ENABLE_INSTRUCTIONS` has no effect — MKL auto-selects one code path regardless. A `libfakeintel.so` shim is built automatically to bypass MKL's Intel-only CPU check, but only a single GFLOPS column is reported.

### 5. Custom AVX-512 MEX Kernel

A custom DGEMM kernel using explicit AVX-512 intrinsics, compiled as a MATLAB MEX function. Based on the OpenBLAS SkylakeX micro-kernel (`dgemm_kernel_4x8_skylakex.c`). Uses OpenMP to parallelize over M-dimension tiles; thread count is controlled via `OMP_NUM_THREADS`.

Usage from MATLAB:
```matlab
C = mex_avx512_dgemm(A, B)
```

```bash
# Compile (requires GCC with AVX-512 support + OpenMP)
matlab -batch "compile_mex_avx512"

# Or use run_matlab_benchmarks.sh, which auto-compiles if source is present
bash run_matlab_benchmarks.sh
```

The shell script `run_matlab_benchmarks.sh` automatically compiles the MEX kernel if `mex_avx512_dgemm.c` is present, and benchmarks it alongside MKL for each size/thread combination.

## Directory Structure

```
matmul-benchmarks/
├── run_numpy_benchmarks.py          # NumPy benchmark (4 arch × 7 sizes × 5 threads)
├── run_mkl_benchmarks.py            # Intel MKL benchmark (direct ctypes, SIMD sweep)
├── run_matlab_benchmarks.sh         # MATLAB + MEX benchmark (shell wrapper, separate processes)
├── run_matlab_benchmarks.m          # MATLAB benchmark (single-session quick test)
├── compile_mex_avx512.m             # Compile AVX-512 MEX kernel (GCC + OpenMP)
├── mex_avx512_dgemm.c               # Custom AVX-512 DGEMM kernel (MEX function, ~1700 lines)
├── mex_avx512_dgemm.mexa64          # Compiled MEX binary (Linux x86-64)
├── matlab_bench.log                 # MATLAB benchmark results log
├── native-openblas/
│   ├── build_all.sh                 # Build OpenBLAS for 4 architecture targets
│   ├── run_benchmarks.sh            # Run all native DGEMM benchmarks
│   ├── OpenBLAS/                    # OpenBLAS source (git submodule/clone)
│   ├── bin/                         # Compiled benchmark executables
│   └── lib/                         # Compiled static libraries
└── README.md
```
