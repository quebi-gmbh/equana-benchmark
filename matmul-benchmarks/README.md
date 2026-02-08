# Native DGEMM Benchmarks

Reference benchmarks for double-precision matrix multiplication (DGEMM) using NumPy/OpenBLAS, native C/OpenBLAS, and MATLAB/MKL. Results are displayed on the [Downloads page](https://benchmark.equana.dev/#/downloads) for comparison with the in-browser WASM benchmarks.

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

## Directory Structure

```
matmul-benchmarks/
├── run_numpy_benchmarks.py          # NumPy benchmark (4 arch × 7 sizes × 5 threads)
├── run_matlab_benchmarks.sh         # MATLAB benchmark (shell wrapper, separate processes)
├── run_matlab_benchmarks.m          # MATLAB benchmark (single-session quick test)
├── native-openblas/
│   ├── build_all.sh                 # Build OpenBLAS for 4 architecture targets
│   ├── run_benchmarks.sh            # Run all native DGEMM benchmarks
│   ├── OpenBLAS/                    # OpenBLAS source (git submodule/clone)
│   ├── bin/                         # Compiled benchmark executables
│   └── lib/                         # Compiled static libraries
└── README.md
```
