# WASM Matrix Multiplication Benchmarks

A comprehensive benchmark suite comparing matrix multiplication performance across JavaScript, WebAssembly (SIMD), WebGPU, Python/NumPy, and OpenBLAS implementations.

**Peak Performance:** 36 GFLOPS (single-threaded WASM SIMD) | 100+ GFLOPS (multi-threaded)

---

## Quick Start

```bash
# Build all WASM variants
./build.sh

# Run browser benchmark
node server.mjs
# Open http://localhost:3000/benchmark-wasm.html

# Run Node.js benchmark
node benchmark.mjs

# Run Python/NumPy benchmark
python run_numpy_benchmarks.py
```

---

## Directory Structure

### C Source Files (WASM Implementations)

| File | Description |
|------|-------------|
| `matmul.c` | **Best performer** - Optimized 4×4 micro-kernel with SIMD + cache blocking |
| `matmul_naive.c` | Naive triple-loop baseline |
| `matmul_cache.c` | Cache-blocking only (no SIMD) |
| `matmul_simd.c` | SIMD only (no cache blocking) |
| `matmul_full_2x2.c` | 2×2 micro-kernel variant |
| `matmul_full_6x8.c` | 6×8 micro-kernel variant |
| `matmul_full_8x8.c` | 8×8 micro-kernel variant |
| `matmul_openblas.c` | OpenBLAS-style kernel ported to WASM SIMD |
| `matmul_sse.c` | Pre-duplicated B packing strategy |
| `matmul_mt.c` | Multi-threaded version (pthreads) |
| `matmul_sse_mt.c` | Multi-threaded SSE-style variant |

### Build Scripts

| File | Description |
|------|-------------|
| `build.sh` | Build single-threaded WASM variants |
| `build_mt.sh` | Build multi-threaded with SharedArrayBuffer support |
| `build_sse_mt.sh` | Build SSE-style multi-threaded variant |
| `build_variants.sh` | Build all variants |

### Node.js Benchmark Scripts

| File | Description |
|------|-------------|
| `benchmark.mjs` | Main CLI benchmark comparing WASM vs NumPy |
| `benchmark_all.mjs` | Extended benchmark suite |
| `matmul_benchmark.mjs` | Alternative Node.js benchmark |
| `matmul_ts_benchmark.mjs` | TypeScript-based benchmark |

### Python Benchmark Scripts

| File | Description |
|------|-------------|
| `matmul_benchmark.py` | NumPy single-threaded benchmark |
| `run_numpy_benchmarks.py` | NumPy/OpenBLAS with multiple SIMD configs (SSE3, SSE4.2, AVX, AVX2) |

### HTTP Servers

| File | Description |
|------|-------------|
| `server.mjs` | Node.js server (port 3000) with COOP/COEP headers |
| `server_mt.py` | Python server (port 8080) with COOP/COEP headers for multi-threaded |

### Browser Benchmark HTML Files

| File | Description |
|------|-------------|
| `benchmark-wasm.html` | **Main browser benchmark** - Compares 19 implementations |
| `benchmark-mt.html` | Multi-threaded WASM benchmark with thread scaling |
| `benchmark-pyodide.html` | Pyodide (NumPy in browser) benchmark |
| `index.html` | Full interactive benchmark UI |
| `index-new.html` | Cleaner benchmark interface |

### WebGPU Implementations

| File | Description |
|------|-------------|
| `webgpu-matmul.js` | WebGPU matrix multiplication |
| `webgpu-matmul-dd.js` | WebGPU double-double precision variant |
| `matmul_webgpu.js` | Alternative WebGPU implementation |

### numwa/ - Pre-built BLAS WASM Modules

Pre-built WASM modules from the numwa BLAS library for comparison:

- `blas.wasm` / `blas.cjs` / `blas.mjs` - Single-threaded BLAS
- `blas_mt.wasm` / `blas_mt.js` - Multi-threaded BLAS
- `matmul_standalone.wasm` / `matmul_standalone.mjs` - Standalone matmul
- `matmul_standalone_mt.wasm` / `matmul_standalone_mt.js` - Multi-threaded standalone

### numwa-src/ - BLAS Source Code

Source code for the numwa BLAS implementation:

| File | Description |
|------|-------------|
| `matmul_sse_standalone.c` | SSE-style standalone matmul implementation |
| `matmul_sse_standalone_mt.c` | Multi-threaded variant |
| `level3.c` | Full BLAS Level 3 implementation |
| `level3_mt.c` | Multi-threaded Level 3 |
| `kernel/` | Micro-kernel implementations |
| `build-blas.sh` | Single-threaded build script |
| `build-blas-mt.sh` | Multi-threaded build script |

### Test/Diagnostic Files

| File | Description |
|------|-------------|
| `test-atomic.html` | Atomic operations testing |
| `test-multiply.html` | General multiply operation tests |
| `test-shared-memory.html` | SharedArrayBuffer testing |
| `timer_check.html` | Performance timer verification |

---

## Performance Results (512×512 matrices)

### Single-Threaded Comparison

| Implementation | GFLOPS | vs JS Naive |
|----------------|--------|-------------|
| JS number[] (naive) | 1.44 | 1.0× |
| JS Cache blocked | 3.12 | 2.2× |
| JS Packed 4×4 | 8.97 | 6.2× |
| WASM Naive | 1.47 | 1.0× |
| WASM Cache | 8.53 | 5.9× |
| WASM Full 2×2 | 20.39 | 14.2× |
| **WASM Full 4×4** | **36.28** | **25.2×** |
| WASM Full 6×8 | 27.93 | 19.4× |
| WASM Full 8×8 | 29.20 | 20.3× |

### vs Native NumPy

| Implementation | GFLOPS |
|----------------|--------|
| NumPy (AVX-512, multi-threaded) | ~105 |
| NumPy (SSE3, 1 thread) | ~28 |
| **Our WASM SIMD 4×4** | **~36** |
| Pyodide NumPy (scalar WASM) | ~1.4 |

---

## Key Optimization Techniques

1. **WASM SIMD128** - 2 doubles per vector operation (`f64x2`)
2. **Three-level cache blocking** - GotoBLAS algorithm (MC=64, KC=128, NC=256)
3. **Panel packing** - Contiguous memory access patterns
4. **4×4 micro-kernel** - Optimal for WASM's ~16 vector registers
5. **Multi-threading** - pthreads with SharedArrayBuffer

---

## Build Requirements

```bash
# Install Emscripten
git clone https://github.com/emscripten-core/emsdk.git
cd emsdk
./emsdk install latest
./emsdk activate latest
source ./emsdk_env.sh
```

### Key Compiler Flags

```bash
emcc matmul.c -o matmul.wasm \
    -O3 -flto \
    -msimd128 \
    --no-entry \
    -Wl,--export-dynamic \
    -s INITIAL_MEMORY=256MB
```

---

## Running Benchmarks

### Browser Benchmarks

```bash
# Single-threaded (port 3000)
node server.mjs
# Open http://localhost:3000/benchmark-wasm.html

# Multi-threaded (port 8080, requires COOP/COEP headers)
python3 server_mt.py
# Open http://localhost:8080/benchmark-mt.html
```

### Node.js Benchmarks

```bash
node benchmark.mjs          # Main WASM vs NumPy comparison
node benchmark_all.mjs      # Extended suite
```

### Python/NumPy Benchmarks

```bash
python matmul_benchmark.py      # Basic NumPy benchmark
python run_numpy_benchmarks.py  # Multiple SIMD configurations
```

---

## References

- [GotoBLAS Paper](https://www.cs.utexas.edu/~flame/pubs/GotoTOMS_revision.pdf) - Cache blocking algorithm
- [WASM SIMD Proposal](https://github.com/WebAssembly/simd) - SIMD128 specification
- [Emscripten SIMD](https://emscripten.org/docs/porting/simd.html) - Compiler documentation
- [OpenBLAS](https://github.com/OpenMMAP/OpenBLAS) - Reference BLAS implementation
