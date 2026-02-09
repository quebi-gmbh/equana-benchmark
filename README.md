# Equana Benchmark

In-browser matrix multiplication benchmark comparing JavaScript, WebAssembly SIMD, multi-threaded WASM (pthreads), and Pyodide (Python/NumPy) implementations. All benchmarks use double-precision (f64) arithmetic.

**Live:** [benchmark.equana.dev](https://benchmark.equana.dev)

## Benchmark Variants

### Pyodide (1)

| Variant | Description |
|---------|-------------|
| Pyodide NumPy | NumPy A @ B via Pyodide (OpenBLAS compiled to scalar WASM) |

### JavaScript (4)

| Variant | Description |
|---------|-------------|
| JS Naive (number[]) | Baseline triple-loop |
| JS Naive (Float64Array) | Typed arrays for better JIT optimization |
| JS Cache-Opt (number[]) | 64x64 loop tiling for cache locality |
| JS Packed 4x4 (number[]) | DGEMM-style: 3-level blocking + packing + 4x4 micro-kernel |

### WebAssembly Single-Threaded (8)

| Variant | Description |
|---------|-------------|
| WASM Naive | Triple-loop compiled to WASM |
| WASM Cache | 64x64 cache blocking, no SIMD |
| WASM SIMD | f64x2 SIMD vectors without cache blocking |
| WASM Full 4x4 | Cache + SIMD + packing — optimal for WASM registers |
| WASM Full 6x8 | 24 f64x2 accumulators — common BLAS size |
| WASM Full 8x8 | 32 f64x2 accumulators — register spilling risk |
| WASM OpenBLAS | OpenBLAS DGEMM compiled to WASM |
| WASM SSE-style | SSE-style lane duplication approach |

### WebAssembly Multi-Threaded (2)

| Variant | Description |
|---------|-------------|
| WASM Full 4x4 MT | Multi-threaded Full 4x4 with Emscripten pthreads |
| WASM SSE-style MT | Multi-threaded SSE-style lane duplication |

Multi-threaded variants require `SharedArrayBuffer`, which needs cross-origin isolation (COOP/COEP headers). This is handled automatically via [coi-serviceworker](https://github.com/nickernickerknickers/nickernickerknickers-coi-serviceworker).

## Tech Stack

- **React 19** + **TypeScript**
- **Vite** (build tool)
- **Tailwind CSS 4**
- **React Aria Components** (accessible UI primitives)
- **React Router** (hash-based routing for GitHub Pages)
- **coi-serviceworker** (cross-origin isolation for `SharedArrayBuffer`)

## Development

```bash
cd app
pnpm install
pnpm dev
```

The dev server serves at `http://localhost:5173` with COOP/COEP headers enabled.

## Production Build

```bash
cd app
pnpm build
```

Output is in `app/dist/`. The build includes all WASM modules in `dist/wasm/` and the `coi-serviceworker.js` for cross-origin isolation on static hosts.

## Project Structure

```
app/
├── public/
│   ├── coi-serviceworker.js          # Cross-origin isolation for SharedArrayBuffer
│   └── wasm/                         # Pre-compiled WASM modules
│       ├── matmul_*.wasm             # Single-threaded WASM variants
│       ├── matmul_mt.js + .wasm      # Multi-threaded (non-modularized)
│       └── matmul_sse_mt.js + .wasm  # Multi-threaded (modularized)
├── src/
│   ├── engine/
│   │   ├── types.ts                  # Core type definitions
│   │   ├── matrixUtils.ts            # Matrix generation and GFLOPS calculation
│   │   ├── benchmarkRegistry.ts      # All 15 benchmark variant definitions
│   │   ├── scheduler.ts              # Sequential benchmark execution with UI yield
│   │   ├── wasmLoader.ts             # Standalone WASM loader (fetch + instantiate)
│   │   ├── mtLoader.ts               # Multi-threaded WASM loader (script injection)
│   │   ├── pyodideLoader.ts          # Pyodide runtime loader (CDN script injection)
│   │   └── runners/
│   │       ├── jsRunner.ts           # JavaScript benchmark runner
│   │       ├── wasmRunner.ts         # Single-threaded WASM runner
│   │       ├── mtRunner.ts           # Multi-threaded WASM runner
│   │       └── pyodideRunner.ts      # Pyodide/NumPy benchmark runner
│   ├── matmul/
│   │   ├── naiveArray.ts             # Triple-loop with number[]
│   │   ├── naiveFloat64.ts           # Triple-loop with Float64Array
│   │   ├── cacheArray.ts             # 64x64 loop tiling
│   │   └── packedArray.ts            # Full DGEMM-style with 4x4 micro-kernel
│   ├── components/
│   │   ├── Layout.tsx                # App shell with nav and footer
│   │   ├── NavBar.tsx                # Navigation links
│   │   ├── MatrixSizePicker.tsx      # Matrix size selector (128–4000)
│   │   ├── RoundsPicker.tsx          # Benchmark rounds selector (1/3/5)
│   │   ├── ThreadCountPicker.tsx     # Thread count selector (1–32)
│   │   ├── RunAllButton.tsx          # Run all benchmarks with progress
│   │   ├── BenchmarkTable.tsx        # Results table with ST/MT sections
│   │   ├── BenchmarkRow.tsx          # Individual benchmark row
│   │   ├── StatusBadge.tsx           # Idle/running/done/error indicator
│   │   ├── CodeBlock.tsx             # Code display with copy button
│   │   ├── PlatformTabs.tsx          # OS-specific build instructions
│   │   ├── ReferenceResultsSection.tsx # Native benchmark reference results display
│   │   └── ReferenceTable.tsx        # GFLOPS comparison table
│   ├── data/
│   │   └── referenceResults.ts       # Hardcoded native DGEMM reference results
│   ├── hooks/
│   │   ├── useBenchmarkStore.ts      # Reducer-based state management
│   │   └── useSEO.ts                 # Dynamic document title, meta tags, and canonical URL
│   ├── pages/
│   │   ├── BenchmarkPage.tsx         # Main benchmark UI
│   │   └── DownloadsPage.tsx         # Native benchmark downloads & setup
│   ├── router.tsx                    # Hash router configuration
│   ├── index.tsx                     # Entry point
│   └── index.css                     # Tailwind imports + theme tokens
└── vite.config.ts                    # Vite config with COOP/COEP dev headers
```

## Native Benchmarks

The `matmul-benchmarks/` directory contains reference DGEMM benchmarks for comparing browser results against native performance. All use identical methodology: 2 warmup runs + 5 timed runs measured together.

- **NumPy / OpenBLAS** — `python run_numpy_benchmarks.py` — sweeps Scalar/SSE/AVX2/AVX-512 via `OPENBLAS_CORETYPE`
- **Intel MKL (direct)** — `python run_mkl_benchmarks.py` — calls `cblas_dgemm` via ctypes; MKL auto-selects AVX2
- **Native C / OpenBLAS** — `bash native-openblas/run_benchmarks.sh` — four architecture-specific OpenBLAS builds
- **MATLAB / MKL** — `bash run_matlab_benchmarks.sh` — Intel MKL (auto-selects best SIMD target)
- **Custom AVX-512 MEX** — `mex_avx512_dgemm.c` + `compile_mex_avx512.m` — AVX-512 DGEMM kernel callable from MATLAB, OpenMP multi-threaded

See the [Downloads page](https://benchmark.equana.dev/#/downloads) in the app for full instructions, or `matmul-benchmarks/README.md` for details.

## License

MIT
