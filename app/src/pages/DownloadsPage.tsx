import { PlatformTabs } from '../components/PlatformTabs';
import { ReferenceResultsSection } from '../components/ReferenceResultsSection';

const REPO = 'https://github.com/quebi-gmbh/equana-benchmark';
const BLOB = `${REPO}/blob/main/matmul-benchmarks`;

export function DownloadsPage() {
  return (
    <div className="space-y-10">
      <div>
        <h1 className="text-2xl font-bold tracking-tight text-gray-100">
          Downloads & Setup
        </h1>
        <p className="mt-1 text-sm text-gray-400">
          Run native benchmarks on your own hardware for comparison with the browser results.
        </p>
      </div>

      {/* Python / NumPy */}
      <section className="space-y-4">
        <h2 className="text-lg font-semibold text-blue-400">Python / NumPy Benchmark</h2>
        <p className="text-sm text-gray-400">
          Benchmark NumPy (backed by OpenBLAS) across different SIMD instruction sets — SSE3, SSE4.2, AVX, AVX2 —
          both single-threaded and multi-threaded. Great for comparing against the WASM results from the browser benchmark.
        </p>
        <div className="flex flex-wrap gap-3">
          <a
            href={`${BLOB}/matmul_benchmark.py`}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 rounded-lg border border-gray-700 bg-gray-900 px-4 py-2 text-sm font-medium text-gray-200 transition-colors hover:border-blue-500/50 hover:text-blue-400"
          >
            <DownloadIcon />
            matmul_benchmark.py
          </a>
          <a
            href={`${BLOB}/run_numpy_benchmarks.py`}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 rounded-lg border border-gray-700 bg-gray-900 px-4 py-2 text-sm font-medium text-gray-200 transition-colors hover:border-blue-500/50 hover:text-blue-400"
          >
            <DownloadIcon />
            run_numpy_benchmarks.py
          </a>
        </div>
        <div className="rounded-lg border border-gray-800 bg-gray-900/30 p-4 text-sm text-gray-400">
          <span className="font-medium text-gray-300">Prerequisites:</span> Python 3.8+, NumPy with OpenBLAS backend (
          <code className="rounded bg-gray-800 px-1.5 py-0.5 text-xs text-gray-300">pip install numpy</code>)
        </div>
      </section>

      {/* Native C / OpenBLAS */}
      <section className="space-y-4">
        <h2 className="text-lg font-semibold text-blue-400">Native C / OpenBLAS Benchmark</h2>
        <p className="text-sm text-gray-400">
          Compile and run the matrix multiplication benchmark natively on your CPU.
          This uses the same GotoBLAS-style algorithm as the WASM versions, but with native SIMD instructions (SSE/AVX/NEON).
        </p>
        <div className="flex flex-wrap gap-3">
          <a
            href={`${BLOB}/matmul_openblas.c`}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 rounded-lg border border-gray-700 bg-gray-900 px-4 py-2 text-sm font-medium text-gray-200 transition-colors hover:border-blue-500/50 hover:text-blue-400"
          >
            <DownloadIcon />
            matmul_openblas.c
          </a>
          <a
            href={`${BLOB}/matmul.c`}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 rounded-lg border border-gray-700 bg-gray-900 px-4 py-2 text-sm font-medium text-gray-200 transition-colors hover:border-blue-500/50 hover:text-blue-400"
          >
            <DownloadIcon />
            matmul.c (4x4 micro-kernel)
          </a>
          <a
            href={REPO}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 rounded-lg border border-gray-700 bg-gray-900 px-4 py-2 text-sm font-medium text-gray-200 transition-colors hover:border-blue-500/50 hover:text-blue-400"
          >
            <GithubIcon />
            Full Repository
          </a>
        </div>
      </section>

      {/* Platform Setup */}
      <section className="space-y-4">
        <h2 className="text-lg font-semibold text-blue-400">Build Instructions</h2>
        <p className="text-sm text-gray-400">
          Platform-specific instructions for compiling and running the native benchmark:
        </p>
        <PlatformTabs />
      </section>

      {/* Reference Results */}
      <ReferenceResultsSection />
    </div>
  );
}

function DownloadIcon() {
  return (
    <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
    </svg>
  );
}

function GithubIcon() {
  return (
    <svg className="h-4 w-4" fill="currentColor" viewBox="0 0 24 24">
      <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z" />
    </svg>
  );
}
