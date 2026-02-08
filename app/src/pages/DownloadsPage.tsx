import { PlatformTabs } from '../components/PlatformTabs';
import { ReferenceResultsSection } from '../components/ReferenceResultsSection';
import { useSEO } from '../hooks/useSEO';

const REPO = 'https://github.com/quebi-gmbh/equana-benchmark';
const BLOB = `${REPO}/blob/main/matmul-benchmarks`;

export function DownloadsPage() {
  useSEO({
    title: 'Downloads & Setup',
    description:
      'Download native DGEMM benchmark scripts for Python/NumPy, C/OpenBLAS, and MATLAB/MKL. Run matrix multiplication benchmarks on your own hardware.',
    path: '/downloads',
  });

  return (
    <div className="space-y-10">
      <div>
        <h1 className="text-2xl font-bold tracking-tight text-gray-100">
          Downloads & Setup
        </h1>
        <p className="mt-1 text-sm text-gray-400">
          Run native DGEMM benchmarks on your own hardware for comparison with the browser results.
        </p>
      </div>

      {/* Python / NumPy */}
      <section className="space-y-4">
        <h2 className="text-lg font-semibold text-blue-400">Python / NumPy Benchmark</h2>
        <p className="text-sm text-gray-400">
          Benchmark NumPy (backed by OpenBLAS) across four SIMD architecture targets — Scalar, SSE, AVX2, AVX-512 —
          with 1 to 16 threads. Each combination runs in a separate subprocess with{' '}
          <code className="rounded bg-gray-800 px-1.5 py-0.5 text-xs text-gray-300">OPENBLAS_CORETYPE</code> set to force
          the SIMD target.
        </p>
        <div className="flex flex-wrap gap-3">
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
          Compiles OpenBLAS from source with four architecture-specific targets (PRESCOTT, NEHALEM, HASWELL, COOPERLAKE),
          then runs the bundled <code className="rounded bg-gray-800 px-1.5 py-0.5 text-xs text-gray-300">gemm.c</code> benchmark
          harness. Produces four statically-linked DGEMM binaries.
        </p>
        <div className="flex flex-wrap gap-3">
          <a
            href={`${BLOB}/native-openblas/build_all.sh`}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 rounded-lg border border-gray-700 bg-gray-900 px-4 py-2 text-sm font-medium text-gray-200 transition-colors hover:border-blue-500/50 hover:text-blue-400"
          >
            <DownloadIcon />
            build_all.sh
          </a>
          <a
            href={`${BLOB}/native-openblas/run_benchmarks.sh`}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 rounded-lg border border-gray-700 bg-gray-900 px-4 py-2 text-sm font-medium text-gray-200 transition-colors hover:border-blue-500/50 hover:text-blue-400"
          >
            <DownloadIcon />
            run_benchmarks.sh
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
        <div className="rounded-lg border border-gray-800 bg-gray-900/30 p-4 text-sm text-gray-400">
          <span className="font-medium text-gray-300">Prerequisites:</span>{' '}
          <code className="rounded bg-gray-800 px-1.5 py-0.5 text-xs text-gray-300">build-essential</code>,{' '}
          <code className="rounded bg-gray-800 px-1.5 py-0.5 text-xs text-gray-300">gfortran</code> (for OpenBLAS compilation)
        </div>
      </section>

      {/* MATLAB / MKL */}
      <section className="space-y-4">
        <h2 className="text-lg font-semibold text-blue-400">MATLAB / MKL Benchmark</h2>
        <p className="text-sm text-gray-400">
          Benchmarks MATLAB's <code className="rounded bg-gray-800 px-1.5 py-0.5 text-xs text-gray-300">A * B</code> operator
          backed by Intel MKL. MKL auto-selects the best SIMD instruction set for the CPU —{' '}
          <code className="rounded bg-gray-800 px-1.5 py-0.5 text-xs text-gray-300">MKL_ENABLE_INSTRUCTIONS</code> only
          sets an upper bound, so a single GFLOPS column is reported. The shell script launches a separate MATLAB process
          per thread count to ensure <code className="rounded bg-gray-800 px-1.5 py-0.5 text-xs text-gray-300">MKL_NUM_THREADS</code> takes
          effect at library load time.
        </p>
        <div className="flex flex-wrap gap-3">
          <a
            href={`${BLOB}/run_matlab_benchmarks.sh`}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 rounded-lg border border-gray-700 bg-gray-900 px-4 py-2 text-sm font-medium text-gray-200 transition-colors hover:border-blue-500/50 hover:text-blue-400"
          >
            <DownloadIcon />
            run_matlab_benchmarks.sh
          </a>
          <a
            href={`${BLOB}/run_matlab_benchmarks.m`}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 rounded-lg border border-gray-700 bg-gray-900 px-4 py-2 text-sm font-medium text-gray-200 transition-colors hover:border-blue-500/50 hover:text-blue-400"
          >
            <DownloadIcon />
            run_matlab_benchmarks.m
          </a>
        </div>
        <div className="rounded-lg border border-gray-800 bg-gray-900/30 p-4 text-sm text-gray-400">
          <span className="font-medium text-gray-300">Prerequisites:</span> MATLAB R2020a+ with a valid license
        </div>
      </section>

      {/* Platform Setup */}
      <section className="space-y-4">
        <h2 className="text-lg font-semibold text-blue-400">Build Instructions</h2>
        <p className="text-sm text-gray-400">
          Platform-specific instructions for cloning the repo and running all benchmarks:
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
