#!/usr/bin/env python3
"""
Intel MKL DGEMM benchmark via direct ctypes calls to libmkl_rt.

Calls cblas_dgemm directly — no NumPy dependency, no backend ambiguity.

MKL auto-selects its SIMD code path internally — in practice it uses
AVX2 regardless, so a single GFLOPS column is reported.

Same methodology as the other benchmarks:
- Sizes: 64, 128, 256, 512, 1024, 2048, 4096
- Thread counts: 1, 2, 4, 8, 16
- 2 untimed warmup runs + 5 timed runs measured together

Prerequisites: pip install mkl

Usage:
  python run_mkl_benchmarks.py
"""

import os
import subprocess
import sys

SIZES = [64, 128, 256, 512, 1024, 2048, 4096]
THREAD_COUNTS = [1, 2, 4, 8, 16]


def get_benchmark_code(N):
    return f'''
import ctypes
import ctypes.util
import os
import sys
import time
import random

# ---------- locate MKL runtime library ----------
def find_mkl():
    """Find libmkl_rt shared library."""
    # 1. Try ctypes.util.find_library
    path = ctypes.util.find_library("mkl_rt")
    if path:
        return ctypes.CDLL(path)

    # 2. Try common pip install locations
    search_dirs = []

    # pip installs MKL libs into {{sys.prefix}}/lib on Linux
    search_dirs.append(os.path.join(sys.prefix, "lib"))

    # Also check relative to site-packages via importlib
    import importlib.util
    spec = importlib.util.find_spec("mkl")
    if spec and spec.origin:
        mkl_dir = os.path.dirname(spec.origin)
        search_dirs.append(mkl_dir)
        search_dirs.append(os.path.join(mkl_dir, "..", "lib"))

    lib_names = ["libmkl_rt.so", "libmkl_rt.so.2", "mkl_rt.2.dll", "mkl_rt.dll"]
    for d in search_dirs:
        for lib in lib_names:
            candidate = os.path.join(d, lib)
            if os.path.isfile(candidate):
                return ctypes.CDLL(candidate)

    # 3. Try direct load (relies on LD_LIBRARY_PATH / system paths)
    for name in lib_names:
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue

    print("ERROR: Could not find libmkl_rt. Install MKL with: pip install mkl", file=sys.stderr)
    sys.exit(1)

mkl = find_mkl()

# ---------- cblas_dgemm signature ----------
CblasRowMajor = 101
CblasNoTrans = 111

cblas_dgemm = mkl.cblas_dgemm
cblas_dgemm.restype = None
cblas_dgemm.argtypes = [
    ctypes.c_int,      # Layout
    ctypes.c_int,      # TransA
    ctypes.c_int,      # TransB
    ctypes.c_int,      # M
    ctypes.c_int,      # N
    ctypes.c_int,      # K
    ctypes.c_double,   # alpha
    ctypes.c_void_p,   # A
    ctypes.c_int,      # lda
    ctypes.c_void_p,   # B
    ctypes.c_int,      # ldb
    ctypes.c_double,   # beta
    ctypes.c_void_p,   # C
    ctypes.c_int,      # ldc
]

# ---------- allocate matrices ----------
N = {N}
n_elements = N * N

ArrayType = ctypes.c_double * n_elements
A = ArrayType()
B = ArrayType()
C = ArrayType()

# Fill with random values
rng = random.Random(42)
for i in range(n_elements):
    A[i] = rng.random()
    B[i] = rng.random()

alpha = ctypes.c_double(1.0)
beta = ctypes.c_double(0.0)

def dgemm():
    cblas_dgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        N, N, N,
        alpha, A, N,
        B, N,
        beta, C, N,
    )

# 2 untimed warmup runs
dgemm()
dgemm()

# 5 runs timed together
RUNS = 5
start = time.perf_counter()
for _ in range(RUNS):
    dgemm()
elapsed = time.perf_counter() - start

avg = elapsed / RUNS
gflops = 2 * N**3 / avg / 1e9
print(f"RESULT|{{avg:.6f}}|{{gflops:.2f}}")
'''


def run_single(N, threads):
    """Run a single benchmark with given size and thread count."""
    env = dict(os.environ)
    env["MKL_NUM_THREADS"] = str(threads)
    env["OMP_NUM_THREADS"] = str(threads)

    try:
        result = subprocess.run(
            [sys.executable, "-c", get_benchmark_code(N)],
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
        for line in result.stdout.strip().split("\n"):
            if line.startswith("RESULT|"):
                parts = line.split("|")
                return float(parts[2])  # gflops
        if result.returncode != 0:
            sys.stderr.write(f"  ERROR: {result.stderr.strip()[:200]}\n")
        return None
    except (subprocess.TimeoutExpired, Exception):
        return None


def main():
    print("=" * 70)
    print("    Intel MKL DGEMM Benchmark")
    print("=" * 70)
    print()
    print(f"Python: {sys.version.split()[0]}")
    print("MKL auto-selects its SIMD code path (falls back to AVX2 in practice).")
    print()

    header = "| Size | GFLOPS (MKL) |"
    sep = "|------|--------------|"

    for threads in THREAD_COUNTS:
        if threads == 1:
            label = "Single-threaded"
        else:
            label = "Multi-threaded"

        print(f"### {label} (MKL_NUM_THREADS={threads})")
        print()
        print(header)
        print(sep)

        for size in SIZES:
            sys.stderr.write(f"  {threads}T / {size}x{size}...\r")
            sys.stderr.flush()
            gflops = run_single(size, threads)
            val = f"{gflops:.1f}" if gflops is not None else "N/A"
            print(f"| {size:>4} | {val:>12} |")

        print()

    print("=" * 70)
    print("Notes:")
    print("- MKL auto-selects SIMD path (AVX2 in practice)")
    print("- Calls cblas_dgemm directly via ctypes (no NumPy dependency)")
    print("=" * 70)


if __name__ == "__main__":
    main()
