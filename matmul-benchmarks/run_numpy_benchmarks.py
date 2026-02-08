#!/usr/bin/env python3
"""
NumPy/OpenBLAS matmul benchmark matching the native OpenBLAS DGEMM variants.

Sweeps the same matrix as the native benchmark:
- Sizes: 64, 128, 256, 512, 1024, 2048, 4096
- Architectures: Scalar (PRESCOTT), SSE (NEHALEM), AVX2 (HASWELL), AVX-512 (COOPERLAKE)
- Thread counts: 1, 2, 4, 8, 16
"""

import os
import subprocess
import sys

SIZES = [64, 128, 256, 512, 1024, 2048, 4096]
THREAD_COUNTS = [1, 2, 4, 8, 16]
ARCH_TARGETS = [
    ("Scalar", "PRESCOTT"),
    ("SSE", "NEHALEM"),
    ("AVX2", "HASWELL"),
    ("AVX-512", "COOPERLAKE"),
]

def get_benchmark_code(N):
    return f'''
import numpy as np
import time

N = {N}
A = np.random.rand(N, N)
B = np.random.rand(N, N)

# 2 untimed warmup runs
_ = A @ B
_ = A @ B

# 5 runs timed together
RUNS = 5
start = time.perf_counter()
for _ in range(RUNS):
    _ = A @ B
elapsed = time.perf_counter() - start

avg = elapsed / RUNS
gflops = 2 * N**3 / avg / 1e9
print(f"RESULT|{{avg:.6f}}|{{gflops:.2f}}")
'''


def run_single(N, threads, coretype):
    """Run a single benchmark with given size, thread count, and OpenBLAS coretype."""
    env = dict(os.environ)
    env["OPENBLAS_NUM_THREADS"] = str(threads)
    env["OMP_NUM_THREADS"] = str(threads)
    env["MKL_NUM_THREADS"] = str(threads)
    env["OPENBLAS_CORETYPE"] = coretype

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
        return None
    except (subprocess.TimeoutExpired, Exception):
        return None


def print_header():
    print("| Size | Scalar (GFLOPS) | SSE (GFLOPS) | AVX2 (GFLOPS) | AVX-512 (GFLOPS) |")
    print("|------|-----------------|--------------|---------------|------------------|")


def main():
    import numpy as np

    print("=" * 70)
    print("    NumPy/OpenBLAS DGEMM Benchmark - Architecture Comparison")
    print("=" * 70)
    print()
    print(f"NumPy version: {np.__version__}")
    print(f"Python: {sys.version.split()[0]}")
    print()

    for threads in THREAD_COUNTS:
        if threads == 1:
            label = "Single-threaded"
        else:
            label = f"Multi-threaded"

        print(f"### {label} (OPENBLAS_NUM_THREADS={threads})")
        print()
        print_header()

        for size in SIZES:
            gflops_row = []
            for arch_name, coretype in ARCH_TARGETS:
                sys.stderr.write(f"  {threads}T / {size}x{size} / {arch_name}...\r")
                sys.stderr.flush()
                gflops = run_single(size, threads, coretype)
                gflops_row.append(gflops)

            vals = []
            for g in gflops_row:
                vals.append(f"{g:.1f}" if g is not None else "N/A")

            print(f"| {size:>4} | {vals[0]:>15} | {vals[1]:>12} | {vals[2]:>13} | {vals[3]:>16} |")

        print()

    print("=" * 70)
    print("Notes:")
    print("- Scalar: PRESCOTT target (no vectorization)")
    print("- SSE: NEHALEM target (128-bit XMM registers)")
    print("- AVX2: HASWELL target (256-bit YMM registers + FMA)")
    print("- AVX-512: COOPERLAKE target (512-bit ZMM registers)")
    print("=" * 70)


if __name__ == "__main__":
    main()
