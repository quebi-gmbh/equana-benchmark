#!/usr/bin/env python3
"""
Intel MKL DGEMM benchmark via direct ctypes calls to libmkl_rt.

Calls cblas_dgemm directly — no NumPy dependency, no backend ambiguity.

On Intel CPUs, sweeps SIMD upper bounds via MKL_ENABLE_INSTRUCTIONS:
- SSE4.2  (MKL_ENABLE_INSTRUCTIONS=SSE4_2)
- AVX2    (MKL_ENABLE_INSTRUCTIONS=AVX2)
- AVX-512 (MKL_ENABLE_INSTRUCTIONS=AVX512)

On AMD CPUs, MKL_ENABLE_INSTRUCTIONS has no effect — MKL auto-selects
one code path regardless. A libfakeintel.so shim is used to bypass MKL's
Intel-only CPU check, but SIMD tier selection is still not available.
A single GFLOPS column is reported on AMD.

Same methodology as the other benchmarks:
- Sizes: 64, 128, 256, 512, 1024, 2048, 4096
- Thread counts: 1, 2, 4, 8, 16
- 2 untimed warmup runs + 5 timed runs measured together

Prerequisites: pip install mkl

Usage:
  python run_mkl_benchmarks.py
"""

import os
import platform
import subprocess
import sys
import tempfile

SIZES = [64, 128, 256, 512, 1024, 2048, 4096]
THREAD_COUNTS = [1, 2, 4, 8, 16]

# Intel: three SIMD tiers via MKL_ENABLE_INSTRUCTIONS
INTEL_TARGETS = [
    ("SSE4.2", "SSE4_2"),
    ("AVX2", "AVX2"),
    ("AVX-512", "AVX512"),
]

# AMD: single auto-select (MKL_ENABLE_INSTRUCTIONS has no effect)
AMD_TARGETS = [
    ("MKL (Auto)", None),
]


def build_fakeintel_lib():
    """Build libfakeintel.so to bypass MKL's Intel-only CPU check on AMD.

    MKL checks mkl_serv_intel_cpu_true() to detect Intel CPUs. On AMD, it falls
    back to generic SSE2 code. This shim overrides that function to always
    return 1, enabling MKL's optimized code paths on AMD.
    """
    if platform.system() != "Linux":
        return None

    cache_dir = os.path.join(tempfile.gettempdir(), "mkl_bench_cache")
    lib_path = os.path.join(cache_dir, "libfakeintel.so")

    if os.path.isfile(lib_path):
        return lib_path

    c_code = """\
int mkl_serv_intel_cpu_true(void) { return 1; }
"""
    os.makedirs(cache_dir, exist_ok=True)
    c_path = os.path.join(cache_dir, "fakeintel.c")
    with open(c_path, "w") as f:
        f.write(c_code)

    try:
        subprocess.run(
            ["gcc", "-shared", "-fPIC", "-o", lib_path, c_path],
            capture_output=True,
            check=True,
        )
        return lib_path
    except (subprocess.CalledProcessError, FileNotFoundError):
        sys.stderr.write("  Warning: could not compile libfakeintel.so (gcc not found?).\n")
        sys.stderr.write("  MKL may use a slow generic code path on AMD CPUs.\n")
        return None


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


def run_single(N, threads, mkl_instruction, fakeintel_path):
    """Run a single benchmark with given size, thread count, and MKL instruction upper bound."""
    env = dict(os.environ)
    env["MKL_NUM_THREADS"] = str(threads)
    env["OMP_NUM_THREADS"] = str(threads)
    if mkl_instruction is not None:
        env["MKL_ENABLE_INSTRUCTIONS"] = mkl_instruction

    # LD_PRELOAD the fakeintel shim so MKL uses optimized paths on AMD
    if fakeintel_path:
        existing = env.get("LD_PRELOAD", "")
        env["LD_PRELOAD"] = f"{fakeintel_path}:{existing}" if existing else fakeintel_path

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


def detect_cpu_vendor():
    """Detect CPU vendor from /proc/cpuinfo."""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("vendor_id"):
                    return line.split(":")[1].strip()
    except (OSError, IndexError):
        pass
    return "unknown"


def main():
    print("=" * 70)
    print("    Intel MKL DGEMM Benchmark")
    print("=" * 70)
    print()
    print(f"Python: {sys.version.split()[0]}")

    vendor = detect_cpu_vendor()
    is_amd = vendor == "AuthenticAMD"
    print(f"CPU vendor: {vendor}")
    print()

    # Choose targets based on CPU vendor
    if is_amd:
        targets = AMD_TARGETS
        print("AMD CPU detected.")
        print("MKL_ENABLE_INSTRUCTIONS has no effect on AMD — MKL auto-selects")
        print("one code path regardless. Reporting a single GFLOPS column.")
    else:
        targets = INTEL_TARGETS
        print("Note: MKL_ENABLE_INSTRUCTIONS sets an UPPER BOUND on instruction usage.")
        print("SSE4_2 = at most SSE4.2, AVX2 = at most AVX2, AVX512 = at most AVX-512.")
    print()

    # Build fakeintel shim for AMD CPUs
    fakeintel_path = None
    if is_amd:
        fakeintel_path = build_fakeintel_lib()
        if fakeintel_path:
            print(f"Using libfakeintel.so shim: {fakeintel_path}")
        else:
            print("Warning: shim build failed. MKL may use a slow generic code path.")
        print()

    # Build header dynamically
    if is_amd:
        header = "| Size | GFLOPS (MKL) |"
        sep = "|------|--------------|"
    else:
        header = "| Size | SSE4.2 (GFLOPS) | AVX2 (GFLOPS) | AVX-512 (GFLOPS) |"
        sep = "|------|-----------------|---------------|------------------|"

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
            gflops_row = []
            for arch_name, mkl_value in targets:
                sys.stderr.write(f"  {threads}T / {size}x{size} / {arch_name}...\r")
                sys.stderr.flush()
                gflops = run_single(size, threads, mkl_value, fakeintel_path)
                gflops_row.append(gflops)

            vals = []
            for g in gflops_row:
                vals.append(f"{g:.1f}" if g is not None else "N/A")

            if is_amd:
                print(f"| {size:>4} | {vals[0]:>12} |")
            else:
                print(f"| {size:>4} | {vals[0]:>15} | {vals[1]:>13} | {vals[2]:>16} |")

        print()

    print("=" * 70)
    print("Notes:")
    if is_amd:
        print("- AMD CPU: libfakeintel.so shim bypasses MKL's Intel-only CPU check")
        print("- MKL_ENABLE_INSTRUCTIONS has no effect on AMD — single column reported")
        print("- MKL uses AVX2 on AMD (AVX-512 disabled on non-Intel CPUs)")
    else:
        print("- SSE4.2: MKL_ENABLE_INSTRUCTIONS=SSE4_2 (at most SSE4.2)")
        print("- AVX2: MKL_ENABLE_INSTRUCTIONS=AVX2 (at most AVX2 + FMA)")
        print("- AVX-512: MKL_ENABLE_INSTRUCTIONS=AVX512 (at most AVX-512)")
        print("- MKL auto-selects the best path within the upper bound")
    print("- Calls cblas_dgemm directly via ctypes (no NumPy dependency)")
    print("=" * 70)


if __name__ == "__main__":
    main()
