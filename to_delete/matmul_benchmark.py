import os
import numpy as np
import time

# Force single-threaded execution
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Optional: Set OPENBLAS_CORETYPE before running to simulate older hardware:
#   OPENBLAS_CORETYPE=HASWELL     -> AVX2 only (~200ms)
#   OPENBLAS_CORETYPE=SANDYBRIDGE -> AVX only (~250ms)
#   OPENBLAS_CORETYPE=NEHALEM     -> SSE4.2 (~500ms)
#   OPENBLAS_CORETYPE=PRESCOTT    -> SSE3 (~530ms)
# Example: OPENBLAS_CORETYPE=PRESCOTT python matmul_benchmark.py

try:
    import threadpoolctl
    threadpoolctl.threadpool_limits(1)
except ImportError:
    pass

coretype = os.environ.get("OPENBLAS_CORETYPE", "auto-detect (likely AVX-512)")
print(f"NumPy version: {np.__version__}")
print(f"OPENBLAS_CORETYPE: {coretype}")
print(f"Threads: 1")
print()

# Create two random 2000x2000 matrices
print("Creating random 2000x2000 matrices...")
A = np.random.rand(2000, 2000)
B = np.random.rand(2000, 2000)

# Warmup round
print("Warmup round...")
_ = A @ B

# Timed rounds
times = []
for i in range(3):
    start = time.perf_counter()
    _ = A @ B
    end = time.perf_counter()
    elapsed = end - start
    times.append(elapsed)
    print(f"Round {i + 1}: {elapsed:.4f} seconds")

avg = np.mean(times)
gflops = 2 * 2000**3 / avg / 1e9
print(f"\nAverage: {avg:.4f} seconds")
print(f"Performance: {gflops:.1f} GFLOPS")
