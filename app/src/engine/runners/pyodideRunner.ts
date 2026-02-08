import type { PyodideConfig, BenchmarkResult } from '../types';
import { computeGflops } from '../matrixUtils';
import { loadPyodideRuntime } from '../pyodideLoader';

export async function runPyodideBenchmark(
  _config: PyodideConfig,
  N: number,
  _A_f64: Float64Array,
  _B_f64: Float64Array,
  rounds: number,
): Promise<BenchmarkResult> {
  const pyodide = await loadPyodideRuntime();

  const result = await pyodide.runPythonAsync(`
import numpy as np
import time

N = ${N}
rounds = ${rounds}

A = np.random.rand(N, N).astype(np.float64)
B = np.random.rand(N, N).astype(np.float64)

# Warmup
_ = A @ B

times = []
for i in range(rounds):
    start = time.perf_counter()
    C = A @ B
    elapsed = time.perf_counter() - start
    times.append(elapsed)

avg = sum(times) / len(times)

{
    'avg_s': float(avg),
    'times_s': [float(t) for t in times],
}
  `);

  const pyResult = result.toJs() as Map<string, unknown>;
  const avgSeconds = pyResult.get('avg_s') as number;
  const timesSeconds = pyResult.get('times_s') as number[];
  result.destroy();

  // Convert seconds to milliseconds to match the convention of all other runners
  const avgMs = avgSeconds * 1000;
  const timesMs = timesSeconds.map((t) => t * 1000);

  return {
    avg: avgMs,
    times: timesMs,
    gflops: computeGflops(N, avgMs),
  };
}
