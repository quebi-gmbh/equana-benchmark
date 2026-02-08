import type { MtConfig, BenchmarkResult } from '../types';
import { computeGflops } from '../matrixUtils';
import { loadMtModule } from '../mtLoader';

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export async function runMtBenchmark(
  config: MtConfig,
  N: number,
  A_f64: Float64Array,
  B_f64: Float64Array,
  rounds: number,
  threadCount: number,
): Promise<BenchmarkResult> {
  const module = await loadMtModule(config.moduleId);
  const size = N * N;

  // Set thread count and allow thread pool to settle
  module._set_num_threads(threadCount);
  await sleep(100);

  const ptrA = module._malloc_f64(size);
  const ptrB = module._malloc_f64(size);
  const ptrC = module._malloc_f64(size);

  // Write to heap — use HEAPF64 (auto-updated by Emscripten on growth)
  const heap = module.HEAPF64;
  const offsetA = ptrA / 8;
  const offsetB = ptrB / 8;

  for (let i = 0; i < size; i++) {
    heap[offsetA + i] = A_f64[i]!;
    heap[offsetB + i] = B_f64[i]!;
  }

  const func = module[`_${config.funcName}`] as (
    M: number, N: number, K: number, pA: number, pB: number, pC: number,
  ) => void;

  if (typeof func !== 'function') {
    module._free_f64(ptrA);
    module._free_f64(ptrB);
    module._free_f64(ptrC);
    throw new Error(`Function _${config.funcName} not found on module ${config.moduleId}`);
  }

  // Warmup
  func(N, N, N, ptrA, ptrB, ptrC);

  const times: number[] = [];
  for (let i = 0; i < rounds; i++) {
    const start = performance.now();
    func(N, N, N, ptrA, ptrB, ptrC);
    times.push(performance.now() - start);
  }

  module._free_f64(ptrA);
  module._free_f64(ptrB);
  module._free_f64(ptrC);

  const avg = times.reduce((a, b) => a + b, 0) / times.length;
  return { avg, times, gflops: computeGflops(N, avg) };
}
