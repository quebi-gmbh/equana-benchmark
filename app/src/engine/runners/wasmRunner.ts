import type { WasmConfig, BenchmarkResult } from '../types';
import { computeGflops } from '../matrixUtils';
import { loadStandaloneWasm } from '../wasmLoader';

export async function runWasmBenchmark(
  config: WasmConfig,
  N: number,
  A_f64: Float64Array,
  B_f64: Float64Array,
  rounds: number,
): Promise<BenchmarkResult> {
  const wasm = await loadStandaloneWasm(config.wasmFile);
  const size = N * N;

  const ptrA = wasm.malloc_f64(size);
  const ptrB = wasm.malloc_f64(size);
  const ptrC = wasm.malloc_f64(size);

  // Write input data to WASM heap
  let heapF64 = new Float64Array(wasm.memory.buffer);
  heapF64.set(A_f64, ptrA / 8);
  heapF64.set(B_f64, ptrB / 8);

  const func = wasm[config.funcName] as (M: number, N: number, K: number, pA: number, pB: number, pC: number) => void;

  // Warmup (may trigger memory growth)
  func(N, N, N, ptrA, ptrB, ptrC);
  // Re-create view after potential memory growth
  heapF64 = new Float64Array(wasm.memory.buffer);

  const times: number[] = [];
  for (let i = 0; i < rounds; i++) {
    const start = performance.now();
    func(N, N, N, ptrA, ptrB, ptrC);
    times.push(performance.now() - start);
  }

  wasm.free_f64(ptrA);
  wasm.free_f64(ptrB);
  wasm.free_f64(ptrC);

  const avg = times.reduce((a, b) => a + b, 0) / times.length;
  return { avg, times, gflops: computeGflops(N, avg) };
}
