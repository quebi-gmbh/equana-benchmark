import type { JsConfig, BenchmarkResult } from '../types';
import { computeGflops } from '../matrixUtils';
import { matmulNaiveArray } from '../../matmul/naiveArray';
import { matmulNaiveFloat64 } from '../../matmul/naiveFloat64';
import { matmulCacheArray } from '../../matmul/cacheArray';
import { matmulPackedArray } from '../../matmul/packedArray';

type ArrayMatmulFn = (A: number[], B: number[], N: number) => number[];
type Float64MatmulFn = (A: Float64Array, B: Float64Array, N: number) => Float64Array;

const JS_ARRAY_FUNCTIONS: Record<string, ArrayMatmulFn> = {
  naiveArray: matmulNaiveArray,
  cacheArray: matmulCacheArray,
  packedArray: matmulPackedArray,
};

const JS_FLOAT64_FUNCTIONS: Record<string, Float64MatmulFn> = {
  naiveFloat64: matmulNaiveFloat64,
};

export function runJsBenchmark(
  config: JsConfig,
  N: number,
  A_f64: Float64Array,
  B_f64: Float64Array,
  A_arr: number[],
  B_arr: number[],
  rounds: number,
): BenchmarkResult {
  const times: number[] = [];

  if (config.inputType === 'float64') {
    const fn = JS_FLOAT64_FUNCTIONS[config.fn]!;
    // JIT warmup
    for (let w = 0; w < 3; w++) fn(A_f64, B_f64, N);
    for (let i = 0; i < rounds; i++) {
      const start = performance.now();
      fn(A_f64, B_f64, N);
      times.push(performance.now() - start);
    }
  } else {
    const fn = JS_ARRAY_FUNCTIONS[config.fn]!;
    // JIT warmup
    for (let w = 0; w < 3; w++) fn(A_arr, B_arr, N);
    for (let i = 0; i < rounds; i++) {
      const start = performance.now();
      fn(A_arr, B_arr, N);
      times.push(performance.now() - start);
    }
  }

  const avg = times.reduce((a, b) => a + b, 0) / times.length;
  return { avg, times, gflops: computeGflops(N, avg) };
}
