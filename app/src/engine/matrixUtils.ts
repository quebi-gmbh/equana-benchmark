import type { MatrixData } from './types';

export function generateMatrix(N: number): Float64Array {
  const size = N * N;
  const M = new Float64Array(size);
  for (let i = 0; i < size; i++) {
    M[i] = Math.random();
  }
  return M;
}

export function toNumberArray(f64: Float64Array): number[] {
  return Array.from(f64);
}

export function computeGflops(N: number, timeMs: number): number {
  return (2 * N * N * N) / (timeMs / 1000) / 1e9;
}

export function generateMatrices(N: number): MatrixData {
  const A_f64 = generateMatrix(N);
  const B_f64 = generateMatrix(N);
  return {
    A_f64,
    B_f64,
    A_arr: toNumberArray(A_f64),
    B_arr: toNumberArray(B_f64),
  };
}
