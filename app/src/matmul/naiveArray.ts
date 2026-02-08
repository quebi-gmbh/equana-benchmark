export function matmulNaiveArray(A: number[], B: number[], N: number): number[] {
  const C = new Array<number>(N * N);
  for (let i = 0; i < N * N; i++) C[i] = 0;

  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      let sum = 0;
      for (let k = 0; k < N; k++) {
        sum += A[i * N + k]! * B[k * N + j]!;
      }
      C[i * N + j] = sum;
    }
  }
  return C;
}
