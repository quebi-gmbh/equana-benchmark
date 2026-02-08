export function matmulNaiveFloat64(A: Float64Array, B: Float64Array, N: number): Float64Array {
  const C = new Float64Array(N * N);

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
