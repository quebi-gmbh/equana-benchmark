export function matmulCacheArray(A: number[], B: number[], N: number): number[] {
  const BLOCK = 64;
  const C = new Array<number>(N * N);
  for (let i = 0; i < N * N; i++) C[i] = 0;

  for (let jj = 0; jj < N; jj += BLOCK) {
    const jEnd = Math.min(jj + BLOCK, N);
    for (let kk = 0; kk < N; kk += BLOCK) {
      const kEnd = Math.min(kk + BLOCK, N);
      for (let ii = 0; ii < N; ii += BLOCK) {
        const iEnd = Math.min(ii + BLOCK, N);
        for (let i = ii; i < iEnd; i++) {
          for (let k = kk; k < kEnd; k++) {
            const aik = A[i * N + k]!;
            for (let j = jj; j < jEnd; j++) {
              C[i * N + j]! += aik * B[k * N + j]!;
            }
          }
        }
      }
    }
  }
  return C;
}
