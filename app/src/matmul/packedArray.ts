export function matmulPackedArray(A: number[], B: number[], N: number): number[] {
  const MC = 64, KC = 128, NC = 256;
  const MR = 4, NR = 4;

  const C = new Array<number>(N * N);
  for (let i = 0; i < N * N; i++) C[i] = 0;

  const packA = new Array<number>((MC + MR) * (KC + 4));
  const packB = new Array<number>((KC + 4) * (NC + NR));

  function packPanelA(ic: number, mc: number, pc: number, kc: number) {
    let pa = 0;
    for (let i = ic; i < ic + mc - MR + 1; i += MR) {
      for (let k = pc; k < pc + kc; k++) {
        packA[pa++] = A[i * N + k]!;
        packA[pa++] = A[(i + 1) * N + k]!;
        packA[pa++] = A[(i + 2) * N + k]!;
        packA[pa++] = A[(i + 3) * N + k]!;
      }
    }
    const iRem = ic + mc - ((ic + mc) % MR || MR);
    if (iRem < ic + mc) {
      const mr = ic + mc - iRem;
      for (let k = pc; k < pc + kc; k++) {
        for (let ii = 0; ii < mr; ii++) {
          packA[pa++] = A[(iRem + ii) * N + k]!;
        }
        for (let ii = mr; ii < MR; ii++) {
          packA[pa++] = 0;
        }
      }
    }
  }

  function packPanelB(pc: number, kc: number, jc: number, nc: number) {
    let pb = 0;
    for (let j = jc; j < jc + nc - NR + 1; j += NR) {
      for (let k = pc; k < pc + kc; k++) {
        packB[pb++] = B[k * N + j]!;
        packB[pb++] = B[k * N + j + 1]!;
        packB[pb++] = B[k * N + j + 2]!;
        packB[pb++] = B[k * N + j + 3]!;
      }
    }
    const jRem = jc + nc - ((jc + nc) % NR || NR);
    if (jRem < jc + nc) {
      const nr = jc + nc - jRem;
      for (let k = pc; k < pc + kc; k++) {
        for (let jj = 0; jj < nr; jj++) {
          packB[pb++] = B[k * N + jRem + jj]!;
        }
        for (let jj = nr; jj < NR; jj++) {
          packB[pb++] = 0;
        }
      }
    }
  }

  function microKernel4x4(pa: number, pb: number, kc: number, cBase: number, ldc: number) {
    let c00 = 0, c01 = 0, c02 = 0, c03 = 0;
    let c10 = 0, c11 = 0, c12 = 0, c13 = 0;
    let c20 = 0, c21 = 0, c22 = 0, c23 = 0;
    let c30 = 0, c31 = 0, c32 = 0, c33 = 0;

    for (let k = 0; k < kc; k++) {
      const a0 = packA[pa]!, a1 = packA[pa + 1]!, a2 = packA[pa + 2]!, a3 = packA[pa + 3]!;
      const b0 = packB[pb]!, b1 = packB[pb + 1]!, b2 = packB[pb + 2]!, b3 = packB[pb + 3]!;

      c00 += a0 * b0; c01 += a0 * b1; c02 += a0 * b2; c03 += a0 * b3;
      c10 += a1 * b0; c11 += a1 * b1; c12 += a1 * b2; c13 += a1 * b3;
      c20 += a2 * b0; c21 += a2 * b1; c22 += a2 * b2; c23 += a2 * b3;
      c30 += a3 * b0; c31 += a3 * b1; c32 += a3 * b2; c33 += a3 * b3;

      pa += MR;
      pb += NR;
    }

    C[cBase]! += c00; C[cBase + 1]! += c01; C[cBase + 2]! += c02; C[cBase + 3]! += c03;
    C[cBase + ldc]! += c10; C[cBase + ldc + 1]! += c11; C[cBase + ldc + 2]! += c12; C[cBase + ldc + 3]! += c13;
    C[cBase + 2 * ldc]! += c20; C[cBase + 2 * ldc + 1]! += c21; C[cBase + 2 * ldc + 2]! += c22; C[cBase + 2 * ldc + 3]! += c23;
    C[cBase + 3 * ldc]! += c30; C[cBase + 3 * ldc + 1]! += c31; C[cBase + 3 * ldc + 2]! += c32; C[cBase + 3 * ldc + 3]! += c33;
  }

  function microKernelEdge(pa: number, pb: number, mr: number, nr: number, kc: number, cBase: number, ldc: number) {
    for (let i = 0; i < mr; i++) {
      for (let j = 0; j < nr; j++) {
        let sum = 0;
        for (let k = 0; k < kc; k++) {
          sum += packA[pa + k * MR + i]! * packB[pb + k * NR + j]!;
        }
        C[cBase + i * ldc + j]! += sum;
      }
    }
  }

  for (let jc = 0; jc < N; jc += NC) {
    const nc = Math.min(NC, N - jc);

    for (let pc = 0; pc < N; pc += KC) {
      const kc = Math.min(KC, N - pc);

      packPanelB(pc, kc, jc, nc);

      for (let ic = 0; ic < N; ic += MC) {
        const mc = Math.min(MC, N - ic);

        packPanelA(ic, mc, pc, kc);

        for (let jr = 0; jr < nc; jr += NR) {
          const nr = Math.min(NR, nc - jr);
          for (let ir = 0; ir < mc; ir += MR) {
            const mr = Math.min(MR, mc - ir);
            const paIdx = ir * kc;
            const pbIdx = jr * kc;
            const cBase = (ic + ir) * N + (jc + jr);

            if (mr === MR && nr === NR) {
              microKernel4x4(paIdx, pbIdx, kc, cBase, N);
            } else {
              microKernelEdge(paIdx, pbIdx, mr, nr, kc, cBase, N);
            }
          }
        }
      }
    }
  }

  return C;
}
