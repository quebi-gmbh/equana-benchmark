/**
 * Matrix Multiplication Benchmark
 * Compares: C/WASM (numwasm), TypeScript implementations, NumPy
 */

import * as nw from "../packages/numwasm/dist/numwasm.mjs";
import { spawn } from "child_process";

const N = 2000;
const ROUNDS = 3;

// ============= TypeScript Matmul Implementations =============

/**
 * Create random matrix (row-major Float64Array)
 */
function createRandomMatrix(rows, cols) {
  const data = new Float64Array(rows * cols);
  for (let i = 0; i < data.length; i++) {
    data[i] = Math.random();
  }
  return data;
}

/**
 * Naive matmul - O(n^3) triple nested loop
 * C[i,j] = sum_k A[i,k] * B[k,j]
 */
function matmulNaive(A, B, C, M, N, K) {
  for (let i = 0; i < M; i++) {
    for (let j = 0; j < N; j++) {
      let sum = 0.0;
      for (let k = 0; k < K; k++) {
        sum += A[i * K + k] * B[k * N + j];
      }
      C[i * N + j] = sum;
    }
  }
}

/**
 * Blocked matmul - cache-friendly tiling
 */
function matmulBlocked(A, B, C, M, N, K, blockSize = 64) {
  C.fill(0);

  for (let ii = 0; ii < M; ii += blockSize) {
    const iEnd = Math.min(ii + blockSize, M);
    for (let kk = 0; kk < K; kk += blockSize) {
      const kEnd = Math.min(kk + blockSize, K);
      for (let jj = 0; jj < N; jj += blockSize) {
        const jEnd = Math.min(jj + blockSize, N);

        // Inner block computation
        for (let i = ii; i < iEnd; i++) {
          const iK = i * K;
          const iN = i * N;
          for (let k = kk; k < kEnd; k++) {
            const aik = A[iK + k];
            const kN = k * N;
            for (let j = jj; j < jEnd; j++) {
              C[iN + j] += aik * B[kN + j];
            }
          }
        }
      }
    }
  }
}

/**
 * Optimized matmul - transpose B + loop unrolling
 */
function matmulOptimized(A, B, C, M, NN, K) {
  const BLOCK = 64;

  // Transpose B for sequential memory access (B^T[j,k] = B[k,j])
  const BT = new Float64Array(NN * K);
  for (let k = 0; k < K; k++) {
    for (let j = 0; j < NN; j++) {
      BT[j * K + k] = B[k * NN + j];
    }
  }

  // Blocked with 4x unrolling
  for (let ii = 0; ii < M; ii += BLOCK) {
    const iEnd = Math.min(ii + BLOCK, M);
    for (let jj = 0; jj < NN; jj += BLOCK) {
      const jEnd = Math.min(jj + BLOCK, NN);

      for (let i = ii; i < iEnd; i++) {
        const rowA = i * K;
        const rowC = i * NN;

        for (let j = jj; j < jEnd; j++) {
          const colBT = j * K;
          let sum = 0.0;

          // 4x unrolled inner loop
          let k = 0;
          const kEnd4 = K - (K % 4);
          for (; k < kEnd4; k += 4) {
            sum +=
              A[rowA + k] * BT[colBT + k] +
              A[rowA + k + 1] * BT[colBT + k + 1] +
              A[rowA + k + 2] * BT[colBT + k + 2] +
              A[rowA + k + 3] * BT[colBT + k + 3];
          }
          for (; k < K; k++) {
            sum += A[rowA + k] * BT[colBT + k];
          }

          C[rowC + j] = sum;
        }
      }
    }
  }
}

/**
 * BLAS-style DGEMM: C = alpha * op(A) * op(B) + beta * C
 * Simplified: only supports transA='N', transB='N' for benchmark
 */
function dgemm(transA, transB, M, NN, K, alpha, A, lda, B, ldb, beta, C, ldc) {
  // Scale C by beta
  if (beta === 0) {
    for (let i = 0; i < M; i++) {
      for (let j = 0; j < NN; j++) {
        C[i * ldc + j] = 0;
      }
    }
  } else if (beta !== 1) {
    for (let i = 0; i < M; i++) {
      for (let j = 0; j < NN; j++) {
        C[i * ldc + j] *= beta;
      }
    }
  }

  if (alpha === 0) return;

  // C += alpha * A * B (row-major)
  for (let i = 0; i < M; i++) {
    for (let k = 0; k < K; k++) {
      const aik = alpha * A[i * lda + k];
      for (let j = 0; j < NN; j++) {
        C[i * ldc + j] += aik * B[k * ldb + j];
      }
    }
  }
}

// ============= Benchmark Functions =============

async function benchmarkWasm(A_data, B_data) {
  await nw.loadWasmModule();

  // Create NDArrays from data
  const A = nw.array(Array.from(A_data)).reshape([N, N]);
  const B = nw.array(Array.from(B_data)).reshape([N, N]);

  // Warmup
  let result = await nw.linalg.matmul(A, B);
  result.dispose();

  const times = [];
  for (let i = 0; i < ROUNDS; i++) {
    const start = performance.now();
    result = await nw.linalg.matmul(A, B);
    times.push(performance.now() - start);
    result.dispose();
  }

  A.dispose();
  B.dispose();

  const avgMs = times.reduce((a, b) => a + b) / times.length;
  const flops = 2 * N * N * N;
  const gflops = flops / (avgMs / 1000) / 1e9;

  return { name: "C/WASM (numwasm)", times, avgMs, gflops };
}

function benchmarkTS(name, fn, A, B) {
  const C = new Float64Array(N * N);

  // Warmup
  fn(A, B, C, N, N, N);

  const times = [];
  for (let i = 0; i < ROUNDS; i++) {
    const start = performance.now();
    fn(A, B, C, N, N, N);
    times.push(performance.now() - start);
  }

  const avgMs = times.reduce((a, b) => a + b) / times.length;
  const flops = 2 * N * N * N;
  const gflops = flops / (avgMs / 1000) / 1e9;

  return { name, times, avgMs, gflops };
}

function benchmarkDgemm(A, B) {
  const C = new Float64Array(N * N);

  // Warmup
  dgemm("N", "N", N, N, N, 1.0, A, N, B, N, 0.0, C, N);

  const times = [];
  for (let i = 0; i < ROUNDS; i++) {
    const start = performance.now();
    dgemm("N", "N", N, N, N, 1.0, A, N, B, N, 0.0, C, N);
    times.push(performance.now() - start);
  }

  const avgMs = times.reduce((a, b) => a + b) / times.length;
  const flops = 2 * N * N * N;
  const gflops = flops / (avgMs / 1000) / 1e9;

  return { name: "TS dgemm", times, avgMs, gflops };
}

async function benchmarkNumPy() {
  return new Promise((resolve) => {
    const pythonCode = `
import numpy as np
import time
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

N = ${N}
ROUNDS = ${ROUNDS}

np.random.seed(42)
A = np.random.rand(N, N)
B = np.random.rand(N, N)

# Warmup
_ = A @ B

times = []
for _ in range(ROUNDS):
    start = time.perf_counter()
    _ = A @ B
    elapsed = (time.perf_counter() - start) * 1000
    times.append(elapsed)

avg = sum(times) / len(times)
gflops = 2 * N**3 / (avg / 1000) / 1e9
print(f"{','.join(map(str, times))}|{avg}|{gflops}")
`;

    const proc = spawn("python3", ["-c", pythonCode]);

    let output = "";
    let stderr = "";
    proc.stdout.on("data", (data) => (output += data));
    proc.stderr.on("data", (data) => (stderr += data));

    proc.on("close", (code) => {
      if (code !== 0 || !output.trim()) {
        console.log("  (NumPy not available or failed)");
        if (stderr) console.log("  Error:", stderr.trim());
        resolve(null);
        return;
      }
      const [timesStr, avgStr, gflopsStr] = output.trim().split("|");
      resolve({
        name: "NumPy (1 thread)",
        times: timesStr.split(",").map(Number),
        avgMs: parseFloat(avgStr),
        gflops: parseFloat(gflopsStr),
      });
    });

    proc.on("error", () => {
      console.log("  (Python not available)");
      resolve(null);
    });
  });
}

// ============= Main =============

async function main() {
  console.log(`\n${"=".repeat(50)}`);
  console.log(`  Matrix Multiplication Benchmark`);
  console.log(`${"=".repeat(50)}`);
  console.log(`Matrix size: ${N}x${N}`);
  console.log(`FLOPS per matmul: ${((2 * N * N * N) / 1e9).toFixed(2)} GFLOP`);
  console.log(`Rounds: ${ROUNDS}\n`);

  // Create shared test data
  console.log("Generating random matrices...");
  const A = createRandomMatrix(N, N);
  const B = createRandomMatrix(N, N);
  console.log("Done.\n");

  const results = [];

  // C/WASM benchmark
  console.log("Running C/WASM (numwasm)...");
  try {
    results.push(await benchmarkWasm(A, B));
    console.log(`  Done: ${results[results.length - 1].avgMs.toFixed(2)} ms`);
  } catch (e) {
    console.log(`  Failed: ${e.message}`);
  }

  // TypeScript benchmarks
  console.log("Running TS optimized...");
  results.push(benchmarkTS("TS optimized", matmulOptimized, A, B));
  console.log(`  Done: ${results[results.length - 1].avgMs.toFixed(2)} ms`);

  console.log("Running TS blocked...");
  results.push(benchmarkTS("TS blocked", matmulBlocked, A, B));
  console.log(`  Done: ${results[results.length - 1].avgMs.toFixed(2)} ms`);

  console.log("Running TS dgemm...");
  results.push(benchmarkDgemm(A, B));
  console.log(`  Done: ${results[results.length - 1].avgMs.toFixed(2)} ms`);

  console.log("Running TS naive...");
  results.push(benchmarkTS("TS naive", matmulNaive, A, B));
  console.log(`  Done: ${results[results.length - 1].avgMs.toFixed(2)} ms`);

  // NumPy benchmark
  console.log("Running NumPy...");
  const numpy = await benchmarkNumPy();
  if (numpy) {
    results.push(numpy);
    console.log(`  Done: ${numpy.avgMs.toFixed(2)} ms`);
  }

  // Sort by performance (fastest first)
  results.sort((a, b) => a.avgMs - b.avgMs);

  // Find baseline (TS naive)
  const baseline =
    results.find((r) => r.name === "TS naive")?.avgMs ||
    results[results.length - 1].avgMs;

  // Print results table
  console.log(`\n${"=".repeat(70)}`);
  console.log("  Results (sorted by speed)");
  console.log("=".repeat(70));
  console.log(
    "| Implementation       | Avg Time (ms) | GFLOPS |  Speedup vs naive |",
  );
  console.log(
    "|----------------------|---------------|--------|-------------------|",
  );

  for (const r of results) {
    const speedup = baseline / r.avgMs;
    console.log(
      `| ${r.name.padEnd(20)} | ${r.avgMs.toFixed(2).padStart(13)} | ${r.gflops.toFixed(2).padStart(6)} | ${speedup.toFixed(2).padStart(17)}x |`,
    );
  }
  console.log("=".repeat(70));

  // Print individual round times
  console.log("\nRound times (ms):");
  for (const r of results) {
    console.log(`  ${r.name}: ${r.times.map((t) => t.toFixed(2)).join(", ")}`);
  }
}

main().catch(console.error);
