/**
 * Matrix multiplication benchmark for numwasm
 * Tests matmul on two random 2000x2000 matrices
 */

import * as nw from "../packages/numwasm/dist/numwasm.mjs";

const N = 2000;
const ROUNDS = 3;

async function benchmark() {
  // Load WASM module first
  console.log("Loading WASM module...");
  await nw.loadWasmModule();
  console.log("WASM module loaded.\n");

  console.log(`Creating two random ${N}x${N} matrices...`);

  // Create random matrices using rand
  const createStart = performance.now();
  const a = nw.rand(N, N);
  const b = nw.rand(N, N);
  const createEnd = performance.now();
  console.log(
    `Matrix creation time: ${(createEnd - createStart).toFixed(2)} ms`,
  );

  console.log(`Matrix A shape: ${a.shape}`);
  console.log(`Matrix B shape: ${b.shape}`);

  // Warmup round
  console.log(`\n--- Warmup round ---`);
  const warmupStart = performance.now();
  await nw.linalg.matmul(a, b);
  const warmupEnd = performance.now();
  console.log(`Warmup time: ${(warmupEnd - warmupStart).toFixed(2)} ms`);

  // Timed rounds
  const times = [];
  for (let i = 0; i < ROUNDS; i++) {
    console.log(`\n--- Round ${i + 1} ---`);
    const start = performance.now();
    const result = await nw.linalg.matmul(a, b);
    const end = performance.now();
    const elapsed = end - start;
    times.push(elapsed);
    console.log(`Time: ${elapsed.toFixed(2)} ms`);
  }

  // Calculate average
  const avgTime = times.reduce((a, b) => a + b, 0) / times.length;
  const flops = 2 * N * N * N;
  const gflops = flops / (avgTime / 1000) / 1e9;

  console.log(`\n========== Results ==========`);
  console.log(`Matrix size: ${N}x${N}`);
  console.log(`Rounds: ${ROUNDS}`);
  console.log(`Times: ${times.map((t) => t.toFixed(2) + " ms").join(", ")}`);
  console.log(`Average time: ${avgTime.toFixed(2)} ms`);
  console.log(`Performance: ${gflops.toFixed(2)} GFLOPS`);
}

benchmark().catch(console.error);
