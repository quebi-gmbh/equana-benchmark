/**
 * Matrix Multiplication Benchmark
 * Compares: Custom WASM (OpenBLAS-style) vs NumPy
 */

import { spawn } from 'child_process';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { readFileSync } from 'fs';

const __dirname = dirname(fileURLToPath(import.meta.url));

const N = 2000;
const ROUNDS = 3;

// ============= WASM Module Loading =============

async function loadWasmModule() {
    const { default: createMatmulModule } = await import('./matmul.mjs');
    const module = await createMatmulModule();
    return module;
}

// ============= WASM Benchmark =============

async function benchmarkWasm() {
    const module = await loadWasmModule();

    // Allocate matrices in WASM memory
    const size = N * N;
    const bytesPerMatrix = size * 8; // Float64

    const ptrA = module._malloc_f64(size);
    const ptrB = module._malloc_f64(size);
    const ptrC = module._malloc_f64(size);

    // Fill with random data
    const heapF64 = module.HEAPF64;
    const offsetA = ptrA / 8;
    const offsetB = ptrB / 8;
    const offsetC = ptrC / 8;

    for (let i = 0; i < size; i++) {
        heapF64[offsetA + i] = Math.random();
        heapF64[offsetB + i] = Math.random();
    }

    // Warmup
    module._matmul_f64(N, N, N, ptrA, ptrB, ptrC);

    // Timed rounds
    const times = [];
    for (let i = 0; i < ROUNDS; i++) {
        const start = performance.now();
        module._matmul_f64(N, N, N, ptrA, ptrB, ptrC);
        times.push(performance.now() - start);
    }

    // Get a sample result for verification
    const sampleResult = heapF64[offsetC];

    // Cleanup
    module._free_f64(ptrA);
    module._free_f64(ptrB);
    module._free_f64(ptrC);
    module._cleanup();

    const avgMs = times.reduce((a, b) => a + b) / times.length;
    const flops = 2 * N * N * N;
    const gflops = flops / (avgMs / 1000) / 1e9;

    return {
        name: 'WASM (OpenBLAS-style)',
        times,
        avgMs,
        gflops,
        sample: sampleResult
    };
}

// ============= NumPy Benchmark =============

async function benchmarkNumPy() {
    return new Promise((resolve) => {
        const pythonCode = `
import numpy as np
import time
import os

# Force single-threaded for fair comparison
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

N = ${N}
ROUNDS = ${ROUNDS}

np.random.seed(42)
A = np.random.rand(N, N)
B = np.random.rand(N, N)

# Warmup
C = A @ B

times = []
for _ in range(ROUNDS):
    start = time.perf_counter()
    C = A @ B
    elapsed = (time.perf_counter() - start) * 1000
    times.append(elapsed)

avg = sum(times) / len(times)
gflops = 2 * N**3 / (avg / 1000) / 1e9
sample = C[0, 0]
print(f"{','.join(map(str, times))}|{avg}|{gflops}|{sample}")
`;

        const proc = spawn('python3', ['-c', pythonCode]);

        let output = '';
        let stderr = '';
        proc.stdout.on('data', (data) => output += data);
        proc.stderr.on('data', (data) => stderr += data);

        proc.on('close', (code) => {
            if (code !== 0 || !output.trim()) {
                console.log('  NumPy benchmark failed');
                if (stderr) console.log('  Error:', stderr.trim());
                resolve(null);
                return;
            }
            const parts = output.trim().split('|');
            resolve({
                name: 'NumPy (1 thread)',
                times: parts[0].split(',').map(Number),
                avgMs: parseFloat(parts[1]),
                gflops: parseFloat(parts[2]),
                sample: parseFloat(parts[3])
            });
        });

        proc.on('error', () => {
            console.log('  Python not available');
            resolve(null);
        });
    });
}

// ============= Main =============

async function main() {
    console.log(`\n${'='.repeat(60)}`);
    console.log(`  Matrix Multiplication Benchmark`);
    console.log(`${'='.repeat(60)}`);
    console.log(`Matrix size: ${N}x${N}`);
    console.log(`FLOPS per matmul: ${(2 * N * N * N / 1e9).toFixed(2)} GFLOP`);
    console.log(`Rounds: ${ROUNDS}\n`);

    const results = [];

    // WASM benchmark
    console.log('Running WASM (OpenBLAS-style)...');
    try {
        const wasmResult = await benchmarkWasm();
        results.push(wasmResult);
        console.log(`  Avg: ${wasmResult.avgMs.toFixed(2)} ms, ${wasmResult.gflops.toFixed(2)} GFLOPS`);
    } catch (e) {
        console.log(`  Failed: ${e.message}`);
    }

    // NumPy benchmark
    console.log('Running NumPy (single-threaded)...');
    const numpyResult = await benchmarkNumPy();
    if (numpyResult) {
        results.push(numpyResult);
        console.log(`  Avg: ${numpyResult.avgMs.toFixed(2)} ms, ${numpyResult.gflops.toFixed(2)} GFLOPS`);
    }

    // Sort by speed
    results.sort((a, b) => a.avgMs - b.avgMs);

    // Print results
    console.log(`\n${'='.repeat(60)}`);
    console.log('  Results');
    console.log('='.repeat(60));
    console.log('| Implementation          | Avg Time (ms) | GFLOPS | Ratio |');
    console.log('|-------------------------|---------------|--------|-------|');

    const baseline = results[results.length - 1]?.avgMs || 1;
    for (const r of results) {
        const ratio = baseline / r.avgMs;
        console.log(
            `| ${r.name.padEnd(23)} | ${r.avgMs.toFixed(2).padStart(13)} | ${r.gflops.toFixed(2).padStart(6)} | ${ratio.toFixed(2).padStart(5)}x |`
        );
    }
    console.log('='.repeat(60));

    // Print round times
    console.log('\nRound times (ms):');
    for (const r of results) {
        console.log(`  ${r.name}: ${r.times.map(t => t.toFixed(2)).join(', ')}`);
    }
}

main().catch(console.error);
