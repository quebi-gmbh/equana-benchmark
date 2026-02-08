/**
 * Terminal benchmark: WASM vs NumPy
 * (WebGPU requires a browser - can't run in Node.js)
 */

import { spawn } from 'child_process';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

const __dirname = dirname(fileURLToPath(import.meta.url));

const N = 2000;
const ROUNDS = 3;

// ============= WASM Benchmark =============

async function benchmarkWasm() {
    const { default: createMatmulModule } = await import('./matmul.mjs');
    const module = await createMatmulModule();

    const size = N * N;
    const ptrA = module._malloc_f64(size);
    const ptrB = module._malloc_f64(size);
    const ptrC = module._malloc_f64(size);

    const heapF64 = module.HEAPF64;
    const offsetA = ptrA / 8;
    const offsetB = ptrB / 8;

    for (let i = 0; i < size; i++) {
        heapF64[offsetA + i] = Math.random();
        heapF64[offsetB + i] = Math.random();
    }

    // Warmup
    module._matmul_f64(N, N, N, ptrA, ptrB, ptrC);

    const times = [];
    for (let i = 0; i < ROUNDS; i++) {
        const start = performance.now();
        module._matmul_f64(N, N, N, ptrA, ptrB, ptrC);
        times.push(performance.now() - start);
    }

    module._free_f64(ptrA);
    module._free_f64(ptrB);
    module._free_f64(ptrC);
    module._cleanup();

    const avg = times.reduce((a, b) => a + b) / times.length;
    const gflops = 2 * N * N * N / (avg / 1000) / 1e9;

    return { name: 'WASM SIMD (f64)', times, avg, gflops };
}

// ============= NumPy Benchmark =============

async function benchmarkNumPy() {
    return new Promise((resolve) => {
        const code = `
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

_ = A @ B  # warmup

times = []
for _ in range(ROUNDS):
    start = time.perf_counter()
    _ = A @ B
    times.append((time.perf_counter() - start) * 1000)

avg = sum(times) / len(times)
gflops = 2 * N**3 / (avg / 1000) / 1e9
print(f"{','.join(map(str, times))}|{avg}|{gflops}")
`;

        const proc = spawn('python3', ['-c', code]);
        let out = '';
        proc.stdout.on('data', d => out += d);
        proc.stderr.on('data', d => process.stderr.write(d));

        proc.on('close', code => {
            if (code !== 0 || !out.trim()) {
                resolve(null);
                return;
            }
            const [timesStr, avgStr, gflopsStr] = out.trim().split('|');
            resolve({
                name: 'NumPy (1 thread, f64)',
                times: timesStr.split(',').map(Number),
                avg: parseFloat(avgStr),
                gflops: parseFloat(gflopsStr)
            });
        });
        proc.on('error', () => resolve(null));
    });
}

// ============= Main =============

async function main() {
    console.log('');
    console.log('='.repeat(65));
    console.log('  Matrix Multiplication Benchmark');
    console.log('='.repeat(65));
    console.log(`Matrix size: ${N}x${N}`);
    console.log(`FLOPS per matmul: ${(2 * N * N * N / 1e9).toFixed(2)} GFLOP`);
    console.log(`Rounds: ${ROUNDS}`);
    console.log('');

    const results = [];

    // WASM
    console.log('Running WASM SIMD...');
    const wasm = await benchmarkWasm();
    results.push(wasm);
    console.log(`  ${wasm.avg.toFixed(2)} ms, ${wasm.gflops.toFixed(2)} GFLOPS`);

    // NumPy
    console.log('Running NumPy...');
    const numpy = await benchmarkNumPy();
    if (numpy) {
        results.push(numpy);
        console.log(`  ${numpy.avg.toFixed(2)} ms, ${numpy.gflops.toFixed(2)} GFLOPS`);
    } else {
        console.log('  (not available)');
    }

    // Results table
    results.sort((a, b) => a.avg - b.avg);
    const baseline = results[results.length - 1]?.avg || 1;

    console.log('');
    console.log('='.repeat(65));
    console.log('  Results');
    console.log('='.repeat(65));
    console.log('| Implementation           | Avg (ms)  | GFLOPS  | Speedup |');
    console.log('|--------------------------|-----------|---------|---------|');

    for (const r of results) {
        const speedup = baseline / r.avg;
        console.log(
            `| ${r.name.padEnd(24)} | ${r.avg.toFixed(2).padStart(9)} | ${r.gflops.toFixed(2).padStart(7)} | ${speedup.toFixed(2).padStart(7)}x |`
        );
    }
    console.log('='.repeat(65));

    console.log('');
    console.log('Round times (ms):');
    for (const r of results) {
        console.log(`  ${r.name}: ${r.times.map(t => t.toFixed(2)).join(', ')}`);
    }

    console.log('');
    console.log('Note: WebGPU benchmark requires a browser. Open http://localhost:3000');
}

main().catch(console.error);
