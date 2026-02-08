// Web Worker for running WASM matrix multiplication
// This isolates WASM execution from the main thread

let wasmModule = null;

// Load and initialize the WASM module
async function loadWasm() {
    if (wasmModule) return wasmModule;

    const response = await fetch('matmul_full_4x4.wasm');
    const bytes = await response.arrayBuffer();

    const { instance } = await WebAssembly.instantiate(bytes, {
        env: {
            emscripten_notify_memory_growth: function(memoryIndex) {
                // Memory growth notification - nothing special needed
            }
        }
    });

    wasmModule = instance.exports;
    return wasmModule;
}

// Run matrix multiplication benchmark
async function runBenchmark(N, A, B, rounds) {
    const wasm = await loadWasm();

    // Allocate memory
    const ptrA = wasm.malloc_f64(N * N);
    const ptrB = wasm.malloc_f64(N * N);
    const ptrC = wasm.malloc_f64(N * N);

    // Copy data to WASM memory
    const mem = new Float64Array(wasm.memory.buffer);
    const offsetA = ptrA / 8;
    const offsetB = ptrB / 8;
    const offsetC = ptrC / 8;

    for (let i = 0; i < N * N; i++) {
        mem[offsetA + i] = A[i];
        mem[offsetB + i] = B[i];
    }

    // Warmup run
    wasm.matmul_f64(N, N, N, ptrA, ptrB, ptrC);

    // Timed runs
    const times = [];
    for (let r = 0; r < rounds; r++) {
        const start = performance.now();
        wasm.matmul_f64(N, N, N, ptrA, ptrB, ptrC);
        times.push(performance.now() - start);
    }

    // Copy result matrix C
    const resultMem = new Float64Array(wasm.memory.buffer);
    const C = new Float64Array(N * N);
    for (let i = 0; i < N * N; i++) {
        C[i] = resultMem[offsetC + i];
    }

    // Cleanup
    wasm.free_f64(ptrA);
    wasm.free_f64(ptrB);
    wasm.free_f64(ptrC);

    return {
        times,
        avg: times.reduce((a, b) => a + b) / times.length,
        C: Array.from(C)  // Convert to regular array for transfer
    };
}

// Handle messages from main thread
self.onmessage = async function(e) {
    const { type, N, A, B, rounds } = e.data;

    if (type === 'init') {
        try {
            await loadWasm();
            self.postMessage({ type: 'ready' });
        } catch (err) {
            self.postMessage({ type: 'error', error: err.message });
        }
    } else if (type === 'benchmark') {
        try {
            const result = await runBenchmark(N, A, B, rounds);
            self.postMessage({ type: 'result', ...result });
        } catch (err) {
            self.postMessage({ type: 'error', error: err.message });
        }
    }
};
