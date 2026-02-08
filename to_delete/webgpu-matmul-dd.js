// Double-Double Precision Matrix Multiplication for WebGPU
// Uses atomic operations to prevent fast-math optimization

/**
 * Create the double-double matmul pipeline with tiling support
 * @param {GPUDevice} device
 * @param {number} N - Matrix dimension
 * @param {number} tileSize - Tile size for processing (default: auto)
 * @returns {Object} Pipeline and bind group layout
 */
export function createDD64MatmulPipeline(device, N, tileSize = 0) {
    // Calculate safe tile size based on temp buffer limits
    // Each output element needs 64 * 4 = 256 bytes of temp storage
    // Limit temp buffer to 128MB -> ~500k elements max
    const maxElements = Math.floor(128 * 1024 * 1024 / 256);
    const autoTileSize = Math.min(N, Math.floor(Math.sqrt(maxElements)));
    const actualTileSize = tileSize > 0 ? tileSize : autoTileSize;

    const shaderCode = `
        // Matrix dimension
        const N: u32 = ${N}u;
        const TILE_SIZE: u32 = ${actualTileSize}u;

        // Input matrices (hi and lo components)
        @group(0) @binding(0) var<storage, read> A_hi: array<f32>;
        @group(0) @binding(1) var<storage, read> A_lo: array<f32>;
        @group(0) @binding(2) var<storage, read> B_hi: array<f32>;
        @group(0) @binding(3) var<storage, read> B_lo: array<f32>;
        @group(0) @binding(4) var<storage, read_write> C_hi: array<f32>;
        @group(0) @binding(5) var<storage, read_write> C_lo: array<f32>;
        @group(0) @binding(6) var<storage, read_write> temp: array<atomic<u32>>;
        @group(0) @binding(7) var<uniform> tileOffset: vec2<u32>;

        // Atomic helpers to prevent fast-math optimization
        fn store(base: u32, idx: u32, val: f32) {
            atomicStore(&temp[base + idx], bitcast<u32>(val));
        }

        fn load(base: u32, idx: u32) -> f32 {
            return bitcast<f32>(atomicLoad(&temp[base + idx]));
        }

        // Add12: Error-free addition
        fn Add12(a: f32, b: f32, base: u32, offset: u32) -> vec2<f32> {
            store(base, offset, a + b);
            let s = load(base, offset);

            store(base, offset + 1u, s - a);
            let v = load(base, offset + 1u);

            store(base, offset + 2u, s - v);
            let s_minus_v = load(base, offset + 2u);

            store(base, offset + 3u, a - s_minus_v);
            let a_minus_smv = load(base, offset + 3u);

            store(base, offset + 4u, b - v);
            let b_minus_v = load(base, offset + 4u);

            store(base, offset + 5u, a_minus_smv + b_minus_v);
            let r = load(base, offset + 5u);

            return vec2<f32>(s, r);
        }

        // Simplified Add22 for accumulation
        fn Add22_fast(ah: f32, al: f32, bh: f32, bl: f32, base: u32, offset: u32) -> vec2<f32> {
            let s = Add12(ah, bh, base, offset);
            let sh = s.x;
            let sl = s.y;

            store(base, offset + 6u, al + bl);
            let ab_lo = load(base, offset + 6u);
            store(base, offset + 7u, ab_lo + sl);
            let rl_temp = load(base, offset + 7u);

            let result = Add12(sh, rl_temp, base, offset + 8u);
            return result;
        }

        // Split: Veltkamp splitting for f32
        fn Split(a: f32, base: u32, offset: u32) -> vec2<f32> {
            let SPLIT_FACTOR: f32 = 4097.0;

            store(base, offset, SPLIT_FACTOR * a);
            let c = load(base, offset);

            store(base, offset + 1u, c - a);
            let c_minus_a = load(base, offset + 1u);

            store(base, offset + 2u, c - c_minus_a);
            let ahi = load(base, offset + 2u);

            store(base, offset + 3u, a - ahi);
            let alo = load(base, offset + 3u);

            return vec2<f32>(ahi, alo);
        }

        // Mul12: Error-free multiplication
        fn Mul12(a: f32, b: f32, base: u32, offset: u32) -> vec2<f32> {
            store(base, offset, a * b);
            let x = load(base, offset);

            let sa = Split(a, base, offset + 1u);
            let ahi = sa.x;
            let alo = sa.y;

            let sb = Split(b, base, offset + 5u);
            let bhi = sb.x;
            let blo = sb.y;

            store(base, offset + 9u, ahi * bhi);
            let ahi_bhi = load(base, offset + 9u);
            store(base, offset + 10u, x - ahi_bhi);
            let err1 = load(base, offset + 10u);

            store(base, offset + 11u, alo * bhi);
            let alo_bhi = load(base, offset + 11u);
            store(base, offset + 12u, err1 - alo_bhi);
            let err2 = load(base, offset + 12u);

            store(base, offset + 13u, ahi * blo);
            let ahi_blo = load(base, offset + 13u);
            store(base, offset + 14u, err2 - ahi_blo);
            let err3 = load(base, offset + 14u);

            store(base, offset + 15u, alo * blo);
            let alo_blo = load(base, offset + 15u);
            store(base, offset + 16u, alo_blo - err3);
            let y = load(base, offset + 16u);

            return vec2<f32>(x, y);
        }

        // Mul22: Double-double multiplication
        fn Mul22(ah: f32, al: f32, bh: f32, bl: f32, base: u32, offset: u32) -> vec2<f32> {
            let t = Mul12(ah, bh, base, offset);
            let t1 = t.x;
            let t2 = t.y;

            store(base, offset + 17u, ah * bl);
            let ah_bl = load(base, offset + 17u);

            store(base, offset + 18u, al * bh);
            let al_bh = load(base, offset + 18u);

            store(base, offset + 19u, ah_bl + al_bh);
            let sum1 = load(base, offset + 19u);

            store(base, offset + 20u, sum1 + t2);
            let t3 = load(base, offset + 20u);

            let result = Add12(t1, t3, base, offset + 21u);

            return result;
        }

        @compute @workgroup_size(1)
        fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
            // Apply tile offset to get global coordinates
            let row = gid.y + tileOffset.y;
            let col = gid.x + tileOffset.x;

            if (row >= N || col >= N) {
                return;
            }

            let idx = row * N + col;

            // Local index within tile for temp buffer
            let localIdx = gid.y * TILE_SIZE + gid.x;
            let base = localIdx * 64u;

            // Initialize accumulator (double-double)
            var acc_hi: f32 = 0.0;
            var acc_lo: f32 = 0.0;

            // Dot product: sum over k of A[row,k] * B[k,col]
            for (var k: u32 = 0u; k < N; k = k + 1u) {
                let a_idx = row * N + k;
                let b_idx = k * N + col;

                let ah = A_hi[a_idx];
                let al = A_lo[a_idx];
                let bh = B_hi[b_idx];
                let bl = B_lo[b_idx];

                // product = Mul22(a, b)
                let prod = Mul22(ah, al, bh, bl, base, 0u);

                // acc = Add22_fast(acc, product)
                let sum = Add22_fast(acc_hi, acc_lo, prod.x, prod.y, base, 30u);
                acc_hi = sum.x;
                acc_lo = sum.y;
            }

            C_hi[idx] = acc_hi;
            C_lo[idx] = acc_lo;
        }
    `;

    const shaderModule = device.createShaderModule({ code: shaderCode });

    const bindGroupLayout = device.createBindGroupLayout({
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        ]
    });

    const pipeline = device.createComputePipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
        compute: { module: shaderModule, entryPoint: 'main' }
    });

    return { pipeline, bindGroupLayout, tileSize: actualTileSize };
}

/**
 * Run double-double matrix multiplication on GPU
 * @param {GPUDevice} device
 * @param {GPUComputePipeline} pipeline
 * @param {GPUBindGroupLayout} bindGroupLayout
 * @param {Float64Array} A_f64 - Input matrix A (f64)
 * @param {Float64Array} B_f64 - Input matrix B (f64)
 * @param {number} N - Matrix dimension
 * @param {number} tileSize - Tile size (from createDD64MatmulPipeline)
 * @returns {Promise<{hi: Float32Array, lo: Float32Array, f64: Float64Array}>}
 */
export async function runDD64Matmul(device, pipeline, bindGroupLayout, A_f64, B_f64, N, tileSize = 0) {
    const size = N * N;

    // Calculate tile size if not provided
    if (tileSize === 0) {
        const maxElements = Math.floor(128 * 1024 * 1024 / 256);
        tileSize = Math.min(N, Math.floor(Math.sqrt(maxElements)));
    }

    // Split f64 matrices into hi/lo f32 components
    const A_hi = new Float32Array(size);
    const A_lo = new Float32Array(size);
    const B_hi = new Float32Array(size);
    const B_lo = new Float32Array(size);

    for (let i = 0; i < size; i++) {
        A_hi[i] = Math.fround(A_f64[i]);
        A_lo[i] = Math.fround(A_f64[i] - A_hi[i]);
        B_hi[i] = Math.fround(B_f64[i]);
        B_lo[i] = Math.fround(B_f64[i] - B_hi[i]);
    }

    // Create GPU buffers
    const bufferA_hi = device.createBuffer({
        size: A_hi.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const bufferA_lo = device.createBuffer({
        size: A_lo.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const bufferB_hi = device.createBuffer({
        size: B_hi.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const bufferB_lo = device.createBuffer({
        size: B_lo.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const bufferC_hi = device.createBuffer({
        size: size * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const bufferC_lo = device.createBuffer({
        size: size * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    // Temp buffer sized for one tile
    const tempBufferSize = tileSize * tileSize * 64 * 4;
    const tempBuffer = device.createBuffer({
        size: tempBufferSize,
        usage: GPUBufferUsage.STORAGE,
    });

    // Uniform buffer for tile offset
    const offsetBuffer = device.createBuffer({
        size: 8, // vec2<u32>
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const readBufferHi = device.createBuffer({
        size: size * 4,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const readBufferLo = device.createBuffer({
        size: size * 4,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    // Upload input data
    device.queue.writeBuffer(bufferA_hi, 0, A_hi);
    device.queue.writeBuffer(bufferA_lo, 0, A_lo);
    device.queue.writeBuffer(bufferB_hi, 0, B_hi);
    device.queue.writeBuffer(bufferB_lo, 0, B_lo);

    const bindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [
            { binding: 0, resource: { buffer: bufferA_hi } },
            { binding: 1, resource: { buffer: bufferA_lo } },
            { binding: 2, resource: { buffer: bufferB_hi } },
            { binding: 3, resource: { buffer: bufferB_lo } },
            { binding: 4, resource: { buffer: bufferC_hi } },
            { binding: 5, resource: { buffer: bufferC_lo } },
            { binding: 6, resource: { buffer: tempBuffer } },
            { binding: 7, resource: { buffer: offsetBuffer } },
        ]
    });

    // Process in tiles - batch multiple tiles per command encoder to reduce overhead
    const numTilesX = Math.ceil(N / tileSize);
    const numTilesY = Math.ceil(N / tileSize);
    const totalTiles = numTilesX * numTilesY;

    // Batch tiles to reduce sync overhead (sync every ~4 tiles or at end)
    const batchSize = Math.min(4, totalTiles);
    let tileCount = 0;

    for (let ty = 0; ty < numTilesY; ty++) {
        for (let tx = 0; tx < numTilesX; tx++) {
            const offsetX = tx * tileSize;
            const offsetY = ty * tileSize;
            const dispatchX = Math.min(tileSize, N - offsetX);
            const dispatchY = Math.min(tileSize, N - offsetY);

            // Update tile offset
            device.queue.writeBuffer(offsetBuffer, 0, new Uint32Array([offsetX, offsetY]));

            const commandEncoder = device.createCommandEncoder();
            const passEncoder = commandEncoder.beginComputePass();
            passEncoder.setPipeline(pipeline);
            passEncoder.setBindGroup(0, bindGroup);
            passEncoder.dispatchWorkgroups(dispatchX, dispatchY);
            passEncoder.end();
            device.queue.submit([commandEncoder.finish()]);

            tileCount++;

            // Sync periodically to avoid temp buffer conflicts, but not after every tile
            if (tileCount % batchSize === 0 && tileCount < totalTiles) {
                await device.queue.onSubmittedWorkDone();
            }
        }
    }

    // Final sync and copy results
    await device.queue.onSubmittedWorkDone();

    const commandEncoder = device.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(bufferC_hi, 0, readBufferHi, 0, size * 4);
    commandEncoder.copyBufferToBuffer(bufferC_lo, 0, readBufferLo, 0, size * 4);
    device.queue.submit([commandEncoder.finish()]);

    // Read results
    await readBufferHi.mapAsync(GPUMapMode.READ);
    await readBufferLo.mapAsync(GPUMapMode.READ);

    const C_hi = new Float32Array(readBufferHi.getMappedRange().slice(0));
    const C_lo = new Float32Array(readBufferLo.getMappedRange().slice(0));

    readBufferHi.unmap();
    readBufferLo.unmap();

    // Assemble f64 result
    const C_f64 = new Float64Array(size);
    for (let i = 0; i < size; i++) {
        C_f64[i] = C_hi[i] + C_lo[i];
    }

    // Cleanup
    bufferA_hi.destroy();
    bufferA_lo.destroy();
    bufferB_hi.destroy();
    bufferB_lo.destroy();
    bufferC_hi.destroy();
    bufferC_lo.destroy();
    tempBuffer.destroy();
    offsetBuffer.destroy();
    readBufferHi.destroy();
    readBufferLo.destroy();

    return { hi: C_hi, lo: C_lo, f64: C_f64 };
}
