// WebGPU Matrix Multiplication Implementations

// ============= Basic f32 Matmul Pipeline =============

export function createMatmulPipeline(device, N) {
    const TILE_SIZE = 16;

    const shaderCode = `
        @group(0) @binding(0) var<storage, read> A: array<f32>;
        @group(0) @binding(1) var<storage, read> B: array<f32>;
        @group(0) @binding(2) var<storage, read_write> C: array<f32>;

        struct Uniforms {
            M: u32,
            N: u32,
            K: u32,
        }
        @group(0) @binding(3) var<uniform> uniforms: Uniforms;

        const TILE_SIZE: u32 = ${TILE_SIZE}u;

        var<workgroup> tileA: array<array<f32, ${TILE_SIZE}>, ${TILE_SIZE}>;
        var<workgroup> tileB: array<array<f32, ${TILE_SIZE}>, ${TILE_SIZE}>;

        @compute @workgroup_size(${TILE_SIZE}, ${TILE_SIZE})
        fn main(
            @builtin(global_invocation_id) global_id: vec3<u32>,
            @builtin(local_invocation_id) local_id: vec3<u32>,
            @builtin(workgroup_id) workgroup_id: vec3<u32>
        ) {
            let row = global_id.y;
            let col = global_id.x;
            let localRow = local_id.y;
            let localCol = local_id.x;

            let M = uniforms.M;
            let N = uniforms.N;
            let K = uniforms.K;

            var sum: f32 = 0.0;

            let numTiles = (K + TILE_SIZE - 1u) / TILE_SIZE;

            for (var t: u32 = 0u; t < numTiles; t = t + 1u) {
                let aRow = row;
                let aCol = t * TILE_SIZE + localCol;
                if (aRow < M && aCol < K) {
                    tileA[localRow][localCol] = A[aRow * K + aCol];
                } else {
                    tileA[localRow][localCol] = 0.0;
                }

                let bRow = t * TILE_SIZE + localRow;
                let bCol = col;
                if (bRow < K && bCol < N) {
                    tileB[localRow][localCol] = B[bRow * N + bCol];
                } else {
                    tileB[localRow][localCol] = 0.0;
                }

                workgroupBarrier();

                for (var k: u32 = 0u; k < TILE_SIZE; k = k + 1u) {
                    sum = sum + tileA[localRow][k] * tileB[k][localCol];
                }

                workgroupBarrier();
            }

            if (row < M && col < N) {
                C[row * N + col] = sum;
            }
        }
    `;

    const shaderModule = device.createShaderModule({ code: shaderCode });

    const bindGroupLayout = device.createBindGroupLayout({
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        ]
    });

    const pipeline = device.createComputePipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
        compute: { module: shaderModule, entryPoint: 'main' }
    });

    return { pipeline, bindGroupLayout, workgroupSize: TILE_SIZE };
}

// ============= Run f32 Matmul =============

export async function runF32Matmul(device, pipeline, bindGroupLayout, A, B, N) {
    const size = N * N;
    const TILE_SIZE = 16;

    const bufferA = device.createBuffer({ size: A.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const bufferB = device.createBuffer({ size: B.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const bufferC = device.createBuffer({ size: size * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const bufferRead = device.createBuffer({ size: size * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const uniformBuffer = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

    device.queue.writeBuffer(bufferA, 0, A);
    device.queue.writeBuffer(bufferB, 0, B);
    device.queue.writeBuffer(uniformBuffer, 0, new Uint32Array([N, N, N, 0]));

    const bindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [
            { binding: 0, resource: { buffer: bufferA } },
            { binding: 1, resource: { buffer: bufferB } },
            { binding: 2, resource: { buffer: bufferC } },
            { binding: 3, resource: { buffer: uniformBuffer } },
        ]
    });

    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(Math.ceil(N / TILE_SIZE), Math.ceil(N / TILE_SIZE));
    passEncoder.end();
    commandEncoder.copyBufferToBuffer(bufferC, 0, bufferRead, 0, size * 4);
    device.queue.submit([commandEncoder.finish()]);

    await bufferRead.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(bufferRead.getMappedRange().slice(0));
    bufferRead.unmap();

    bufferA.destroy();
    bufferB.destroy();
    bufferC.destroy();
    bufferRead.destroy();
    uniformBuffer.destroy();

    return result;
}

// ============= Split Matmul for df64 precision =============
// Computes C = A.hi*B.hi + A.hi*B.lo + A.lo*B.hi using 3 f32 matmuls
// Results assembled on CPU in f64 precision

export async function runSplitMatmul(device, pipeline, bindGroupLayout, A_f64, B_f64, N) {
    const size = N * N;
    const TILE_SIZE = 16;

    // Split inputs into hi/lo components
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
    const createBuffer = (data) => {
        const buf = device.createBuffer({
            size: data.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(buf, 0, data);
        return buf;
    };

    const bufA_hi = createBuffer(A_hi);
    const bufA_lo = createBuffer(A_lo);
    const bufB_hi = createBuffer(B_hi);
    const bufB_lo = createBuffer(B_lo);

    // Output and read buffers
    const bufC = device.createBuffer({ size: size * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const bufRead = device.createBuffer({ size: size * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const uniformBuffer = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(uniformBuffer, 0, new Uint32Array([N, N, N, 0]));

    // Helper to run one matmul and get result
    const runOneMatmul = async (bufA, bufB) => {
        const bindGroup = device.createBindGroup({
            layout: bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: bufA } },
                { binding: 1, resource: { buffer: bufB } },
                { binding: 2, resource: { buffer: bufC } },
                { binding: 3, resource: { buffer: uniformBuffer } },
            ]
        });

        const commandEncoder = device.createCommandEncoder();
        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setPipeline(pipeline);
        passEncoder.setBindGroup(0, bindGroup);
        passEncoder.dispatchWorkgroups(Math.ceil(N / TILE_SIZE), Math.ceil(N / TILE_SIZE));
        passEncoder.end();
        commandEncoder.copyBufferToBuffer(bufC, 0, bufRead, 0, size * 4);
        device.queue.submit([commandEncoder.finish()]);

        await bufRead.mapAsync(GPUMapMode.READ);
        const result = new Float32Array(bufRead.getMappedRange().slice(0));
        bufRead.unmap();
        return result;
    };

    // Compute 3 matmuls
    const C_hh = await runOneMatmul(bufA_hi, bufB_hi);  // A.hi * B.hi
    const C_hl = await runOneMatmul(bufA_hi, bufB_lo);  // A.hi * B.lo
    const C_lh = await runOneMatmul(bufA_lo, bufB_hi);  // A.lo * B.hi

    // Assemble in f64 on CPU
    const C_f64 = new Float64Array(size);
    for (let i = 0; i < size; i++) {
        C_f64[i] = C_hh[i] + C_hl[i] + C_lh[i];
    }

    // Cleanup
    bufA_hi.destroy();
    bufA_lo.destroy();
    bufB_hi.destroy();
    bufB_lo.destroy();
    bufC.destroy();
    bufRead.destroy();
    uniformBuffer.destroy();

    return C_f64;
}

// ============= GPU Diagnostics =============

export async function checkGPUStatus() {
    const results = {
        available: false,
        adapters: [],
        error: null
    };

    if (!navigator.gpu) {
        results.error = 'WebGPU not supported';
        return results;
    }

    results.available = true;

    try {
        const highPerf = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
        if (highPerf) {
            const info = highPerf.info || {};
            results.adapters.push({
                type: 'high-performance',
                vendor: info.vendor || 'Unknown',
                device: info.device || 'Unknown',
                architecture: info.architecture || 'Unknown',
                limits: {
                    maxBufferSize: highPerf.limits.maxBufferSize,
                    maxWorkgroupSize: highPerf.limits.maxComputeWorkgroupSizeX
                }
            });
        }

        const lowPower = await navigator.gpu.requestAdapter({ powerPreference: 'low-power' });
        if (lowPower && lowPower !== highPerf) {
            const info = lowPower.info || {};
            results.adapters.push({
                type: 'low-power',
                vendor: info.vendor || 'Unknown',
                device: info.device || 'Unknown'
            });
        }
    } catch (e) {
        results.error = e.message;
    }

    return results;
}
