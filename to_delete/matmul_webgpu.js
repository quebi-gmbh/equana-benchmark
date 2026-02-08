/**
 * WebGPU Matrix Multiplication
 * Uses compute shaders for GPU-accelerated matmul
 */

export async function initWebGPU() {
    if (!navigator.gpu) {
        throw new Error('WebGPU not supported');
    }

    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
        throw new Error('No GPU adapter found');
    }

    const device = await adapter.requestDevice();
    return { adapter, device };
}

/**
 * Create matmul compute pipeline
 */
export function createMatmulPipeline(device, M, N, K) {
    const workgroupSize = 16; // 16x16 workgroup

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

        const TILE_SIZE: u32 = ${workgroupSize}u;

        var<workgroup> tileA: array<array<f32, ${workgroupSize}>, ${workgroupSize}>;
        var<workgroup> tileB: array<array<f32, ${workgroupSize}>, ${workgroupSize}>;

        @compute @workgroup_size(${workgroupSize}, ${workgroupSize})
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
                // Load tile of A into shared memory
                let aRow = row;
                let aCol = t * TILE_SIZE + localCol;
                if (aRow < M && aCol < K) {
                    tileA[localRow][localCol] = A[aRow * K + aCol];
                } else {
                    tileA[localRow][localCol] = 0.0;
                }

                // Load tile of B into shared memory
                let bRow = t * TILE_SIZE + localRow;
                let bCol = col;
                if (bRow < K && bCol < N) {
                    tileB[localRow][localCol] = B[bRow * N + bCol];
                } else {
                    tileB[localRow][localCol] = 0.0;
                }

                workgroupBarrier();

                // Compute partial dot product
                for (var k: u32 = 0u; k < TILE_SIZE; k = k + 1u) {
                    sum = sum + tileA[localRow][k] * tileB[k][localCol];
                }

                workgroupBarrier();
            }

            // Write result
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

    const pipelineLayout = device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout]
    });

    const pipeline = device.createComputePipeline({
        layout: pipelineLayout,
        compute: {
            module: shaderModule,
            entryPoint: 'main'
        }
    });

    return { pipeline, bindGroupLayout, workgroupSize };
}

/**
 * Run matmul on GPU
 */
export async function matmulGPU(device, pipeline, bindGroupLayout, workgroupSize, A, B, M, N, K) {
    // Create buffers
    const bufferA = device.createBuffer({
        size: A.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    const bufferB = device.createBuffer({
        size: B.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    const bufferC = device.createBuffer({
        size: M * N * 4, // f32
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    const uniformBuffer = device.createBuffer({
        size: 16, // 3 u32 + padding
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Upload data
    device.queue.writeBuffer(bufferA, 0, A);
    device.queue.writeBuffer(bufferB, 0, B);
    device.queue.writeBuffer(uniformBuffer, 0, new Uint32Array([M, N, K, 0]));

    // Create bind group
    const bindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [
            { binding: 0, resource: { buffer: bufferA } },
            { binding: 1, resource: { buffer: bufferB } },
            { binding: 2, resource: { buffer: bufferC } },
            { binding: 3, resource: { buffer: uniformBuffer } },
        ]
    });

    // Dispatch
    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);

    const workgroupsX = Math.ceil(N / workgroupSize);
    const workgroupsY = Math.ceil(M / workgroupSize);
    passEncoder.dispatchWorkgroups(workgroupsX, workgroupsY);
    passEncoder.end();

    // Read back result
    const readBuffer = device.createBuffer({
        size: M * N * 4,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    commandEncoder.copyBufferToBuffer(bufferC, 0, readBuffer, 0, M * N * 4);

    device.queue.submit([commandEncoder.finish()]);

    // Wait and read
    await readBuffer.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(readBuffer.getMappedRange().slice(0));
    readBuffer.unmap();

    // Cleanup
    bufferA.destroy();
    bufferB.destroy();
    bufferC.destroy();
    uniformBuffer.destroy();
    readBuffer.destroy();

    return result;
}

/**
 * Benchmark helper - runs matmul without reading back (faster for benchmarking)
 */
export async function matmulGPUBenchmark(device, pipeline, bindGroupLayout, workgroupSize, bufferA, bufferB, bufferC, uniformBuffer, M, N, K) {
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

    const workgroupsX = Math.ceil(N / workgroupSize);
    const workgroupsY = Math.ceil(M / workgroupSize);
    passEncoder.dispatchWorkgroups(workgroupsX, workgroupsY);
    passEncoder.end();

    device.queue.submit([commandEncoder.finish()]);
    await device.queue.onSubmittedWorkDone();
}
