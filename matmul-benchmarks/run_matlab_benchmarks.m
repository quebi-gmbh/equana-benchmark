% MATLAB DGEMM Benchmark (single-architecture, quick test)
%
% For the full architecture sweep (SSE4_2 / AVX2 / AVX-512), use:
%   bash run_matlab_benchmarks.sh
%
% This script runs in a single MATLAB session using whatever SIMD
% target MKL selected at startup. Useful for quick sanity checks.
%
% If the AVX-512 MEX kernel (mex_avx512_dgemm) is compiled and on the
% path, an additional column reports its GFLOPS. The MEX kernel uses
% OMP_NUM_THREADS (set before launching MATLAB) for thread control.
%
% Methodology matches NumPy and native OpenBLAS benchmarks:
% - Sizes: 64, 128, 256, 512, 1024, 2048, 4096
% - Thread counts: 1, 2, 4, 8, 16
% - 2 untimed warmup runs + 5 timed runs measured together
%
% Usage:
%   matlab -batch "run_matlab_benchmarks"

sizes = [64, 128, 256, 512, 1024, 2048, 4096];
thread_counts = [1, 2, 4, 8, 16];
RUNS = 5;

has_avx512_mex = exist('mex_avx512_dgemm', 'file') == 3;

fprintf('======================================================================\n');
fprintf('    MATLAB/MKL DGEMM Benchmark\n');
fprintf('======================================================================\n\n');
fprintf('MATLAB version: %s\n', version);
if has_avx512_mex
    fprintf('AVX-512 MEX kernel: available\n');
    fprintf('OMP_NUM_THREADS: %s\n', getenv('OMP_NUM_THREADS'));
else
    fprintf('AVX-512 MEX kernel: not found (run compile_mex_avx512 to build)\n');
end
fprintf('\n');

for ti = 1:length(thread_counts)
    threads = thread_counts(ti);

    % Set thread count for MKL
    maxNumCompThreads(threads);

    if threads == 1
        label = 'Single-threaded';
    else
        label = 'Multi-threaded';
    end

    fprintf('### %s (threads=%d)\n\n', label, threads);

    if has_avx512_mex
        fprintf('| Size | GFLOPS (MKL) | GFLOPS (AVX-512 MEX) |\n');
        fprintf('|------|--------------|----------------------|\n');
    else
        fprintf('| Size | GFLOPS (MKL) |\n');
        fprintf('|------|--------------|\n');
    end

    for si = 1:length(sizes)
        N = sizes(si);

        A = rand(N, N);
        B = rand(N, N);

        % --- MKL benchmark ---
        % 2 untimed warmup runs
        C = A * B; %#ok<NASGU>
        C = A * B; %#ok<NASGU>

        % 5 runs timed together
        tic;
        for r = 1:RUNS
            C = A * B; %#ok<NASGU>
        end
        elapsed = toc;

        mkl_avg = elapsed / RUNS;
        mkl_gflops = 2 * N^3 / mkl_avg / 1e9;

        if has_avx512_mex
            % --- AVX-512 MEX benchmark ---
            % 2 untimed warmup runs
            C = mex_avx512_dgemm(A, B); %#ok<NASGU>
            C = mex_avx512_dgemm(A, B); %#ok<NASGU>

            % 5 runs timed together
            tic;
            for r = 1:RUNS
                C = mex_avx512_dgemm(A, B); %#ok<NASGU>
            end
            elapsed = toc;

            mex_avg = elapsed / RUNS;
            mex_gflops = 2 * N^3 / mex_avg / 1e9;

            fprintf('| %4d | %12.1f | %20.1f |\n', N, mkl_gflops, mex_gflops);
        else
            fprintf('| %4d | %12.1f |\n', N, mkl_gflops);
        end
    end

    fprintf('\n');
end

fprintf('======================================================================\n');
fprintf('Notes:\n');
fprintf('- MATLAB uses Intel MKL which auto-selects the best SIMD target\n');
fprintf('- MKL thread count controlled via maxNumCompThreads()\n');
if has_avx512_mex
    fprintf('- AVX-512 MEX thread count controlled via OMP_NUM_THREADS env var\n');
    fprintf('- OMP_NUM_THREADS must be set before launching MATLAB\n');
end
fprintf('======================================================================\n');
