% MATLAB DGEMM Benchmark (single-architecture, quick test)
%
% For the full architecture sweep (SSE4_2 / AVX2 / AVX-512), use:
%   bash run_matlab_benchmarks.sh
%
% This script runs in a single MATLAB session using whatever SIMD
% target MKL selected at startup. Useful for quick sanity checks.
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

fprintf('======================================================================\n');
fprintf('    MATLAB/MKL DGEMM Benchmark\n');
fprintf('======================================================================\n\n');
fprintf('MATLAB version: %s\n', version);
fprintf('\n');

for ti = 1:length(thread_counts)
    threads = thread_counts(ti);

    % Set thread count
    maxNumCompThreads(threads);

    if threads == 1
        label = 'Single-threaded';
    else
        label = 'Multi-threaded';
    end

    fprintf('### %s (maxNumCompThreads=%d)\n\n', label, threads);
    fprintf('| Size | GFLOPS (MKL) |\n');
    fprintf('|------|--------------|\n');

    for si = 1:length(sizes)
        N = sizes(si);

        A = rand(N, N);
        B = rand(N, N);

        % 2 untimed warmup runs
        C = A * B; %#ok<NASGU>
        C = A * B; %#ok<NASGU>

        % 5 runs timed together
        tic;
        for r = 1:RUNS
            C = A * B; %#ok<NASGU>
        end
        elapsed = toc;

        avg_time = elapsed / RUNS;
        gflops = 2 * N^3 / avg_time / 1e9;

        fprintf('| %4d | %12.1f |\n', N, gflops);
    end

    fprintf('\n');
end

fprintf('======================================================================\n');
fprintf('Notes:\n');
fprintf('- MATLAB uses Intel MKL which auto-selects the best SIMD target\n');
fprintf('- Thread count controlled via maxNumCompThreads()\n');
fprintf('======================================================================\n');
