% Compile the AVX-512 DGEMM MEX kernel
%
% Requires: GCC with AVX-512 support, OpenMP
%
% Usage:
%   matlab -batch "compile_mex_avx512"
%   — or from the MATLAB command window:
%   compile_mex_avx512

fprintf('Compiling mex_avx512_dgemm.c with AVX-512 + OpenMP...\n');

mex('-v', ...
    'CFLAGS=$CFLAGS -mavx512f -mavx512vl -mfma -O3 -fopenmp', ...
    'LDFLAGS=$LDFLAGS -fopenmp', ...
    'mex_avx512_dgemm.c');

fprintf('Done. MEX file: %s\n', which('mex_avx512_dgemm'));
