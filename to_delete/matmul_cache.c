/**
 * Cache-Optimized Matrix Multiplication (NO SIMD)
 * Uses blocking/tiling to improve cache utilization
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* Blocking sizes tuned for L1/L2 cache */
#define BLOCK_SIZE 64

/**
 * Cache-blocked matrix multiplication
 * Uses loop tiling to keep data in cache
 */
__attribute__((export_name("matmul_cache")))
void matmul_cache(int32_t M, int32_t N, int32_t K,
                  const double *A, const double *B, double *C) {
    // Initialize C to zero
    for (int32_t i = 0; i < M * N; i++) {
        C[i] = 0.0;
    }

    // Blocked/tiled matrix multiplication
    // Loop order: jj, kk, ii (good for row-major C and B access)
    for (int32_t jj = 0; jj < N; jj += BLOCK_SIZE) {
        int32_t j_end = (jj + BLOCK_SIZE < N) ? jj + BLOCK_SIZE : N;

        for (int32_t kk = 0; kk < K; kk += BLOCK_SIZE) {
            int32_t k_end = (kk + BLOCK_SIZE < K) ? kk + BLOCK_SIZE : K;

            for (int32_t ii = 0; ii < M; ii += BLOCK_SIZE) {
                int32_t i_end = (ii + BLOCK_SIZE < M) ? ii + BLOCK_SIZE : M;

                // Micro-block multiplication
                for (int32_t i = ii; i < i_end; i++) {
                    for (int32_t k = kk; k < k_end; k++) {
                        double a_ik = A[i * K + k];
                        for (int32_t j = jj; j < j_end; j++) {
                            C[i * N + j] += a_ik * B[k * N + j];
                        }
                    }
                }
            }
        }
    }
}

__attribute__((export_name("malloc_f64")))
double* malloc_f64(int32_t count) {
    return (double*)malloc(count * sizeof(double));
}

__attribute__((export_name("free_f64")))
void free_f64(double* ptr) {
    free(ptr);
}
