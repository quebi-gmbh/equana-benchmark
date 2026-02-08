/**
 * SIMD-Only Matrix Multiplication (NO cache blocking)
 * Uses WASM SIMD128 for vectorization but naive loop order
 */

#include <stdint.h>
#include <stdlib.h>
#include <wasm_simd128.h>

/**
 * SIMD matrix multiplication without cache blocking
 * Processes 2 doubles at a time using f64x2 vectors
 */
__attribute__((export_name("matmul_simd")))
void matmul_simd(int32_t M, int32_t N, int32_t K,
                 const double *A, const double *B, double *C) {
    // Initialize C to zero
    for (int32_t i = 0; i < M * N; i++) {
        C[i] = 0.0;
    }

    // Process 2 columns at a time with SIMD
    for (int32_t i = 0; i < M; i++) {
        for (int32_t j = 0; j < N - 1; j += 2) {
            v128_t sum = wasm_f64x2_splat(0.0);

            for (int32_t k = 0; k < K; k++) {
                // Broadcast A[i,k]
                v128_t a_ik = wasm_f64x2_splat(A[i * K + k]);
                // Load B[k,j] and B[k,j+1]
                v128_t b_kj = wasm_v128_load(&B[k * N + j]);
                // sum += a * b
                sum = wasm_f64x2_add(sum, wasm_f64x2_mul(a_ik, b_kj));
            }

            // Store result
            wasm_v128_store(&C[i * N + j], sum);
        }

        // Handle odd column if N is odd
        if (N % 2 == 1) {
            int32_t j = N - 1;
            double sum = 0.0;
            for (int32_t k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
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
