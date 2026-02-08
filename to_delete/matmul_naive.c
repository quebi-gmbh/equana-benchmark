/**
 * Naive Matrix Multiplication - baseline for benchmarking
 */

#include <stdint.h>
#include <stdlib.h>

/**
 * Naive O(n³) matrix multiplication
 * C = A * B where A is MxK, B is KxN, C is MxN (row-major)
 */
__attribute__((export_name("matmul_naive")))
void matmul_naive(int32_t M, int32_t N, int32_t K,
                  const double *A, const double *B, double *C) {
    // Initialize C to zero
    for (int32_t i = 0; i < M * N; i++) {
        C[i] = 0.0;
    }

    // Triple nested loop - worst cache behavior
    for (int32_t i = 0; i < M; i++) {
        for (int32_t j = 0; j < N; j++) {
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
