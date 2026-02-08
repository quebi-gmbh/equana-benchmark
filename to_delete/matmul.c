/**
 * Optimized Matrix Multiplication for WebAssembly
 *
 * Implements cache-blocked DGEMM following OpenBLAS/GotoBLAS algorithm:
 * - 3-level blocking for cache hierarchy (L1/L2/L3)
 * - Panel packing for sequential memory access
 * - SIMD micro-kernels (AVX2 -> translated to WASM SIMD by emscripten)
 * - 4x4 register blocking
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* AVX2 intrinsics - Emscripten translates these to WASM SIMD */
#ifdef __AVX2__
#include <immintrin.h>
#define USE_AVX2 1
#elif defined(__wasm_simd128__)
#include <wasm_simd128.h>
#define USE_SIMD128 1
#endif

/* ============= Blocking Parameters ============= */

/* Cache blocking sizes - tuned for L1=32KB, L2=256KB */
#define MC 64     /* Block rows of A: 64*128*8 = 64KB in L2 */
#define KC 128    /* Block cols of A/rows of B: fits in L1 */
#define NC 256    /* Block cols of B: 128*256*8 = 256KB in L2 */

/* Micro-kernel dimensions */
#define MR 4      /* Rows per micro-kernel */
#define NR 4      /* Cols per micro-kernel */

/* ============= Memory Buffers ============= */

static double *pack_a = NULL;  /* Packed A panel: MC x KC */
static double *pack_b = NULL;  /* Packed B panel: KC x NC */
static size_t pack_a_size = 0;
static size_t pack_b_size = 0;

static void ensure_buffers(int64_t mc, int64_t kc, int64_t nc) {
    size_t need_a = (mc + MR) * (kc + 4) * sizeof(double);
    size_t need_b = (kc + 4) * (nc + NR) * sizeof(double);

    if (pack_a_size < need_a) {
        free(pack_a);
        pack_a = (double*)malloc(need_a);
        pack_a_size = need_a;
    }
    if (pack_b_size < need_b) {
        free(pack_b);
        pack_b = (double*)malloc(need_b);
        pack_b_size = need_b;
    }
}

/* ============= Panel Packing ============= */

/**
 * Pack A panel (mc x kc) into contiguous MR-strided format
 * Input: column-major A with leading dimension lda
 * Output: pack_a in format suitable for micro-kernel
 *
 * Layout: For each MR rows, store kc elements contiguously
 * [a00 a10 a20 a30] [a01 a11 a21 a31] ...
 */
__attribute__((noinline))
static void pack_panel_a(int64_t mc, int64_t kc, const double *A, int64_t lda, double *pa) {
    int64_t i, j;

    /* Pack full MR-row panels */
    for (i = 0; i + MR <= mc; i += MR) {
        const double *a_ptr = A + i;
        for (j = 0; j < kc; j++) {
            pa[0] = a_ptr[0];
            pa[1] = a_ptr[1];
            pa[2] = a_ptr[2];
            pa[3] = a_ptr[3];
            a_ptr += lda;  /* Next column */
            pa += MR;
        }
    }

    /* Pack remaining rows (< MR) with zero padding */
    if (i < mc) {
        int64_t mr_rem = mc - i;
        const double *a_ptr = A + i;
        for (j = 0; j < kc; j++) {
            for (int64_t ii = 0; ii < mr_rem; ii++) {
                pa[ii] = a_ptr[ii];
            }
            for (int64_t ii = mr_rem; ii < MR; ii++) {
                pa[ii] = 0.0;
            }
            a_ptr += lda;
            pa += MR;
        }
    }
}

/**
 * Pack B panel (kc x nc) into contiguous NR-strided format
 * Input: column-major B with leading dimension ldb
 * Output: pack_b in format suitable for micro-kernel
 *
 * Layout: For each NR cols, store kc elements interleaved
 * [b00 b01 b02 b03] [b10 b11 b12 b13] ...
 */
__attribute__((noinline))
static void pack_panel_b(int64_t kc, int64_t nc, const double *B, int64_t ldb, double *pb) {
    int64_t i, j;

    /* Pack full NR-column panels */
    for (j = 0; j + NR <= nc; j += NR) {
        const double *b0 = B + j * ldb;
        const double *b1 = b0 + ldb;
        const double *b2 = b1 + ldb;
        const double *b3 = b2 + ldb;

        for (i = 0; i < kc; i++) {
            pb[0] = b0[i];
            pb[1] = b1[i];
            pb[2] = b2[i];
            pb[3] = b3[i];
            pb += NR;
        }
    }

    /* Pack remaining columns (< NR) with zero padding */
    if (j < nc) {
        int64_t nr_rem = nc - j;
        for (i = 0; i < kc; i++) {
            for (int64_t jj = 0; jj < nr_rem; jj++) {
                pb[jj] = B[(j + jj) * ldb + i];
            }
            for (int64_t jj = nr_rem; jj < NR; jj++) {
                pb[jj] = 0.0;
            }
            pb += NR;
        }
    }
}

/* ============= Micro-Kernels ============= */

#ifdef USE_AVX2
/**
 * 4x4 micro-kernel using AVX2 (4 doubles per vector)
 * Computes: C[0:4, 0:4] += A[0:4, 0:k] * B[0:k, 0:4]
 *
 * Uses 4 accumulator vectors (one per column of C)
 * Each __m256d holds all 4 rows of one column
 */
__attribute__((noinline))
static void micro_kernel_4x4_avx2(int64_t kc, double alpha,
                                   const double *pa, const double *pb,
                                   double *C, int64_t ldc) {
    /* 4 accumulators for 4 columns of C (each holds 4 rows) */
    __m256d c0 = _mm256_setzero_pd();
    __m256d c1 = _mm256_setzero_pd();
    __m256d c2 = _mm256_setzero_pd();
    __m256d c3 = _mm256_setzero_pd();

    /* Main loop */
    for (int64_t k = 0; k < kc; k++) {
        __m256d a = _mm256_loadu_pd(pa);           /* [a0, a1, a2, a3] */
        __m256d b0 = _mm256_set1_pd(pb[0]);        /* broadcast b[0] */
        __m256d b1 = _mm256_set1_pd(pb[1]);        /* broadcast b[1] */
        __m256d b2 = _mm256_set1_pd(pb[2]);        /* broadcast b[2] */
        __m256d b3 = _mm256_set1_pd(pb[3]);        /* broadcast b[3] */

        c0 = _mm256_add_pd(c0, _mm256_mul_pd(a, b0));
        c1 = _mm256_add_pd(c1, _mm256_mul_pd(a, b1));
        c2 = _mm256_add_pd(c2, _mm256_mul_pd(a, b2));
        c3 = _mm256_add_pd(c3, _mm256_mul_pd(a, b3));

        pa += MR;
        pb += NR;
    }

    /* Scale by alpha */
    __m256d alpha_v = _mm256_set1_pd(alpha);
    c0 = _mm256_mul_pd(c0, alpha_v);
    c1 = _mm256_mul_pd(c1, alpha_v);
    c2 = _mm256_mul_pd(c2, alpha_v);
    c3 = _mm256_mul_pd(c3, alpha_v);

    /* Load C columns, add results, store back */
    __m256d t;
    t = _mm256_loadu_pd(C);
    _mm256_storeu_pd(C, _mm256_add_pd(t, c0));

    t = _mm256_loadu_pd(C + ldc);
    _mm256_storeu_pd(C + ldc, _mm256_add_pd(t, c1));

    t = _mm256_loadu_pd(C + 2 * ldc);
    _mm256_storeu_pd(C + 2 * ldc, _mm256_add_pd(t, c2));

    t = _mm256_loadu_pd(C + 3 * ldc);
    _mm256_storeu_pd(C + 3 * ldc, _mm256_add_pd(t, c3));
}

#elif defined(USE_SIMD128)
/**
 * 4x4 micro-kernel using WASM SIMD128 (2 doubles per vector)
 */
static void micro_kernel_4x4_simd128(int64_t kc, double alpha,
                                      const double *pa, const double *pb,
                                      double *C, int64_t ldc) {
    /* 4 columns of C, each split into 2 SIMD vectors (lo=rows 0-1, hi=rows 2-3) */
    v128_t c0_lo = wasm_f64x2_splat(0.0);
    v128_t c0_hi = wasm_f64x2_splat(0.0);
    v128_t c1_lo = wasm_f64x2_splat(0.0);
    v128_t c1_hi = wasm_f64x2_splat(0.0);
    v128_t c2_lo = wasm_f64x2_splat(0.0);
    v128_t c2_hi = wasm_f64x2_splat(0.0);
    v128_t c3_lo = wasm_f64x2_splat(0.0);
    v128_t c3_hi = wasm_f64x2_splat(0.0);

    for (int64_t k = 0; k < kc; k++) {
        v128_t a_lo = wasm_v128_load(pa);
        v128_t a_hi = wasm_v128_load(pa + 2);

        v128_t b0 = wasm_f64x2_splat(pb[0]);
        v128_t b1 = wasm_f64x2_splat(pb[1]);
        v128_t b2 = wasm_f64x2_splat(pb[2]);
        v128_t b3 = wasm_f64x2_splat(pb[3]);

        c0_lo = wasm_f64x2_add(c0_lo, wasm_f64x2_mul(a_lo, b0));
        c0_hi = wasm_f64x2_add(c0_hi, wasm_f64x2_mul(a_hi, b0));
        c1_lo = wasm_f64x2_add(c1_lo, wasm_f64x2_mul(a_lo, b1));
        c1_hi = wasm_f64x2_add(c1_hi, wasm_f64x2_mul(a_hi, b1));
        c2_lo = wasm_f64x2_add(c2_lo, wasm_f64x2_mul(a_lo, b2));
        c2_hi = wasm_f64x2_add(c2_hi, wasm_f64x2_mul(a_hi, b2));
        c3_lo = wasm_f64x2_add(c3_lo, wasm_f64x2_mul(a_lo, b3));
        c3_hi = wasm_f64x2_add(c3_hi, wasm_f64x2_mul(a_hi, b3));

        pa += MR;
        pb += NR;
    }

    v128_t alpha_v = wasm_f64x2_splat(alpha);
    c0_lo = wasm_f64x2_mul(c0_lo, alpha_v);
    c0_hi = wasm_f64x2_mul(c0_hi, alpha_v);
    c1_lo = wasm_f64x2_mul(c1_lo, alpha_v);
    c1_hi = wasm_f64x2_mul(c1_hi, alpha_v);
    c2_lo = wasm_f64x2_mul(c2_lo, alpha_v);
    c2_hi = wasm_f64x2_mul(c2_hi, alpha_v);
    c3_lo = wasm_f64x2_mul(c3_lo, alpha_v);
    c3_hi = wasm_f64x2_mul(c3_hi, alpha_v);

    double *c0 = C;
    double *c1 = C + ldc;
    double *c2 = C + 2 * ldc;
    double *c3 = C + 3 * ldc;

    v128_t t0, t1;
    t0 = wasm_v128_load(c0);
    t1 = wasm_v128_load(c0 + 2);
    wasm_v128_store(c0, wasm_f64x2_add(t0, c0_lo));
    wasm_v128_store(c0 + 2, wasm_f64x2_add(t1, c0_hi));

    t0 = wasm_v128_load(c1);
    t1 = wasm_v128_load(c1 + 2);
    wasm_v128_store(c1, wasm_f64x2_add(t0, c1_lo));
    wasm_v128_store(c1 + 2, wasm_f64x2_add(t1, c1_hi));

    t0 = wasm_v128_load(c2);
    t1 = wasm_v128_load(c2 + 2);
    wasm_v128_store(c2, wasm_f64x2_add(t0, c2_lo));
    wasm_v128_store(c2 + 2, wasm_f64x2_add(t1, c2_hi));

    t0 = wasm_v128_load(c3);
    t1 = wasm_v128_load(c3 + 2);
    wasm_v128_store(c3, wasm_f64x2_add(t0, c3_lo));
    wasm_v128_store(c3 + 2, wasm_f64x2_add(t1, c3_hi));
}
#endif

/**
 * 4x4 micro-kernel - scalar fallback
 */
static void micro_kernel_4x4_scalar(int64_t kc, double alpha,
                                     const double *pa, const double *pb,
                                     double *C, int64_t ldc) {
    /* 16 accumulators for 4x4 tile */
    double c00 = 0, c10 = 0, c20 = 0, c30 = 0;
    double c01 = 0, c11 = 0, c21 = 0, c31 = 0;
    double c02 = 0, c12 = 0, c22 = 0, c32 = 0;
    double c03 = 0, c13 = 0, c23 = 0, c33 = 0;

    for (int64_t k = 0; k < kc; k++) {
        double a0 = pa[0], a1 = pa[1], a2 = pa[2], a3 = pa[3];
        double b0 = pb[0], b1 = pb[1], b2 = pb[2], b3 = pb[3];

        c00 += a0 * b0; c10 += a1 * b0; c20 += a2 * b0; c30 += a3 * b0;
        c01 += a0 * b1; c11 += a1 * b1; c21 += a2 * b1; c31 += a3 * b1;
        c02 += a0 * b2; c12 += a1 * b2; c22 += a2 * b2; c32 += a3 * b2;
        c03 += a0 * b3; c13 += a1 * b3; c23 += a2 * b3; c33 += a3 * b3;

        pa += MR;
        pb += NR;
    }

    /* Scale and accumulate to C */
    C[0]         += alpha * c00; C[1]         += alpha * c10;
    C[2]         += alpha * c20; C[3]         += alpha * c30;
    C[ldc]       += alpha * c01; C[ldc + 1]   += alpha * c11;
    C[ldc + 2]   += alpha * c21; C[ldc + 3]   += alpha * c31;
    C[2*ldc]     += alpha * c02; C[2*ldc + 1] += alpha * c12;
    C[2*ldc + 2] += alpha * c22; C[2*ldc + 3] += alpha * c32;
    C[3*ldc]     += alpha * c03; C[3*ldc + 1] += alpha * c13;
    C[3*ldc + 2] += alpha * c23; C[3*ldc + 3] += alpha * c33;
}

/**
 * Edge micro-kernel for partial tiles
 */
static void micro_kernel_edge(int64_t mr, int64_t nr, int64_t kc, double alpha,
                               const double *pa, const double *pb,
                               double *C, int64_t ldc) {
    for (int64_t j = 0; j < nr; j++) {
        for (int64_t i = 0; i < mr; i++) {
            double sum = 0.0;
            for (int64_t k = 0; k < kc; k++) {
                sum += pa[k * MR + i] * pb[k * NR + j];
            }
            C[i + j * ldc] += alpha * sum;
        }
    }
}

/* ============= Macro Kernel ============= */

/**
 * Macro kernel: multiply packed panels
 * C[0:mc, 0:nc] += alpha * pack_a[0:mc, 0:kc] * pack_b[0:kc, 0:nc]
 */
__attribute__((noinline))
static void macro_kernel(int64_t mc, int64_t nc, int64_t kc, double alpha,
                          const double *pa, const double *pb,
                          double *C, int64_t ldc) {
    int64_t i, j;

    for (j = 0; j + NR <= nc; j += NR) {
        for (i = 0; i + MR <= mc; i += MR) {
            #ifdef USE_AVX2
            micro_kernel_4x4_avx2(kc, alpha,
                                   pa + i * kc,
                                   pb + j * kc,
                                   C + i + j * ldc, ldc);
            #elif defined(USE_SIMD128)
            micro_kernel_4x4_simd128(kc, alpha,
                                      pa + i * kc,
                                      pb + j * kc,
                                      C + i + j * ldc, ldc);
            #else
            micro_kernel_4x4_scalar(kc, alpha,
                                     pa + i * kc,
                                     pb + j * kc,
                                     C + i + j * ldc, ldc);
            #endif
        }
        /* Handle remaining rows */
        if (i < mc) {
            micro_kernel_edge(mc - i, NR, kc, alpha,
                              pa + i * kc,
                              pb + j * kc,
                              C + i + j * ldc, ldc);
        }
    }

    /* Handle remaining columns */
    if (j < nc) {
        for (i = 0; i < mc; i += MR) {
            int64_t mr = (i + MR <= mc) ? MR : (mc - i);
            micro_kernel_edge(mr, nc - j, kc, alpha,
                              pa + i * kc,
                              pb + j * kc,
                              C + i + j * ldc, ldc);
        }
    }
}

/* ============= DGEMM Interface ============= */

/**
 * DGEMM: C = alpha * A * B + beta * C
 *
 * Column-major layout (Fortran/BLAS convention)
 * A: m x k, B: k x n, C: m x n
 *
 * Implements GotoBLAS 3-level blocking algorithm:
 * 1. Loop over NC blocks of B columns
 * 2. Loop over KC blocks of shared dimension
 * 3. Loop over MC blocks of A rows
 */
__attribute__((export_name("dgemm")))
void dgemm(int32_t m, int32_t n, int32_t k,
           double alpha, const double *A, int32_t lda,
           const double *B, int32_t ldb,
           double beta, double *C, int32_t ldc) {

    if (m == 0 || n == 0 || k == 0) return;

    /* Scale C by beta */
    if (beta == 0.0) {
        for (int32_t j = 0; j < n; j++) {
            for (int32_t i = 0; i < m; i++) {
                C[i + j * ldc] = 0.0;
            }
        }
    } else if (beta != 1.0) {
        for (int32_t j = 0; j < n; j++) {
            for (int32_t i = 0; i < m; i++) {
                C[i + j * ldc] *= beta;
            }
        }
    }

    if (alpha == 0.0) return;

    /* Ensure packing buffers */
    ensure_buffers(MC, KC, NC);

    /* 3-level blocking loops */
    for (int64_t jc = 0; jc < n; jc += NC) {
        int64_t nc = (jc + NC <= n) ? NC : (n - jc);

        for (int64_t pc = 0; pc < k; pc += KC) {
            int64_t kc = (pc + KC <= k) ? KC : (k - pc);

            /* Pack B panel: B[pc:pc+kc, jc:jc+nc] */
            pack_panel_b(kc, nc, B + pc + jc * ldb, ldb, pack_b);

            for (int64_t ic = 0; ic < m; ic += MC) {
                int64_t mc = (ic + MC <= m) ? MC : (m - ic);

                /* Pack A panel: A[ic:ic+mc, pc:pc+kc] */
                pack_panel_a(mc, kc, A + ic + pc * lda, lda, pack_a);

                /* Compute C[ic:ic+mc, jc:jc+nc] += alpha * pack_a * pack_b */
                macro_kernel(mc, nc, kc, alpha, pack_a, pack_b,
                             C + ic + jc * ldc, ldc);
            }
        }
    }
}

/* ============= Row-Major Matmul Interface ============= */

/**
 * Matrix multiplication with row-major input/output
 * C = A * B where A is MxK, B is KxN, C is MxN
 *
 * Converts to column-major and calls DGEMM
 */
__attribute__((export_name("matmul_f64")))
void matmul_f64(int32_t M, int32_t N, int32_t K,
                const double *A, const double *B, double *C) {

    /* For row-major: C = A * B is equivalent to C^T = B^T * A^T in column-major
     * But since we want row-major output, we can use:
     * Row-major C[i,j] = Column-major C[j,i]
     *
     * Actually simpler: treat row-major as column-major with transposed dimensions
     * Row-major A (MxK) = Column-major A^T (KxM)
     *
     * So: C_row(MxN) = A_row(MxK) * B_row(KxN)
     * becomes: C_col(NxM) = B_col(NxK) * A_col(KxM)
     * i.e.: dgemm(N, M, K, B, A, C) with swapped dimensions
     */

    dgemm(N, M, K,
          1.0, B, N,   /* B treated as NxK column-major */
          A, K,        /* A treated as KxM column-major */
          0.0, C, N);  /* C is NxM column-major = MxN row-major */
}

/* ============= Memory Management ============= */

__attribute__((export_name("malloc_f64")))
double* malloc_f64(int32_t count) {
    return (double*)malloc(count * sizeof(double));
}

__attribute__((export_name("free_f64")))
void free_f64(double* ptr) {
    free(ptr);
}

__attribute__((export_name("cleanup")))
void cleanup(void) {
    free(pack_a);
    free(pack_b);
    pack_a = NULL;
    pack_b = NULL;
    pack_a_size = 0;
    pack_b_size = 0;
}
