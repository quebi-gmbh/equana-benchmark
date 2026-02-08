/**
 * Optimized Matrix Multiplication for WebAssembly - 2x2 Micro-kernel Version
 *
 * Same algorithm as matmul.c but with MR=2, NR=2 micro-kernel
 * for comparison with the 4x4 version.
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef __wasm_simd128__
#include <wasm_simd128.h>
#define USE_SIMD128 1
#endif

/* ============= Blocking Parameters ============= */

/* Cache blocking sizes */
#define MC 64
#define KC 128
#define NC 256

/* Micro-kernel dimensions - 2x2 */
#define MR 2
#define NR 2

/* ============= Memory Buffers ============= */

static double *pack_a = NULL;
static double *pack_b = NULL;
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
 */
__attribute__((noinline))
static void pack_panel_a(int64_t mc, int64_t kc, const double *A, int64_t lda, double *pa) {
    int64_t i, j;

    /* Pack full MR-row panels (2 rows at a time) */
    for (i = 0; i + MR <= mc; i += MR) {
        const double *a_ptr = A + i;
        for (j = 0; j < kc; j++) {
            pa[0] = a_ptr[0];
            pa[1] = a_ptr[1];
            a_ptr += lda;
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
 */
__attribute__((noinline))
static void pack_panel_b(int64_t kc, int64_t nc, const double *B, int64_t ldb, double *pb) {
    int64_t i, j;

    /* Pack full NR-column panels (2 columns at a time) */
    for (j = 0; j + NR <= nc; j += NR) {
        const double *b0 = B + j * ldb;
        const double *b1 = b0 + ldb;

        for (i = 0; i < kc; i++) {
            pb[0] = b0[i];
            pb[1] = b1[i];
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

#ifdef USE_SIMD128
/**
 * 2x2 micro-kernel using WASM SIMD128
 * Uses f64x2 perfectly - one vector holds 2 doubles
 */
static void micro_kernel_2x2_simd128(int64_t kc, double alpha,
                                      const double *pa, const double *pb,
                                      double *C, int64_t ldc) {
    /* 2 columns of C, each is a f64x2 vector (2 rows) */
    v128_t c0 = wasm_f64x2_splat(0.0);  /* C[0:2, 0] */
    v128_t c1 = wasm_f64x2_splat(0.0);  /* C[0:2, 1] */

    for (int64_t k = 0; k < kc; k++) {
        /* Load packed A: [a0, a1] */
        v128_t a = wasm_v128_load(pa);

        /* Load packed B and broadcast: b0, b1 */
        v128_t b0 = wasm_f64x2_splat(pb[0]);
        v128_t b1 = wasm_f64x2_splat(pb[1]);

        /* Accumulate */
        c0 = wasm_f64x2_add(c0, wasm_f64x2_mul(a, b0));
        c1 = wasm_f64x2_add(c1, wasm_f64x2_mul(a, b1));

        pa += MR;
        pb += NR;
    }

    /* Scale by alpha */
    v128_t alpha_v = wasm_f64x2_splat(alpha);
    c0 = wasm_f64x2_mul(c0, alpha_v);
    c1 = wasm_f64x2_mul(c1, alpha_v);

    /* Load C, add results, store back */
    v128_t t0 = wasm_v128_load(C);
    v128_t t1 = wasm_v128_load(C + ldc);
    wasm_v128_store(C, wasm_f64x2_add(t0, c0));
    wasm_v128_store(C + ldc, wasm_f64x2_add(t1, c1));
}
#endif

/**
 * 2x2 micro-kernel - scalar fallback
 */
static void micro_kernel_2x2_scalar(int64_t kc, double alpha,
                                     const double *pa, const double *pb,
                                     double *C, int64_t ldc) {
    /* 4 accumulators for 2x2 tile */
    double c00 = 0, c10 = 0;
    double c01 = 0, c11 = 0;

    for (int64_t k = 0; k < kc; k++) {
        double a0 = pa[0], a1 = pa[1];
        double b0 = pb[0], b1 = pb[1];

        c00 += a0 * b0; c10 += a1 * b0;
        c01 += a0 * b1; c11 += a1 * b1;

        pa += MR;
        pb += NR;
    }

    /* Scale and accumulate to C */
    C[0]       += alpha * c00;
    C[1]       += alpha * c10;
    C[ldc]     += alpha * c01;
    C[ldc + 1] += alpha * c11;
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

__attribute__((noinline))
static void macro_kernel(int64_t mc, int64_t nc, int64_t kc, double alpha,
                          const double *pa, const double *pb,
                          double *C, int64_t ldc) {
    int64_t i, j;

    for (j = 0; j + NR <= nc; j += NR) {
        for (i = 0; i + MR <= mc; i += MR) {
            #ifdef USE_SIMD128
            micro_kernel_2x2_simd128(kc, alpha,
                                      pa + i * kc,
                                      pb + j * kc,
                                      C + i + j * ldc, ldc);
            #else
            micro_kernel_2x2_scalar(kc, alpha,
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

__attribute__((export_name("dgemm_2x2")))
void dgemm_2x2(int32_t m, int32_t n, int32_t k,
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

    ensure_buffers(MC, KC, NC);

    /* 3-level blocking loops */
    for (int64_t jc = 0; jc < n; jc += NC) {
        int64_t nc = (jc + NC <= n) ? NC : (n - jc);

        for (int64_t pc = 0; pc < k; pc += KC) {
            int64_t kc = (pc + KC <= k) ? KC : (k - pc);

            pack_panel_b(kc, nc, B + pc + jc * ldb, ldb, pack_b);

            for (int64_t ic = 0; ic < m; ic += MC) {
                int64_t mc = (ic + MC <= m) ? MC : (m - ic);

                pack_panel_a(mc, kc, A + ic + pc * lda, lda, pack_a);

                macro_kernel(mc, nc, kc, alpha, pack_a, pack_b,
                             C + ic + jc * ldc, ldc);
            }
        }
    }
}

/* ============= Row-Major Matmul Interface ============= */

__attribute__((export_name("matmul_f64_2x2")))
void matmul_f64_2x2(int32_t M, int32_t N, int32_t K,
                    const double *A, const double *B, double *C) {
    dgemm_2x2(N, M, K,
              1.0, B, N,
              A, K,
              0.0, C, N);
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
