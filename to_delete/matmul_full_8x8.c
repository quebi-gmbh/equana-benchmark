/**
 * Optimized Matrix Multiplication - 8x8 Micro-kernel Version
 *
 * Larger micro-kernel for potentially better compute/memory ratio
 * BUT requires 64 accumulators (32 f64x2 vectors) - may cause register spilling
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef __wasm_simd128__
#include <wasm_simd128.h>
#define USE_SIMD128 1
#endif

/* ============= Blocking Parameters ============= */

#define MC 64
#define KC 128
#define NC 256

/* 8x8 micro-kernel */
#define MR 8
#define NR 8

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

__attribute__((noinline))
static void pack_panel_a(int64_t mc, int64_t kc, const double *A, int64_t lda, double *pa) {
    int64_t i, j;

    for (i = 0; i + MR <= mc; i += MR) {
        const double *a_ptr = A + i;
        for (j = 0; j < kc; j++) {
            for (int ii = 0; ii < MR; ii++) {
                pa[ii] = a_ptr[ii];
            }
            a_ptr += lda;
            pa += MR;
        }
    }

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

__attribute__((noinline))
static void pack_panel_b(int64_t kc, int64_t nc, const double *B, int64_t ldb, double *pb) {
    int64_t i, j;

    for (j = 0; j + NR <= nc; j += NR) {
        for (i = 0; i < kc; i++) {
            for (int jj = 0; jj < NR; jj++) {
                pb[jj] = B[(j + jj) * ldb + i];
            }
            pb += NR;
        }
    }

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
 * 8x8 micro-kernel using WASM SIMD128
 * 8 columns × (8 rows / 2 per vector) = 32 accumulator vectors
 * This is a lot of register pressure!
 */
static void micro_kernel_8x8_simd128(int64_t kc, double alpha,
                                      const double *pa, const double *pb,
                                      double *C, int64_t ldc) {
    /* 8 columns, each split into 4 f64x2 vectors (rows 0-1, 2-3, 4-5, 6-7) */
    v128_t c00 = wasm_f64x2_splat(0.0), c01 = wasm_f64x2_splat(0.0);
    v128_t c02 = wasm_f64x2_splat(0.0), c03 = wasm_f64x2_splat(0.0);
    v128_t c10 = wasm_f64x2_splat(0.0), c11 = wasm_f64x2_splat(0.0);
    v128_t c12 = wasm_f64x2_splat(0.0), c13 = wasm_f64x2_splat(0.0);
    v128_t c20 = wasm_f64x2_splat(0.0), c21 = wasm_f64x2_splat(0.0);
    v128_t c22 = wasm_f64x2_splat(0.0), c23 = wasm_f64x2_splat(0.0);
    v128_t c30 = wasm_f64x2_splat(0.0), c31 = wasm_f64x2_splat(0.0);
    v128_t c32 = wasm_f64x2_splat(0.0), c33 = wasm_f64x2_splat(0.0);
    v128_t c40 = wasm_f64x2_splat(0.0), c41 = wasm_f64x2_splat(0.0);
    v128_t c42 = wasm_f64x2_splat(0.0), c43 = wasm_f64x2_splat(0.0);
    v128_t c50 = wasm_f64x2_splat(0.0), c51 = wasm_f64x2_splat(0.0);
    v128_t c52 = wasm_f64x2_splat(0.0), c53 = wasm_f64x2_splat(0.0);
    v128_t c60 = wasm_f64x2_splat(0.0), c61 = wasm_f64x2_splat(0.0);
    v128_t c62 = wasm_f64x2_splat(0.0), c63 = wasm_f64x2_splat(0.0);
    v128_t c70 = wasm_f64x2_splat(0.0), c71 = wasm_f64x2_splat(0.0);
    v128_t c72 = wasm_f64x2_splat(0.0), c73 = wasm_f64x2_splat(0.0);

    for (int64_t k = 0; k < kc; k++) {
        /* Load 8 rows of A as 4 vectors */
        v128_t a0 = wasm_v128_load(pa);
        v128_t a1 = wasm_v128_load(pa + 2);
        v128_t a2 = wasm_v128_load(pa + 4);
        v128_t a3 = wasm_v128_load(pa + 6);

        /* Broadcast each B element and accumulate */
        v128_t b;

        b = wasm_f64x2_splat(pb[0]);
        c00 = wasm_f64x2_add(c00, wasm_f64x2_mul(a0, b));
        c01 = wasm_f64x2_add(c01, wasm_f64x2_mul(a1, b));
        c02 = wasm_f64x2_add(c02, wasm_f64x2_mul(a2, b));
        c03 = wasm_f64x2_add(c03, wasm_f64x2_mul(a3, b));

        b = wasm_f64x2_splat(pb[1]);
        c10 = wasm_f64x2_add(c10, wasm_f64x2_mul(a0, b));
        c11 = wasm_f64x2_add(c11, wasm_f64x2_mul(a1, b));
        c12 = wasm_f64x2_add(c12, wasm_f64x2_mul(a2, b));
        c13 = wasm_f64x2_add(c13, wasm_f64x2_mul(a3, b));

        b = wasm_f64x2_splat(pb[2]);
        c20 = wasm_f64x2_add(c20, wasm_f64x2_mul(a0, b));
        c21 = wasm_f64x2_add(c21, wasm_f64x2_mul(a1, b));
        c22 = wasm_f64x2_add(c22, wasm_f64x2_mul(a2, b));
        c23 = wasm_f64x2_add(c23, wasm_f64x2_mul(a3, b));

        b = wasm_f64x2_splat(pb[3]);
        c30 = wasm_f64x2_add(c30, wasm_f64x2_mul(a0, b));
        c31 = wasm_f64x2_add(c31, wasm_f64x2_mul(a1, b));
        c32 = wasm_f64x2_add(c32, wasm_f64x2_mul(a2, b));
        c33 = wasm_f64x2_add(c33, wasm_f64x2_mul(a3, b));

        b = wasm_f64x2_splat(pb[4]);
        c40 = wasm_f64x2_add(c40, wasm_f64x2_mul(a0, b));
        c41 = wasm_f64x2_add(c41, wasm_f64x2_mul(a1, b));
        c42 = wasm_f64x2_add(c42, wasm_f64x2_mul(a2, b));
        c43 = wasm_f64x2_add(c43, wasm_f64x2_mul(a3, b));

        b = wasm_f64x2_splat(pb[5]);
        c50 = wasm_f64x2_add(c50, wasm_f64x2_mul(a0, b));
        c51 = wasm_f64x2_add(c51, wasm_f64x2_mul(a1, b));
        c52 = wasm_f64x2_add(c52, wasm_f64x2_mul(a2, b));
        c53 = wasm_f64x2_add(c53, wasm_f64x2_mul(a3, b));

        b = wasm_f64x2_splat(pb[6]);
        c60 = wasm_f64x2_add(c60, wasm_f64x2_mul(a0, b));
        c61 = wasm_f64x2_add(c61, wasm_f64x2_mul(a1, b));
        c62 = wasm_f64x2_add(c62, wasm_f64x2_mul(a2, b));
        c63 = wasm_f64x2_add(c63, wasm_f64x2_mul(a3, b));

        b = wasm_f64x2_splat(pb[7]);
        c70 = wasm_f64x2_add(c70, wasm_f64x2_mul(a0, b));
        c71 = wasm_f64x2_add(c71, wasm_f64x2_mul(a1, b));
        c72 = wasm_f64x2_add(c72, wasm_f64x2_mul(a2, b));
        c73 = wasm_f64x2_add(c73, wasm_f64x2_mul(a3, b));

        pa += MR;
        pb += NR;
    }

    /* Scale by alpha */
    v128_t alpha_v = wasm_f64x2_splat(alpha);

    #define SCALE(c) c = wasm_f64x2_mul(c, alpha_v)
    SCALE(c00); SCALE(c01); SCALE(c02); SCALE(c03);
    SCALE(c10); SCALE(c11); SCALE(c12); SCALE(c13);
    SCALE(c20); SCALE(c21); SCALE(c22); SCALE(c23);
    SCALE(c30); SCALE(c31); SCALE(c32); SCALE(c33);
    SCALE(c40); SCALE(c41); SCALE(c42); SCALE(c43);
    SCALE(c50); SCALE(c51); SCALE(c52); SCALE(c53);
    SCALE(c60); SCALE(c61); SCALE(c62); SCALE(c63);
    SCALE(c70); SCALE(c71); SCALE(c72); SCALE(c73);
    #undef SCALE

    /* Store results - load, add, store for each column */
    #define STORE_COL(col, c0, c1, c2, c3) do { \
        double *p = C + col * ldc; \
        v128_t t0 = wasm_v128_load(p); \
        v128_t t1 = wasm_v128_load(p + 2); \
        v128_t t2 = wasm_v128_load(p + 4); \
        v128_t t3 = wasm_v128_load(p + 6); \
        wasm_v128_store(p, wasm_f64x2_add(t0, c0)); \
        wasm_v128_store(p + 2, wasm_f64x2_add(t1, c1)); \
        wasm_v128_store(p + 4, wasm_f64x2_add(t2, c2)); \
        wasm_v128_store(p + 6, wasm_f64x2_add(t3, c3)); \
    } while(0)

    STORE_COL(0, c00, c01, c02, c03);
    STORE_COL(1, c10, c11, c12, c13);
    STORE_COL(2, c20, c21, c22, c23);
    STORE_COL(3, c30, c31, c32, c33);
    STORE_COL(4, c40, c41, c42, c43);
    STORE_COL(5, c50, c51, c52, c53);
    STORE_COL(6, c60, c61, c62, c63);
    STORE_COL(7, c70, c71, c72, c73);
    #undef STORE_COL
}
#endif

/**
 * 8x8 micro-kernel - scalar fallback
 */
static void micro_kernel_8x8_scalar(int64_t kc, double alpha,
                                     const double *pa, const double *pb,
                                     double *C, int64_t ldc) {
    double c[8][8] = {{0}};

    for (int64_t k = 0; k < kc; k++) {
        for (int i = 0; i < 8; i++) {
            double ai = pa[i];
            for (int j = 0; j < 8; j++) {
                c[j][i] += ai * pb[j];
            }
        }
        pa += MR;
        pb += NR;
    }

    for (int j = 0; j < 8; j++) {
        for (int i = 0; i < 8; i++) {
            C[i + j * ldc] += alpha * c[j][i];
        }
    }
}

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
            micro_kernel_8x8_simd128(kc, alpha,
                                      pa + i * kc,
                                      pb + j * kc,
                                      C + i + j * ldc, ldc);
            #else
            micro_kernel_8x8_scalar(kc, alpha,
                                     pa + i * kc,
                                     pb + j * kc,
                                     C + i + j * ldc, ldc);
            #endif
        }
        if (i < mc) {
            micro_kernel_edge(mc - i, NR, kc, alpha,
                              pa + i * kc,
                              pb + j * kc,
                              C + i + j * ldc, ldc);
        }
    }

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

__attribute__((export_name("dgemm_8x8")))
void dgemm_8x8(int32_t m, int32_t n, int32_t k,
               double alpha, const double *A, int32_t lda,
               const double *B, int32_t ldb,
               double beta, double *C, int32_t ldc) {

    if (m == 0 || n == 0 || k == 0) return;

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

__attribute__((export_name("matmul_f64_8x8")))
void matmul_f64_8x8(int32_t M, int32_t N, int32_t K,
                    const double *A, const double *B, double *C) {
    dgemm_8x8(N, M, K,
              1.0, B, N,
              A, K,
              0.0, C, N);
}

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
