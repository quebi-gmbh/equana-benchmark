/**
 * SSE-Style Matrix Multiplication for WebAssembly
 *
 * Uses x86 SSE2 approach: pre-duplicate B values during packing
 * to eliminate runtime broadcast operations (extract_lane + splat).
 *
 * Key insight from OpenBLAS gemm_kernel_4x4_sse2.S:
 * - Original B packing: [b0, b1, b2, b3] per k iteration
 * - SSE-style packing:  [b0, b0, b1, b1, b2, b2, b3, b3] per k iteration
 *
 * This allows the kernel to load pre-duplicated vectors directly:
 *   v128_t b0 = wasm_v128_load(pb);  // [b0, b0] ready to use!
 * Instead of:
 *   v128_t b = wasm_v128_load(pb);
 *   v128_t b0 = wasm_f64x2_splat(wasm_f64x2_extract_lane(b, 0));  // 2 ops
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <wasm_simd128.h>

/* ============= Blocking Parameters ============= */

#define MC 64     /* Block rows of A */
#define KC 128    /* Block of K dimension */
#define NC 256    /* Block cols of B */

#define MR 4      /* Micro-kernel rows */
#define NR 4      /* Micro-kernel cols */

/* SSE-style: B buffer needs 2x space due to pre-duplication */
#define NR_DUP 8  /* NR * 2 for pre-duplicated B */

/* ============= Memory Buffers ============= */

static double *pack_a = NULL;  /* Packed A panel: MC x KC */
static double *pack_b = NULL;  /* Packed B panel: KC x NC (pre-duplicated) */
static size_t pack_a_size = 0;
static size_t pack_b_size = 0;

static void ensure_buffers(int64_t mc, int64_t kc, int64_t nc) {
    size_t need_a = (mc + MR) * (kc + 4) * sizeof(double);
    /* SSE-style: 2x B buffer for pre-duplication */
    size_t need_b = (kc + 4) * (nc + NR) * 2 * sizeof(double);

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
 * Pack A panel (mc x kc) - same as standard version
 * Layout: [a00 a10 a20 a30] [a01 a11 a21 a31] ...
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
            a_ptr += lda;
            pa += MR;
        }
    }

    /* Pack remaining rows with zero padding */
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
 * Pack B panel with PRE-DUPLICATION (SSE-style)
 *
 * Standard layout:  [b0, b1, b2, b3] per k iteration (4 doubles)
 * SSE-style layout: [b0, b0, b1, b1, b2, b2, b3, b3] per k iteration (8 doubles)
 *
 * This allows the micro-kernel to load [bi, bi] directly as a v128_t
 * without needing extract_lane + splat operations.
 */
__attribute__((noinline))
static void pack_panel_b_sse(int64_t kc, int64_t nc, const double *B, int64_t ldb, double *pb) {
    int64_t i, j;

    /* Pack full NR-column panels with pre-duplication */
    for (j = 0; j + NR <= nc; j += NR) {
        const double *b0 = B + j * ldb;
        const double *b1 = b0 + ldb;
        const double *b2 = b1 + ldb;
        const double *b3 = b2 + ldb;

        for (i = 0; i < kc; i++) {
            /* Pre-duplicate each B value for SIMD loads */
            pb[0] = b0[i];  pb[1] = b0[i];  /* [b0, b0] */
            pb[2] = b1[i];  pb[3] = b1[i];  /* [b1, b1] */
            pb[4] = b2[i];  pb[5] = b2[i];  /* [b2, b2] */
            pb[6] = b3[i];  pb[7] = b3[i];  /* [b3, b3] */
            pb += NR_DUP;  /* 8 doubles per k iteration */
        }
    }

    /* Pack remaining columns with pre-duplication and zero padding */
    if (j < nc) {
        int64_t nr_rem = nc - j;
        for (i = 0; i < kc; i++) {
            for (int64_t jj = 0; jj < nr_rem; jj++) {
                double val = B[(j + jj) * ldb + i];
                pb[jj * 2] = val;
                pb[jj * 2 + 1] = val;
            }
            for (int64_t jj = nr_rem; jj < NR; jj++) {
                pb[jj * 2] = 0.0;
                pb[jj * 2 + 1] = 0.0;
            }
            pb += NR_DUP;
        }
    }
}

/* ============= Micro-Kernels ============= */

/**
 * 4x4 micro-kernel using SSE-style pre-duplicated B
 *
 * Key difference from standard kernel:
 * - B is already duplicated: [b0,b0,b1,b1,b2,b2,b3,b3] per k
 * - We load directly: v128_t b0 = wasm_v128_load(pb); // gets [b0, b0]
 * - No extract_lane or splat needed!
 *
 * Operations per k iteration:
 * - Standard: 2 loads from B + 4 extract_lane + 4 splat + 8 mul + 8 add
 * - SSE-style: 4 loads from B + 8 mul + 8 add (no extract/splat)
 */
static void micro_kernel_4x4_sse(int64_t kc, double alpha,
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
        /* Load A: 4 rows, 1 column */
        v128_t a_lo = wasm_v128_load(pa);      /* [a0, a1] */
        v128_t a_hi = wasm_v128_load(pa + 2);  /* [a2, a3] */

        /* Load B: pre-duplicated, no extract_lane needed! */
        v128_t b0 = wasm_v128_load(pb);      /* [b0, b0] already duplicated */
        v128_t b1 = wasm_v128_load(pb + 2);  /* [b1, b1] already duplicated */
        v128_t b2 = wasm_v128_load(pb + 4);  /* [b2, b2] already duplicated */
        v128_t b3 = wasm_v128_load(pb + 6);  /* [b3, b3] already duplicated */

        /* Column 0: C[*,0] += A[*] * b0 */
        c0_lo = wasm_f64x2_add(c0_lo, wasm_f64x2_mul(a_lo, b0));
        c0_hi = wasm_f64x2_add(c0_hi, wasm_f64x2_mul(a_hi, b0));

        /* Column 1: C[*,1] += A[*] * b1 */
        c1_lo = wasm_f64x2_add(c1_lo, wasm_f64x2_mul(a_lo, b1));
        c1_hi = wasm_f64x2_add(c1_hi, wasm_f64x2_mul(a_hi, b1));

        /* Column 2: C[*,2] += A[*] * b2 */
        c2_lo = wasm_f64x2_add(c2_lo, wasm_f64x2_mul(a_lo, b2));
        c2_hi = wasm_f64x2_add(c2_hi, wasm_f64x2_mul(a_hi, b2));

        /* Column 3: C[*,3] += A[*] * b3 */
        c3_lo = wasm_f64x2_add(c3_lo, wasm_f64x2_mul(a_lo, b3));
        c3_hi = wasm_f64x2_add(c3_hi, wasm_f64x2_mul(a_hi, b3));

        pa += MR;       /* 4 doubles */
        pb += NR_DUP;   /* 8 doubles (pre-duplicated) */
    }

    /* Scale by alpha */
    v128_t alpha_v = wasm_f64x2_splat(alpha);
    c0_lo = wasm_f64x2_mul(c0_lo, alpha_v);
    c0_hi = wasm_f64x2_mul(c0_hi, alpha_v);
    c1_lo = wasm_f64x2_mul(c1_lo, alpha_v);
    c1_hi = wasm_f64x2_mul(c1_hi, alpha_v);
    c2_lo = wasm_f64x2_mul(c2_lo, alpha_v);
    c2_hi = wasm_f64x2_mul(c2_hi, alpha_v);
    c3_lo = wasm_f64x2_mul(c3_lo, alpha_v);
    c3_hi = wasm_f64x2_mul(c3_hi, alpha_v);

    /* Store: C += accumulators */
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

/**
 * Edge micro-kernel for partial tiles (scalar fallback)
 */
static void micro_kernel_edge(int64_t mr, int64_t nr, int64_t kc, double alpha,
                               const double *pa, const double *pb,
                               double *C, int64_t ldc) {
    for (int64_t j = 0; j < nr; j++) {
        for (int64_t i = 0; i < mr; i++) {
            double sum = 0.0;
            for (int64_t k = 0; k < kc; k++) {
                /* Note: pb uses NR_DUP stride with duplicated values */
                sum += pa[k * MR + i] * pb[k * NR_DUP + j * 2];
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
            micro_kernel_4x4_sse(kc, alpha,
                                  pa + i * kc,
                                  pb + j * kc * 2,  /* 2x for pre-duplication */
                                  C + i + j * ldc, ldc);
        }
        /* Handle remaining rows */
        if (i < mc) {
            micro_kernel_edge(mc - i, NR, kc, alpha,
                              pa + i * kc,
                              pb + j * kc * 2,
                              C + i + j * ldc, ldc);
        }
    }

    /* Handle remaining columns */
    if (j < nc) {
        for (i = 0; i < mc; i += MR) {
            int64_t mr = (i + MR <= mc) ? MR : (mc - i);
            micro_kernel_edge(mr, nc - j, kc, alpha,
                              pa + i * kc,
                              pb + j * kc * 2,
                              C + i + j * ldc, ldc);
        }
    }
}

/* ============= DGEMM Interface ============= */

__attribute__((export_name("dgemm_sse")))
void dgemm_sse(int32_t m, int32_t n, int32_t k,
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

            /* Pack B with pre-duplication */
            pack_panel_b_sse(kc, nc, B + pc + jc * ldb, ldb, pack_b);

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

__attribute__((export_name("matmul_sse")))
void matmul_sse(int32_t M, int32_t N, int32_t K,
                const double *A, const double *B, double *C) {
    /* Row-major to column-major conversion:
     * C_row(MxN) = A_row(MxK) * B_row(KxN)
     * becomes: C_col(NxM) = B_col(NxK) * A_col(KxM)
     */
    dgemm_sse(N, M, K,
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
