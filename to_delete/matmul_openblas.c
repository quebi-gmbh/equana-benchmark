/**
 * OpenBLAS-style DGEMM for WebAssembly
 *
 * Uses the dgemm_kernel_4x4_wasm.c micro-kernels with proper packing.
 * This implementation follows the GotoBLAS/OpenBLAS blocking strategy.
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <wasm_simd128.h>

typedef double FLOAT;
typedef long long BLASLONG;

/* ============= Blocking Parameters ============= */
/* Cache blocking sizes - tuned for typical browser WASM environment */
#define MC 64     /* Block rows of A */
#define KC 128    /* Block cols of A / rows of B */
#define NC 512    /* Block cols of B */

/* Micro-kernel register blocking */
#define MR 4      /* Rows per micro-kernel */
#define NR 4      /* Cols per micro-kernel (primary) */
#define NR8 8     /* Cols for 8-wide kernel */

/* ============= Packing Buffers ============= */
static double *pack_a = NULL;
static double *pack_b = NULL;
static size_t pack_a_size = 0;
static size_t pack_b_size = 0;

static void ensure_buffers(BLASLONG mc, BLASLONG kc, BLASLONG nc) {
    size_t need_a = (mc + MR) * (kc + 4) * sizeof(double);
    size_t need_b = (kc + 4) * (nc + NR8) * sizeof(double);

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

/* ============= Include the micro-kernels ============= */

/*===========================================================================
 * Helper: Store 4 rows to 1 column of C
 *===========================================================================*/
static inline void dgemm_store_m4n1(FLOAT *C, v128_t up, v128_t down, FLOAT alpha) {
    v128_t alpha_v = wasm_f64x2_splat(alpha);
    v128_t t1 = wasm_v128_load(C);
    v128_t t2 = wasm_v128_load(C + 2);
    t1 = wasm_f64x2_add(t1, wasm_f64x2_mul(up, alpha_v));
    t2 = wasm_f64x2_add(t2, wasm_f64x2_mul(down, alpha_v));
    wasm_v128_store(C, t1);
    wasm_v128_store(C + 2, t2);
}

/*===========================================================================
 * Helper: Store 1 row to 2 columns of C
 *===========================================================================*/
static inline void dgemm_store_m1n2(FLOAT *C, v128_t vc, FLOAT alpha, BLASLONG LDC) {
    double c0 = wasm_f64x2_extract_lane(vc, 0);
    double c1 = wasm_f64x2_extract_lane(vc, 1);
    C[0] += c0 * alpha;
    C[LDC] += c1 * alpha;
}

/*===========================================================================
 * m4n4: 4 rows x 4 columns kernel (primary workhorse)
 *===========================================================================*/
static inline void dgemm_kernel_wasm_m4n4(
    const FLOAT *sa, const FLOAT *sb, FLOAT *C,
    BLASLONG K, BLASLONG LDC, FLOAT alpha) {

    v128_t c11, c21, c12, c22, c13, c23, c14, c24;
    c11 = c21 = c12 = c22 = c13 = c23 = c14 = c24 = wasm_f64x2_splat(0.0);

    for (BLASLONG k = 0; k < K; k++) {
        v128_t a1 = wasm_v128_load(sa);
        v128_t a2 = wasm_v128_load(sa + 2);
        sa += 4;

        v128_t b1 = wasm_v128_load(sb);
        v128_t b2 = wasm_v128_load(sb + 2);
        sb += 4;

        v128_t b0_splat = wasm_f64x2_splat(wasm_f64x2_extract_lane(b1, 0));
        c11 = wasm_f64x2_add(c11, wasm_f64x2_mul(a1, b0_splat));
        c21 = wasm_f64x2_add(c21, wasm_f64x2_mul(a2, b0_splat));

        v128_t b1_splat = wasm_f64x2_splat(wasm_f64x2_extract_lane(b1, 1));
        c12 = wasm_f64x2_add(c12, wasm_f64x2_mul(a1, b1_splat));
        c22 = wasm_f64x2_add(c22, wasm_f64x2_mul(a2, b1_splat));

        v128_t b2_splat = wasm_f64x2_splat(wasm_f64x2_extract_lane(b2, 0));
        c13 = wasm_f64x2_add(c13, wasm_f64x2_mul(a1, b2_splat));
        c23 = wasm_f64x2_add(c23, wasm_f64x2_mul(a2, b2_splat));

        v128_t b3_splat = wasm_f64x2_splat(wasm_f64x2_extract_lane(b2, 1));
        c14 = wasm_f64x2_add(c14, wasm_f64x2_mul(a1, b3_splat));
        c24 = wasm_f64x2_add(c24, wasm_f64x2_mul(a2, b3_splat));
    }

    dgemm_store_m4n1(C, c11, c21, alpha); C += LDC;
    dgemm_store_m4n1(C, c12, c22, alpha); C += LDC;
    dgemm_store_m4n1(C, c13, c23, alpha); C += LDC;
    dgemm_store_m4n1(C, c14, c24, alpha);
}

/*===========================================================================
 * m4n2: 4 rows x 2 columns kernel
 *===========================================================================*/
static inline void dgemm_kernel_wasm_m4n2(
    const FLOAT *sa, const FLOAT *sb, FLOAT *C,
    BLASLONG K, BLASLONG LDC, FLOAT alpha) {

    v128_t c11, c21, c12, c22;
    c11 = c21 = c12 = c22 = wasm_f64x2_splat(0.0);

    for (BLASLONG k = 0; k < K; k++) {
        v128_t a1 = wasm_v128_load(sa);
        v128_t a2 = wasm_v128_load(sa + 2);
        sa += 4;

        v128_t b1 = wasm_v128_load(sb);
        sb += 2;

        v128_t b0_splat = wasm_f64x2_splat(wasm_f64x2_extract_lane(b1, 0));
        v128_t b1_splat = wasm_f64x2_splat(wasm_f64x2_extract_lane(b1, 1));

        c11 = wasm_f64x2_add(c11, wasm_f64x2_mul(a1, b0_splat));
        c21 = wasm_f64x2_add(c21, wasm_f64x2_mul(a2, b0_splat));
        c12 = wasm_f64x2_add(c12, wasm_f64x2_mul(a1, b1_splat));
        c22 = wasm_f64x2_add(c22, wasm_f64x2_mul(a2, b1_splat));
    }

    dgemm_store_m4n1(C, c11, c21, alpha); C += LDC;
    dgemm_store_m4n1(C, c12, c22, alpha);
}

/*===========================================================================
 * m4n1: 4 rows x 1 column kernel
 *===========================================================================*/
static inline void dgemm_kernel_wasm_m4n1(
    const FLOAT *sa, const FLOAT *sb, FLOAT *C,
    BLASLONG K, BLASLONG LDC, FLOAT alpha) {

    v128_t c11, c21;
    c11 = c21 = wasm_f64x2_splat(0.0);

    for (BLASLONG k = 0; k < K; k++) {
        v128_t a1 = wasm_v128_load(sa);
        v128_t a2 = wasm_v128_load(sa + 2);
        sa += 4;

        v128_t b_splat = wasm_f64x2_splat(*sb++);

        c11 = wasm_f64x2_add(c11, wasm_f64x2_mul(a1, b_splat));
        c21 = wasm_f64x2_add(c21, wasm_f64x2_mul(a2, b_splat));
    }

    dgemm_store_m4n1(C, c11, c21, alpha);
}

/*===========================================================================
 * m2n4: 2 rows x 4 columns kernel
 *===========================================================================*/
static inline void dgemm_kernel_wasm_m2n4(
    const FLOAT *sa, const FLOAT *sb, FLOAT *C,
    BLASLONG K, BLASLONG LDC, FLOAT alpha) {

    v128_t c1, c2, c3, c4;
    c1 = c2 = c3 = c4 = wasm_f64x2_splat(0.0);

    for (BLASLONG k = 0; k < K; k++) {
        v128_t a1 = wasm_v128_load(sa);
        sa += 2;

        v128_t b1 = wasm_v128_load(sb);
        v128_t b2 = wasm_v128_load(sb + 2);
        sb += 4;

        v128_t b0_splat = wasm_f64x2_splat(wasm_f64x2_extract_lane(b1, 0));
        v128_t b1_splat = wasm_f64x2_splat(wasm_f64x2_extract_lane(b1, 1));
        v128_t b2_splat = wasm_f64x2_splat(wasm_f64x2_extract_lane(b2, 0));
        v128_t b3_splat = wasm_f64x2_splat(wasm_f64x2_extract_lane(b2, 1));

        c1 = wasm_f64x2_add(c1, wasm_f64x2_mul(a1, b0_splat));
        c2 = wasm_f64x2_add(c2, wasm_f64x2_mul(a1, b1_splat));
        c3 = wasm_f64x2_add(c3, wasm_f64x2_mul(a1, b2_splat));
        c4 = wasm_f64x2_add(c4, wasm_f64x2_mul(a1, b3_splat));
    }

    v128_t alpha_v = wasm_f64x2_splat(alpha);
    v128_t t;

    t = wasm_v128_load(C);
    wasm_v128_store(C, wasm_f64x2_add(t, wasm_f64x2_mul(c1, alpha_v)));
    C += LDC;
    t = wasm_v128_load(C);
    wasm_v128_store(C, wasm_f64x2_add(t, wasm_f64x2_mul(c2, alpha_v)));
    C += LDC;
    t = wasm_v128_load(C);
    wasm_v128_store(C, wasm_f64x2_add(t, wasm_f64x2_mul(c3, alpha_v)));
    C += LDC;
    t = wasm_v128_load(C);
    wasm_v128_store(C, wasm_f64x2_add(t, wasm_f64x2_mul(c4, alpha_v)));
}

/*===========================================================================
 * m2n2: 2 rows x 2 columns kernel
 *===========================================================================*/
static inline void dgemm_kernel_wasm_m2n2(
    const FLOAT *sa, const FLOAT *sb, FLOAT *C,
    BLASLONG K, BLASLONG LDC, FLOAT alpha) {

    v128_t c1, c2;
    c1 = c2 = wasm_f64x2_splat(0.0);

    for (BLASLONG k = 0; k < K; k++) {
        v128_t a1 = wasm_v128_load(sa);
        sa += 2;

        v128_t b1 = wasm_v128_load(sb);
        sb += 2;

        v128_t b0_splat = wasm_f64x2_splat(wasm_f64x2_extract_lane(b1, 0));
        v128_t b1_splat = wasm_f64x2_splat(wasm_f64x2_extract_lane(b1, 1));

        c1 = wasm_f64x2_add(c1, wasm_f64x2_mul(a1, b0_splat));
        c2 = wasm_f64x2_add(c2, wasm_f64x2_mul(a1, b1_splat));
    }

    v128_t alpha_v = wasm_f64x2_splat(alpha);
    v128_t t;

    t = wasm_v128_load(C);
    wasm_v128_store(C, wasm_f64x2_add(t, wasm_f64x2_mul(c1, alpha_v)));
    C += LDC;
    t = wasm_v128_load(C);
    wasm_v128_store(C, wasm_f64x2_add(t, wasm_f64x2_mul(c2, alpha_v)));
}

/*===========================================================================
 * m2n1: 2 rows x 1 column kernel
 *===========================================================================*/
static inline void dgemm_kernel_wasm_m2n1(
    const FLOAT *sa, const FLOAT *sb, FLOAT *C,
    BLASLONG K, BLASLONG LDC, FLOAT alpha) {

    v128_t c1 = wasm_f64x2_splat(0.0);

    for (BLASLONG k = 0; k < K; k++) {
        v128_t a1 = wasm_v128_load(sa);
        sa += 2;
        v128_t b_splat = wasm_f64x2_splat(*sb++);
        c1 = wasm_f64x2_add(c1, wasm_f64x2_mul(a1, b_splat));
    }

    v128_t alpha_v = wasm_f64x2_splat(alpha);
    v128_t t = wasm_v128_load(C);
    wasm_v128_store(C, wasm_f64x2_add(t, wasm_f64x2_mul(c1, alpha_v)));
}

/*===========================================================================
 * m1n4: 1 row x 4 columns kernel
 *===========================================================================*/
static inline void dgemm_kernel_wasm_m1n4(
    const FLOAT *sa, const FLOAT *sb, FLOAT *C,
    BLASLONG K, BLASLONG LDC, FLOAT alpha) {

    v128_t c1, c2;
    c1 = c2 = wasm_f64x2_splat(0.0);

    for (BLASLONG k = 0; k < K; k++) {
        double a1 = *sa++;
        v128_t a_splat = wasm_f64x2_splat(a1);
        c1 = wasm_f64x2_add(c1, wasm_f64x2_mul(wasm_v128_load(sb), a_splat));
        c2 = wasm_f64x2_add(c2, wasm_f64x2_mul(wasm_v128_load(sb + 2), a_splat));
        sb += 4;
    }

    dgemm_store_m1n2(C, c1, alpha, LDC); C += LDC * 2;
    dgemm_store_m1n2(C, c2, alpha, LDC);
}

/*===========================================================================
 * m1n2: 1 row x 2 columns kernel
 *===========================================================================*/
static inline void dgemm_kernel_wasm_m1n2(
    const FLOAT *sa, const FLOAT *sb, FLOAT *C,
    BLASLONG K, BLASLONG LDC, FLOAT alpha) {

    v128_t c1 = wasm_f64x2_splat(0.0);

    for (BLASLONG k = 0; k < K; k++) {
        v128_t a_splat = wasm_f64x2_splat(*sa++);
        c1 = wasm_f64x2_add(c1, wasm_f64x2_mul(wasm_v128_load(sb), a_splat));
        sb += 2;
    }

    dgemm_store_m1n2(C, c1, alpha, LDC);
}

/*===========================================================================
 * m1n1: 1 row x 1 column kernel (dot product)
 *===========================================================================*/
static inline void dgemm_kernel_wasm_m1n1(
    const FLOAT *sa, const FLOAT *sb, FLOAT *C,
    BLASLONG K, BLASLONG LDC, FLOAT alpha) {

    double sum = 0.0;
    for (BLASLONG k = 0; k < K; k++) {
        sum += (*sa++) * (*sb++);
    }
    C[0] += sum * alpha;
}

/* ============= Packing Functions ============= */

/**
 * Pack A panel (mc x kc) into MR-strided column-major format
 * Input: column-major A with leading dimension lda
 * Output: pack_a in format [a00, a10, a20, a30, a01, a11, a21, a31, ...]
 */
__attribute__((noinline))
static void pack_panel_a_openblas(BLASLONG mc, BLASLONG kc, const double *A, BLASLONG lda, double *pa) {
    BLASLONG i, k;

    /* Pack full MR-row panels */
    for (i = 0; i + MR <= mc; i += MR) {
        const double *a_ptr = A + i;
        for (k = 0; k < kc; k++) {
            pa[0] = a_ptr[0];
            pa[1] = a_ptr[1];
            pa[2] = a_ptr[2];
            pa[3] = a_ptr[3];
            a_ptr += lda;
            pa += MR;
        }
    }

    /* Pack remaining 2 rows */
    if (i + 2 <= mc) {
        const double *a_ptr = A + i;
        for (k = 0; k < kc; k++) {
            pa[0] = a_ptr[0];
            pa[1] = a_ptr[1];
            a_ptr += lda;
            pa += 2;
        }
        i += 2;
    }

    /* Pack remaining 1 row */
    if (i < mc) {
        const double *a_ptr = A + i;
        for (k = 0; k < kc; k++) {
            pa[0] = a_ptr[0];
            a_ptr += lda;
            pa += 1;
        }
    }
}

/**
 * Pack B panel (kc x nc) into NR-strided row-major format
 * Input: column-major B with leading dimension ldb
 * Output: pack_b in format [b00, b01, b02, b03, b10, b11, b12, b13, ...]
 */
__attribute__((noinline))
static void pack_panel_b_openblas(BLASLONG kc, BLASLONG nc, const double *B, BLASLONG ldb, double *pb) {
    BLASLONG j, k;

    /* Pack full NR-column panels */
    for (j = 0; j + NR <= nc; j += NR) {
        const double *b0 = B + j * ldb;
        const double *b1 = b0 + ldb;
        const double *b2 = b1 + ldb;
        const double *b3 = b2 + ldb;

        for (k = 0; k < kc; k++) {
            pb[0] = b0[k];
            pb[1] = b1[k];
            pb[2] = b2[k];
            pb[3] = b3[k];
            pb += NR;
        }
    }

    /* Pack remaining 2 columns */
    if (j + 2 <= nc) {
        const double *b0 = B + j * ldb;
        const double *b1 = b0 + ldb;
        for (k = 0; k < kc; k++) {
            pb[0] = b0[k];
            pb[1] = b1[k];
            pb += 2;
        }
        j += 2;
    }

    /* Pack remaining 1 column */
    if (j < nc) {
        const double *b0 = B + j * ldb;
        for (k = 0; k < kc; k++) {
            pb[0] = b0[k];
            pb += 1;
        }
    }
}

/* ============= Macro Kernel ============= */

/**
 * Macro kernel: multiply packed panels using micro-kernels
 * C[0:mc, 0:nc] += alpha * pack_a[0:mc, 0:kc] * pack_b[0:kc, 0:nc]
 */
__attribute__((noinline))
static void macro_kernel_openblas(BLASLONG mc, BLASLONG nc, BLASLONG kc, double alpha,
                                   const double *pa, const double *pb,
                                   double *C, BLASLONG ldc) {
    BLASLONG i, j;

    /* Process N in blocks of 4 */
    for (j = 0; j + NR <= nc; j += NR) {
        const double *pb_j = pb + j * kc;

        /* Process M in blocks of 4 */
        for (i = 0; i + MR <= mc; i += MR) {
            dgemm_kernel_wasm_m4n4(pa + i * kc, pb_j, C + i + j * ldc, kc, ldc, alpha);
        }
        /* Remaining 2 rows */
        if (i + 2 <= mc) {
            dgemm_kernel_wasm_m2n4(pa + i * kc, pb_j, C + i + j * ldc, kc, ldc, alpha);
            i += 2;
        }
        /* Remaining 1 row */
        if (i < mc) {
            dgemm_kernel_wasm_m1n4(pa + i * kc, pb_j, C + i + j * ldc, kc, ldc, alpha);
        }
    }

    /* Remaining 2 columns */
    if (j + 2 <= nc) {
        const double *pb_j = pb + j * kc;

        for (i = 0; i + MR <= mc; i += MR) {
            dgemm_kernel_wasm_m4n2(pa + i * kc, pb_j, C + i + j * ldc, kc, ldc, alpha);
        }
        if (i + 2 <= mc) {
            dgemm_kernel_wasm_m2n2(pa + i * kc, pb_j, C + i + j * ldc, kc, ldc, alpha);
            i += 2;
        }
        if (i < mc) {
            dgemm_kernel_wasm_m1n2(pa + i * kc, pb_j, C + i + j * ldc, kc, ldc, alpha);
        }
        j += 2;
    }

    /* Remaining 1 column */
    if (j < nc) {
        const double *pb_j = pb + j * kc;

        for (i = 0; i + MR <= mc; i += MR) {
            dgemm_kernel_wasm_m4n1(pa + i * kc, pb_j, C + i + j * ldc, kc, ldc, alpha);
        }
        if (i + 2 <= mc) {
            dgemm_kernel_wasm_m2n1(pa + i * kc, pb_j, C + i + j * ldc, kc, ldc, alpha);
            i += 2;
        }
        if (i < mc) {
            dgemm_kernel_wasm_m1n1(pa + i * kc, pb_j, C + i + j * ldc, kc, ldc, alpha);
        }
    }
}

/* ============= DGEMM Interface ============= */

/**
 * DGEMM: C = alpha * A * B + beta * C
 * Column-major layout (Fortran/BLAS convention)
 */
__attribute__((export_name("dgemm_openblas")))
void dgemm_openblas(int32_t m, int32_t n, int32_t k,
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

    /* 3-level blocking loops (GotoBLAS algorithm) */
    for (BLASLONG jc = 0; jc < n; jc += NC) {
        BLASLONG nc = (jc + NC <= n) ? NC : (n - jc);

        for (BLASLONG pc = 0; pc < k; pc += KC) {
            BLASLONG kc = (pc + KC <= k) ? KC : (k - pc);

            /* Pack B panel: B[pc:pc+kc, jc:jc+nc] */
            pack_panel_b_openblas(kc, nc, B + pc + jc * ldb, ldb, pack_b);

            for (BLASLONG ic = 0; ic < m; ic += MC) {
                BLASLONG mc = (ic + MC <= m) ? MC : (m - ic);

                /* Pack A panel: A[ic:ic+mc, pc:pc+kc] */
                pack_panel_a_openblas(mc, kc, A + ic + pc * lda, lda, pack_a);

                /* Compute C[ic:ic+mc, jc:jc+nc] += alpha * pack_a * pack_b */
                macro_kernel_openblas(mc, nc, kc, alpha, pack_a, pack_b,
                                      C + ic + jc * ldc, ldc);
            }
        }
    }
}

/* ============= Row-Major Matmul Interface ============= */

/**
 * Matrix multiplication with row-major input/output
 * C = A * B where A is MxK, B is KxN, C is MxN
 */
__attribute__((export_name("matmul_openblas")))
void matmul_openblas(int32_t M, int32_t N, int32_t K,
                     const double *A, const double *B, double *C) {
    /* Row-major to column-major conversion:
     * C_row(MxN) = A_row(MxK) * B_row(KxN)
     * becomes: C_col(NxM) = B_col(NxK) * A_col(KxM)
     */
    dgemm_openblas(N, M, K,
                   1.0, B, N,    /* B treated as NxK column-major */
                   A, K,         /* A treated as KxM column-major */
                   0.0, C, N);   /* C is NxM column-major = MxN row-major */
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
