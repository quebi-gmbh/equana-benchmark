/**
 * OpenBLAS-style Level 3 BLAS Driver for WebAssembly
 *
 * This implements the GotoBLAS/OpenBLAS cache-blocking algorithm:
 * - 3-level blocking (GEMM_P × GEMM_Q for L2, GEMM_R for L3)
 * - Panel packing (ICOPY for A, OCOPY for B)
 * - Micro-kernel dispatch
 *
 * Operation: C = alpha * op(A) * op(B) + beta * C
 *
 * Based on OpenBLAS driver/level3/level3.c
 * Simplified for single-threaded WebAssembly with SIMD128
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef __wasm_simd128__
#include <wasm_simd128.h>
#endif

/*===========================================================================
 * Type Definitions
 *===========================================================================*/

typedef double FLOAT;
typedef int64_t BLASLONG;

/*===========================================================================
 * Blocking Parameters
 *
 * Tuned for typical browser/WASM environment:
 * - L1 cache: ~32KB (KC × MR × 8 bytes should fit)
 * - L2 cache: ~256KB (GEMM_P × GEMM_Q × 8 bytes)
 * - L3/memory: GEMM_R controls outer blocking
 *===========================================================================*/

#define GEMM_P      64      /* Block rows of A (M dimension) */
#define GEMM_Q      128     /* Block of K dimension */
#define GEMM_R      512     /* Block cols of B (N dimension) */

#define GEMM_UNROLL_M   4   /* Micro-kernel M tile */
#define GEMM_UNROLL_N   4   /* Micro-kernel N tile */

/*===========================================================================
 * BLAS Argument Structure
 *===========================================================================*/

typedef struct {
    BLASLONG m, n, k;
    BLASLONG lda, ldb, ldc;
    FLOAT alpha;
    FLOAT beta;
    const FLOAT *a;
    const FLOAT *b;
    FLOAT *c;
} blas_arg_t;

/*===========================================================================
 * Work Buffers
 *===========================================================================*/

static FLOAT *sa_buffer = NULL;     /* Packed A panel */
static FLOAT *sb_buffer = NULL;     /* Packed B panel */
static size_t sa_size = 0;
static size_t sb_size = 0;

static void ensure_buffers(BLASLONG p, BLASLONG q, BLASLONG r) {
    size_t need_sa = ((size_t)p + GEMM_UNROLL_M) * ((size_t)q + 4) * sizeof(FLOAT);
    size_t need_sb = ((size_t)q + 4) * ((size_t)r + GEMM_UNROLL_N) * sizeof(FLOAT);

    if (sa_size < need_sa) {
        free(sa_buffer);
        sa_buffer = (FLOAT *)malloc(need_sa);
        sa_size = need_sa;
    }
    if (sb_size < need_sb) {
        free(sb_buffer);
        sb_buffer = (FLOAT *)malloc(need_sb);
        sb_size = need_sb;
    }
}

/*===========================================================================
 * External Kernel Declaration
 * (Implemented in dgemm_kernel_4x4_wasm.c)
 *===========================================================================*/

extern int dgemm_kernel(BLASLONG M, BLASLONG N, BLASLONG K, FLOAT alpha,
                        FLOAT *sa, FLOAT *sb, FLOAT *C, BLASLONG LDC);

/*===========================================================================
 * BETA Operation: C = beta * C
 *===========================================================================*/

static void gemm_beta(BLASLONG m, BLASLONG n, FLOAT beta, FLOAT *c, BLASLONG ldc) {
    BLASLONG i, j;

    if (beta == 0.0) {
        for (j = 0; j < n; j++) {
            for (i = 0; i < m; i++) {
                c[i + j * ldc] = 0.0;
            }
        }
    } else if (beta != 1.0) {
        for (j = 0; j < n; j++) {
            for (i = 0; i < m; i++) {
                c[i + j * ldc] *= beta;
            }
        }
    }
    /* beta == 1.0: do nothing */
}

/*===========================================================================
 * ICOPY Operation: Pack A panel (column-major input → packed format)
 *
 * Packs min_l columns × min_i rows of A into sa buffer
 * A is at position (is, ls) in the original matrix
 *
 * Packed format: MR rows interleaved
 * [a00 a10 a20 a30 | a01 a11 a21 a31 | ...]
 *===========================================================================*/

static void gemm_incopy(BLASLONG min_l, BLASLONG min_i,
                        const FLOAT *a, BLASLONG lda,
                        BLASLONG ls, BLASLONG is, FLOAT *sa) {
    BLASLONG i, j;
    const FLOAT *a_offset;
    FLOAT *b_offset = sa;

    /* Adjust a to start position */
    a_offset = a + is + ls * lda;

    /* Pack full MR-row panels */
    for (i = 0; i + GEMM_UNROLL_M <= min_i; i += GEMM_UNROLL_M) {
        const FLOAT *a_ptr = a_offset + i;

        for (j = 0; j < min_l; j++) {
            b_offset[0] = a_ptr[0];
            b_offset[1] = a_ptr[1];
            b_offset[2] = a_ptr[2];
            b_offset[3] = a_ptr[3];
            a_ptr += lda;
            b_offset += GEMM_UNROLL_M;
        }
    }

    /* Pack remaining rows (< MR) with zero padding */
    if (i < min_i) {
        BLASLONG mr_rem = min_i - i;
        const FLOAT *a_ptr = a_offset + i;

        for (j = 0; j < min_l; j++) {
            BLASLONG ii;
            for (ii = 0; ii < mr_rem; ii++) {
                b_offset[ii] = a_ptr[ii];
            }
            for (ii = mr_rem; ii < GEMM_UNROLL_M; ii++) {
                b_offset[ii] = 0.0;
            }
            a_ptr += lda;
            b_offset += GEMM_UNROLL_M;
        }
    }
}

/*===========================================================================
 * ICOPY for Transposed A: Pack A^T panel
 *
 * A is stored as K×M (row-major when viewed as M×K)
 * We read rows of A and pack as columns
 *===========================================================================*/

static void gemm_itcopy(BLASLONG min_l, BLASLONG min_i,
                        const FLOAT *a, BLASLONG lda,
                        BLASLONG ls, BLASLONG is, FLOAT *sa) {
    BLASLONG i, j;
    FLOAT *b_offset = sa;

    /* Pack full MR-row panels */
    for (i = 0; i + GEMM_UNROLL_M <= min_i; i += GEMM_UNROLL_M) {
        const FLOAT *a_ptr0 = a + (is + i) * lda + ls;
        const FLOAT *a_ptr1 = a + (is + i + 1) * lda + ls;
        const FLOAT *a_ptr2 = a + (is + i + 2) * lda + ls;
        const FLOAT *a_ptr3 = a + (is + i + 3) * lda + ls;

        for (j = 0; j < min_l; j++) {
            b_offset[0] = a_ptr0[j];
            b_offset[1] = a_ptr1[j];
            b_offset[2] = a_ptr2[j];
            b_offset[3] = a_ptr3[j];
            b_offset += GEMM_UNROLL_M;
        }
    }

    /* Pack remaining rows with zero padding */
    if (i < min_i) {
        BLASLONG mr_rem = min_i - i;

        for (j = 0; j < min_l; j++) {
            BLASLONG ii;
            for (ii = 0; ii < mr_rem; ii++) {
                b_offset[ii] = a[(is + i + ii) * lda + ls + j];
            }
            for (ii = mr_rem; ii < GEMM_UNROLL_M; ii++) {
                b_offset[ii] = 0.0;
            }
            b_offset += GEMM_UNROLL_M;
        }
    }
}

/*===========================================================================
 * OCOPY Operation: Pack B panel (column-major input → packed format)
 *
 * Packs min_l rows × min_j columns of B into sb buffer
 * B is at position (ls, js) in the original matrix
 *
 * Packed format: NR columns interleaved
 * [b00 b01 b02 b03 | b10 b11 b12 b13 | ...]
 *===========================================================================*/

static void gemm_oncopy(BLASLONG min_l, BLASLONG min_j,
                        const FLOAT *b, BLASLONG ldb,
                        BLASLONG ls, BLASLONG js, FLOAT *sb) {
    BLASLONG i, j;
    FLOAT *d_offset = sb;

    /* Pack full NR-column panels */
    for (j = 0; j + GEMM_UNROLL_N <= min_j; j += GEMM_UNROLL_N) {
        const FLOAT *b_ptr0 = b + ls + (js + j) * ldb;
        const FLOAT *b_ptr1 = b + ls + (js + j + 1) * ldb;
        const FLOAT *b_ptr2 = b + ls + (js + j + 2) * ldb;
        const FLOAT *b_ptr3 = b + ls + (js + j + 3) * ldb;

        for (i = 0; i < min_l; i++) {
            d_offset[0] = b_ptr0[i];
            d_offset[1] = b_ptr1[i];
            d_offset[2] = b_ptr2[i];
            d_offset[3] = b_ptr3[i];
            d_offset += GEMM_UNROLL_N;
        }
    }

    /* Pack remaining columns with zero padding */
    if (j < min_j) {
        BLASLONG nr_rem = min_j - j;

        for (i = 0; i < min_l; i++) {
            BLASLONG jj;
            for (jj = 0; jj < nr_rem; jj++) {
                d_offset[jj] = b[ls + i + (js + j + jj) * ldb];
            }
            for (jj = nr_rem; jj < GEMM_UNROLL_N; jj++) {
                d_offset[jj] = 0.0;
            }
            d_offset += GEMM_UNROLL_N;
        }
    }
}

/*===========================================================================
 * OCOPY for Transposed B: Pack B^T panel
 *
 * B is stored as N×K (row-major when viewed as K×N)
 *===========================================================================*/

static void gemm_otcopy(BLASLONG min_l, BLASLONG min_j,
                        const FLOAT *b, BLASLONG ldb,
                        BLASLONG ls, BLASLONG js, FLOAT *sb) {
    BLASLONG i, j;
    FLOAT *d_offset = sb;

    /* Pack full NR-column panels */
    for (j = 0; j + GEMM_UNROLL_N <= min_j; j += GEMM_UNROLL_N) {
        const FLOAT *b_ptr = b + (js + j) + ls * ldb;

        for (i = 0; i < min_l; i++) {
            d_offset[0] = b_ptr[0];
            d_offset[1] = b_ptr[1];
            d_offset[2] = b_ptr[2];
            d_offset[3] = b_ptr[3];
            b_ptr += ldb;
            d_offset += GEMM_UNROLL_N;
        }
    }

    /* Pack remaining columns with zero padding */
    if (j < min_j) {
        BLASLONG nr_rem = min_j - j;
        const FLOAT *b_ptr = b + (js + j) + ls * ldb;

        for (i = 0; i < min_l; i++) {
            BLASLONG jj;
            for (jj = 0; jj < nr_rem; jj++) {
                d_offset[jj] = b_ptr[jj];
            }
            for (jj = nr_rem; jj < GEMM_UNROLL_N; jj++) {
                d_offset[jj] = 0.0;
            }
            b_ptr += ldb;
            d_offset += GEMM_UNROLL_N;
        }
    }
}

/*===========================================================================
 * Main GEMM Driver (Single-Threaded)
 *
 * Implements GotoBLAS 3-level blocking:
 *
 * for js = 0 to N by GEMM_R:           // L3 blocking (N)
 *   for ls = 0 to K by GEMM_Q:         // L2 blocking (K)
 *     for is = 0 to M by GEMM_P:       // L2 blocking (M)
 *       ICOPY: pack A[is:is+P, ls:ls+Q] → sa
 *       OCOPY: pack B[ls:ls+Q, js:js+R] → sb
 *       KERNEL: C[is:is+P, js:js+R] += alpha * sa * sb
 *===========================================================================*/

static int gemm_nn(blas_arg_t *args, BLASLONG *range_m, BLASLONG *range_n,
                   FLOAT *sa, FLOAT *sb) {
    BLASLONG m_from, m_to, n_from, n_to;
    BLASLONG k = args->k;
    BLASLONG lda = args->lda;
    BLASLONG ldb = args->ldb;
    BLASLONG ldc = args->ldc;
    FLOAT alpha = args->alpha;
    FLOAT beta = args->beta;
    const FLOAT *a = args->a;
    const FLOAT *b = args->b;
    FLOAT *c = args->c;

    BLASLONG ls, is, js;
    BLASLONG min_l, min_i, min_j;
    BLASLONG jjs, min_jj;

    /* Determine working range */
    m_from = 0;
    m_to = args->m;
    if (range_m) {
        m_from = range_m[0];
        m_to = range_m[1];
    }

    n_from = 0;
    n_to = args->n;
    if (range_n) {
        n_from = range_n[0];
        n_to = range_n[1];
    }

    /* Apply beta to C */
    if (beta != 1.0) {
        gemm_beta(m_to - m_from, n_to - n_from, beta,
                  c + m_from + n_from * ldc, ldc);
    }

    /* Early exit */
    if (k == 0 || alpha == 0.0) return 0;

    /* Main 3-level blocking loop */
    for (js = n_from; js < n_to; js += GEMM_R) {
        min_j = n_to - js;
        if (min_j > GEMM_R) min_j = GEMM_R;

        for (ls = 0; ls < k; ls += min_l) {
            min_l = k - ls;
            if (min_l >= GEMM_Q * 2) {
                min_l = GEMM_Q;
            } else if (min_l > GEMM_Q) {
                min_l = ((min_l / 2 + GEMM_UNROLL_M - 1) / GEMM_UNROLL_M) * GEMM_UNROLL_M;
            }

            /* First M block - special handling */
            min_i = m_to - m_from;
            if (min_i >= GEMM_P * 2) {
                min_i = GEMM_P;
            } else if (min_i > GEMM_P) {
                min_i = ((min_i / 2 + GEMM_UNROLL_M - 1) / GEMM_UNROLL_M) * GEMM_UNROLL_M;
            }

            /* Pack first A panel */
            gemm_incopy(min_l, min_i, a, lda, ls, m_from, sa);

            /* Process B in chunks of GEMM_UNROLL_N * 3 for better cache use */
            for (jjs = js; jjs < js + min_j; jjs += min_jj) {
                min_jj = min_j + js - jjs;
                if (min_jj >= 3 * GEMM_UNROLL_N) {
                    min_jj = 3 * GEMM_UNROLL_N;
                } else if (min_jj > GEMM_UNROLL_N) {
                    min_jj = GEMM_UNROLL_N;
                }

                /* Pack B panel */
                gemm_oncopy(min_l, min_jj, b, ldb, ls, jjs,
                            sb + min_l * (jjs - js));

                /* Call micro-kernel */
                dgemm_kernel(min_i, min_jj, min_l, alpha,
                             sa, sb + min_l * (jjs - js),
                             c + m_from + jjs * ldc, ldc);
            }

            /* Remaining M blocks */
            for (is = m_from + min_i; is < m_to; is += min_i) {
                min_i = m_to - is;
                if (min_i >= GEMM_P * 2) {
                    min_i = GEMM_P;
                } else if (min_i > GEMM_P) {
                    min_i = ((min_i / 2 + GEMM_UNROLL_M - 1) / GEMM_UNROLL_M) * GEMM_UNROLL_M;
                }

                /* Pack A panel */
                gemm_incopy(min_l, min_i, a, lda, ls, is, sa);

                /* Call micro-kernel (B is already packed) */
                dgemm_kernel(min_i, min_j, min_l, alpha,
                             sa, sb, c + is + js * ldc, ldc);
            }
        }
    }

    return 0;
}

/*===========================================================================
 * Transposed Variants
 *===========================================================================*/

/* C = alpha * A^T * B + beta * C */
static int gemm_tn(blas_arg_t *args, BLASLONG *range_m, BLASLONG *range_n,
                   FLOAT *sa, FLOAT *sb) {
    BLASLONG m_from, m_to, n_from, n_to;
    BLASLONG k = args->k;
    BLASLONG lda = args->lda;
    BLASLONG ldb = args->ldb;
    BLASLONG ldc = args->ldc;
    FLOAT alpha = args->alpha;
    FLOAT beta = args->beta;
    const FLOAT *a = args->a;
    const FLOAT *b = args->b;
    FLOAT *c = args->c;

    BLASLONG ls, is, js;
    BLASLONG min_l, min_i, min_j;
    BLASLONG jjs, min_jj;

    m_from = range_m ? range_m[0] : 0;
    m_to = range_m ? range_m[1] : args->m;
    n_from = range_n ? range_n[0] : 0;
    n_to = range_n ? range_n[1] : args->n;

    if (beta != 1.0) {
        gemm_beta(m_to - m_from, n_to - n_from, beta,
                  c + m_from + n_from * ldc, ldc);
    }

    if (k == 0 || alpha == 0.0) return 0;

    for (js = n_from; js < n_to; js += GEMM_R) {
        min_j = n_to - js;
        if (min_j > GEMM_R) min_j = GEMM_R;

        for (ls = 0; ls < k; ls += min_l) {
            min_l = k - ls;
            if (min_l >= GEMM_Q * 2) min_l = GEMM_Q;
            else if (min_l > GEMM_Q) {
                min_l = ((min_l / 2 + GEMM_UNROLL_M - 1) / GEMM_UNROLL_M) * GEMM_UNROLL_M;
            }

            min_i = m_to - m_from;
            if (min_i >= GEMM_P * 2) min_i = GEMM_P;
            else if (min_i > GEMM_P) {
                min_i = ((min_i / 2 + GEMM_UNROLL_M - 1) / GEMM_UNROLL_M) * GEMM_UNROLL_M;
            }

            /* Pack A^T */
            gemm_itcopy(min_l, min_i, a, lda, ls, m_from, sa);

            for (jjs = js; jjs < js + min_j; jjs += min_jj) {
                min_jj = min_j + js - jjs;
                if (min_jj >= 3 * GEMM_UNROLL_N) min_jj = 3 * GEMM_UNROLL_N;
                else if (min_jj > GEMM_UNROLL_N) min_jj = GEMM_UNROLL_N;

                gemm_oncopy(min_l, min_jj, b, ldb, ls, jjs,
                            sb + min_l * (jjs - js));

                dgemm_kernel(min_i, min_jj, min_l, alpha,
                             sa, sb + min_l * (jjs - js),
                             c + m_from + jjs * ldc, ldc);
            }

            for (is = m_from + min_i; is < m_to; is += min_i) {
                min_i = m_to - is;
                if (min_i >= GEMM_P * 2) min_i = GEMM_P;
                else if (min_i > GEMM_P) {
                    min_i = ((min_i / 2 + GEMM_UNROLL_M - 1) / GEMM_UNROLL_M) * GEMM_UNROLL_M;
                }

                gemm_itcopy(min_l, min_i, a, lda, ls, is, sa);
                dgemm_kernel(min_i, min_j, min_l, alpha,
                             sa, sb, c + is + js * ldc, ldc);
            }
        }
    }

    return 0;
}

/* C = alpha * A * B^T + beta * C */
static int gemm_nt(blas_arg_t *args, BLASLONG *range_m, BLASLONG *range_n,
                   FLOAT *sa, FLOAT *sb) {
    BLASLONG m_from, m_to, n_from, n_to;
    BLASLONG k = args->k;
    BLASLONG lda = args->lda;
    BLASLONG ldb = args->ldb;
    BLASLONG ldc = args->ldc;
    FLOAT alpha = args->alpha;
    FLOAT beta = args->beta;
    const FLOAT *a = args->a;
    const FLOAT *b = args->b;
    FLOAT *c = args->c;

    BLASLONG ls, is, js;
    BLASLONG min_l, min_i, min_j;
    BLASLONG jjs, min_jj;

    m_from = range_m ? range_m[0] : 0;
    m_to = range_m ? range_m[1] : args->m;
    n_from = range_n ? range_n[0] : 0;
    n_to = range_n ? range_n[1] : args->n;

    if (beta != 1.0) {
        gemm_beta(m_to - m_from, n_to - n_from, beta,
                  c + m_from + n_from * ldc, ldc);
    }

    if (k == 0 || alpha == 0.0) return 0;

    for (js = n_from; js < n_to; js += GEMM_R) {
        min_j = n_to - js;
        if (min_j > GEMM_R) min_j = GEMM_R;

        for (ls = 0; ls < k; ls += min_l) {
            min_l = k - ls;
            if (min_l >= GEMM_Q * 2) min_l = GEMM_Q;
            else if (min_l > GEMM_Q) {
                min_l = ((min_l / 2 + GEMM_UNROLL_M - 1) / GEMM_UNROLL_M) * GEMM_UNROLL_M;
            }

            min_i = m_to - m_from;
            if (min_i >= GEMM_P * 2) min_i = GEMM_P;
            else if (min_i > GEMM_P) {
                min_i = ((min_i / 2 + GEMM_UNROLL_M - 1) / GEMM_UNROLL_M) * GEMM_UNROLL_M;
            }

            gemm_incopy(min_l, min_i, a, lda, ls, m_from, sa);

            for (jjs = js; jjs < js + min_j; jjs += min_jj) {
                min_jj = min_j + js - jjs;
                if (min_jj >= 3 * GEMM_UNROLL_N) min_jj = 3 * GEMM_UNROLL_N;
                else if (min_jj > GEMM_UNROLL_N) min_jj = GEMM_UNROLL_N;

                /* Pack B^T */
                gemm_otcopy(min_l, min_jj, b, ldb, ls, jjs,
                            sb + min_l * (jjs - js));

                dgemm_kernel(min_i, min_jj, min_l, alpha,
                             sa, sb + min_l * (jjs - js),
                             c + m_from + jjs * ldc, ldc);
            }

            for (is = m_from + min_i; is < m_to; is += min_i) {
                min_i = m_to - is;
                if (min_i >= GEMM_P * 2) min_i = GEMM_P;
                else if (min_i > GEMM_P) {
                    min_i = ((min_i / 2 + GEMM_UNROLL_M - 1) / GEMM_UNROLL_M) * GEMM_UNROLL_M;
                }

                gemm_incopy(min_l, min_i, a, lda, ls, is, sa);
                dgemm_kernel(min_i, min_j, min_l, alpha,
                             sa, sb, c + is + js * ldc, ldc);
            }
        }
    }

    return 0;
}

/* C = alpha * A^T * B^T + beta * C */
static int gemm_tt(blas_arg_t *args, BLASLONG *range_m, BLASLONG *range_n,
                   FLOAT *sa, FLOAT *sb) {
    BLASLONG m_from, m_to, n_from, n_to;
    BLASLONG k = args->k;
    BLASLONG lda = args->lda;
    BLASLONG ldb = args->ldb;
    BLASLONG ldc = args->ldc;
    FLOAT alpha = args->alpha;
    FLOAT beta = args->beta;
    const FLOAT *a = args->a;
    const FLOAT *b = args->b;
    FLOAT *c = args->c;

    BLASLONG ls, is, js;
    BLASLONG min_l, min_i, min_j;
    BLASLONG jjs, min_jj;

    m_from = range_m ? range_m[0] : 0;
    m_to = range_m ? range_m[1] : args->m;
    n_from = range_n ? range_n[0] : 0;
    n_to = range_n ? range_n[1] : args->n;

    if (beta != 1.0) {
        gemm_beta(m_to - m_from, n_to - n_from, beta,
                  c + m_from + n_from * ldc, ldc);
    }

    if (k == 0 || alpha == 0.0) return 0;

    for (js = n_from; js < n_to; js += GEMM_R) {
        min_j = n_to - js;
        if (min_j > GEMM_R) min_j = GEMM_R;

        for (ls = 0; ls < k; ls += min_l) {
            min_l = k - ls;
            if (min_l >= GEMM_Q * 2) min_l = GEMM_Q;
            else if (min_l > GEMM_Q) {
                min_l = ((min_l / 2 + GEMM_UNROLL_M - 1) / GEMM_UNROLL_M) * GEMM_UNROLL_M;
            }

            min_i = m_to - m_from;
            if (min_i >= GEMM_P * 2) min_i = GEMM_P;
            else if (min_i > GEMM_P) {
                min_i = ((min_i / 2 + GEMM_UNROLL_M - 1) / GEMM_UNROLL_M) * GEMM_UNROLL_M;
            }

            gemm_itcopy(min_l, min_i, a, lda, ls, m_from, sa);

            for (jjs = js; jjs < js + min_j; jjs += min_jj) {
                min_jj = min_j + js - jjs;
                if (min_jj >= 3 * GEMM_UNROLL_N) min_jj = 3 * GEMM_UNROLL_N;
                else if (min_jj > GEMM_UNROLL_N) min_jj = GEMM_UNROLL_N;

                gemm_otcopy(min_l, min_jj, b, ldb, ls, jjs,
                            sb + min_l * (jjs - js));

                dgemm_kernel(min_i, min_jj, min_l, alpha,
                             sa, sb + min_l * (jjs - js),
                             c + m_from + jjs * ldc, ldc);
            }

            for (is = m_from + min_i; is < m_to; is += min_i) {
                min_i = m_to - is;
                if (min_i >= GEMM_P * 2) min_i = GEMM_P;
                else if (min_i > GEMM_P) {
                    min_i = ((min_i / 2 + GEMM_UNROLL_M - 1) / GEMM_UNROLL_M) * GEMM_UNROLL_M;
                }

                gemm_itcopy(min_l, min_i, a, lda, ls, is, sa);
                dgemm_kernel(min_i, min_j, min_l, alpha,
                             sa, sb, c + is + js * ldc, ldc);
            }
        }
    }

    return 0;
}

/*===========================================================================
 * Public DGEMM Interface
 *
 * C = alpha * op(A) * op(B) + beta * C
 *
 * transA: 'N' = no transpose, 'T' = transpose
 * transB: 'N' = no transpose, 'T' = transpose
 *
 * All matrices are column-major (Fortran/BLAS convention)
 *===========================================================================*/

__attribute__((export_name("dgemm")))
void dgemm(char transA, char transB,
           int32_t m, int32_t n, int32_t k,
           double alpha, const double *A, int32_t lda,
           const double *B, int32_t ldb,
           double beta, double *C, int32_t ldc) {

    if (m <= 0 || n <= 0) return;

    /* Ensure work buffers */
    ensure_buffers(GEMM_P, GEMM_Q, GEMM_R);

    /* Set up arguments */
    blas_arg_t args;
    args.m = m;
    args.n = n;
    args.k = k;
    args.alpha = alpha;
    args.beta = beta;
    args.a = A;
    args.b = B;
    args.c = C;
    args.lda = lda;
    args.ldb = ldb;
    args.ldc = ldc;

    int trans_a = (transA == 'T' || transA == 't' || transA == 'C' || transA == 'c');
    int trans_b = (transB == 'T' || transB == 't' || transB == 'C' || transB == 'c');

    /* Dispatch based on transpose flags */
    if (!trans_a && !trans_b) {
        gemm_nn(&args, NULL, NULL, sa_buffer, sb_buffer);
    } else if (trans_a && !trans_b) {
        gemm_tn(&args, NULL, NULL, sa_buffer, sb_buffer);
    } else if (!trans_a && trans_b) {
        gemm_nt(&args, NULL, NULL, sa_buffer, sb_buffer);
    } else {
        gemm_tt(&args, NULL, NULL, sa_buffer, sb_buffer);
    }
}

/*===========================================================================
 * Row-Major Matrix Multiply Interface
 *
 * C = A * B where A is MxK, B is KxN, C is MxN (all row-major)
 *
 * Converts to column-major and calls dgemm:
 * Row-major C = A * B  <==>  Column-major C^T = B^T * A^T
 *===========================================================================*/

__attribute__((export_name("matmul_f64")))
void matmul_f64(int32_t M, int32_t N, int32_t K,
                const double *A, const double *B, double *C) {
    /* Row-major A(MxK) * B(KxN) = C(MxN)
     * is equivalent to:
     * Column-major B^T(NxK) * A^T(KxM) = C^T(NxM)
     *
     * Treating row-major as column-major with swapped dimensions:
     * A_row(MxK) viewed as A_col^T with lda=K
     * B_row(KxN) viewed as B_col^T with ldb=N
     *
     * So we call: dgemm('N', 'N', N, M, K, 1, B, N, A, K, 0, C, N)
     */
    dgemm('N', 'N', N, M, K,
          1.0, B, N,
          A, K,
          0.0, C, N);
}

/*===========================================================================
 * SSE-Style DGEMM Implementation (Pre-Duplicated B Packing)
 *
 * Key optimization: B values are pre-duplicated during packing so the
 * micro-kernel can load [bi, bi] directly without extract_lane + splat.
 *
 * Standard B packing:  [b0, b1, b2, b3] per k iteration (4 doubles)
 * SSE-style B packing: [b0, b0, b1, b1, b2, b2, b3, b3] per k iteration (8 doubles)
 *===========================================================================*/

/*===========================================================================
 * SSE Work Buffers (2x size for B due to pre-duplication)
 *===========================================================================*/

static FLOAT *sa_buffer_sse = NULL;
static FLOAT *sb_buffer_sse = NULL;
static size_t sa_size_sse = 0;
static size_t sb_size_sse = 0;

static void ensure_buffers_sse(BLASLONG p, BLASLONG q, BLASLONG r) {
    size_t need_sa = ((size_t)p + GEMM_UNROLL_M) * ((size_t)q + 4) * sizeof(FLOAT);
    /* SSE-style: 2x B buffer for pre-duplication */
    size_t need_sb = ((size_t)q + 4) * ((size_t)r + GEMM_UNROLL_N) * 2 * sizeof(FLOAT);

    if (sa_size_sse < need_sa) {
        free(sa_buffer_sse);
        sa_buffer_sse = (FLOAT *)malloc(need_sa);
        sa_size_sse = need_sa;
    }
    if (sb_size_sse < need_sb) {
        free(sb_buffer_sse);
        sb_buffer_sse = (FLOAT *)malloc(need_sb);
        sb_size_sse = need_sb;
    }
}

/*===========================================================================
 * External SSE Kernel Declaration
 * (Implemented in dgemm_kernel_sse_4x4_wasm.c)
 *===========================================================================*/

extern int dgemm_kernel_sse(BLASLONG M, BLASLONG N, BLASLONG K, FLOAT alpha,
                            FLOAT *sa, FLOAT *sb, FLOAT *C, BLASLONG LDC);

/*===========================================================================
 * SSE-Style OCOPY: Pack B panel with PRE-DUPLICATION
 *
 * Standard packed:     [b00, b01, b02, b03, b10, b11, ...] (4 doubles/k)
 * SSE-style packed:    [b00, b00, b01, b01, b02, b02, b03, b03, ...] (8 doubles/k)
 *
 * This allows the micro-kernel to load [bi, bi] directly as a v128_t
 * without needing extract_lane + splat operations.
 *===========================================================================*/

static void gemm_oncopy_sse(BLASLONG min_l, BLASLONG min_j,
                            const FLOAT *b, BLASLONG ldb,
                            BLASLONG ls, BLASLONG js, FLOAT *sb) {
    BLASLONG i, j;
    FLOAT *d_offset = sb;

    /* Pack full NR-column panels with pre-duplication */
    for (j = 0; j + GEMM_UNROLL_N <= min_j; j += GEMM_UNROLL_N) {
        const FLOAT *b_ptr0 = b + ls + (js + j) * ldb;
        const FLOAT *b_ptr1 = b + ls + (js + j + 1) * ldb;
        const FLOAT *b_ptr2 = b + ls + (js + j + 2) * ldb;
        const FLOAT *b_ptr3 = b + ls + (js + j + 3) * ldb;

        for (i = 0; i < min_l; i++) {
            /* Pre-duplicate each B value */
            d_offset[0] = d_offset[1] = b_ptr0[i];  /* [b0, b0] */
            d_offset[2] = d_offset[3] = b_ptr1[i];  /* [b1, b1] */
            d_offset[4] = d_offset[5] = b_ptr2[i];  /* [b2, b2] */
            d_offset[6] = d_offset[7] = b_ptr3[i];  /* [b3, b3] */
            d_offset += GEMM_UNROLL_N * 2;  /* 8 doubles per k */
        }
    }

    /* Pack remaining columns with pre-duplication and zero padding */
    if (j < min_j) {
        BLASLONG nr_rem = min_j - j;

        for (i = 0; i < min_l; i++) {
            BLASLONG jj;
            for (jj = 0; jj < nr_rem; jj++) {
                FLOAT val = b[ls + i + (js + j + jj) * ldb];
                d_offset[jj * 2] = val;
                d_offset[jj * 2 + 1] = val;
            }
            for (jj = nr_rem; jj < GEMM_UNROLL_N; jj++) {
                d_offset[jj * 2] = 0.0;
                d_offset[jj * 2 + 1] = 0.0;
            }
            d_offset += GEMM_UNROLL_N * 2;
        }
    }
}

/*===========================================================================
 * SSE-Style OTCOPY: Pack B^T panel with PRE-DUPLICATION
 *===========================================================================*/

static void gemm_otcopy_sse(BLASLONG min_l, BLASLONG min_j,
                            const FLOAT *b, BLASLONG ldb,
                            BLASLONG ls, BLASLONG js, FLOAT *sb) {
    BLASLONG i, j;
    FLOAT *d_offset = sb;

    /* Pack full NR-column panels with pre-duplication */
    for (j = 0; j + GEMM_UNROLL_N <= min_j; j += GEMM_UNROLL_N) {
        const FLOAT *b_ptr = b + (js + j) + ls * ldb;

        for (i = 0; i < min_l; i++) {
            /* Pre-duplicate each B value */
            d_offset[0] = d_offset[1] = b_ptr[0];  /* [b0, b0] */
            d_offset[2] = d_offset[3] = b_ptr[1];  /* [b1, b1] */
            d_offset[4] = d_offset[5] = b_ptr[2];  /* [b2, b2] */
            d_offset[6] = d_offset[7] = b_ptr[3];  /* [b3, b3] */
            b_ptr += ldb;
            d_offset += GEMM_UNROLL_N * 2;
        }
    }

    /* Pack remaining columns with pre-duplication and zero padding */
    if (j < min_j) {
        BLASLONG nr_rem = min_j - j;
        const FLOAT *b_ptr = b + (js + j) + ls * ldb;

        for (i = 0; i < min_l; i++) {
            BLASLONG jj;
            for (jj = 0; jj < nr_rem; jj++) {
                d_offset[jj * 2] = b_ptr[jj];
                d_offset[jj * 2 + 1] = b_ptr[jj];
            }
            for (jj = nr_rem; jj < GEMM_UNROLL_N; jj++) {
                d_offset[jj * 2] = 0.0;
                d_offset[jj * 2 + 1] = 0.0;
            }
            b_ptr += ldb;
            d_offset += GEMM_UNROLL_N * 2;
        }
    }
}

/*===========================================================================
 * SSE-Style GEMM Driver (No Transpose)
 *===========================================================================*/

static int gemm_nn_sse(blas_arg_t *args, BLASLONG *range_m, BLASLONG *range_n,
                       FLOAT *sa, FLOAT *sb) {
    BLASLONG m_from, m_to, n_from, n_to;
    BLASLONG k = args->k;
    BLASLONG lda = args->lda;
    BLASLONG ldb = args->ldb;
    BLASLONG ldc = args->ldc;
    FLOAT alpha = args->alpha;
    FLOAT beta = args->beta;
    const FLOAT *a = args->a;
    const FLOAT *b = args->b;
    FLOAT *c = args->c;

    BLASLONG ls, is, js;
    BLASLONG min_l, min_i, min_j;
    BLASLONG jjs, min_jj;

    /* Determine working range */
    m_from = 0;
    m_to = args->m;
    if (range_m) {
        m_from = range_m[0];
        m_to = range_m[1];
    }

    n_from = 0;
    n_to = args->n;
    if (range_n) {
        n_from = range_n[0];
        n_to = range_n[1];
    }

    /* Apply beta to C */
    if (beta != 1.0) {
        gemm_beta(m_to - m_from, n_to - n_from, beta,
                  c + m_from + n_from * ldc, ldc);
    }

    /* Early exit */
    if (k == 0 || alpha == 0.0) return 0;

    /* Main 3-level blocking loop */
    for (js = n_from; js < n_to; js += GEMM_R) {
        min_j = n_to - js;
        if (min_j > GEMM_R) min_j = GEMM_R;

        for (ls = 0; ls < k; ls += min_l) {
            min_l = k - ls;
            if (min_l >= GEMM_Q * 2) {
                min_l = GEMM_Q;
            } else if (min_l > GEMM_Q) {
                min_l = ((min_l / 2 + GEMM_UNROLL_M - 1) / GEMM_UNROLL_M) * GEMM_UNROLL_M;
            }

            /* First M block - special handling */
            min_i = m_to - m_from;
            if (min_i >= GEMM_P * 2) {
                min_i = GEMM_P;
            } else if (min_i > GEMM_P) {
                min_i = ((min_i / 2 + GEMM_UNROLL_M - 1) / GEMM_UNROLL_M) * GEMM_UNROLL_M;
            }

            /* Pack first A panel (same as standard) */
            gemm_incopy(min_l, min_i, a, lda, ls, m_from, sa);

            /* Process B in chunks of GEMM_UNROLL_N * 3 for better cache use */
            for (jjs = js; jjs < js + min_j; jjs += min_jj) {
                min_jj = min_j + js - jjs;
                if (min_jj >= 3 * GEMM_UNROLL_N) {
                    min_jj = 3 * GEMM_UNROLL_N;
                } else if (min_jj > GEMM_UNROLL_N) {
                    min_jj = GEMM_UNROLL_N;
                }

                /* Pack B panel with SSE-style pre-duplication */
                gemm_oncopy_sse(min_l, min_jj, b, ldb, ls, jjs,
                                sb + min_l * (jjs - js) * 2);  /* 2x offset for SSE */

                /* Call SSE micro-kernel */
                dgemm_kernel_sse(min_i, min_jj, min_l, alpha,
                                 sa, sb + min_l * (jjs - js) * 2,
                                 c + m_from + jjs * ldc, ldc);
            }

            /* Remaining M blocks */
            for (is = m_from + min_i; is < m_to; is += min_i) {
                min_i = m_to - is;
                if (min_i >= GEMM_P * 2) {
                    min_i = GEMM_P;
                } else if (min_i > GEMM_P) {
                    min_i = ((min_i / 2 + GEMM_UNROLL_M - 1) / GEMM_UNROLL_M) * GEMM_UNROLL_M;
                }

                /* Pack A panel */
                gemm_incopy(min_l, min_i, a, lda, ls, is, sa);

                /* Call SSE micro-kernel (B is already packed) */
                dgemm_kernel_sse(min_i, min_j, min_l, alpha,
                                 sa, sb, c + is + js * ldc, ldc);
            }
        }
    }

    return 0;
}

/*===========================================================================
 * SSE-Style GEMM Driver (A Transposed)
 *===========================================================================*/

static int gemm_tn_sse(blas_arg_t *args, BLASLONG *range_m, BLASLONG *range_n,
                       FLOAT *sa, FLOAT *sb) {
    BLASLONG m_from, m_to, n_from, n_to;
    BLASLONG k = args->k;
    BLASLONG lda = args->lda;
    BLASLONG ldb = args->ldb;
    BLASLONG ldc = args->ldc;
    FLOAT alpha = args->alpha;
    FLOAT beta = args->beta;
    const FLOAT *a = args->a;
    const FLOAT *b = args->b;
    FLOAT *c = args->c;

    BLASLONG ls, is, js;
    BLASLONG min_l, min_i, min_j;
    BLASLONG jjs, min_jj;

    m_from = range_m ? range_m[0] : 0;
    m_to = range_m ? range_m[1] : args->m;
    n_from = range_n ? range_n[0] : 0;
    n_to = range_n ? range_n[1] : args->n;

    if (beta != 1.0) {
        gemm_beta(m_to - m_from, n_to - n_from, beta,
                  c + m_from + n_from * ldc, ldc);
    }

    if (k == 0 || alpha == 0.0) return 0;

    for (js = n_from; js < n_to; js += GEMM_R) {
        min_j = n_to - js;
        if (min_j > GEMM_R) min_j = GEMM_R;

        for (ls = 0; ls < k; ls += min_l) {
            min_l = k - ls;
            if (min_l >= GEMM_Q * 2) min_l = GEMM_Q;
            else if (min_l > GEMM_Q) {
                min_l = ((min_l / 2 + GEMM_UNROLL_M - 1) / GEMM_UNROLL_M) * GEMM_UNROLL_M;
            }

            min_i = m_to - m_from;
            if (min_i >= GEMM_P * 2) min_i = GEMM_P;
            else if (min_i > GEMM_P) {
                min_i = ((min_i / 2 + GEMM_UNROLL_M - 1) / GEMM_UNROLL_M) * GEMM_UNROLL_M;
            }

            /* Pack A^T */
            gemm_itcopy(min_l, min_i, a, lda, ls, m_from, sa);

            for (jjs = js; jjs < js + min_j; jjs += min_jj) {
                min_jj = min_j + js - jjs;
                if (min_jj >= 3 * GEMM_UNROLL_N) min_jj = 3 * GEMM_UNROLL_N;
                else if (min_jj > GEMM_UNROLL_N) min_jj = GEMM_UNROLL_N;

                gemm_oncopy_sse(min_l, min_jj, b, ldb, ls, jjs,
                                sb + min_l * (jjs - js) * 2);

                dgemm_kernel_sse(min_i, min_jj, min_l, alpha,
                                 sa, sb + min_l * (jjs - js) * 2,
                                 c + m_from + jjs * ldc, ldc);
            }

            for (is = m_from + min_i; is < m_to; is += min_i) {
                min_i = m_to - is;
                if (min_i >= GEMM_P * 2) min_i = GEMM_P;
                else if (min_i > GEMM_P) {
                    min_i = ((min_i / 2 + GEMM_UNROLL_M - 1) / GEMM_UNROLL_M) * GEMM_UNROLL_M;
                }

                gemm_itcopy(min_l, min_i, a, lda, ls, is, sa);
                dgemm_kernel_sse(min_i, min_j, min_l, alpha,
                                 sa, sb, c + is + js * ldc, ldc);
            }
        }
    }

    return 0;
}

/*===========================================================================
 * SSE-Style GEMM Driver (B Transposed)
 *===========================================================================*/

static int gemm_nt_sse(blas_arg_t *args, BLASLONG *range_m, BLASLONG *range_n,
                       FLOAT *sa, FLOAT *sb) {
    BLASLONG m_from, m_to, n_from, n_to;
    BLASLONG k = args->k;
    BLASLONG lda = args->lda;
    BLASLONG ldb = args->ldb;
    BLASLONG ldc = args->ldc;
    FLOAT alpha = args->alpha;
    FLOAT beta = args->beta;
    const FLOAT *a = args->a;
    const FLOAT *b = args->b;
    FLOAT *c = args->c;

    BLASLONG ls, is, js;
    BLASLONG min_l, min_i, min_j;
    BLASLONG jjs, min_jj;

    m_from = range_m ? range_m[0] : 0;
    m_to = range_m ? range_m[1] : args->m;
    n_from = range_n ? range_n[0] : 0;
    n_to = range_n ? range_n[1] : args->n;

    if (beta != 1.0) {
        gemm_beta(m_to - m_from, n_to - n_from, beta,
                  c + m_from + n_from * ldc, ldc);
    }

    if (k == 0 || alpha == 0.0) return 0;

    for (js = n_from; js < n_to; js += GEMM_R) {
        min_j = n_to - js;
        if (min_j > GEMM_R) min_j = GEMM_R;

        for (ls = 0; ls < k; ls += min_l) {
            min_l = k - ls;
            if (min_l >= GEMM_Q * 2) min_l = GEMM_Q;
            else if (min_l > GEMM_Q) {
                min_l = ((min_l / 2 + GEMM_UNROLL_M - 1) / GEMM_UNROLL_M) * GEMM_UNROLL_M;
            }

            min_i = m_to - m_from;
            if (min_i >= GEMM_P * 2) min_i = GEMM_P;
            else if (min_i > GEMM_P) {
                min_i = ((min_i / 2 + GEMM_UNROLL_M - 1) / GEMM_UNROLL_M) * GEMM_UNROLL_M;
            }

            gemm_incopy(min_l, min_i, a, lda, ls, m_from, sa);

            for (jjs = js; jjs < js + min_j; jjs += min_jj) {
                min_jj = min_j + js - jjs;
                if (min_jj >= 3 * GEMM_UNROLL_N) min_jj = 3 * GEMM_UNROLL_N;
                else if (min_jj > GEMM_UNROLL_N) min_jj = GEMM_UNROLL_N;

                /* Pack B^T with SSE-style pre-duplication */
                gemm_otcopy_sse(min_l, min_jj, b, ldb, ls, jjs,
                                sb + min_l * (jjs - js) * 2);

                dgemm_kernel_sse(min_i, min_jj, min_l, alpha,
                                 sa, sb + min_l * (jjs - js) * 2,
                                 c + m_from + jjs * ldc, ldc);
            }

            for (is = m_from + min_i; is < m_to; is += min_i) {
                min_i = m_to - is;
                if (min_i >= GEMM_P * 2) min_i = GEMM_P;
                else if (min_i > GEMM_P) {
                    min_i = ((min_i / 2 + GEMM_UNROLL_M - 1) / GEMM_UNROLL_M) * GEMM_UNROLL_M;
                }

                gemm_incopy(min_l, min_i, a, lda, ls, is, sa);
                dgemm_kernel_sse(min_i, min_j, min_l, alpha,
                                 sa, sb, c + is + js * ldc, ldc);
            }
        }
    }

    return 0;
}

/*===========================================================================
 * SSE-Style GEMM Driver (A^T * B^T)
 *===========================================================================*/

static int gemm_tt_sse(blas_arg_t *args, BLASLONG *range_m, BLASLONG *range_n,
                       FLOAT *sa, FLOAT *sb) {
    BLASLONG m_from, m_to, n_from, n_to;
    BLASLONG k = args->k;
    BLASLONG lda = args->lda;
    BLASLONG ldb = args->ldb;
    BLASLONG ldc = args->ldc;
    FLOAT alpha = args->alpha;
    FLOAT beta = args->beta;
    const FLOAT *a = args->a;
    const FLOAT *b = args->b;
    FLOAT *c = args->c;

    BLASLONG ls, is, js;
    BLASLONG min_l, min_i, min_j;
    BLASLONG jjs, min_jj;

    m_from = range_m ? range_m[0] : 0;
    m_to = range_m ? range_m[1] : args->m;
    n_from = range_n ? range_n[0] : 0;
    n_to = range_n ? range_n[1] : args->n;

    if (beta != 1.0) {
        gemm_beta(m_to - m_from, n_to - n_from, beta,
                  c + m_from + n_from * ldc, ldc);
    }

    if (k == 0 || alpha == 0.0) return 0;

    for (js = n_from; js < n_to; js += GEMM_R) {
        min_j = n_to - js;
        if (min_j > GEMM_R) min_j = GEMM_R;

        for (ls = 0; ls < k; ls += min_l) {
            min_l = k - ls;
            if (min_l >= GEMM_Q * 2) min_l = GEMM_Q;
            else if (min_l > GEMM_Q) {
                min_l = ((min_l / 2 + GEMM_UNROLL_M - 1) / GEMM_UNROLL_M) * GEMM_UNROLL_M;
            }

            min_i = m_to - m_from;
            if (min_i >= GEMM_P * 2) min_i = GEMM_P;
            else if (min_i > GEMM_P) {
                min_i = ((min_i / 2 + GEMM_UNROLL_M - 1) / GEMM_UNROLL_M) * GEMM_UNROLL_M;
            }

            gemm_itcopy(min_l, min_i, a, lda, ls, m_from, sa);

            for (jjs = js; jjs < js + min_j; jjs += min_jj) {
                min_jj = min_j + js - jjs;
                if (min_jj >= 3 * GEMM_UNROLL_N) min_jj = 3 * GEMM_UNROLL_N;
                else if (min_jj > GEMM_UNROLL_N) min_jj = GEMM_UNROLL_N;

                gemm_otcopy_sse(min_l, min_jj, b, ldb, ls, jjs,
                                sb + min_l * (jjs - js) * 2);

                dgemm_kernel_sse(min_i, min_jj, min_l, alpha,
                                 sa, sb + min_l * (jjs - js) * 2,
                                 c + m_from + jjs * ldc, ldc);
            }

            for (is = m_from + min_i; is < m_to; is += min_i) {
                min_i = m_to - is;
                if (min_i >= GEMM_P * 2) min_i = GEMM_P;
                else if (min_i > GEMM_P) {
                    min_i = ((min_i / 2 + GEMM_UNROLL_M - 1) / GEMM_UNROLL_M) * GEMM_UNROLL_M;
                }

                gemm_itcopy(min_l, min_i, a, lda, ls, is, sa);
                dgemm_kernel_sse(min_i, min_j, min_l, alpha,
                                 sa, sb, c + is + js * ldc, ldc);
            }
        }
    }

    return 0;
}

/*===========================================================================
 * Public SSE-Style DGEMM Interface
 *
 * C = alpha * op(A) * op(B) + beta * C
 *
 * Same interface as dgemm but uses pre-duplicated B packing for better
 * SIMD performance (eliminates extract_lane + splat in inner loop).
 *===========================================================================*/

__attribute__((export_name("dgemm_sse")))
void dgemm_sse(char transA, char transB,
               int32_t m, int32_t n, int32_t k,
               double alpha, const double *A, int32_t lda,
               const double *B, int32_t ldb,
               double beta, double *C, int32_t ldc) {

    if (m <= 0 || n <= 0) return;

    /* Ensure work buffers (2x B size for SSE) */
    ensure_buffers_sse(GEMM_P, GEMM_Q, GEMM_R);

    /* Set up arguments */
    blas_arg_t args;
    args.m = m;
    args.n = n;
    args.k = k;
    args.alpha = alpha;
    args.beta = beta;
    args.a = A;
    args.b = B;
    args.c = C;
    args.lda = lda;
    args.ldb = ldb;
    args.ldc = ldc;

    int trans_a = (transA == 'T' || transA == 't' || transA == 'C' || transA == 'c');
    int trans_b = (transB == 'T' || transB == 't' || transB == 'C' || transB == 'c');

    /* Dispatch based on transpose flags */
    if (!trans_a && !trans_b) {
        gemm_nn_sse(&args, NULL, NULL, sa_buffer_sse, sb_buffer_sse);
    } else if (trans_a && !trans_b) {
        gemm_tn_sse(&args, NULL, NULL, sa_buffer_sse, sb_buffer_sse);
    } else if (!trans_a && trans_b) {
        gemm_nt_sse(&args, NULL, NULL, sa_buffer_sse, sb_buffer_sse);
    } else {
        gemm_tt_sse(&args, NULL, NULL, sa_buffer_sse, sb_buffer_sse);
    }
}

/*===========================================================================
 * Row-Major Matrix Multiply Interface (SSE-Style)
 *===========================================================================*/

__attribute__((export_name("matmul_f64_sse")))
void matmul_f64_sse(int32_t M, int32_t N, int32_t K,
                    const double *A, const double *B, double *C) {
    /* Row-major to column-major conversion:
     * C_row(MxN) = A_row(MxK) * B_row(KxN)
     * becomes: C_col(NxM) = B_col(NxK) * A_col(KxM)
     */
    dgemm_sse('N', 'N', N, M, K,
              1.0, B, N,
              A, K,
              0.0, C, N);
}

/*===========================================================================
 * Memory Management
 *===========================================================================*/

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
    free(sa_buffer);
    free(sb_buffer);
    sa_buffer = NULL;
    sb_buffer = NULL;
    sa_size = 0;
    sb_size = 0;

    /* Also cleanup SSE buffers */
    free(sa_buffer_sse);
    free(sb_buffer_sse);
    sa_buffer_sse = NULL;
    sb_buffer_sse = NULL;
    sa_size_sse = 0;
    sb_size_sse = 0;
}

/*===========================================================================
 * Query Functions
 *===========================================================================*/

__attribute__((export_name("get_gemm_p")))
int32_t get_gemm_p(void) { return GEMM_P; }

__attribute__((export_name("get_gemm_q")))
int32_t get_gemm_q(void) { return GEMM_Q; }

__attribute__((export_name("get_gemm_r")))
int32_t get_gemm_r(void) { return GEMM_R; }

/*===========================================================================
 * Native WASM SIMD DGEMM Implementation
 *
 * Key optimization: Uses i8x16.shuffle with compile-time constant indices
 * to duplicate f64 lanes, avoiding the expensive extract_lane + splat pattern.
 *
 * Shuffle maps to single pshufb instruction on x86 and constant shuffle on ARM.
 * Standard B packing (same as standard kernel).
 *===========================================================================*/

/*===========================================================================
 * External Native Kernel Declaration
 * (Implemented in dgemm_kernel_native_4x4_wasm.c)
 *===========================================================================*/

extern int dgemm_kernel_native(BLASLONG M, BLASLONG N, BLASLONG K, FLOAT alpha,
                               FLOAT *sa, FLOAT *sb, FLOAT *C, BLASLONG LDC);

/*===========================================================================
 * Native GEMM Driver (No Transpose)
 * Uses same packing as standard but calls native kernel
 *===========================================================================*/

static int gemm_nn_native(blas_arg_t *args, BLASLONG *range_m, BLASLONG *range_n,
                          FLOAT *sa, FLOAT *sb) {
    BLASLONG m_from, m_to, n_from, n_to;
    BLASLONG k = args->k;
    BLASLONG lda = args->lda;
    BLASLONG ldb = args->ldb;
    BLASLONG ldc = args->ldc;
    FLOAT alpha = args->alpha;
    FLOAT beta = args->beta;
    const FLOAT *a = args->a;
    const FLOAT *b = args->b;
    FLOAT *c = args->c;

    BLASLONG ls, is, js;
    BLASLONG min_l, min_i, min_j;
    BLASLONG jjs, min_jj;

    m_from = 0;
    m_to = args->m;
    if (range_m) {
        m_from = range_m[0];
        m_to = range_m[1];
    }

    n_from = 0;
    n_to = args->n;
    if (range_n) {
        n_from = range_n[0];
        n_to = range_n[1];
    }

    if (beta != 1.0) {
        gemm_beta(m_to - m_from, n_to - n_from, beta,
                  c + m_from + n_from * ldc, ldc);
    }

    if (k == 0 || alpha == 0.0) return 0;

    for (js = n_from; js < n_to; js += GEMM_R) {
        min_j = n_to - js;
        if (min_j > GEMM_R) min_j = GEMM_R;

        for (ls = 0; ls < k; ls += min_l) {
            min_l = k - ls;
            if (min_l >= GEMM_Q * 2) {
                min_l = GEMM_Q;
            } else if (min_l > GEMM_Q) {
                min_l = ((min_l / 2 + GEMM_UNROLL_M - 1) / GEMM_UNROLL_M) * GEMM_UNROLL_M;
            }

            min_i = m_to - m_from;
            if (min_i >= GEMM_P * 2) {
                min_i = GEMM_P;
            } else if (min_i > GEMM_P) {
                min_i = ((min_i / 2 + GEMM_UNROLL_M - 1) / GEMM_UNROLL_M) * GEMM_UNROLL_M;
            }

            gemm_incopy(min_l, min_i, a, lda, ls, m_from, sa);

            for (jjs = js; jjs < js + min_j; jjs += min_jj) {
                min_jj = min_j + js - jjs;
                if (min_jj >= 3 * GEMM_UNROLL_N) {
                    min_jj = 3 * GEMM_UNROLL_N;
                } else if (min_jj > GEMM_UNROLL_N) {
                    min_jj = GEMM_UNROLL_N;
                }

                gemm_oncopy(min_l, min_jj, b, ldb, ls, jjs,
                            sb + min_l * (jjs - js));

                dgemm_kernel_native(min_i, min_jj, min_l, alpha,
                                    sa, sb + min_l * (jjs - js),
                                    c + m_from + jjs * ldc, ldc);
            }

            for (is = m_from + min_i; is < m_to; is += min_i) {
                min_i = m_to - is;
                if (min_i >= GEMM_P * 2) {
                    min_i = GEMM_P;
                } else if (min_i > GEMM_P) {
                    min_i = ((min_i / 2 + GEMM_UNROLL_M - 1) / GEMM_UNROLL_M) * GEMM_UNROLL_M;
                }

                gemm_incopy(min_l, min_i, a, lda, ls, is, sa);
                dgemm_kernel_native(min_i, min_j, min_l, alpha,
                                    sa, sb, c + is + js * ldc, ldc);
            }
        }
    }

    return 0;
}

/*===========================================================================
 * Native GEMM Driver (A Transposed)
 *===========================================================================*/

static int gemm_tn_native(blas_arg_t *args, BLASLONG *range_m, BLASLONG *range_n,
                          FLOAT *sa, FLOAT *sb) {
    BLASLONG m_from, m_to, n_from, n_to;
    BLASLONG k = args->k;
    BLASLONG lda = args->lda;
    BLASLONG ldb = args->ldb;
    BLASLONG ldc = args->ldc;
    FLOAT alpha = args->alpha;
    FLOAT beta = args->beta;
    const FLOAT *a = args->a;
    const FLOAT *b = args->b;
    FLOAT *c = args->c;

    BLASLONG ls, is, js;
    BLASLONG min_l, min_i, min_j;
    BLASLONG jjs, min_jj;

    m_from = range_m ? range_m[0] : 0;
    m_to = range_m ? range_m[1] : args->m;
    n_from = range_n ? range_n[0] : 0;
    n_to = range_n ? range_n[1] : args->n;

    if (beta != 1.0) {
        gemm_beta(m_to - m_from, n_to - n_from, beta,
                  c + m_from + n_from * ldc, ldc);
    }

    if (k == 0 || alpha == 0.0) return 0;

    for (js = n_from; js < n_to; js += GEMM_R) {
        min_j = n_to - js;
        if (min_j > GEMM_R) min_j = GEMM_R;

        for (ls = 0; ls < k; ls += min_l) {
            min_l = k - ls;
            if (min_l >= GEMM_Q * 2) min_l = GEMM_Q;
            else if (min_l > GEMM_Q) {
                min_l = ((min_l / 2 + GEMM_UNROLL_M - 1) / GEMM_UNROLL_M) * GEMM_UNROLL_M;
            }

            min_i = m_to - m_from;
            if (min_i >= GEMM_P * 2) min_i = GEMM_P;
            else if (min_i > GEMM_P) {
                min_i = ((min_i / 2 + GEMM_UNROLL_M - 1) / GEMM_UNROLL_M) * GEMM_UNROLL_M;
            }

            gemm_itcopy(min_l, min_i, a, lda, ls, m_from, sa);

            for (jjs = js; jjs < js + min_j; jjs += min_jj) {
                min_jj = min_j + js - jjs;
                if (min_jj >= 3 * GEMM_UNROLL_N) min_jj = 3 * GEMM_UNROLL_N;
                else if (min_jj > GEMM_UNROLL_N) min_jj = GEMM_UNROLL_N;

                gemm_oncopy(min_l, min_jj, b, ldb, ls, jjs,
                            sb + min_l * (jjs - js));

                dgemm_kernel_native(min_i, min_jj, min_l, alpha,
                                    sa, sb + min_l * (jjs - js),
                                    c + m_from + jjs * ldc, ldc);
            }

            for (is = m_from + min_i; is < m_to; is += min_i) {
                min_i = m_to - is;
                if (min_i >= GEMM_P * 2) min_i = GEMM_P;
                else if (min_i > GEMM_P) {
                    min_i = ((min_i / 2 + GEMM_UNROLL_M - 1) / GEMM_UNROLL_M) * GEMM_UNROLL_M;
                }

                gemm_itcopy(min_l, min_i, a, lda, ls, is, sa);
                dgemm_kernel_native(min_i, min_j, min_l, alpha,
                                    sa, sb, c + is + js * ldc, ldc);
            }
        }
    }

    return 0;
}

/*===========================================================================
 * Native GEMM Driver (B Transposed)
 *===========================================================================*/

static int gemm_nt_native(blas_arg_t *args, BLASLONG *range_m, BLASLONG *range_n,
                          FLOAT *sa, FLOAT *sb) {
    BLASLONG m_from, m_to, n_from, n_to;
    BLASLONG k = args->k;
    BLASLONG lda = args->lda;
    BLASLONG ldb = args->ldb;
    BLASLONG ldc = args->ldc;
    FLOAT alpha = args->alpha;
    FLOAT beta = args->beta;
    const FLOAT *a = args->a;
    const FLOAT *b = args->b;
    FLOAT *c = args->c;

    BLASLONG ls, is, js;
    BLASLONG min_l, min_i, min_j;
    BLASLONG jjs, min_jj;

    m_from = range_m ? range_m[0] : 0;
    m_to = range_m ? range_m[1] : args->m;
    n_from = range_n ? range_n[0] : 0;
    n_to = range_n ? range_n[1] : args->n;

    if (beta != 1.0) {
        gemm_beta(m_to - m_from, n_to - n_from, beta,
                  c + m_from + n_from * ldc, ldc);
    }

    if (k == 0 || alpha == 0.0) return 0;

    for (js = n_from; js < n_to; js += GEMM_R) {
        min_j = n_to - js;
        if (min_j > GEMM_R) min_j = GEMM_R;

        for (ls = 0; ls < k; ls += min_l) {
            min_l = k - ls;
            if (min_l >= GEMM_Q * 2) min_l = GEMM_Q;
            else if (min_l > GEMM_Q) {
                min_l = ((min_l / 2 + GEMM_UNROLL_M - 1) / GEMM_UNROLL_M) * GEMM_UNROLL_M;
            }

            min_i = m_to - m_from;
            if (min_i >= GEMM_P * 2) min_i = GEMM_P;
            else if (min_i > GEMM_P) {
                min_i = ((min_i / 2 + GEMM_UNROLL_M - 1) / GEMM_UNROLL_M) * GEMM_UNROLL_M;
            }

            gemm_incopy(min_l, min_i, a, lda, ls, m_from, sa);

            for (jjs = js; jjs < js + min_j; jjs += min_jj) {
                min_jj = min_j + js - jjs;
                if (min_jj >= 3 * GEMM_UNROLL_N) min_jj = 3 * GEMM_UNROLL_N;
                else if (min_jj > GEMM_UNROLL_N) min_jj = GEMM_UNROLL_N;

                gemm_otcopy(min_l, min_jj, b, ldb, ls, jjs,
                            sb + min_l * (jjs - js));

                dgemm_kernel_native(min_i, min_jj, min_l, alpha,
                                    sa, sb + min_l * (jjs - js),
                                    c + m_from + jjs * ldc, ldc);
            }

            for (is = m_from + min_i; is < m_to; is += min_i) {
                min_i = m_to - is;
                if (min_i >= GEMM_P * 2) min_i = GEMM_P;
                else if (min_i > GEMM_P) {
                    min_i = ((min_i / 2 + GEMM_UNROLL_M - 1) / GEMM_UNROLL_M) * GEMM_UNROLL_M;
                }

                gemm_incopy(min_l, min_i, a, lda, ls, is, sa);
                dgemm_kernel_native(min_i, min_j, min_l, alpha,
                                    sa, sb, c + is + js * ldc, ldc);
            }
        }
    }

    return 0;
}

/*===========================================================================
 * Native GEMM Driver (A^T * B^T)
 *===========================================================================*/

static int gemm_tt_native(blas_arg_t *args, BLASLONG *range_m, BLASLONG *range_n,
                          FLOAT *sa, FLOAT *sb) {
    BLASLONG m_from, m_to, n_from, n_to;
    BLASLONG k = args->k;
    BLASLONG lda = args->lda;
    BLASLONG ldb = args->ldb;
    BLASLONG ldc = args->ldc;
    FLOAT alpha = args->alpha;
    FLOAT beta = args->beta;
    const FLOAT *a = args->a;
    const FLOAT *b = args->b;
    FLOAT *c = args->c;

    BLASLONG ls, is, js;
    BLASLONG min_l, min_i, min_j;
    BLASLONG jjs, min_jj;

    m_from = range_m ? range_m[0] : 0;
    m_to = range_m ? range_m[1] : args->m;
    n_from = range_n ? range_n[0] : 0;
    n_to = range_n ? range_n[1] : args->n;

    if (beta != 1.0) {
        gemm_beta(m_to - m_from, n_to - n_from, beta,
                  c + m_from + n_from * ldc, ldc);
    }

    if (k == 0 || alpha == 0.0) return 0;

    for (js = n_from; js < n_to; js += GEMM_R) {
        min_j = n_to - js;
        if (min_j > GEMM_R) min_j = GEMM_R;

        for (ls = 0; ls < k; ls += min_l) {
            min_l = k - ls;
            if (min_l >= GEMM_Q * 2) min_l = GEMM_Q;
            else if (min_l > GEMM_Q) {
                min_l = ((min_l / 2 + GEMM_UNROLL_M - 1) / GEMM_UNROLL_M) * GEMM_UNROLL_M;
            }

            min_i = m_to - m_from;
            if (min_i >= GEMM_P * 2) min_i = GEMM_P;
            else if (min_i > GEMM_P) {
                min_i = ((min_i / 2 + GEMM_UNROLL_M - 1) / GEMM_UNROLL_M) * GEMM_UNROLL_M;
            }

            gemm_itcopy(min_l, min_i, a, lda, ls, m_from, sa);

            for (jjs = js; jjs < js + min_j; jjs += min_jj) {
                min_jj = min_j + js - jjs;
                if (min_jj >= 3 * GEMM_UNROLL_N) min_jj = 3 * GEMM_UNROLL_N;
                else if (min_jj > GEMM_UNROLL_N) min_jj = GEMM_UNROLL_N;

                gemm_otcopy(min_l, min_jj, b, ldb, ls, jjs,
                            sb + min_l * (jjs - js));

                dgemm_kernel_native(min_i, min_jj, min_l, alpha,
                                    sa, sb + min_l * (jjs - js),
                                    c + m_from + jjs * ldc, ldc);
            }

            for (is = m_from + min_i; is < m_to; is += min_i) {
                min_i = m_to - is;
                if (min_i >= GEMM_P * 2) min_i = GEMM_P;
                else if (min_i > GEMM_P) {
                    min_i = ((min_i / 2 + GEMM_UNROLL_M - 1) / GEMM_UNROLL_M) * GEMM_UNROLL_M;
                }

                gemm_itcopy(min_l, min_i, a, lda, ls, is, sa);
                dgemm_kernel_native(min_i, min_j, min_l, alpha,
                                    sa, sb, c + is + js * ldc, ldc);
            }
        }
    }

    return 0;
}

/*===========================================================================
 * Public Native DGEMM Interface
 *
 * C = alpha * op(A) * op(B) + beta * C
 *
 * Uses i8x16.shuffle for lane duplication instead of extract_lane + splat.
 *===========================================================================*/

__attribute__((export_name("dgemm_native")))
void dgemm_native(char transA, char transB,
                  int32_t m, int32_t n, int32_t k,
                  double alpha, const double *A, int32_t lda,
                  const double *B, int32_t ldb,
                  double beta, double *C, int32_t ldc) {

    if (m <= 0 || n <= 0) return;

    /* Ensure work buffers (same size as standard) */
    ensure_buffers(GEMM_P, GEMM_Q, GEMM_R);

    /* Set up arguments */
    blas_arg_t args;
    args.m = m;
    args.n = n;
    args.k = k;
    args.alpha = alpha;
    args.beta = beta;
    args.a = A;
    args.b = B;
    args.c = C;
    args.lda = lda;
    args.ldb = ldb;
    args.ldc = ldc;

    int trans_a = (transA == 'T' || transA == 't' || transA == 'C' || transA == 'c');
    int trans_b = (transB == 'T' || transB == 't' || transB == 'C' || transB == 'c');

    /* Dispatch based on transpose flags */
    if (!trans_a && !trans_b) {
        gemm_nn_native(&args, NULL, NULL, sa_buffer, sb_buffer);
    } else if (trans_a && !trans_b) {
        gemm_tn_native(&args, NULL, NULL, sa_buffer, sb_buffer);
    } else if (!trans_a && trans_b) {
        gemm_nt_native(&args, NULL, NULL, sa_buffer, sb_buffer);
    } else {
        gemm_tt_native(&args, NULL, NULL, sa_buffer, sb_buffer);
    }
}

/*===========================================================================
 * Row-Major Matrix Multiply Interface (Native)
 *===========================================================================*/

__attribute__((export_name("matmul_f64_native")))
void matmul_f64_native(int32_t M, int32_t N, int32_t K,
                       const double *A, const double *B, double *C) {
    dgemm_native('N', 'N', N, M, K,
                 1.0, B, N,
                 A, K,
                 0.0, C, N);
}
