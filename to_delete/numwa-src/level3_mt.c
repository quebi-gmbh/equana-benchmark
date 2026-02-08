/**
 * Multi-threaded Level 3 BLAS Driver for WebAssembly
 *
 * This implements pthread-based parallel DGEMM, parallelizing over the
 * N dimension (column blocks of C). Each thread gets its own packing
 * buffers and processes a range of column blocks independently.
 *
 * Threading model:
 * - Parallelize outer N-blocking loop (GEMM_R column blocks)
 * - Per-thread pack_a and pack_b buffers (no sharing)
 * - Simple pthread_create/join synchronization
 * - Beta scaling done before thread spawn to avoid races
 *
 * Based on:
 * - level3.c (single-threaded numwa BLAS driver)
 * - matmul_mt.c (proven pthread pattern from matmul_wasm)
 */

#include <pthread.h>
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
 * Blocking Parameters (same as level3.c)
 *===========================================================================*/

#define GEMM_P      64      /* Block rows of A (M dimension) */
#define GEMM_Q      128     /* Block of K dimension */
#define GEMM_R      512     /* Block cols of B (N dimension) */

#define GEMM_UNROLL_M   4   /* Micro-kernel M tile */
#define GEMM_UNROLL_N   4   /* Micro-kernel N tile */

/*===========================================================================
 * Thread Configuration
 *===========================================================================*/

static int num_threads = 4;
#define MAX_THREADS 32

/*===========================================================================
 * Per-Thread Buffer Management
 *===========================================================================*/

typedef struct {
    FLOAT *pack_a;          /* Thread-local A packing buffer */
    FLOAT *pack_b;          /* Thread-local B packing buffer */
    size_t pack_a_size;     /* Current A allocation size */
    size_t pack_b_size;     /* Current B allocation size */
} thread_buffers_t;

static thread_buffers_t thread_bufs[MAX_THREADS];

static void ensure_thread_buffers(int tid, int is_sse) {
    size_t need_a = ((size_t)GEMM_P + GEMM_UNROLL_M) * ((size_t)GEMM_Q + 4) * sizeof(FLOAT);
    size_t need_b;

    if (is_sse) {
        /* SSE-style: 2x B buffer for pre-duplication */
        need_b = ((size_t)GEMM_Q + 4) * ((size_t)GEMM_R + GEMM_UNROLL_N) * 2 * sizeof(FLOAT);
    } else {
        need_b = ((size_t)GEMM_Q + 4) * ((size_t)GEMM_R + GEMM_UNROLL_N) * sizeof(FLOAT);
    }

    if (thread_bufs[tid].pack_a_size < need_a) {
        free(thread_bufs[tid].pack_a);
        thread_bufs[tid].pack_a = (FLOAT *)malloc(need_a);
        thread_bufs[tid].pack_a_size = need_a;
    }
    if (thread_bufs[tid].pack_b_size < need_b) {
        free(thread_bufs[tid].pack_b);
        thread_bufs[tid].pack_b = (FLOAT *)malloc(need_b);
        thread_bufs[tid].pack_b_size = need_b;
    }
}

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
 * Thread Work Structure
 *===========================================================================*/

typedef struct {
    int tid;                /* Thread ID (0..nthreads-1) */
    blas_arg_t *args;       /* Shared problem parameters */
    BLASLONG js_start;      /* Starting N column for this thread */
    BLASLONG js_end;        /* Ending N column for this thread */
    int is_sse;             /* Use SSE-style packing? */
    int trans_a;            /* A transposed? */
    int trans_b;            /* B transposed? */
} thread_work_t;

/*===========================================================================
 * External Kernel Declarations
 *===========================================================================*/

extern int dgemm_kernel(BLASLONG M, BLASLONG N, BLASLONG K, FLOAT alpha,
                        FLOAT *sa, FLOAT *sb, FLOAT *C, BLASLONG LDC);

extern int dgemm_kernel_sse(BLASLONG M, BLASLONG N, BLASLONG K, FLOAT alpha,
                            FLOAT *sa, FLOAT *sb, FLOAT *C, BLASLONG LDC);

/*===========================================================================
 * BETA Operation: C = beta * C
 * Note: This is called BEFORE spawning threads
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
}

/*===========================================================================
 * ICOPY: Pack A panel (column-major input → packed format)
 *===========================================================================*/

static void gemm_incopy(BLASLONG min_l, BLASLONG min_i,
                        const FLOAT *a, BLASLONG lda,
                        BLASLONG ls, BLASLONG is, FLOAT *sa) {
    BLASLONG i, j;
    const FLOAT *a_offset;
    FLOAT *b_offset = sa;

    a_offset = a + is + ls * lda;

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
 *===========================================================================*/

static void gemm_itcopy(BLASLONG min_l, BLASLONG min_i,
                        const FLOAT *a, BLASLONG lda,
                        BLASLONG ls, BLASLONG is, FLOAT *sa) {
    BLASLONG i, j;
    FLOAT *b_offset = sa;

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
 * OCOPY: Pack B panel (standard)
 *===========================================================================*/

static void gemm_oncopy(BLASLONG min_l, BLASLONG min_j,
                        const FLOAT *b, BLASLONG ldb,
                        BLASLONG ls, BLASLONG js, FLOAT *sb) {
    BLASLONG i, j;
    FLOAT *d_offset = sb;

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
 * OCOPY for Transposed B: Pack B^T panel (standard)
 *===========================================================================*/

static void gemm_otcopy(BLASLONG min_l, BLASLONG min_j,
                        const FLOAT *b, BLASLONG ldb,
                        BLASLONG ls, BLASLONG js, FLOAT *sb) {
    BLASLONG i, j;
    FLOAT *d_offset = sb;

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
 * SSE-Style OCOPY: Pack B panel with pre-duplication
 *===========================================================================*/

static void gemm_oncopy_sse(BLASLONG min_l, BLASLONG min_j,
                            const FLOAT *b, BLASLONG ldb,
                            BLASLONG ls, BLASLONG js, FLOAT *sb) {
    BLASLONG i, j;
    FLOAT *d_offset = sb;

    for (j = 0; j + GEMM_UNROLL_N <= min_j; j += GEMM_UNROLL_N) {
        const FLOAT *b_ptr0 = b + ls + (js + j) * ldb;
        const FLOAT *b_ptr1 = b + ls + (js + j + 1) * ldb;
        const FLOAT *b_ptr2 = b + ls + (js + j + 2) * ldb;
        const FLOAT *b_ptr3 = b + ls + (js + j + 3) * ldb;

        for (i = 0; i < min_l; i++) {
            d_offset[0] = d_offset[1] = b_ptr0[i];
            d_offset[2] = d_offset[3] = b_ptr1[i];
            d_offset[4] = d_offset[5] = b_ptr2[i];
            d_offset[6] = d_offset[7] = b_ptr3[i];
            d_offset += GEMM_UNROLL_N * 2;
        }
    }

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
 * SSE-Style OTCOPY: Pack B^T panel with pre-duplication
 *===========================================================================*/

static void gemm_otcopy_sse(BLASLONG min_l, BLASLONG min_j,
                            const FLOAT *b, BLASLONG ldb,
                            BLASLONG ls, BLASLONG js, FLOAT *sb) {
    BLASLONG i, j;
    FLOAT *d_offset = sb;

    for (j = 0; j + GEMM_UNROLL_N <= min_j; j += GEMM_UNROLL_N) {
        const FLOAT *b_ptr = b + (js + j) + ls * ldb;

        for (i = 0; i < min_l; i++) {
            d_offset[0] = d_offset[1] = b_ptr[0];
            d_offset[2] = d_offset[3] = b_ptr[1];
            d_offset[4] = d_offset[5] = b_ptr[2];
            d_offset[6] = d_offset[7] = b_ptr[3];
            b_ptr += ldb;
            d_offset += GEMM_UNROLL_N * 2;
        }
    }

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
 * Thread Worker: Process assigned N-block range
 *
 * Each thread processes columns [js_start, js_end) of the output matrix C.
 * Uses thread-local packing buffers.
 *===========================================================================*/

static void* thread_gemm_worker(void *arg) {
    thread_work_t *work = (thread_work_t *)arg;
    int tid = work->tid;
    blas_arg_t *args = work->args;

    BLASLONG m = args->m;
    BLASLONG k = args->k;
    BLASLONG lda = args->lda;
    BLASLONG ldb = args->ldb;
    BLASLONG ldc = args->ldc;
    FLOAT alpha = args->alpha;
    const FLOAT *a = args->a;
    const FLOAT *b = args->b;
    FLOAT *c = args->c;

    int is_sse = work->is_sse;
    int trans_a = work->trans_a;
    int trans_b = work->trans_b;

    BLASLONG js_start = work->js_start;
    BLASLONG js_end = work->js_end;

    /* Get thread-local buffers */
    ensure_thread_buffers(tid, is_sse);
    FLOAT *sa = thread_bufs[tid].pack_a;
    FLOAT *sb = thread_bufs[tid].pack_b;

    BLASLONG ls, is, js;
    BLASLONG min_l, min_i, min_j;
    BLASLONG jjs, min_jj;

    /* Process assigned N-block range */
    for (js = js_start; js < js_end; js += GEMM_R) {
        min_j = js_end - js;
        if (min_j > GEMM_R) min_j = GEMM_R;

        for (ls = 0; ls < k; ls += min_l) {
            min_l = k - ls;
            if (min_l >= GEMM_Q * 2) {
                min_l = GEMM_Q;
            } else if (min_l > GEMM_Q) {
                min_l = ((min_l / 2 + GEMM_UNROLL_M - 1) / GEMM_UNROLL_M) * GEMM_UNROLL_M;
            }

            /* First M block */
            min_i = m;
            if (min_i >= GEMM_P * 2) {
                min_i = GEMM_P;
            } else if (min_i > GEMM_P) {
                min_i = ((min_i / 2 + GEMM_UNROLL_M - 1) / GEMM_UNROLL_M) * GEMM_UNROLL_M;
            }

            /* Pack A */
            if (trans_a) {
                gemm_itcopy(min_l, min_i, a, lda, ls, 0, sa);
            } else {
                gemm_incopy(min_l, min_i, a, lda, ls, 0, sa);
            }

            /* Process B in chunks */
            for (jjs = js; jjs < js + min_j; jjs += min_jj) {
                min_jj = min_j + js - jjs;
                if (min_jj >= 3 * GEMM_UNROLL_N) {
                    min_jj = 3 * GEMM_UNROLL_N;
                } else if (min_jj > GEMM_UNROLL_N) {
                    min_jj = GEMM_UNROLL_N;
                }

                /* Pack B */
                if (is_sse) {
                    if (trans_b) {
                        gemm_otcopy_sse(min_l, min_jj, b, ldb, ls, jjs,
                                        sb + min_l * (jjs - js) * 2);
                    } else {
                        gemm_oncopy_sse(min_l, min_jj, b, ldb, ls, jjs,
                                        sb + min_l * (jjs - js) * 2);
                    }
                    dgemm_kernel_sse(min_i, min_jj, min_l, alpha,
                                     sa, sb + min_l * (jjs - js) * 2,
                                     c + jjs * ldc, ldc);
                } else {
                    if (trans_b) {
                        gemm_otcopy(min_l, min_jj, b, ldb, ls, jjs,
                                    sb + min_l * (jjs - js));
                    } else {
                        gemm_oncopy(min_l, min_jj, b, ldb, ls, jjs,
                                    sb + min_l * (jjs - js));
                    }
                    dgemm_kernel(min_i, min_jj, min_l, alpha,
                                 sa, sb + min_l * (jjs - js),
                                 c + jjs * ldc, ldc);
                }
            }

            /* Remaining M blocks */
            for (is = min_i; is < m; is += min_i) {
                min_i = m - is;
                if (min_i >= GEMM_P * 2) {
                    min_i = GEMM_P;
                } else if (min_i > GEMM_P) {
                    min_i = ((min_i / 2 + GEMM_UNROLL_M - 1) / GEMM_UNROLL_M) * GEMM_UNROLL_M;
                }

                /* Pack A for this M block */
                if (trans_a) {
                    gemm_itcopy(min_l, min_i, a, lda, ls, is, sa);
                } else {
                    gemm_incopy(min_l, min_i, a, lda, ls, is, sa);
                }

                /* Call kernel (B already packed) */
                if (is_sse) {
                    dgemm_kernel_sse(min_i, min_j, min_l, alpha,
                                     sa, sb, c + is + js * ldc, ldc);
                } else {
                    dgemm_kernel(min_i, min_j, min_l, alpha,
                                 sa, sb, c + is + js * ldc, ldc);
                }
            }
        }
    }

    return NULL;
}

/*===========================================================================
 * Multi-threaded DGEMM Driver
 *===========================================================================*/

static void dgemm_mt_internal(char transA, char transB,
                              BLASLONG m, BLASLONG n, BLASLONG k,
                              FLOAT alpha, const FLOAT *A, BLASLONG lda,
                              const FLOAT *B, BLASLONG ldb,
                              FLOAT beta, FLOAT *C, BLASLONG ldc,
                              int is_sse) {
    if (m <= 0 || n <= 0) return;

    int trans_a = (transA == 'T' || transA == 't' || transA == 'C' || transA == 'c');
    int trans_b = (transB == 'T' || transB == 't' || transB == 'C' || transB == 'c');

    /* Apply beta to C (before spawning threads to avoid races) */
    if (beta != 1.0) {
        gemm_beta(m, n, beta, C, ldc);
    }

    /* Early exit */
    if (k == 0 || alpha == 0.0) return;

    /* Calculate work distribution */
    int nthreads = num_threads;
    BLASLONG total_nc_blocks = (n + GEMM_R - 1) / GEMM_R;

    /* Don't use more threads than blocks */
    if (nthreads > total_nc_blocks) {
        nthreads = (int)total_nc_blocks;
    }
    if (nthreads < 1) nthreads = 1;

    /* For small matrices, use single thread */
    if (n < 256 || total_nc_blocks < 2) {
        nthreads = 1;
    }

    /* Set up arguments */
    blas_arg_t args;
    args.m = m;
    args.n = n;
    args.k = k;
    args.alpha = alpha;
    args.beta = 1.0;  /* Beta already applied */
    args.a = A;
    args.b = B;
    args.c = C;
    args.lda = lda;
    args.ldb = ldb;
    args.ldc = ldc;

    /* Distribute work */
    thread_work_t work[MAX_THREADS];
    pthread_t threads[MAX_THREADS];

    BLASLONG blocks_per_thread = (total_nc_blocks + nthreads - 1) / nthreads;

    for (int t = 0; t < nthreads; t++) {
        work[t].tid = t;
        work[t].args = &args;
        work[t].js_start = t * blocks_per_thread * GEMM_R;
        work[t].js_end = (t + 1) * blocks_per_thread * GEMM_R;
        if (work[t].js_end > n) work[t].js_end = n;
        work[t].is_sse = is_sse;
        work[t].trans_a = trans_a;
        work[t].trans_b = trans_b;
    }

    /* Spawn threads (main thread does work[0]) */
    for (int t = 1; t < nthreads; t++) {
        pthread_create(&threads[t], NULL, thread_gemm_worker, &work[t]);
    }

    /* Main thread does first chunk */
    thread_gemm_worker(&work[0]);

    /* Join threads */
    for (int t = 1; t < nthreads; t++) {
        pthread_join(threads[t], NULL);
    }
}

/*===========================================================================
 * Public Multi-threaded DGEMM Interfaces
 *===========================================================================*/

__attribute__((export_name("dgemm_mt")))
void dgemm_mt(char transA, char transB,
              int32_t m, int32_t n, int32_t k,
              double alpha, const double *A, int32_t lda,
              const double *B, int32_t ldb,
              double beta, double *C, int32_t ldc) {
    dgemm_mt_internal(transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, 0);
}

__attribute__((export_name("dgemm_sse_mt")))
void dgemm_sse_mt(char transA, char transB,
                  int32_t m, int32_t n, int32_t k,
                  double alpha, const double *A, int32_t lda,
                  const double *B, int32_t ldb,
                  double beta, double *C, int32_t ldc) {
    dgemm_mt_internal(transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, 1);
}

/*===========================================================================
 * Row-Major Matrix Multiply Interfaces
 *===========================================================================*/

__attribute__((export_name("matmul_f64_mt")))
void matmul_f64_mt(int32_t M, int32_t N, int32_t K,
                   const double *A, const double *B, double *C) {
    dgemm_mt('N', 'N', N, M, K, 1.0, B, N, A, K, 0.0, C, N);
}

__attribute__((export_name("matmul_f64_sse_mt")))
void matmul_f64_sse_mt(int32_t M, int32_t N, int32_t K,
                       const double *A, const double *B, double *C) {
    dgemm_sse_mt('N', 'N', N, M, K, 1.0, B, N, A, K, 0.0, C, N);
}

/*===========================================================================
 * Thread Control
 *===========================================================================*/

__attribute__((export_name("set_num_threads")))
void set_num_threads(int32_t n) {
    if (n < 1) n = 1;
    if (n > MAX_THREADS) n = MAX_THREADS;
    num_threads = n;
}

__attribute__((export_name("get_num_threads")))
int32_t get_num_threads(void) {
    return num_threads;
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
    /* Free all thread buffers */
    for (int t = 0; t < MAX_THREADS; t++) {
        free(thread_bufs[t].pack_a);
        free(thread_bufs[t].pack_b);
        thread_bufs[t].pack_a = NULL;
        thread_bufs[t].pack_b = NULL;
        thread_bufs[t].pack_a_size = 0;
        thread_bufs[t].pack_b_size = 0;
    }
}
