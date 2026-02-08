/**
 * Multithreaded SSE-Style Matrix Multiplication for WebAssembly
 *
 * Combines:
 * - SSE2-style pre-duplicated B packing (no runtime broadcasts)
 * - Pthread parallelization over NC blocks
 *
 * Pre-duplicates B during packing: [b0,b0,b1,b1,b2,b2,b3,b3]
 * Kernel loads directly without extract_lane + splat.
 *
 * Build with:
 *   emcc matmul_sse_mt.c -o matmul_sse_mt.js -O3 -flto -msimd128 \
 *     -pthread -sPTHREAD_POOL_SIZE=navigator.hardwareConcurrency \
 *     -sMALLOC=mimalloc \
 *     -sINITIAL_MEMORY=512MB -sALLOW_MEMORY_GROWTH=1 \
 *     -sEXPORTED_FUNCTIONS='[...]'
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <wasm_simd128.h>

/* ============= Blocking Parameters ============= */

#define MC 64     /* Block rows of A */
#define KC 128    /* Block of K dimension */
#define NC 256    /* Block cols of B */

#define MR 4      /* Micro-kernel rows */
#define NR 4      /* Micro-kernel cols */

/* SSE-style: B buffer needs 2x space due to pre-duplication */
#define NR_DUP 8  /* NR * 2 for pre-duplicated B */

/* ============= Thread Configuration ============= */

static int num_threads = 4;

__attribute__((export_name("set_num_threads")))
void set_num_threads(int n) {
    if (n > 0 && n <= 32) {
        num_threads = n;
    }
}

__attribute__((export_name("get_num_threads")))
int get_num_threads(void) {
    return num_threads;
}

/* ============= Thread-Local Packing Buffers ============= */

typedef struct {
    double *pack_a;
    double *pack_b;
    size_t pack_a_size;
    size_t pack_b_size;
} thread_buffers_t;

#define MAX_THREADS 32
static thread_buffers_t thread_bufs[MAX_THREADS];

static void ensure_thread_buffers(int tid, int64_t mc, int64_t kc, int64_t nc) {
    size_t need_a = (mc + MR) * (kc + 4) * sizeof(double);
    /* SSE-style: 2x B buffer for pre-duplication */
    size_t need_b = (kc + 4) * (nc + NR) * 2 * sizeof(double);

    if (thread_bufs[tid].pack_a_size < need_a) {
        free(thread_bufs[tid].pack_a);
        thread_bufs[tid].pack_a = (double*)malloc(need_a);
        thread_bufs[tid].pack_a_size = need_a;
    }
    if (thread_bufs[tid].pack_b_size < need_b) {
        free(thread_bufs[tid].pack_b);
        thread_bufs[tid].pack_b = (double*)malloc(need_b);
        thread_bufs[tid].pack_b_size = need_b;
    }
}

/* ============= Panel Packing ============= */

static void pack_panel_a(int64_t mc, int64_t kc, const double *A, int64_t lda, double *pa) {
    int64_t i, j;

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
 */
static void pack_panel_b_sse(int64_t kc, int64_t nc, const double *B, int64_t ldb, double *pb) {
    int64_t i, j;

    for (j = 0; j + NR <= nc; j += NR) {
        const double *b0 = B + j * ldb;
        const double *b1 = b0 + ldb;
        const double *b2 = b1 + ldb;
        const double *b3 = b2 + ldb;

        for (i = 0; i < kc; i++) {
            /* Pre-duplicate each B value */
            pb[0] = b0[i];  pb[1] = b0[i];  /* [b0, b0] */
            pb[2] = b1[i];  pb[3] = b1[i];  /* [b1, b1] */
            pb[4] = b2[i];  pb[5] = b2[i];  /* [b2, b2] */
            pb[6] = b3[i];  pb[7] = b3[i];  /* [b3, b3] */
            pb += NR_DUP;
        }
    }

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
 * No extract_lane or splat needed - B values are already [bi, bi]
 */
static void micro_kernel_4x4_sse(int64_t kc, double alpha,
                                  const double *pa, const double *pb,
                                  double *C, int64_t ldc) {
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
        v128_t a_lo = wasm_v128_load(pa);
        v128_t a_hi = wasm_v128_load(pa + 2);

        /* Load B: pre-duplicated, no extract_lane needed! */
        v128_t b0 = wasm_v128_load(pb);      /* [b0, b0] */
        v128_t b1 = wasm_v128_load(pb + 2);  /* [b1, b1] */
        v128_t b2 = wasm_v128_load(pb + 4);  /* [b2, b2] */
        v128_t b3 = wasm_v128_load(pb + 6);  /* [b3, b3] */

        c0_lo = wasm_f64x2_add(c0_lo, wasm_f64x2_mul(a_lo, b0));
        c0_hi = wasm_f64x2_add(c0_hi, wasm_f64x2_mul(a_hi, b0));
        c1_lo = wasm_f64x2_add(c1_lo, wasm_f64x2_mul(a_lo, b1));
        c1_hi = wasm_f64x2_add(c1_hi, wasm_f64x2_mul(a_hi, b1));
        c2_lo = wasm_f64x2_add(c2_lo, wasm_f64x2_mul(a_lo, b2));
        c2_hi = wasm_f64x2_add(c2_hi, wasm_f64x2_mul(a_hi, b2));
        c3_lo = wasm_f64x2_add(c3_lo, wasm_f64x2_mul(a_lo, b3));
        c3_hi = wasm_f64x2_add(c3_hi, wasm_f64x2_mul(a_hi, b3));

        pa += MR;
        pb += NR_DUP;
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
        if (i < mc) {
            micro_kernel_edge(mc - i, NR, kc, alpha,
                              pa + i * kc,
                              pb + j * kc * 2,
                              C + i + j * ldc, ldc);
        }
    }

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

/* ============= Thread Work Structure ============= */

typedef struct {
    int tid;
    int32_t m, n, k;
    double alpha;
    const double *A;
    int32_t lda;
    const double *B;
    int32_t ldb;
    double *C;
    int32_t ldc;
    int64_t jc_start;
    int64_t jc_end;
} thread_work_t;

static void* thread_dgemm_sse(void *arg) {
    thread_work_t *work = (thread_work_t*)arg;

    int tid = work->tid;
    int32_t m = work->m;
    int32_t n = work->n;
    int32_t k = work->k;
    double alpha = work->alpha;
    const double *A = work->A;
    int32_t lda = work->lda;
    const double *B = work->B;
    int32_t ldb = work->ldb;
    double *C = work->C;
    int32_t ldc = work->ldc;

    ensure_thread_buffers(tid, MC, KC, NC);
    double *pack_a = thread_bufs[tid].pack_a;
    double *pack_b = thread_bufs[tid].pack_b;

    for (int64_t jc = work->jc_start; jc < work->jc_end && jc < n; jc += NC) {
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

    return NULL;
}

/* ============= Multithreaded DGEMM ============= */

__attribute__((export_name("dgemm_sse_mt")))
void dgemm_sse_mt(int32_t m, int32_t n, int32_t k,
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

    int nthreads = num_threads;
    int64_t total_nc_blocks = (n + NC - 1) / NC;

    if (nthreads > total_nc_blocks) {
        nthreads = total_nc_blocks;
    }
    if (nthreads < 1) nthreads = 1;

    /* For small matrices, use single thread */
    if (n < 256 || m < 256 || nthreads == 1) {
        ensure_thread_buffers(0, MC, KC, NC);
        double *pack_a = thread_bufs[0].pack_a;
        double *pack_b = thread_bufs[0].pack_b;

        for (int64_t jc = 0; jc < n; jc += NC) {
            int64_t nc = (jc + NC <= n) ? NC : (n - jc);

            for (int64_t pc = 0; pc < k; pc += KC) {
                int64_t kc = (pc + KC <= k) ? KC : (k - pc);

                pack_panel_b_sse(kc, nc, B + pc + jc * ldb, ldb, pack_b);

                for (int64_t ic = 0; ic < m; ic += MC) {
                    int64_t mc = (ic + MC <= m) ? MC : (m - ic);

                    pack_panel_a(mc, kc, A + ic + pc * lda, lda, pack_a);
                    macro_kernel(mc, nc, kc, alpha, pack_a, pack_b,
                                 C + ic + jc * ldc, ldc);
                }
            }
        }
        return;
    }

    /* Create threads */
    pthread_t threads[MAX_THREADS];
    thread_work_t work[MAX_THREADS];

    int64_t blocks_per_thread = (total_nc_blocks + nthreads - 1) / nthreads;

    for (int t = 0; t < nthreads; t++) {
        work[t].tid = t;
        work[t].m = m;
        work[t].n = n;
        work[t].k = k;
        work[t].alpha = alpha;
        work[t].A = A;
        work[t].lda = lda;
        work[t].B = B;
        work[t].ldb = ldb;
        work[t].C = C;
        work[t].ldc = ldc;

        work[t].jc_start = t * blocks_per_thread * NC;
        work[t].jc_end = (t + 1) * blocks_per_thread * NC;
        if (work[t].jc_end > n) work[t].jc_end = n;

        if (t == 0) {
            continue;  /* Main thread does first chunk */
        }

        pthread_create(&threads[t], NULL, thread_dgemm_sse, &work[t]);
    }

    /* Main thread processes its chunk */
    thread_dgemm_sse(&work[0]);

    /* Wait for other threads */
    for (int t = 1; t < nthreads; t++) {
        pthread_join(threads[t], NULL);
    }
}

/* ============= Row-Major Interface ============= */

__attribute__((export_name("matmul_sse_mt")))
void matmul_sse_mt(int32_t M, int32_t N, int32_t K,
                   const double *A, const double *B, double *C) {
    dgemm_sse_mt(N, M, K,
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
    for (int i = 0; i < MAX_THREADS; i++) {
        free(thread_bufs[i].pack_a);
        free(thread_bufs[i].pack_b);
        thread_bufs[i].pack_a = NULL;
        thread_bufs[i].pack_b = NULL;
        thread_bufs[i].pack_a_size = 0;
        thread_bufs[i].pack_b_size = 0;
    }
}
