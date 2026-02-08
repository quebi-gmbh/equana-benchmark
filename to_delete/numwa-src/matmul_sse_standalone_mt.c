/**
 * Standalone SSE-style matmul - Multi-threaded version
 * Single TU for full inlining, with pthread parallelization.
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <wasm_simd128.h>
#include <pthread.h>

#define MC 64
#define KC 128
#define NC 256
#define MR 4
#define NR 4
#define NR_DUP 8  /* Pre-duplicated */

/* Thread-local storage for work buffers */
static __thread double *tls_pack_a = NULL;
static __thread double *tls_pack_b = NULL;
static __thread size_t tls_pack_a_size = 0;
static __thread size_t tls_pack_b_size = 0;

/* Global thread count */
static int num_threads = 4;

static void ensure_tls_buffers(int64_t mc, int64_t kc, int64_t nc) {
    size_t need_a = (mc + MR) * (kc + 4) * sizeof(double);
    size_t need_b = (kc + 4) * (nc + NR) * 2 * sizeof(double);

    if (tls_pack_a_size < need_a) {
        free(tls_pack_a);
        tls_pack_a = (double*)malloc(need_a);
        tls_pack_a_size = need_a;
    }
    if (tls_pack_b_size < need_b) {
        free(tls_pack_b);
        tls_pack_b = (double*)malloc(need_b);
        tls_pack_b_size = need_b;
    }
}

static void pack_panel_a(int64_t mc, int64_t kc, const double *A, int64_t lda, double *pa) {
    for (int64_t i = 0; i + MR <= mc; i += MR) {
        const double *a_ptr = A + i;
        for (int64_t j = 0; j < kc; j++) {
            pa[0] = a_ptr[0];
            pa[1] = a_ptr[1];
            pa[2] = a_ptr[2];
            pa[3] = a_ptr[3];
            a_ptr += lda;
            pa += MR;
        }
    }
    int64_t rem = mc % MR;
    if (rem) {
        int64_t i = mc - rem;
        const double *a_ptr = A + i;
        for (int64_t j = 0; j < kc; j++) {
            for (int64_t ii = 0; ii < rem; ii++) pa[ii] = a_ptr[ii];
            for (int64_t ii = rem; ii < MR; ii++) pa[ii] = 0.0;
            a_ptr += lda;
            pa += MR;
        }
    }
}

static void pack_panel_b_sse(int64_t kc, int64_t nc, const double *B, int64_t ldb, double *pb) {
    for (int64_t j = 0; j + NR <= nc; j += NR) {
        const double *b0 = B + j * ldb;
        const double *b1 = b0 + ldb;
        const double *b2 = b1 + ldb;
        const double *b3 = b2 + ldb;

        for (int64_t i = 0; i < kc; i++) {
            pb[0] = pb[1] = b0[i];
            pb[2] = pb[3] = b1[i];
            pb[4] = pb[5] = b2[i];
            pb[6] = pb[7] = b3[i];
            pb += NR_DUP;
        }
    }
    int64_t rem = nc % NR;
    if (rem) {
        int64_t j = nc - rem;
        for (int64_t i = 0; i < kc; i++) {
            for (int64_t jj = 0; jj < rem; jj++) {
                double val = B[(j + jj) * ldb + i];
                pb[jj * 2] = pb[jj * 2 + 1] = val;
            }
            for (int64_t jj = rem; jj < NR; jj++) {
                pb[jj * 2] = pb[jj * 2 + 1] = 0.0;
            }
            pb += NR_DUP;
        }
    }
}

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
        v128_t a_lo = wasm_v128_load(pa);
        v128_t a_hi = wasm_v128_load(pa + 2);

        v128_t b0 = wasm_v128_load(pb);
        v128_t b1 = wasm_v128_load(pb + 2);
        v128_t b2 = wasm_v128_load(pb + 4);
        v128_t b3 = wasm_v128_load(pb + 6);

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
    t0 = wasm_v128_load(c0); t1 = wasm_v128_load(c0 + 2);
    wasm_v128_store(c0, wasm_f64x2_add(t0, c0_lo));
    wasm_v128_store(c0 + 2, wasm_f64x2_add(t1, c0_hi));

    t0 = wasm_v128_load(c1); t1 = wasm_v128_load(c1 + 2);
    wasm_v128_store(c1, wasm_f64x2_add(t0, c1_lo));
    wasm_v128_store(c1 + 2, wasm_f64x2_add(t1, c1_hi));

    t0 = wasm_v128_load(c2); t1 = wasm_v128_load(c2 + 2);
    wasm_v128_store(c2, wasm_f64x2_add(t0, c2_lo));
    wasm_v128_store(c2 + 2, wasm_f64x2_add(t1, c2_hi));

    t0 = wasm_v128_load(c3); t1 = wasm_v128_load(c3 + 2);
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
                sum += pa[k * MR + i] * pb[k * NR_DUP + j * 2];
            }
            C[i + j * ldc] += alpha * sum;
        }
    }
}

static void macro_kernel(int64_t mc, int64_t nc, int64_t kc, double alpha,
                          const double *pa, const double *pb,
                          double *C, int64_t ldc) {
    int64_t i, j;

    for (j = 0; j + NR <= nc; j += NR) {
        for (i = 0; i + MR <= mc; i += MR) {
            micro_kernel_4x4_sse(kc, alpha,
                                  pa + i * kc,
                                  pb + j * kc * 2,
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

/* Thread work structure */
typedef struct {
    int32_t m, n, k;
    double alpha;
    const double *A;
    int32_t lda;
    const double *B;
    int32_t ldb;
    double *C;
    int32_t ldc;
    int64_t n_start, n_end;
} thread_work_t;

static void* thread_gemm(void *arg) {
    thread_work_t *work = (thread_work_t*)arg;

    int32_t m = work->m;
    int32_t k = work->k;
    double alpha = work->alpha;
    const double *A = work->A;
    int32_t lda = work->lda;
    const double *B = work->B;
    int32_t ldb = work->ldb;
    double *C = work->C;
    int32_t ldc = work->ldc;
    int64_t n_start = work->n_start;
    int64_t n_end = work->n_end;

    ensure_tls_buffers(MC, KC, NC);

    for (int64_t jc = n_start; jc < n_end; jc += NC) {
        int64_t nc = (jc + NC <= n_end) ? NC : (n_end - jc);

        for (int64_t pc = 0; pc < k; pc += KC) {
            int64_t kc = (pc + KC <= k) ? KC : (k - pc);

            pack_panel_b_sse(kc, nc, B + pc + jc * ldb, ldb, tls_pack_b);

            for (int64_t ic = 0; ic < m; ic += MC) {
                int64_t mc = (ic + MC <= m) ? MC : (m - ic);

                pack_panel_a(mc, kc, A + ic + pc * lda, lda, tls_pack_a);

                macro_kernel(mc, nc, kc, alpha, tls_pack_a, tls_pack_b,
                             C + ic + jc * ldc, ldc);
            }
        }
    }

    return NULL;
}

__attribute__((export_name("dgemm_standalone_mt")))
void dgemm_standalone_mt(int32_t m, int32_t n, int32_t k,
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
    if (nthreads <= 1 || n < NC) {
        /* Single-threaded fallback */
        ensure_tls_buffers(MC, KC, NC);

        for (int64_t jc = 0; jc < n; jc += NC) {
            int64_t nc = (jc + NC <= n) ? NC : (n - jc);

            for (int64_t pc = 0; pc < k; pc += KC) {
                int64_t kc = (pc + KC <= k) ? KC : (k - pc);

                pack_panel_b_sse(kc, nc, B + pc + jc * ldb, ldb, tls_pack_b);

                for (int64_t ic = 0; ic < m; ic += MC) {
                    int64_t mc = (ic + MC <= m) ? MC : (m - ic);

                    pack_panel_a(mc, kc, A + ic + pc * lda, lda, tls_pack_a);

                    macro_kernel(mc, nc, kc, alpha, tls_pack_a, tls_pack_b,
                                 C + ic + jc * ldc, ldc);
                }
            }
        }
        return;
    }

    /* Multi-threaded: partition N dimension */
    pthread_t *threads = (pthread_t*)malloc(nthreads * sizeof(pthread_t));
    thread_work_t *works = (thread_work_t*)malloc(nthreads * sizeof(thread_work_t));

    int64_t n_per_thread = (n + nthreads - 1) / nthreads;
    /* Round to NC boundary for better cache behavior */
    n_per_thread = ((n_per_thread + NC - 1) / NC) * NC;

    int actual_threads = 0;
    for (int t = 0; t < nthreads; t++) {
        int64_t n_start = t * n_per_thread;
        if (n_start >= n) break;
        int64_t n_end = n_start + n_per_thread;
        if (n_end > n) n_end = n;

        works[t].m = m;
        works[t].n = n;
        works[t].k = k;
        works[t].alpha = alpha;
        works[t].A = A;
        works[t].lda = lda;
        works[t].B = B;
        works[t].ldb = ldb;
        works[t].C = C;
        works[t].ldc = ldc;
        works[t].n_start = n_start;
        works[t].n_end = n_end;

        pthread_create(&threads[t], NULL, thread_gemm, &works[t]);
        actual_threads++;
    }

    for (int t = 0; t < actual_threads; t++) {
        pthread_join(threads[t], NULL);
    }

    free(threads);
    free(works);
}

__attribute__((export_name("matmul_standalone_mt")))
void matmul_standalone_mt(int32_t M, int32_t N, int32_t K,
                const double *A, const double *B, double *C) {
    dgemm_standalone_mt(N, M, K, 1.0, B, N, A, K, 0.0, C, N);
}

__attribute__((export_name("set_num_threads")))
void set_num_threads(int32_t n) {
    if (n < 1) n = 1;
    num_threads = n;
}

__attribute__((export_name("get_num_threads")))
int32_t get_num_threads(void) {
    return num_threads;
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
    free(tls_pack_a);
    free(tls_pack_b);
    tls_pack_a = NULL;
    tls_pack_b = NULL;
    tls_pack_a_size = 0;
    tls_pack_b_size = 0;
}
