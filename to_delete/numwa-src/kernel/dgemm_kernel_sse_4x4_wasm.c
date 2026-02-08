/**
 * WebAssembly SIMD DGEMM Micro-Kernel (SSE-style with Pre-Duplicated B)
 *
 * Key optimization: B is pre-duplicated during packing, so we can load
 * [bi, bi] directly without extract_lane + splat operations.
 *
 * Standard B packing:  [b0, b1, b2, b3] per k iteration (4 doubles)
 * SSE-style B packing: [b0, b0, b1, b1, b2, b2, b3, b3] per k iteration (8 doubles)
 *
 * This trades 2x memory bandwidth for B against eliminating expensive
 * extract_lane + splat operations in the inner loop.
 *
 * Operation: C[M][N] += alpha * sa[M][K] * sb[K][N]
 *
 * Data Layout:
 *   sa: column-major packed (leading dimension = 4), same as standard kernel
 *       [a00, a10, a20, a30, a01, a11, a21, a31, ...]
 *   sb: row-major packed with pre-duplication
 *       [b00, b00, b01, b01, b02, b02, b03, b03, b10, b10, ...]
 *   C:  column-major (leading dimension = LDC)
 */

#include <wasm_simd128.h>

typedef double FLOAT;
typedef long long BLASLONG;

/*===========================================================================
 * Helper: Store 4 rows to 1 column of C
 * C[0:4] += alpha * [up[0], up[1], down[0], down[1]]
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
 * C[0] += alpha * vc[0];  C[LDC] += alpha * vc[1];
 *===========================================================================*/
static inline void dgemm_store_m1n2(FLOAT *C, v128_t vc, FLOAT alpha, BLASLONG LDC) {
    double c0 = wasm_f64x2_extract_lane(vc, 0);
    double c1 = wasm_f64x2_extract_lane(vc, 1);
    C[0] += c0 * alpha;
    C[LDC] += c1 * alpha;
}

/*===========================================================================
 * m4n8_sse: 4 rows x 8 columns kernel (SSE-style)
 * Uses 16 v128 accumulators (8 columns x 2 vectors per column)
 * B is stored as 2 concatenated 4-column blocks, each pre-duplicated
 *===========================================================================*/
static inline void dgemm_kernel_wasm_m4n8_sse(
    const FLOAT *sa, const FLOAT *sb, FLOAT *C,
    BLASLONG K, BLASLONG LDC, FLOAT alpha) {

    /* For 8 columns, sb has layout:
     * First 4 cols: [b0,b0,b1,b1,b2,b2,b3,b3] per k (8 doubles)
     * Next 4 cols:  [b4,b4,b5,b5,b6,b6,b7,b7] per k (8 doubles)
     */
    const FLOAT *b1_ = sb;
    const FLOAT *b2_ = sb + K * 8;  /* 8 doubles per k for 4 cols */

    /* 8 columns, each needs 2 v128 (up=rows0-1, down=rows2-3) */
    v128_t c11, c21, c12, c22, c13, c23, c14, c24;
    v128_t c15, c25, c16, c26, c17, c27, c18, c28;

    c11 = c21 = c12 = c22 = c13 = c23 = c14 = c24 = wasm_f64x2_splat(0.0);
    c15 = c25 = c16 = c26 = c17 = c27 = c18 = c28 = wasm_f64x2_splat(0.0);

    for (BLASLONG k = 0; k < K; k++) {
        v128_t a1 = wasm_v128_load(sa);      /* rows 0-1 */
        v128_t a2 = wasm_v128_load(sa + 2);  /* rows 2-3 */
        sa += 4;

        /* First 4 columns from b1_ (pre-duplicated) */
        v128_t b0 = wasm_v128_load(b1_);      /* [b0, b0] */
        v128_t b1 = wasm_v128_load(b1_ + 2);  /* [b1, b1] */
        v128_t b2 = wasm_v128_load(b1_ + 4);  /* [b2, b2] */
        v128_t b3 = wasm_v128_load(b1_ + 6);  /* [b3, b3] */
        b1_ += 8;

        c11 = wasm_f64x2_add(c11, wasm_f64x2_mul(a1, b0));
        c21 = wasm_f64x2_add(c21, wasm_f64x2_mul(a2, b0));
        c12 = wasm_f64x2_add(c12, wasm_f64x2_mul(a1, b1));
        c22 = wasm_f64x2_add(c22, wasm_f64x2_mul(a2, b1));
        c13 = wasm_f64x2_add(c13, wasm_f64x2_mul(a1, b2));
        c23 = wasm_f64x2_add(c23, wasm_f64x2_mul(a2, b2));
        c14 = wasm_f64x2_add(c14, wasm_f64x2_mul(a1, b3));
        c24 = wasm_f64x2_add(c24, wasm_f64x2_mul(a2, b3));

        /* Second 4 columns from b2_ (pre-duplicated) */
        v128_t b4 = wasm_v128_load(b2_);      /* [b4, b4] */
        v128_t b5 = wasm_v128_load(b2_ + 2);  /* [b5, b5] */
        v128_t b6 = wasm_v128_load(b2_ + 4);  /* [b6, b6] */
        v128_t b7 = wasm_v128_load(b2_ + 6);  /* [b7, b7] */
        b2_ += 8;

        c15 = wasm_f64x2_add(c15, wasm_f64x2_mul(a1, b4));
        c25 = wasm_f64x2_add(c25, wasm_f64x2_mul(a2, b4));
        c16 = wasm_f64x2_add(c16, wasm_f64x2_mul(a1, b5));
        c26 = wasm_f64x2_add(c26, wasm_f64x2_mul(a2, b5));
        c17 = wasm_f64x2_add(c17, wasm_f64x2_mul(a1, b6));
        c27 = wasm_f64x2_add(c27, wasm_f64x2_mul(a2, b6));
        c18 = wasm_f64x2_add(c18, wasm_f64x2_mul(a1, b7));
        c28 = wasm_f64x2_add(c28, wasm_f64x2_mul(a2, b7));
    }

    /* Store results */
    dgemm_store_m4n1(C, c11, c21, alpha); C += LDC;
    dgemm_store_m4n1(C, c12, c22, alpha); C += LDC;
    dgemm_store_m4n1(C, c13, c23, alpha); C += LDC;
    dgemm_store_m4n1(C, c14, c24, alpha); C += LDC;
    dgemm_store_m4n1(C, c15, c25, alpha); C += LDC;
    dgemm_store_m4n1(C, c16, c26, alpha); C += LDC;
    dgemm_store_m4n1(C, c17, c27, alpha); C += LDC;
    dgemm_store_m4n1(C, c18, c28, alpha);
}

/*===========================================================================
 * m4n4_sse: 4 rows x 4 columns kernel (SSE-style, primary workhorse)
 * Uses 8 v128 accumulators
 * B pre-duplicated: [b0,b0,b1,b1,b2,b2,b3,b3] per k
 *===========================================================================*/
static inline void dgemm_kernel_wasm_m4n4_sse(
    const FLOAT *sa, const FLOAT *sb, FLOAT *C,
    BLASLONG K, BLASLONG LDC, FLOAT alpha) {

    /* 4 columns, each stored as 2 vectors (up=rows0-1, down=rows2-3) */
    v128_t c11, c21, c12, c22, c13, c23, c14, c24;
    c11 = c21 = c12 = c22 = c13 = c23 = c14 = c24 = wasm_f64x2_splat(0.0);

    for (BLASLONG k = 0; k < K; k++) {
        v128_t a1 = wasm_v128_load(sa);      /* rows 0-1 */
        v128_t a2 = wasm_v128_load(sa + 2);  /* rows 2-3 */
        sa += 4;

        /* Load B: pre-duplicated, no extract_lane needed! */
        v128_t b0 = wasm_v128_load(sb);      /* [b0, b0] */
        v128_t b1 = wasm_v128_load(sb + 2);  /* [b1, b1] */
        v128_t b2 = wasm_v128_load(sb + 4);  /* [b2, b2] */
        v128_t b3 = wasm_v128_load(sb + 6);  /* [b3, b3] */
        sb += 8;  /* 8 doubles per k for SSE-style */

        /* Column 0: c[*,0] += a[*] * b0 */
        c11 = wasm_f64x2_add(c11, wasm_f64x2_mul(a1, b0));
        c21 = wasm_f64x2_add(c21, wasm_f64x2_mul(a2, b0));

        /* Column 1: c[*,1] += a[*] * b1 */
        c12 = wasm_f64x2_add(c12, wasm_f64x2_mul(a1, b1));
        c22 = wasm_f64x2_add(c22, wasm_f64x2_mul(a2, b1));

        /* Column 2: c[*,2] += a[*] * b2 */
        c13 = wasm_f64x2_add(c13, wasm_f64x2_mul(a1, b2));
        c23 = wasm_f64x2_add(c23, wasm_f64x2_mul(a2, b2));

        /* Column 3: c[*,3] += a[*] * b3 */
        c14 = wasm_f64x2_add(c14, wasm_f64x2_mul(a1, b3));
        c24 = wasm_f64x2_add(c24, wasm_f64x2_mul(a2, b3));
    }

    /* Store results */
    dgemm_store_m4n1(C, c11, c21, alpha); C += LDC;
    dgemm_store_m4n1(C, c12, c22, alpha); C += LDC;
    dgemm_store_m4n1(C, c13, c23, alpha); C += LDC;
    dgemm_store_m4n1(C, c14, c24, alpha);
}

/*===========================================================================
 * m4n2_sse: 4 rows x 2 columns kernel (SSE-style)
 * B pre-duplicated: [b0,b0,b1,b1] per k (4 doubles)
 *===========================================================================*/
static inline void dgemm_kernel_wasm_m4n2_sse(
    const FLOAT *sa, const FLOAT *sb, FLOAT *C,
    BLASLONG K, BLASLONG LDC, FLOAT alpha) {

    v128_t c11_1, c11_2, c21_1, c21_2, c12_1, c12_2, c22_1, c22_2;
    c11_1 = c11_2 = c21_1 = c21_2 = wasm_f64x2_splat(0.0);
    c12_1 = c12_2 = c22_1 = c22_2 = wasm_f64x2_splat(0.0);

    BLASLONG k = K;
    for (; k > 1; k -= 2) {
        /* Iteration 1 */
        v128_t a1_1 = wasm_v128_load(sa);
        v128_t a2_1 = wasm_v128_load(sa + 2);
        /* Iteration 2 */
        v128_t a1_2 = wasm_v128_load(sa + 4);
        v128_t a2_2 = wasm_v128_load(sa + 6);
        sa += 8;

        /* B: [b0,b0,b1,b1] for k=0, then [b0,b0,b1,b1] for k=1 */
        v128_t b0_1 = wasm_v128_load(sb);      /* [b0, b0] k=0 */
        v128_t b1_1 = wasm_v128_load(sb + 2);  /* [b1, b1] k=0 */
        v128_t b0_2 = wasm_v128_load(sb + 4);  /* [b0, b0] k=1 */
        v128_t b1_2 = wasm_v128_load(sb + 6);  /* [b1, b1] k=1 */
        sb += 8;  /* 4 doubles per k * 2 iterations */

        c11_1 = wasm_f64x2_add(c11_1, wasm_f64x2_mul(a1_1, b0_1));
        c21_1 = wasm_f64x2_add(c21_1, wasm_f64x2_mul(a2_1, b0_1));
        c12_1 = wasm_f64x2_add(c12_1, wasm_f64x2_mul(a1_1, b1_1));
        c22_1 = wasm_f64x2_add(c22_1, wasm_f64x2_mul(a2_1, b1_1));

        c11_2 = wasm_f64x2_add(c11_2, wasm_f64x2_mul(a1_2, b0_2));
        c21_2 = wasm_f64x2_add(c21_2, wasm_f64x2_mul(a2_2, b0_2));
        c12_2 = wasm_f64x2_add(c12_2, wasm_f64x2_mul(a1_2, b1_2));
        c22_2 = wasm_f64x2_add(c22_2, wasm_f64x2_mul(a2_2, b1_2));
    }

    /* Combine accumulators */
    c11_1 = wasm_f64x2_add(c11_1, c11_2);
    c21_1 = wasm_f64x2_add(c21_1, c21_2);
    c12_1 = wasm_f64x2_add(c12_1, c12_2);
    c22_1 = wasm_f64x2_add(c22_1, c22_2);

    /* Handle remaining k=1 */
    if (k) {
        v128_t a1 = wasm_v128_load(sa);
        v128_t a2 = wasm_v128_load(sa + 2);
        sa += 4;

        v128_t b0 = wasm_v128_load(sb);      /* [b0, b0] */
        v128_t b1 = wasm_v128_load(sb + 2);  /* [b1, b1] */
        sb += 4;

        c11_1 = wasm_f64x2_add(c11_1, wasm_f64x2_mul(a1, b0));
        c21_1 = wasm_f64x2_add(c21_1, wasm_f64x2_mul(a2, b0));
        c12_1 = wasm_f64x2_add(c12_1, wasm_f64x2_mul(a1, b1));
        c22_1 = wasm_f64x2_add(c22_1, wasm_f64x2_mul(a2, b1));
    }

    dgemm_store_m4n1(C, c11_1, c21_1, alpha); C += LDC;
    dgemm_store_m4n1(C, c12_1, c22_1, alpha);
}

/*===========================================================================
 * m4n1_sse: 4 rows x 1 column kernel (SSE-style)
 * B pre-duplicated: [b0,b0] per k (2 doubles)
 *===========================================================================*/
static inline void dgemm_kernel_wasm_m4n1_sse(
    const FLOAT *sa, const FLOAT *sb, FLOAT *C,
    BLASLONG K, BLASLONG LDC, FLOAT alpha) {

    v128_t c11_1, c11_2, c21_1, c21_2;
    c11_1 = c11_2 = c21_1 = c21_2 = wasm_f64x2_splat(0.0);

    BLASLONG k = K;
    for (; k > 1; k -= 2) {
        v128_t b0 = wasm_v128_load(sb);  /* [b0_k0, b0_k0] then [b0_k1, b0_k1] */
        v128_t b1 = wasm_v128_load(sb + 2);
        sb += 4;

        v128_t b0_splat = wasm_f64x2_splat(wasm_f64x2_extract_lane(b0, 0));
        v128_t b1_splat = wasm_f64x2_splat(wasm_f64x2_extract_lane(b1, 0));

        c11_1 = wasm_f64x2_add(c11_1, wasm_f64x2_mul(wasm_v128_load(sa), b0_splat));
        c21_1 = wasm_f64x2_add(c21_1, wasm_f64x2_mul(wasm_v128_load(sa + 2), b0_splat));
        c11_2 = wasm_f64x2_add(c11_2, wasm_f64x2_mul(wasm_v128_load(sa + 4), b1_splat));
        c21_2 = wasm_f64x2_add(c21_2, wasm_f64x2_mul(wasm_v128_load(sa + 6), b1_splat));
        sa += 8;
    }

    c11_1 = wasm_f64x2_add(c11_1, c11_2);
    c21_1 = wasm_f64x2_add(c21_1, c21_2);

    if (k) {
        double b1 = sb[0];  /* Pre-duplicated, just take first */
        v128_t b_splat = wasm_f64x2_splat(b1);
        c11_1 = wasm_f64x2_add(c11_1, wasm_f64x2_mul(wasm_v128_load(sa), b_splat));
        c21_1 = wasm_f64x2_add(c21_1, wasm_f64x2_mul(wasm_v128_load(sa + 2), b_splat));
        sa += 4;
    }

    dgemm_store_m4n1(C, c11_1, c21_1, alpha);
}

/*===========================================================================
 * m2n8_sse: 2 rows x 8 columns kernel (SSE-style)
 *===========================================================================*/
static inline void dgemm_kernel_wasm_m2n8_sse(
    const FLOAT *sa, const FLOAT *sb, FLOAT *C,
    BLASLONG K, BLASLONG LDC, FLOAT alpha) {

    v128_t c01, c02, c03, c04, c11, c12, c13, c14;
    c01 = c02 = c03 = c04 = c11 = c12 = c13 = c14 = wasm_f64x2_splat(0.0);

    const FLOAT *b1_ = sb;
    const FLOAT *b2_ = sb + 8 * K;  /* 8 doubles per k for first 4 cols */

    for (BLASLONG k = 0; k < K; k++) {
        v128_t a1 = wasm_v128_load(sa);
        sa += 2;

        /* First 4 columns (pre-duplicated) */
        v128_t b0 = wasm_v128_load(b1_);      /* [b0, b0] */
        v128_t b1 = wasm_v128_load(b1_ + 2);  /* [b1, b1] */
        v128_t b2 = wasm_v128_load(b1_ + 4);  /* [b2, b2] */
        v128_t b3 = wasm_v128_load(b1_ + 6);  /* [b3, b3] */
        b1_ += 8;

        c01 = wasm_f64x2_add(c01, wasm_f64x2_mul(a1, b0));
        c02 = wasm_f64x2_add(c02, wasm_f64x2_mul(a1, b1));
        c03 = wasm_f64x2_add(c03, wasm_f64x2_mul(a1, b2));
        c04 = wasm_f64x2_add(c04, wasm_f64x2_mul(a1, b3));

        /* Second 4 columns (pre-duplicated) */
        v128_t b4 = wasm_v128_load(b2_);      /* [b4, b4] */
        v128_t b5 = wasm_v128_load(b2_ + 2);  /* [b5, b5] */
        v128_t b6 = wasm_v128_load(b2_ + 4);  /* [b6, b6] */
        v128_t b7 = wasm_v128_load(b2_ + 6);  /* [b7, b7] */
        b2_ += 8;

        c11 = wasm_f64x2_add(c11, wasm_f64x2_mul(a1, b4));
        c12 = wasm_f64x2_add(c12, wasm_f64x2_mul(a1, b5));
        c13 = wasm_f64x2_add(c13, wasm_f64x2_mul(a1, b6));
        c14 = wasm_f64x2_add(c14, wasm_f64x2_mul(a1, b7));
    }

    v128_t alpha_v = wasm_f64x2_splat(alpha);
    v128_t t;

    t = wasm_v128_load(C);
    wasm_v128_store(C, wasm_f64x2_add(t, wasm_f64x2_mul(c01, alpha_v)));
    C += LDC;
    t = wasm_v128_load(C);
    wasm_v128_store(C, wasm_f64x2_add(t, wasm_f64x2_mul(c02, alpha_v)));
    C += LDC;
    t = wasm_v128_load(C);
    wasm_v128_store(C, wasm_f64x2_add(t, wasm_f64x2_mul(c03, alpha_v)));
    C += LDC;
    t = wasm_v128_load(C);
    wasm_v128_store(C, wasm_f64x2_add(t, wasm_f64x2_mul(c04, alpha_v)));
    C += LDC;
    t = wasm_v128_load(C);
    wasm_v128_store(C, wasm_f64x2_add(t, wasm_f64x2_mul(c11, alpha_v)));
    C += LDC;
    t = wasm_v128_load(C);
    wasm_v128_store(C, wasm_f64x2_add(t, wasm_f64x2_mul(c12, alpha_v)));
    C += LDC;
    t = wasm_v128_load(C);
    wasm_v128_store(C, wasm_f64x2_add(t, wasm_f64x2_mul(c13, alpha_v)));
    C += LDC;
    t = wasm_v128_load(C);
    wasm_v128_store(C, wasm_f64x2_add(t, wasm_f64x2_mul(c14, alpha_v)));
}

/*===========================================================================
 * m2n4_sse: 2 rows x 4 columns kernel (SSE-style)
 *===========================================================================*/
static inline void dgemm_kernel_wasm_m2n4_sse(
    const FLOAT *sa, const FLOAT *sb, FLOAT *C,
    BLASLONG K, BLASLONG LDC, FLOAT alpha) {

    v128_t c1_1, c1_2, c2_1, c2_2, c3_1, c3_2, c4_1, c4_2;
    c1_1 = c1_2 = c2_1 = c2_2 = c3_1 = c3_2 = c4_1 = c4_2 = wasm_f64x2_splat(0.0);

    BLASLONG k = K;
    for (; k > 1; k -= 2) {
        v128_t a1 = wasm_v128_load(sa);
        v128_t a2 = wasm_v128_load(sa + 2);
        sa += 4;

        /* k=0: [b0,b0,b1,b1,b2,b2,b3,b3] */
        v128_t b0_1 = wasm_v128_load(sb);
        v128_t b1_1 = wasm_v128_load(sb + 2);
        v128_t b2_1 = wasm_v128_load(sb + 4);
        v128_t b3_1 = wasm_v128_load(sb + 6);
        /* k=1: [b0,b0,b1,b1,b2,b2,b3,b3] */
        v128_t b0_2 = wasm_v128_load(sb + 8);
        v128_t b1_2 = wasm_v128_load(sb + 10);
        v128_t b2_2 = wasm_v128_load(sb + 12);
        v128_t b3_2 = wasm_v128_load(sb + 14);
        sb += 16;

        c1_1 = wasm_f64x2_add(c1_1, wasm_f64x2_mul(a1, b0_1));
        c2_1 = wasm_f64x2_add(c2_1, wasm_f64x2_mul(a1, b1_1));
        c3_1 = wasm_f64x2_add(c3_1, wasm_f64x2_mul(a1, b2_1));
        c4_1 = wasm_f64x2_add(c4_1, wasm_f64x2_mul(a1, b3_1));

        c1_2 = wasm_f64x2_add(c1_2, wasm_f64x2_mul(a2, b0_2));
        c2_2 = wasm_f64x2_add(c2_2, wasm_f64x2_mul(a2, b1_2));
        c3_2 = wasm_f64x2_add(c3_2, wasm_f64x2_mul(a2, b2_2));
        c4_2 = wasm_f64x2_add(c4_2, wasm_f64x2_mul(a2, b3_2));
    }

    c1_1 = wasm_f64x2_add(c1_1, c1_2);
    c2_1 = wasm_f64x2_add(c2_1, c2_2);
    c3_1 = wasm_f64x2_add(c3_1, c3_2);
    c4_1 = wasm_f64x2_add(c4_1, c4_2);

    if (k) {
        v128_t a1 = wasm_v128_load(sa);
        sa += 2;
        v128_t b0 = wasm_v128_load(sb);
        v128_t b1 = wasm_v128_load(sb + 2);
        v128_t b2 = wasm_v128_load(sb + 4);
        v128_t b3 = wasm_v128_load(sb + 6);
        sb += 8;

        c1_1 = wasm_f64x2_add(c1_1, wasm_f64x2_mul(a1, b0));
        c2_1 = wasm_f64x2_add(c2_1, wasm_f64x2_mul(a1, b1));
        c3_1 = wasm_f64x2_add(c3_1, wasm_f64x2_mul(a1, b2));
        c4_1 = wasm_f64x2_add(c4_1, wasm_f64x2_mul(a1, b3));
    }

    v128_t alpha_v = wasm_f64x2_splat(alpha);
    v128_t t;

    t = wasm_v128_load(C);
    wasm_v128_store(C, wasm_f64x2_add(t, wasm_f64x2_mul(c1_1, alpha_v)));
    C += LDC;
    t = wasm_v128_load(C);
    wasm_v128_store(C, wasm_f64x2_add(t, wasm_f64x2_mul(c2_1, alpha_v)));
    C += LDC;
    t = wasm_v128_load(C);
    wasm_v128_store(C, wasm_f64x2_add(t, wasm_f64x2_mul(c3_1, alpha_v)));
    C += LDC;
    t = wasm_v128_load(C);
    wasm_v128_store(C, wasm_f64x2_add(t, wasm_f64x2_mul(c4_1, alpha_v)));
}

/*===========================================================================
 * m2n2_sse: 2 rows x 2 columns kernel (SSE-style)
 *===========================================================================*/
static inline void dgemm_kernel_wasm_m2n2_sse(
    const FLOAT *sa, const FLOAT *sb, FLOAT *C,
    BLASLONG K, BLASLONG LDC, FLOAT alpha) {

    v128_t c1_1, c1_2, c2_1, c2_2;
    c1_1 = c1_2 = c2_1 = c2_2 = wasm_f64x2_splat(0.0);

    BLASLONG k = K;
    for (; k > 1; k -= 2) {
        v128_t a1 = wasm_v128_load(sa);
        v128_t a2 = wasm_v128_load(sa + 2);
        sa += 4;

        /* [b0,b0,b1,b1] per k, 2 k iterations */
        v128_t b0_1 = wasm_v128_load(sb);
        v128_t b1_1 = wasm_v128_load(sb + 2);
        v128_t b0_2 = wasm_v128_load(sb + 4);
        v128_t b1_2 = wasm_v128_load(sb + 6);
        sb += 8;

        c1_1 = wasm_f64x2_add(c1_1, wasm_f64x2_mul(a1, b0_1));
        c2_1 = wasm_f64x2_add(c2_1, wasm_f64x2_mul(a1, b1_1));

        c1_2 = wasm_f64x2_add(c1_2, wasm_f64x2_mul(a2, b0_2));
        c2_2 = wasm_f64x2_add(c2_2, wasm_f64x2_mul(a2, b1_2));
    }

    c1_1 = wasm_f64x2_add(c1_1, c1_2);
    c2_1 = wasm_f64x2_add(c2_1, c2_2);

    if (k) {
        v128_t a1 = wasm_v128_load(sa);
        sa += 2;
        v128_t b0 = wasm_v128_load(sb);
        v128_t b1 = wasm_v128_load(sb + 2);
        sb += 4;

        c1_1 = wasm_f64x2_add(c1_1, wasm_f64x2_mul(a1, b0));
        c2_1 = wasm_f64x2_add(c2_1, wasm_f64x2_mul(a1, b1));
    }

    v128_t alpha_v = wasm_f64x2_splat(alpha);
    v128_t t;

    t = wasm_v128_load(C);
    wasm_v128_store(C, wasm_f64x2_add(t, wasm_f64x2_mul(c1_1, alpha_v)));
    C += LDC;
    t = wasm_v128_load(C);
    wasm_v128_store(C, wasm_f64x2_add(t, wasm_f64x2_mul(c2_1, alpha_v)));
}

/*===========================================================================
 * m2n1_sse: 2 rows x 1 column kernel (SSE-style)
 *===========================================================================*/
static inline void dgemm_kernel_wasm_m2n1_sse(
    const FLOAT *sa, const FLOAT *sb, FLOAT *C,
    BLASLONG K, BLASLONG LDC, FLOAT alpha) {

    v128_t c1, c2, c3, c4;
    c1 = c2 = c3 = c4 = wasm_f64x2_splat(0.0);

    BLASLONG k = K;
    for (; k > 3; k -= 4) {
        /* Pre-duplicated: [b0,b0], [b1,b1], [b2,b2], [b3,b3] */
        v128_t b0 = wasm_v128_load(sb);
        v128_t b1 = wasm_v128_load(sb + 2);
        v128_t b2 = wasm_v128_load(sb + 4);
        v128_t b3 = wasm_v128_load(sb + 6);
        sb += 8;

        c1 = wasm_f64x2_add(c1, wasm_f64x2_mul(wasm_v128_load(sa), b0));
        c2 = wasm_f64x2_add(c2, wasm_f64x2_mul(wasm_v128_load(sa + 2), b1));
        c3 = wasm_f64x2_add(c3, wasm_f64x2_mul(wasm_v128_load(sa + 4), b2));
        c4 = wasm_f64x2_add(c4, wasm_f64x2_mul(wasm_v128_load(sa + 6), b3));
        sa += 8;
    }

    c1 = wasm_f64x2_add(c1, c2);
    c3 = wasm_f64x2_add(c3, c4);
    c1 = wasm_f64x2_add(c1, c3);

    for (; k; k--) {
        v128_t b = wasm_v128_load(sb);  /* [b, b] */
        sb += 2;
        c1 = wasm_f64x2_add(c1, wasm_f64x2_mul(wasm_v128_load(sa), b));
        sa += 2;
    }

    v128_t alpha_v = wasm_f64x2_splat(alpha);
    v128_t t = wasm_v128_load(C);
    wasm_v128_store(C, wasm_f64x2_add(t, wasm_f64x2_mul(c1, alpha_v)));
}

/*===========================================================================
 * m1n8_sse: 1 row x 8 columns kernel (SSE-style)
 *===========================================================================*/
static inline void dgemm_kernel_wasm_m1n8_sse(
    const FLOAT *sa, const FLOAT *sb, FLOAT *C,
    BLASLONG K, BLASLONG LDC, FLOAT alpha) {

    v128_t c1, c2, c3, c4;
    c1 = c2 = c3 = c4 = wasm_f64x2_splat(0.0);

    const FLOAT *b1_ = sb;
    const FLOAT *b2_ = sb + 8 * K;

    for (BLASLONG k = 0; k < K; k++) {
        double a1 = *sa++;
        v128_t a_splat = wasm_f64x2_splat(a1);

        /* For 1 row, we accumulate pairs of columns */
        /* First 4 cols: b is [b0,b0,b1,b1,b2,b2,b3,b3] but we only need one of each pair */
        c1 = wasm_f64x2_add(c1, wasm_f64x2_mul(wasm_v128_load(b1_), a_splat));
        c2 = wasm_f64x2_add(c2, wasm_f64x2_mul(wasm_v128_load(b1_ + 4), a_splat));
        b1_ += 8;
        c3 = wasm_f64x2_add(c3, wasm_f64x2_mul(wasm_v128_load(b2_), a_splat));
        c4 = wasm_f64x2_add(c4, wasm_f64x2_mul(wasm_v128_load(b2_ + 4), a_splat));
        b2_ += 8;
    }

    /* Extract and store - c1 has [col0*a_sum + col0*a_sum, col1*a_sum + col1*a_sum]
       but actually c1[0]=sum(b0*a), c1[1]=sum(b0*a) - both same!
       Need to rethink: with pre-duplicated B, c1 = sum([b0,b0] * [a,a]) = [sum(b0*a), sum(b0*a)]
       So c1[0] is col0 result, c1[1] is also col0. We need to interleave.

       Actually for m1, standard approach is better. Let's use scalar here. */
    double c0 = wasm_f64x2_extract_lane(c1, 0);
    double c1_ = wasm_f64x2_extract_lane(c1, 1);
    double c2_ = wasm_f64x2_extract_lane(c2, 0);
    double c3_ = wasm_f64x2_extract_lane(c2, 1);
    double c4_ = wasm_f64x2_extract_lane(c3, 0);
    double c5_ = wasm_f64x2_extract_lane(c3, 1);
    double c6_ = wasm_f64x2_extract_lane(c4, 0);
    double c7_ = wasm_f64x2_extract_lane(c4, 1);

    /* Wait - with duplicated B: [b0,b0,b1,b1,b2,b2,b3,b3]
       loading [b0,b0] and multiplying by [a,a] gives [b0*a, b0*a]
       So c1 accumulates [sum(b0*a), sum(b0*a)] - both lanes are same!

       For m1, the SSE layout doesn't help. Let's read B differently:
       Load [b0,b0] -> just use lane 0, load [b1,b1] -> use lane 0, etc.
       But that's wasteful. Better to just not use SSE packing advantage for m1. */

    /* Re-accumulate properly for 1-row case */
    c1 = c2 = c3 = c4 = wasm_f64x2_splat(0.0);
    b1_ = sb;
    b2_ = sb + 8 * K;
    const FLOAT *sa2 = sa - K;

    for (BLASLONG k = 0; k < K; k++) {
        double a1 = *sa2++;
        /* Extract individual B values from pre-duplicated layout */
        double b0 = b1_[0];
        double b1 = b1_[2];
        double b2 = b1_[4];
        double b3 = b1_[6];
        double b4 = b2_[0];
        double b5 = b2_[2];
        double b6 = b2_[4];
        double b7 = b2_[6];
        b1_ += 8;
        b2_ += 8;

        c0 += b0 * a1;
        c1_ += b1 * a1;
        c2_ += b2 * a1;
        c3_ += b3 * a1;
        c4_ += b4 * a1;
        c5_ += b5 * a1;
        c6_ += b6 * a1;
        c7_ += b7 * a1;
    }

    C[0] += c0 * alpha; C += LDC;
    C[0] += c1_ * alpha; C += LDC;
    C[0] += c2_ * alpha; C += LDC;
    C[0] += c3_ * alpha; C += LDC;
    C[0] += c4_ * alpha; C += LDC;
    C[0] += c5_ * alpha; C += LDC;
    C[0] += c6_ * alpha; C += LDC;
    C[0] += c7_ * alpha;
}

/*===========================================================================
 * m1n4_sse: 1 row x 4 columns kernel (SSE-style)
 *===========================================================================*/
static inline void dgemm_kernel_wasm_m1n4_sse(
    const FLOAT *sa, const FLOAT *sb, FLOAT *C,
    BLASLONG K, BLASLONG LDC, FLOAT alpha) {

    double c0 = 0.0, c1 = 0.0, c2 = 0.0, c3 = 0.0;

    for (BLASLONG k = 0; k < K; k++) {
        double a1 = *sa++;
        /* Extract from pre-duplicated: [b0,b0,b1,b1,b2,b2,b3,b3] */
        c0 += sb[0] * a1;
        c1 += sb[2] * a1;
        c2 += sb[4] * a1;
        c3 += sb[6] * a1;
        sb += 8;
    }

    C[0] += c0 * alpha; C += LDC;
    C[0] += c1 * alpha; C += LDC;
    C[0] += c2 * alpha; C += LDC;
    C[0] += c3 * alpha;
}

/*===========================================================================
 * m1n2_sse: 1 row x 2 columns kernel (SSE-style)
 *===========================================================================*/
static inline void dgemm_kernel_wasm_m1n2_sse(
    const FLOAT *sa, const FLOAT *sb, FLOAT *C,
    BLASLONG K, BLASLONG LDC, FLOAT alpha) {

    double c0 = 0.0, c1 = 0.0;

    for (BLASLONG k = 0; k < K; k++) {
        double a1 = *sa++;
        /* Extract from pre-duplicated: [b0,b0,b1,b1] */
        c0 += sb[0] * a1;
        c1 += sb[2] * a1;
        sb += 4;
    }

    C[0] += c0 * alpha; C += LDC;
    C[0] += c1 * alpha;
}

/*===========================================================================
 * m1n1_sse: 1 row x 1 column kernel (dot product, SSE-style)
 *===========================================================================*/
static inline void dgemm_kernel_wasm_m1n1_sse(
    const FLOAT *sa, const FLOAT *sb, FLOAT *C,
    BLASLONG K, BLASLONG LDC, FLOAT alpha) {

    v128_t c1, c2, c3, c4;
    c1 = c2 = c3 = c4 = wasm_f64x2_splat(0.0);

    BLASLONG k = K;
    for (; k > 7; k -= 8) {
        /* Pre-duplicated B: [b0,b0], [b1,b1], ... */
        c1 = wasm_f64x2_add(c1, wasm_f64x2_mul(wasm_v128_load(sa), wasm_v128_load(sb)));
        c2 = wasm_f64x2_add(c2, wasm_f64x2_mul(wasm_v128_load(sa + 2), wasm_v128_load(sb + 2)));
        c3 = wasm_f64x2_add(c3, wasm_f64x2_mul(wasm_v128_load(sa + 4), wasm_v128_load(sb + 4)));
        c4 = wasm_f64x2_add(c4, wasm_f64x2_mul(wasm_v128_load(sa + 6), wasm_v128_load(sb + 6)));
        sa += 8;
        sb += 16;  /* 8 k values * 2 for duplication */
    }

    c1 = wasm_f64x2_add(c1, c2);
    c3 = wasm_f64x2_add(c3, c4);
    c1 = wasm_f64x2_add(c1, c3);

    /* Horizontal sum */
    double cs1 = wasm_f64x2_extract_lane(c1, 0) + wasm_f64x2_extract_lane(c1, 1);

    for (; k; k--) {
        cs1 += (*sa++) * sb[0];  /* sb[0] = sb[1] due to pre-duplication */
        sb += 2;
    }

    C[0] += cs1 * alpha;
}

/*===========================================================================
 * Main dispatch function (SSE-style)
 * Processes matrix in blocks: N in 8->4->2->1, M in 4->2->1
 *===========================================================================*/
int dgemm_kernel_sse(BLASLONG M, BLASLONG N, BLASLONG K, FLOAT alpha,
                     FLOAT *sa, FLOAT *sb, FLOAT *C, BLASLONG LDC) {

    /* Process N in blocks of 8 */
    for (; N >= 8; N -= 8) {
        BLASLONG m_left = M;
        const FLOAT *a_ = sa;
        FLOAT *c_ = C;

        for (; m_left >= 4; m_left -= 4) {
            dgemm_kernel_wasm_m4n8_sse(a_, sb, c_, K, LDC, alpha);
            c_ += 4;
            a_ += 4 * K;
        }
        if (m_left >= 2) {
            m_left -= 2;
            dgemm_kernel_wasm_m2n8_sse(a_, sb, c_, K, LDC, alpha);
            c_ += 2;
            a_ += 2 * K;
        }
        if (m_left) {
            dgemm_kernel_wasm_m1n8_sse(a_, sb, c_, K, LDC, alpha);
        }
        sb += 16 * K;  /* 8 cols * 2 for pre-duplication */
        C += 8 * LDC;
    }

    /* Process remaining N in block of 4 */
    if (N >= 4) {
        N -= 4;
        BLASLONG m_left = M;
        const FLOAT *a_ = sa;
        FLOAT *c_ = C;

        for (; m_left >= 4; m_left -= 4) {
            dgemm_kernel_wasm_m4n4_sse(a_, sb, c_, K, LDC, alpha);
            c_ += 4;
            a_ += 4 * K;
        }
        if (m_left >= 2) {
            m_left -= 2;
            dgemm_kernel_wasm_m2n4_sse(a_, sb, c_, K, LDC, alpha);
            c_ += 2;
            a_ += 2 * K;
        }
        if (m_left) {
            dgemm_kernel_wasm_m1n4_sse(a_, sb, c_, K, LDC, alpha);
        }
        sb += 8 * K;  /* 4 cols * 2 for pre-duplication */
        C += 4 * LDC;
    }

    /* Process remaining N in block of 2 */
    if (N >= 2) {
        N -= 2;
        BLASLONG m_left = M;
        const FLOAT *a_ = sa;
        FLOAT *c_ = C;

        for (; m_left >= 4; m_left -= 4) {
            dgemm_kernel_wasm_m4n2_sse(a_, sb, c_, K, LDC, alpha);
            c_ += 4;
            a_ += 4 * K;
        }
        if (m_left >= 2) {
            m_left -= 2;
            dgemm_kernel_wasm_m2n2_sse(a_, sb, c_, K, LDC, alpha);
            c_ += 2;
            a_ += 2 * K;
        }
        if (m_left) {
            dgemm_kernel_wasm_m1n2_sse(a_, sb, c_, K, LDC, alpha);
        }
        sb += 4 * K;  /* 2 cols * 2 for pre-duplication */
        C += 2 * LDC;
    }

    /* Process remaining N = 1 */
    if (N) {
        BLASLONG m_left = M;
        const FLOAT *a_ = sa;
        FLOAT *c_ = C;

        for (; m_left >= 4; m_left -= 4) {
            dgemm_kernel_wasm_m4n1_sse(a_, sb, c_, K, LDC, alpha);
            c_ += 4;
            a_ += 4 * K;
        }
        if (m_left >= 2) {
            m_left -= 2;
            dgemm_kernel_wasm_m2n1_sse(a_, sb, c_, K, LDC, alpha);
            c_ += 2;
            a_ += 2 * K;
        }
        if (m_left) {
            dgemm_kernel_wasm_m1n1_sse(a_, sb, c_, K, LDC, alpha);
        }
    }

    return 0;
}
