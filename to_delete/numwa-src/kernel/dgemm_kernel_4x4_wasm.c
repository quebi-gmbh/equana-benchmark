/**
 * WebAssembly SIMD DGEMM Micro-Kernel (4x4 base)
 *
 * Direct port of OpenBLAS ARM64 NEON dgemm_kernel_4x4_cortexa53.c
 * Both WASM SIMD and NEON use 128-bit vectors with f64x2 (2 doubles).
 *
 * Operation: C[M][N] += alpha * sa[M][K] * sb[K][N]
 *
 * Data Layout:
 *   sa: column-major packed (leading dimension = 4)
 *       [a00, a10, a20, a30, a01, a11, a21, a31, ...]
 *   sb: row-major packed in 4-column blocks
 *       [b00, b01, b02, b03, b10, b11, b12, b13, ...]
 *   C:  column-major (leading dimension = LDC)
 *
 * Copyright (c) 2021, The OpenBLAS Project (original NEON version)
 * WebAssembly port maintains the same algorithm and data layout.
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
 * m4n8: 4 rows x 8 columns kernel
 * Uses 16 v128 accumulators (8 columns x 2 vectors per column)
 * B is stored as 2 concatenated 4-column blocks
 *===========================================================================*/
static inline void dgemm_kernel_wasm_m4n8(
    const FLOAT *sa, const FLOAT *sb, FLOAT *C,
    BLASLONG K, BLASLONG LDC, FLOAT alpha) {

    const FLOAT *b1_ = sb;
    const FLOAT *b2_ = sb + K * 4;

    /* 8 columns, each needs 2 v128 (up=rows0-1, down=rows2-3) */
    v128_t c11, c21, c12, c22, c13, c23, c14, c24;
    v128_t c15, c25, c16, c26, c17, c27, c18, c28;

    c11 = c21 = c12 = c22 = c13 = c23 = c14 = c24 = wasm_f64x2_splat(0.0);
    c15 = c25 = c16 = c26 = c17 = c27 = c18 = c28 = wasm_f64x2_splat(0.0);

    for (BLASLONG k = 0; k < K; k++) {
        v128_t a1 = wasm_v128_load(sa);      /* rows 0-1 */
        v128_t a2 = wasm_v128_load(sa + 2);  /* rows 2-3 */
        sa += 4;

        /* First 4 columns from b1_ */
        v128_t b1 = wasm_v128_load(b1_);
        v128_t b0_splat = wasm_f64x2_splat(wasm_f64x2_extract_lane(b1, 0));
        v128_t b1_splat = wasm_f64x2_splat(wasm_f64x2_extract_lane(b1, 1));
        c11 = wasm_f64x2_add(c11, wasm_f64x2_mul(a1, b0_splat));
        c21 = wasm_f64x2_add(c21, wasm_f64x2_mul(a2, b0_splat));
        c12 = wasm_f64x2_add(c12, wasm_f64x2_mul(a1, b1_splat));
        c22 = wasm_f64x2_add(c22, wasm_f64x2_mul(a2, b1_splat));

        v128_t b2 = wasm_v128_load(b1_ + 2);
        b1_ += 4;
        v128_t b2_splat = wasm_f64x2_splat(wasm_f64x2_extract_lane(b2, 0));
        v128_t b3_splat = wasm_f64x2_splat(wasm_f64x2_extract_lane(b2, 1));
        c13 = wasm_f64x2_add(c13, wasm_f64x2_mul(a1, b2_splat));
        c23 = wasm_f64x2_add(c23, wasm_f64x2_mul(a2, b2_splat));
        c14 = wasm_f64x2_add(c14, wasm_f64x2_mul(a1, b3_splat));
        c24 = wasm_f64x2_add(c24, wasm_f64x2_mul(a2, b3_splat));

        /* Second 4 columns from b2_ */
        v128_t b3 = wasm_v128_load(b2_);
        v128_t b4_splat = wasm_f64x2_splat(wasm_f64x2_extract_lane(b3, 0));
        v128_t b5_splat = wasm_f64x2_splat(wasm_f64x2_extract_lane(b3, 1));
        c15 = wasm_f64x2_add(c15, wasm_f64x2_mul(a1, b4_splat));
        c25 = wasm_f64x2_add(c25, wasm_f64x2_mul(a2, b4_splat));
        c16 = wasm_f64x2_add(c16, wasm_f64x2_mul(a1, b5_splat));
        c26 = wasm_f64x2_add(c26, wasm_f64x2_mul(a2, b5_splat));

        v128_t b4 = wasm_v128_load(b2_ + 2);
        b2_ += 4;
        v128_t b6_splat = wasm_f64x2_splat(wasm_f64x2_extract_lane(b4, 0));
        v128_t b7_splat = wasm_f64x2_splat(wasm_f64x2_extract_lane(b4, 1));
        c17 = wasm_f64x2_add(c17, wasm_f64x2_mul(a1, b6_splat));
        c27 = wasm_f64x2_add(c27, wasm_f64x2_mul(a2, b6_splat));
        c18 = wasm_f64x2_add(c18, wasm_f64x2_mul(a1, b7_splat));
        c28 = wasm_f64x2_add(c28, wasm_f64x2_mul(a2, b7_splat));
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
 * m4n4: 4 rows x 4 columns kernel (primary workhorse)
 * Uses 8 v128 accumulators
 *===========================================================================*/
static inline void dgemm_kernel_wasm_m4n4(
    const FLOAT *sa, const FLOAT *sb, FLOAT *C,
    BLASLONG K, BLASLONG LDC, FLOAT alpha) {

    /* 4 columns, each stored as 2 vectors (up=rows0-1, down=rows2-3) */
    v128_t c11, c21, c12, c22, c13, c23, c14, c24;
    c11 = c21 = c12 = c22 = c13 = c23 = c14 = c24 = wasm_f64x2_splat(0.0);

    for (BLASLONG k = 0; k < K; k++) {
        v128_t a1 = wasm_v128_load(sa);      /* rows 0-1 */
        v128_t a2 = wasm_v128_load(sa + 2);  /* rows 2-3 */
        sa += 4;

        v128_t b1 = wasm_v128_load(sb);      /* cols 0-1 */
        v128_t b2 = wasm_v128_load(sb + 2);  /* cols 2-3 */
        sb += 4;

        /* Column 0: c[*,0] += a[*] * b[0] */
        v128_t b0_splat = wasm_f64x2_splat(wasm_f64x2_extract_lane(b1, 0));
        c11 = wasm_f64x2_add(c11, wasm_f64x2_mul(a1, b0_splat));
        c21 = wasm_f64x2_add(c21, wasm_f64x2_mul(a2, b0_splat));

        /* Column 1: c[*,1] += a[*] * b[1] */
        v128_t b1_splat = wasm_f64x2_splat(wasm_f64x2_extract_lane(b1, 1));
        c12 = wasm_f64x2_add(c12, wasm_f64x2_mul(a1, b1_splat));
        c22 = wasm_f64x2_add(c22, wasm_f64x2_mul(a2, b1_splat));

        /* Column 2: c[*,2] += a[*] * b[2] */
        v128_t b2_splat = wasm_f64x2_splat(wasm_f64x2_extract_lane(b2, 0));
        c13 = wasm_f64x2_add(c13, wasm_f64x2_mul(a1, b2_splat));
        c23 = wasm_f64x2_add(c23, wasm_f64x2_mul(a2, b2_splat));

        /* Column 3: c[*,3] += a[*] * b[3] */
        v128_t b3_splat = wasm_f64x2_splat(wasm_f64x2_extract_lane(b2, 1));
        c14 = wasm_f64x2_add(c14, wasm_f64x2_mul(a1, b3_splat));
        c24 = wasm_f64x2_add(c24, wasm_f64x2_mul(a2, b3_splat));
    }

    /* Store results */
    dgemm_store_m4n1(C, c11, c21, alpha); C += LDC;
    dgemm_store_m4n1(C, c12, c22, alpha); C += LDC;
    dgemm_store_m4n1(C, c13, c23, alpha); C += LDC;
    dgemm_store_m4n1(C, c14, c24, alpha);
}

/*===========================================================================
 * m4n2: 4 rows x 2 columns kernel
 * Uses loop unrolling by 2 for better performance
 *===========================================================================*/
static inline void dgemm_kernel_wasm_m4n2(
    const FLOAT *sa, const FLOAT *sb, FLOAT *C,
    BLASLONG K, BLASLONG LDC, FLOAT alpha) {

    v128_t c11_1, c11_2, c21_1, c21_2, c12_1, c12_2, c22_1, c22_2;
    c11_1 = c11_2 = c21_1 = c21_2 = wasm_f64x2_splat(0.0);
    c12_1 = c12_2 = c22_1 = c22_2 = wasm_f64x2_splat(0.0);

    BLASLONG k = K;
    for (; k > 1; k -= 2) {
        v128_t b1 = wasm_v128_load(sb);
        v128_t b2 = wasm_v128_load(sb + 2);
        sb += 4;

        v128_t a1_1 = wasm_v128_load(sa);
        v128_t a2_1 = wasm_v128_load(sa + 2);
        v128_t a1_2 = wasm_v128_load(sa + 4);
        v128_t a2_2 = wasm_v128_load(sa + 6);
        sa += 8;

        /* Iteration 1 */
        v128_t b0_splat = wasm_f64x2_splat(wasm_f64x2_extract_lane(b1, 0));
        v128_t b1_splat = wasm_f64x2_splat(wasm_f64x2_extract_lane(b1, 1));
        c11_1 = wasm_f64x2_add(c11_1, wasm_f64x2_mul(a1_1, b0_splat));
        c21_1 = wasm_f64x2_add(c21_1, wasm_f64x2_mul(a2_1, b0_splat));
        c12_1 = wasm_f64x2_add(c12_1, wasm_f64x2_mul(a1_1, b1_splat));
        c22_1 = wasm_f64x2_add(c22_1, wasm_f64x2_mul(a2_1, b1_splat));

        /* Iteration 2 */
        v128_t b2_splat = wasm_f64x2_splat(wasm_f64x2_extract_lane(b2, 0));
        v128_t b3_splat = wasm_f64x2_splat(wasm_f64x2_extract_lane(b2, 1));
        c11_2 = wasm_f64x2_add(c11_2, wasm_f64x2_mul(a1_2, b2_splat));
        c21_2 = wasm_f64x2_add(c21_2, wasm_f64x2_mul(a2_2, b2_splat));
        c12_2 = wasm_f64x2_add(c12_2, wasm_f64x2_mul(a1_2, b3_splat));
        c22_2 = wasm_f64x2_add(c22_2, wasm_f64x2_mul(a2_2, b3_splat));
    }

    /* Combine accumulators */
    c11_1 = wasm_f64x2_add(c11_1, c11_2);
    c21_1 = wasm_f64x2_add(c21_1, c21_2);
    c12_1 = wasm_f64x2_add(c12_1, c12_2);
    c22_1 = wasm_f64x2_add(c22_1, c22_2);

    /* Handle remaining k=1 */
    if (k) {
        v128_t b1 = wasm_v128_load(sb);
        sb += 2;
        v128_t a1 = wasm_v128_load(sa);
        v128_t a2 = wasm_v128_load(sa + 2);
        sa += 4;

        v128_t b0_splat = wasm_f64x2_splat(wasm_f64x2_extract_lane(b1, 0));
        v128_t b1_splat = wasm_f64x2_splat(wasm_f64x2_extract_lane(b1, 1));
        c11_1 = wasm_f64x2_add(c11_1, wasm_f64x2_mul(a1, b0_splat));
        c21_1 = wasm_f64x2_add(c21_1, wasm_f64x2_mul(a2, b0_splat));
        c12_1 = wasm_f64x2_add(c12_1, wasm_f64x2_mul(a1, b1_splat));
        c22_1 = wasm_f64x2_add(c22_1, wasm_f64x2_mul(a2, b1_splat));
    }

    dgemm_store_m4n1(C, c11_1, c21_1, alpha); C += LDC;
    dgemm_store_m4n1(C, c12_1, c22_1, alpha);
}

/*===========================================================================
 * m4n1: 4 rows x 1 column kernel
 *===========================================================================*/
static inline void dgemm_kernel_wasm_m4n1(
    const FLOAT *sa, const FLOAT *sb, FLOAT *C,
    BLASLONG K, BLASLONG LDC, FLOAT alpha) {

    v128_t c11_1, c11_2, c21_1, c21_2;
    c11_1 = c11_2 = c21_1 = c21_2 = wasm_f64x2_splat(0.0);

    BLASLONG k = K;
    for (; k > 1; k -= 2) {
        v128_t b1 = wasm_v128_load(sb);
        sb += 2;

        v128_t b0_splat = wasm_f64x2_splat(wasm_f64x2_extract_lane(b1, 0));
        v128_t b1_splat = wasm_f64x2_splat(wasm_f64x2_extract_lane(b1, 1));

        c11_1 = wasm_f64x2_add(c11_1, wasm_f64x2_mul(wasm_v128_load(sa), b0_splat));
        c21_1 = wasm_f64x2_add(c21_1, wasm_f64x2_mul(wasm_v128_load(sa + 2), b0_splat));
        c11_2 = wasm_f64x2_add(c11_2, wasm_f64x2_mul(wasm_v128_load(sa + 4), b1_splat));
        c21_2 = wasm_f64x2_add(c21_2, wasm_f64x2_mul(wasm_v128_load(sa + 6), b1_splat));
        sa += 8;
    }

    c11_1 = wasm_f64x2_add(c11_1, c11_2);
    c21_1 = wasm_f64x2_add(c21_1, c21_2);

    if (k) {
        double b1 = *sb++;
        v128_t b_splat = wasm_f64x2_splat(b1);
        c11_1 = wasm_f64x2_add(c11_1, wasm_f64x2_mul(wasm_v128_load(sa), b_splat));
        c21_1 = wasm_f64x2_add(c21_1, wasm_f64x2_mul(wasm_v128_load(sa + 2), b_splat));
        sa += 4;
    }

    dgemm_store_m4n1(C, c11_1, c21_1, alpha);
}

/*===========================================================================
 * m2n8: 2 rows x 8 columns kernel
 *===========================================================================*/
static inline void dgemm_kernel_wasm_m2n8(
    const FLOAT *sa, const FLOAT *sb, FLOAT *C,
    BLASLONG K, BLASLONG LDC, FLOAT alpha) {

    v128_t c01, c02, c03, c04, c11, c12, c13, c14;
    c01 = c02 = c03 = c04 = c11 = c12 = c13 = c14 = wasm_f64x2_splat(0.0);

    const FLOAT *b1_ = sb;
    const FLOAT *b2_ = sb + 4 * K;

    for (BLASLONG k = 0; k < K; k++) {
        v128_t a1 = wasm_v128_load(sa);
        sa += 2;

        /* First 4 columns */
        v128_t b1 = wasm_v128_load(b1_);
        v128_t b2 = wasm_v128_load(b1_ + 2);
        b1_ += 4;

        v128_t b0_splat = wasm_f64x2_splat(wasm_f64x2_extract_lane(b1, 0));
        v128_t b1_splat = wasm_f64x2_splat(wasm_f64x2_extract_lane(b1, 1));
        v128_t b2_splat = wasm_f64x2_splat(wasm_f64x2_extract_lane(b2, 0));
        v128_t b3_splat = wasm_f64x2_splat(wasm_f64x2_extract_lane(b2, 1));

        c01 = wasm_f64x2_add(c01, wasm_f64x2_mul(a1, b0_splat));
        c02 = wasm_f64x2_add(c02, wasm_f64x2_mul(a1, b1_splat));
        c03 = wasm_f64x2_add(c03, wasm_f64x2_mul(a1, b2_splat));
        c04 = wasm_f64x2_add(c04, wasm_f64x2_mul(a1, b3_splat));

        /* Second 4 columns */
        b1 = wasm_v128_load(b2_);
        b2 = wasm_v128_load(b2_ + 2);
        b2_ += 4;

        b0_splat = wasm_f64x2_splat(wasm_f64x2_extract_lane(b1, 0));
        b1_splat = wasm_f64x2_splat(wasm_f64x2_extract_lane(b1, 1));
        b2_splat = wasm_f64x2_splat(wasm_f64x2_extract_lane(b2, 0));
        b3_splat = wasm_f64x2_splat(wasm_f64x2_extract_lane(b2, 1));

        c11 = wasm_f64x2_add(c11, wasm_f64x2_mul(a1, b0_splat));
        c12 = wasm_f64x2_add(c12, wasm_f64x2_mul(a1, b1_splat));
        c13 = wasm_f64x2_add(c13, wasm_f64x2_mul(a1, b2_splat));
        c14 = wasm_f64x2_add(c14, wasm_f64x2_mul(a1, b3_splat));
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
 * m2n4: 2 rows x 4 columns kernel
 *===========================================================================*/
static inline void dgemm_kernel_wasm_m2n4(
    const FLOAT *sa, const FLOAT *sb, FLOAT *C,
    BLASLONG K, BLASLONG LDC, FLOAT alpha) {

    v128_t c1_1, c1_2, c2_1, c2_2, c3_1, c3_2, c4_1, c4_2;
    c1_1 = c1_2 = c2_1 = c2_2 = c3_1 = c3_2 = c4_1 = c4_2 = wasm_f64x2_splat(0.0);

    BLASLONG k = K;
    for (; k > 1; k -= 2) {
        v128_t a1 = wasm_v128_load(sa);
        v128_t a2 = wasm_v128_load(sa + 2);
        sa += 4;

        v128_t b1_1 = wasm_v128_load(sb);
        v128_t b2_1 = wasm_v128_load(sb + 2);
        v128_t b1_2 = wasm_v128_load(sb + 4);
        v128_t b2_2 = wasm_v128_load(sb + 6);
        sb += 8;

        /* Iteration 1 */
        v128_t b0_splat = wasm_f64x2_splat(wasm_f64x2_extract_lane(b1_1, 0));
        v128_t b1_splat = wasm_f64x2_splat(wasm_f64x2_extract_lane(b1_1, 1));
        v128_t b2_splat = wasm_f64x2_splat(wasm_f64x2_extract_lane(b2_1, 0));
        v128_t b3_splat = wasm_f64x2_splat(wasm_f64x2_extract_lane(b2_1, 1));

        c1_1 = wasm_f64x2_add(c1_1, wasm_f64x2_mul(a1, b0_splat));
        c2_1 = wasm_f64x2_add(c2_1, wasm_f64x2_mul(a1, b1_splat));
        c3_1 = wasm_f64x2_add(c3_1, wasm_f64x2_mul(a1, b2_splat));
        c4_1 = wasm_f64x2_add(c4_1, wasm_f64x2_mul(a1, b3_splat));

        /* Iteration 2 */
        b0_splat = wasm_f64x2_splat(wasm_f64x2_extract_lane(b1_2, 0));
        b1_splat = wasm_f64x2_splat(wasm_f64x2_extract_lane(b1_2, 1));
        b2_splat = wasm_f64x2_splat(wasm_f64x2_extract_lane(b2_2, 0));
        b3_splat = wasm_f64x2_splat(wasm_f64x2_extract_lane(b2_2, 1));

        c1_2 = wasm_f64x2_add(c1_2, wasm_f64x2_mul(a2, b0_splat));
        c2_2 = wasm_f64x2_add(c2_2, wasm_f64x2_mul(a2, b1_splat));
        c3_2 = wasm_f64x2_add(c3_2, wasm_f64x2_mul(a2, b2_splat));
        c4_2 = wasm_f64x2_add(c4_2, wasm_f64x2_mul(a2, b3_splat));
    }

    c1_1 = wasm_f64x2_add(c1_1, c1_2);
    c2_1 = wasm_f64x2_add(c2_1, c2_2);
    c3_1 = wasm_f64x2_add(c3_1, c3_2);
    c4_1 = wasm_f64x2_add(c4_1, c4_2);

    if (k) {
        v128_t a1 = wasm_v128_load(sa);
        sa += 2;
        v128_t b1 = wasm_v128_load(sb);
        v128_t b2 = wasm_v128_load(sb + 2);
        sb += 4;

        v128_t b0_splat = wasm_f64x2_splat(wasm_f64x2_extract_lane(b1, 0));
        v128_t b1_splat = wasm_f64x2_splat(wasm_f64x2_extract_lane(b1, 1));
        v128_t b2_splat = wasm_f64x2_splat(wasm_f64x2_extract_lane(b2, 0));
        v128_t b3_splat = wasm_f64x2_splat(wasm_f64x2_extract_lane(b2, 1));

        c1_1 = wasm_f64x2_add(c1_1, wasm_f64x2_mul(a1, b0_splat));
        c2_1 = wasm_f64x2_add(c2_1, wasm_f64x2_mul(a1, b1_splat));
        c3_1 = wasm_f64x2_add(c3_1, wasm_f64x2_mul(a1, b2_splat));
        c4_1 = wasm_f64x2_add(c4_1, wasm_f64x2_mul(a1, b3_splat));
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
 * m2n2: 2 rows x 2 columns kernel
 *===========================================================================*/
static inline void dgemm_kernel_wasm_m2n2(
    const FLOAT *sa, const FLOAT *sb, FLOAT *C,
    BLASLONG K, BLASLONG LDC, FLOAT alpha) {

    v128_t c1_1, c1_2, c2_1, c2_2;
    c1_1 = c1_2 = c2_1 = c2_2 = wasm_f64x2_splat(0.0);

    BLASLONG k = K;
    for (; k > 1; k -= 2) {
        v128_t a1 = wasm_v128_load(sa);
        v128_t a2 = wasm_v128_load(sa + 2);
        sa += 4;

        v128_t b1 = wasm_v128_load(sb);
        v128_t b2 = wasm_v128_load(sb + 2);
        sb += 4;

        v128_t b0_splat = wasm_f64x2_splat(wasm_f64x2_extract_lane(b1, 0));
        v128_t b1_splat = wasm_f64x2_splat(wasm_f64x2_extract_lane(b1, 1));
        c1_1 = wasm_f64x2_add(c1_1, wasm_f64x2_mul(a1, b0_splat));
        c2_1 = wasm_f64x2_add(c2_1, wasm_f64x2_mul(a1, b1_splat));

        b0_splat = wasm_f64x2_splat(wasm_f64x2_extract_lane(b2, 0));
        b1_splat = wasm_f64x2_splat(wasm_f64x2_extract_lane(b2, 1));
        c1_2 = wasm_f64x2_add(c1_2, wasm_f64x2_mul(a2, b0_splat));
        c2_2 = wasm_f64x2_add(c2_2, wasm_f64x2_mul(a2, b1_splat));
    }

    c1_1 = wasm_f64x2_add(c1_1, c1_2);
    c2_1 = wasm_f64x2_add(c2_1, c2_2);

    if (k) {
        v128_t a1 = wasm_v128_load(sa);
        sa += 2;
        v128_t b1 = wasm_v128_load(sb);
        sb += 2;

        v128_t b0_splat = wasm_f64x2_splat(wasm_f64x2_extract_lane(b1, 0));
        v128_t b1_splat = wasm_f64x2_splat(wasm_f64x2_extract_lane(b1, 1));
        c1_1 = wasm_f64x2_add(c1_1, wasm_f64x2_mul(a1, b0_splat));
        c2_1 = wasm_f64x2_add(c2_1, wasm_f64x2_mul(a1, b1_splat));
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
 * m2n1: 2 rows x 1 column kernel
 *===========================================================================*/
static inline void dgemm_kernel_wasm_m2n1(
    const FLOAT *sa, const FLOAT *sb, FLOAT *C,
    BLASLONG K, BLASLONG LDC, FLOAT alpha) {

    v128_t c1, c2, c3, c4;
    c1 = c2 = c3 = c4 = wasm_f64x2_splat(0.0);

    BLASLONG k = K;
    for (; k > 3; k -= 4) {
        v128_t b12 = wasm_v128_load(sb);
        v128_t b34 = wasm_v128_load(sb + 2);
        sb += 4;

        v128_t b0_splat = wasm_f64x2_splat(wasm_f64x2_extract_lane(b12, 0));
        v128_t b1_splat = wasm_f64x2_splat(wasm_f64x2_extract_lane(b12, 1));
        v128_t b2_splat = wasm_f64x2_splat(wasm_f64x2_extract_lane(b34, 0));
        v128_t b3_splat = wasm_f64x2_splat(wasm_f64x2_extract_lane(b34, 1));

        c1 = wasm_f64x2_add(c1, wasm_f64x2_mul(wasm_v128_load(sa), b0_splat));
        c2 = wasm_f64x2_add(c2, wasm_f64x2_mul(wasm_v128_load(sa + 2), b1_splat));
        c3 = wasm_f64x2_add(c3, wasm_f64x2_mul(wasm_v128_load(sa + 4), b2_splat));
        c4 = wasm_f64x2_add(c4, wasm_f64x2_mul(wasm_v128_load(sa + 6), b3_splat));
        sa += 8;
    }

    c1 = wasm_f64x2_add(c1, c2);
    c3 = wasm_f64x2_add(c3, c4);
    c1 = wasm_f64x2_add(c1, c3);

    for (; k; k--) {
        v128_t b_splat = wasm_f64x2_splat(*sb++);
        c1 = wasm_f64x2_add(c1, wasm_f64x2_mul(wasm_v128_load(sa), b_splat));
        sa += 2;
    }

    v128_t alpha_v = wasm_f64x2_splat(alpha);
    v128_t t = wasm_v128_load(C);
    wasm_v128_store(C, wasm_f64x2_add(t, wasm_f64x2_mul(c1, alpha_v)));
}

/*===========================================================================
 * m1n8: 1 row x 8 columns kernel
 *===========================================================================*/
static inline void dgemm_kernel_wasm_m1n8(
    const FLOAT *sa, const FLOAT *sb, FLOAT *C,
    BLASLONG K, BLASLONG LDC, FLOAT alpha) {

    v128_t c1, c2, c3, c4;
    c1 = c2 = c3 = c4 = wasm_f64x2_splat(0.0);

    const FLOAT *b1_ = sb;
    const FLOAT *b2_ = sb + 4 * K;

    for (BLASLONG k = 0; k < K; k++) {
        double a1 = *sa++;
        v128_t a_splat = wasm_f64x2_splat(a1);

        c1 = wasm_f64x2_add(c1, wasm_f64x2_mul(wasm_v128_load(b1_), a_splat));
        c2 = wasm_f64x2_add(c2, wasm_f64x2_mul(wasm_v128_load(b1_ + 2), a_splat));
        b1_ += 4;
        c3 = wasm_f64x2_add(c3, wasm_f64x2_mul(wasm_v128_load(b2_), a_splat));
        c4 = wasm_f64x2_add(c4, wasm_f64x2_mul(wasm_v128_load(b2_ + 2), a_splat));
        b2_ += 4;
    }

    dgemm_store_m1n2(C, c1, alpha, LDC); C += LDC * 2;
    dgemm_store_m1n2(C, c2, alpha, LDC); C += LDC * 2;
    dgemm_store_m1n2(C, c3, alpha, LDC); C += LDC * 2;
    dgemm_store_m1n2(C, c4, alpha, LDC);
}

/*===========================================================================
 * m1n4: 1 row x 4 columns kernel
 *===========================================================================*/
static inline void dgemm_kernel_wasm_m1n4(
    const FLOAT *sa, const FLOAT *sb, FLOAT *C,
    BLASLONG K, BLASLONG LDC, FLOAT alpha) {

    v128_t c1_1, c1_2, c2_1, c2_2;
    c1_1 = c1_2 = c2_1 = c2_2 = wasm_f64x2_splat(0.0);

    BLASLONG k = K;
    for (; k > 1; k -= 2) {
        v128_t a1 = wasm_v128_load(sa);
        sa += 2;

        v128_t a0_splat = wasm_f64x2_splat(wasm_f64x2_extract_lane(a1, 0));
        v128_t a1_splat = wasm_f64x2_splat(wasm_f64x2_extract_lane(a1, 1));

        c1_1 = wasm_f64x2_add(c1_1, wasm_f64x2_mul(wasm_v128_load(sb), a0_splat));
        c2_1 = wasm_f64x2_add(c2_1, wasm_f64x2_mul(wasm_v128_load(sb + 2), a0_splat));
        c1_2 = wasm_f64x2_add(c1_2, wasm_f64x2_mul(wasm_v128_load(sb + 4), a1_splat));
        c2_2 = wasm_f64x2_add(c2_2, wasm_f64x2_mul(wasm_v128_load(sb + 6), a1_splat));
        sb += 8;
    }

    c1_1 = wasm_f64x2_add(c1_1, c1_2);
    c2_1 = wasm_f64x2_add(c2_1, c2_2);

    if (k) {
        double a1 = *sa++;
        v128_t a_splat = wasm_f64x2_splat(a1);
        c1_1 = wasm_f64x2_add(c1_1, wasm_f64x2_mul(wasm_v128_load(sb), a_splat));
        c2_1 = wasm_f64x2_add(c2_1, wasm_f64x2_mul(wasm_v128_load(sb + 2), a_splat));
        sb += 4;
    }

    dgemm_store_m1n2(C, c1_1, alpha, LDC); C += LDC * 2;
    dgemm_store_m1n2(C, c2_1, alpha, LDC);
}

/*===========================================================================
 * m1n2: 1 row x 2 columns kernel
 *===========================================================================*/
static inline void dgemm_kernel_wasm_m1n2(
    const FLOAT *sa, const FLOAT *sb, FLOAT *C,
    BLASLONG K, BLASLONG LDC, FLOAT alpha) {

    v128_t c1, c2, c3, c4;
    c1 = c2 = c3 = c4 = wasm_f64x2_splat(0.0);

    BLASLONG k = K;
    for (; k > 3; k -= 4) {
        v128_t a12 = wasm_v128_load(sa);
        v128_t a34 = wasm_v128_load(sa + 2);
        sa += 4;

        v128_t a0_splat = wasm_f64x2_splat(wasm_f64x2_extract_lane(a12, 0));
        v128_t a1_splat = wasm_f64x2_splat(wasm_f64x2_extract_lane(a12, 1));
        v128_t a2_splat = wasm_f64x2_splat(wasm_f64x2_extract_lane(a34, 0));
        v128_t a3_splat = wasm_f64x2_splat(wasm_f64x2_extract_lane(a34, 1));

        c1 = wasm_f64x2_add(c1, wasm_f64x2_mul(wasm_v128_load(sb), a0_splat));
        c2 = wasm_f64x2_add(c2, wasm_f64x2_mul(wasm_v128_load(sb + 2), a1_splat));
        c3 = wasm_f64x2_add(c3, wasm_f64x2_mul(wasm_v128_load(sb + 4), a2_splat));
        c4 = wasm_f64x2_add(c4, wasm_f64x2_mul(wasm_v128_load(sb + 6), a3_splat));
        sb += 8;
    }

    c1 = wasm_f64x2_add(c1, c2);
    c3 = wasm_f64x2_add(c3, c4);
    c1 = wasm_f64x2_add(c1, c3);

    for (; k; k--) {
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

    v128_t c1, c2, c3, c4;
    c1 = c2 = c3 = c4 = wasm_f64x2_splat(0.0);

    BLASLONG k = K;
    for (; k > 7; k -= 8) {
        c1 = wasm_f64x2_add(c1, wasm_f64x2_mul(wasm_v128_load(sa), wasm_v128_load(sb)));
        c2 = wasm_f64x2_add(c2, wasm_f64x2_mul(wasm_v128_load(sa + 2), wasm_v128_load(sb + 2)));
        c3 = wasm_f64x2_add(c3, wasm_f64x2_mul(wasm_v128_load(sa + 4), wasm_v128_load(sb + 4)));
        c4 = wasm_f64x2_add(c4, wasm_f64x2_mul(wasm_v128_load(sa + 6), wasm_v128_load(sb + 6)));
        sa += 8;
        sb += 8;
    }

    c1 = wasm_f64x2_add(c1, c2);
    c3 = wasm_f64x2_add(c3, c4);
    c1 = wasm_f64x2_add(c1, c3);

    /* Horizontal sum */
    double cs1 = wasm_f64x2_extract_lane(c1, 0) + wasm_f64x2_extract_lane(c1, 1);

    for (; k; k--) {
        cs1 += (*sa++) * (*sb++);
    }

    C[0] += cs1 * alpha;
}

/*===========================================================================
 * Main dispatch function
 * Processes matrix in blocks: N in 8->4->2->1, M in 4->2->1
 *===========================================================================*/
int dgemm_kernel(BLASLONG M, BLASLONG N, BLASLONG K, FLOAT alpha,
                 FLOAT *sa, FLOAT *sb, FLOAT *C, BLASLONG LDC) {

    /* Process N in blocks of 8 */
    for (; N >= 8; N -= 8) {
        BLASLONG m_left = M;
        const FLOAT *a_ = sa;
        FLOAT *c_ = C;

        for (; m_left >= 4; m_left -= 4) {
            dgemm_kernel_wasm_m4n8(a_, sb, c_, K, LDC, alpha);
            c_ += 4;
            a_ += 4 * K;
        }
        if (m_left >= 2) {
            m_left -= 2;
            dgemm_kernel_wasm_m2n8(a_, sb, c_, K, LDC, alpha);
            c_ += 2;
            a_ += 2 * K;
        }
        if (m_left) {
            dgemm_kernel_wasm_m1n8(a_, sb, c_, K, LDC, alpha);
        }
        sb += 8 * K;
        C += 8 * LDC;
    }

    /* Process remaining N in block of 4 */
    if (N >= 4) {
        N -= 4;
        BLASLONG m_left = M;
        const FLOAT *a_ = sa;
        FLOAT *c_ = C;

        for (; m_left >= 4; m_left -= 4) {
            dgemm_kernel_wasm_m4n4(a_, sb, c_, K, LDC, alpha);
            c_ += 4;
            a_ += 4 * K;
        }
        if (m_left >= 2) {
            m_left -= 2;
            dgemm_kernel_wasm_m2n4(a_, sb, c_, K, LDC, alpha);
            c_ += 2;
            a_ += 2 * K;
        }
        if (m_left) {
            dgemm_kernel_wasm_m1n4(a_, sb, c_, K, LDC, alpha);
        }
        sb += 4 * K;
        C += 4 * LDC;
    }

    /* Process remaining N in block of 2 */
    if (N >= 2) {
        N -= 2;
        BLASLONG m_left = M;
        const FLOAT *a_ = sa;
        FLOAT *c_ = C;

        for (; m_left >= 4; m_left -= 4) {
            dgemm_kernel_wasm_m4n2(a_, sb, c_, K, LDC, alpha);
            c_ += 4;
            a_ += 4 * K;
        }
        if (m_left >= 2) {
            m_left -= 2;
            dgemm_kernel_wasm_m2n2(a_, sb, c_, K, LDC, alpha);
            c_ += 2;
            a_ += 2 * K;
        }
        if (m_left) {
            dgemm_kernel_wasm_m1n2(a_, sb, c_, K, LDC, alpha);
        }
        sb += 2 * K;
        C += 2 * LDC;
    }

    /* Process remaining N = 1 */
    if (N) {
        BLASLONG m_left = M;
        const FLOAT *a_ = sa;
        FLOAT *c_ = C;

        for (; m_left >= 4; m_left -= 4) {
            dgemm_kernel_wasm_m4n1(a_, sb, c_, K, LDC, alpha);
            c_ += 4;
            a_ += 4 * K;
        }
        if (m_left >= 2) {
            m_left -= 2;
            dgemm_kernel_wasm_m2n1(a_, sb, c_, K, LDC, alpha);
            c_ += 2;
            a_ += 2 * K;
        }
        if (m_left) {
            dgemm_kernel_wasm_m1n1(a_, sb, c_, K, LDC, alpha);
        }
    }

    return 0;
}
