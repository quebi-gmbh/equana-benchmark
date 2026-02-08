/**
 * Native WebAssembly SIMD DGEMM Micro-Kernel (4x4 base)
 *
 * Optimized for native WASM SIMD - avoids expensive extract_lane + splat patterns.
 *
 * Key optimization: Use i8x16.shuffle with compile-time constant indices to
 * duplicate f64 lanes. This maps to a single pshufb instruction on x86 and
 * a constant shuffle on ARM, avoiding the 2-3 instruction extract+splat sequence.
 *
 * Operation: C[M][N] += alpha * sa[M][K] * sb[K][N]
 *
 * Data Layout (same as standard kernel for compatibility):
 *   sa: column-major packed (leading dimension = 4)
 *       [a00, a10, a20, a30, a01, a11, a21, a31, ...]
 *   sb: row-major packed in 4-column blocks
 *       [b00, b01, b02, b03, b10, b11, b12, b13, ...]
 *   C:  column-major (leading dimension = LDC)
 */

#include <wasm_simd128.h>

typedef double FLOAT;
typedef long long BLASLONG;

/*===========================================================================
 * Lane duplication using i8x16.shuffle with constant indices
 *
 * This is the key optimization: instead of extract_lane + splat (2-3 ops),
 * use shuffle with compile-time constant indices (1 op on x86/ARM).
 *
 * For f64x2 vectors (16 bytes total, 8 bytes per double):
 *   Lane 0: bytes 0-7
 *   Lane 1: bytes 8-15
 *===========================================================================*/

/* Duplicate lane 0 to both lanes: [a, b] -> [a, a] */
static inline v128_t dup_lo(v128_t v) {
    return wasm_i8x16_shuffle(v, v,
        0, 1, 2, 3, 4, 5, 6, 7,    /* bytes 0-7: lane 0 */
        0, 1, 2, 3, 4, 5, 6, 7);   /* bytes 8-15: lane 0 again */
}

/* Duplicate lane 1 to both lanes: [a, b] -> [b, b] */
static inline v128_t dup_hi(v128_t v) {
    return wasm_i8x16_shuffle(v, v,
        8, 9, 10, 11, 12, 13, 14, 15,   /* bytes 0-7: lane 1 */
        8, 9, 10, 11, 12, 13, 14, 15);  /* bytes 8-15: lane 1 again */
}

/*===========================================================================
 * Helper: Store 4 rows to 1 column of C
 * C[0:4] += alpha * [up[0], up[1], down[0], down[1]]
 *===========================================================================*/
static inline void store_m4n1(FLOAT *C, v128_t up, v128_t down, v128_t alpha_v) {
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
static inline void store_m1n2(FLOAT *C, v128_t vc, FLOAT alpha, BLASLONG LDC) {
    double c0 = wasm_f64x2_extract_lane(vc, 0);
    double c1 = wasm_f64x2_extract_lane(vc, 1);
    C[0] += c0 * alpha;
    C[LDC] += c1 * alpha;
}

/*===========================================================================
 * m4n8: 4 rows x 8 columns kernel
 * Uses 16 v128 accumulators
 *===========================================================================*/
static inline void dgemm_kernel_native_m4n8(
    const FLOAT *sa, const FLOAT *sb, FLOAT *C,
    BLASLONG K, BLASLONG LDC, FLOAT alpha) {

    const FLOAT *b1_ = sb;
    const FLOAT *b2_ = sb + K * 4;

    v128_t c11, c21, c12, c22, c13, c23, c14, c24;
    v128_t c15, c25, c16, c26, c17, c27, c18, c28;

    c11 = c21 = c12 = c22 = c13 = c23 = c14 = c24 = wasm_f64x2_splat(0.0);
    c15 = c25 = c16 = c26 = c17 = c27 = c18 = c28 = wasm_f64x2_splat(0.0);

    for (BLASLONG k = 0; k < K; k++) {
        /* Load A: 4 rows */
        v128_t a_lo = wasm_v128_load(sa);      /* [a0, a1] */
        v128_t a_hi = wasm_v128_load(sa + 2);  /* [a2, a3] */
        sa += 4;

        /* First 4 columns from b1_ */
        v128_t b01 = wasm_v128_load(b1_);      /* [b0, b1] */
        v128_t b23 = wasm_v128_load(b1_ + 2);  /* [b2, b3] */
        b1_ += 4;

        /* Duplicate B using shuffle (1 instruction each) */
        v128_t b0 = dup_lo(b01);
        v128_t b1 = dup_hi(b01);
        v128_t b2 = dup_lo(b23);
        v128_t b3 = dup_hi(b23);

        c11 = wasm_f64x2_add(c11, wasm_f64x2_mul(a_lo, b0));
        c21 = wasm_f64x2_add(c21, wasm_f64x2_mul(a_hi, b0));
        c12 = wasm_f64x2_add(c12, wasm_f64x2_mul(a_lo, b1));
        c22 = wasm_f64x2_add(c22, wasm_f64x2_mul(a_hi, b1));
        c13 = wasm_f64x2_add(c13, wasm_f64x2_mul(a_lo, b2));
        c23 = wasm_f64x2_add(c23, wasm_f64x2_mul(a_hi, b2));
        c14 = wasm_f64x2_add(c14, wasm_f64x2_mul(a_lo, b3));
        c24 = wasm_f64x2_add(c24, wasm_f64x2_mul(a_hi, b3));

        /* Second 4 columns from b2_ */
        b01 = wasm_v128_load(b2_);
        b23 = wasm_v128_load(b2_ + 2);
        b2_ += 4;

        b0 = dup_lo(b01);
        b1 = dup_hi(b01);
        b2 = dup_lo(b23);
        b3 = dup_hi(b23);

        c15 = wasm_f64x2_add(c15, wasm_f64x2_mul(a_lo, b0));
        c25 = wasm_f64x2_add(c25, wasm_f64x2_mul(a_hi, b0));
        c16 = wasm_f64x2_add(c16, wasm_f64x2_mul(a_lo, b1));
        c26 = wasm_f64x2_add(c26, wasm_f64x2_mul(a_hi, b1));
        c17 = wasm_f64x2_add(c17, wasm_f64x2_mul(a_lo, b2));
        c27 = wasm_f64x2_add(c27, wasm_f64x2_mul(a_hi, b2));
        c18 = wasm_f64x2_add(c18, wasm_f64x2_mul(a_lo, b3));
        c28 = wasm_f64x2_add(c28, wasm_f64x2_mul(a_hi, b3));
    }

    v128_t alpha_v = wasm_f64x2_splat(alpha);
    store_m4n1(C, c11, c21, alpha_v); C += LDC;
    store_m4n1(C, c12, c22, alpha_v); C += LDC;
    store_m4n1(C, c13, c23, alpha_v); C += LDC;
    store_m4n1(C, c14, c24, alpha_v); C += LDC;
    store_m4n1(C, c15, c25, alpha_v); C += LDC;
    store_m4n1(C, c16, c26, alpha_v); C += LDC;
    store_m4n1(C, c17, c27, alpha_v); C += LDC;
    store_m4n1(C, c18, c28, alpha_v);
}

/*===========================================================================
 * m4n4: 4 rows x 4 columns kernel (primary workhorse)
 *===========================================================================*/
static inline void dgemm_kernel_native_m4n4(
    const FLOAT *sa, const FLOAT *sb, FLOAT *C,
    BLASLONG K, BLASLONG LDC, FLOAT alpha) {

    v128_t c11, c21, c12, c22, c13, c23, c14, c24;
    c11 = c21 = c12 = c22 = c13 = c23 = c14 = c24 = wasm_f64x2_splat(0.0);

    for (BLASLONG k = 0; k < K; k++) {
        v128_t a_lo = wasm_v128_load(sa);      /* [a0, a1] */
        v128_t a_hi = wasm_v128_load(sa + 2);  /* [a2, a3] */
        sa += 4;

        v128_t b01 = wasm_v128_load(sb);       /* [b0, b1] */
        v128_t b23 = wasm_v128_load(sb + 2);   /* [b2, b3] */
        sb += 4;

        /* Key optimization: shuffle instead of extract+splat */
        v128_t b0 = dup_lo(b01);  /* [b0, b0] */
        v128_t b1 = dup_hi(b01);  /* [b1, b1] */
        v128_t b2 = dup_lo(b23);  /* [b2, b2] */
        v128_t b3 = dup_hi(b23);  /* [b3, b3] */

        c11 = wasm_f64x2_add(c11, wasm_f64x2_mul(a_lo, b0));
        c21 = wasm_f64x2_add(c21, wasm_f64x2_mul(a_hi, b0));

        c12 = wasm_f64x2_add(c12, wasm_f64x2_mul(a_lo, b1));
        c22 = wasm_f64x2_add(c22, wasm_f64x2_mul(a_hi, b1));

        c13 = wasm_f64x2_add(c13, wasm_f64x2_mul(a_lo, b2));
        c23 = wasm_f64x2_add(c23, wasm_f64x2_mul(a_hi, b2));

        c14 = wasm_f64x2_add(c14, wasm_f64x2_mul(a_lo, b3));
        c24 = wasm_f64x2_add(c24, wasm_f64x2_mul(a_hi, b3));
    }

    v128_t alpha_v = wasm_f64x2_splat(alpha);
    store_m4n1(C, c11, c21, alpha_v); C += LDC;
    store_m4n1(C, c12, c22, alpha_v); C += LDC;
    store_m4n1(C, c13, c23, alpha_v); C += LDC;
    store_m4n1(C, c14, c24, alpha_v);
}

/*===========================================================================
 * m4n2: 4 rows x 2 columns kernel
 *===========================================================================*/
static inline void dgemm_kernel_native_m4n2(
    const FLOAT *sa, const FLOAT *sb, FLOAT *C,
    BLASLONG K, BLASLONG LDC, FLOAT alpha) {

    v128_t c11_1, c11_2, c21_1, c21_2, c12_1, c12_2, c22_1, c22_2;
    c11_1 = c11_2 = c21_1 = c21_2 = wasm_f64x2_splat(0.0);
    c12_1 = c12_2 = c22_1 = c22_2 = wasm_f64x2_splat(0.0);

    BLASLONG k = K;
    for (; k > 1; k -= 2) {
        /* Load 2 iterations of B */
        v128_t b1 = wasm_v128_load(sb);      /* [b0_k0, b1_k0] */
        v128_t b2 = wasm_v128_load(sb + 2);  /* [b0_k1, b1_k1] */
        sb += 4;

        /* Load A for both k iterations */
        v128_t a1_lo = wasm_v128_load(sa);
        v128_t a1_hi = wasm_v128_load(sa + 2);
        v128_t a2_lo = wasm_v128_load(sa + 4);
        v128_t a2_hi = wasm_v128_load(sa + 6);
        sa += 8;

        /* Iteration 1 */
        v128_t b0 = dup_lo(b1);
        v128_t b1_dup = dup_hi(b1);
        c11_1 = wasm_f64x2_add(c11_1, wasm_f64x2_mul(a1_lo, b0));
        c21_1 = wasm_f64x2_add(c21_1, wasm_f64x2_mul(a1_hi, b0));
        c12_1 = wasm_f64x2_add(c12_1, wasm_f64x2_mul(a1_lo, b1_dup));
        c22_1 = wasm_f64x2_add(c22_1, wasm_f64x2_mul(a1_hi, b1_dup));

        /* Iteration 2 */
        b0 = dup_lo(b2);
        b1_dup = dup_hi(b2);
        c11_2 = wasm_f64x2_add(c11_2, wasm_f64x2_mul(a2_lo, b0));
        c21_2 = wasm_f64x2_add(c21_2, wasm_f64x2_mul(a2_hi, b0));
        c12_2 = wasm_f64x2_add(c12_2, wasm_f64x2_mul(a2_lo, b1_dup));
        c22_2 = wasm_f64x2_add(c22_2, wasm_f64x2_mul(a2_hi, b1_dup));
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
        v128_t a_lo = wasm_v128_load(sa);
        v128_t a_hi = wasm_v128_load(sa + 2);
        sa += 4;

        v128_t b0 = dup_lo(b1);
        v128_t b1_dup = dup_hi(b1);
        c11_1 = wasm_f64x2_add(c11_1, wasm_f64x2_mul(a_lo, b0));
        c21_1 = wasm_f64x2_add(c21_1, wasm_f64x2_mul(a_hi, b0));
        c12_1 = wasm_f64x2_add(c12_1, wasm_f64x2_mul(a_lo, b1_dup));
        c22_1 = wasm_f64x2_add(c22_1, wasm_f64x2_mul(a_hi, b1_dup));
    }

    v128_t alpha_v = wasm_f64x2_splat(alpha);
    store_m4n1(C, c11_1, c21_1, alpha_v); C += LDC;
    store_m4n1(C, c12_1, c22_1, alpha_v);
}

/*===========================================================================
 * m4n1: 4 rows x 1 column kernel
 *===========================================================================*/
static inline void dgemm_kernel_native_m4n1(
    const FLOAT *sa, const FLOAT *sb, FLOAT *C,
    BLASLONG K, BLASLONG LDC, FLOAT alpha) {

    v128_t c11_1, c11_2, c21_1, c21_2;
    c11_1 = c11_2 = c21_1 = c21_2 = wasm_f64x2_splat(0.0);

    BLASLONG k = K;
    for (; k > 1; k -= 2) {
        v128_t b1 = wasm_v128_load(sb);  /* [b_k0, b_k1] */
        sb += 2;

        v128_t b0 = dup_lo(b1);
        v128_t b1_dup = dup_hi(b1);

        c11_1 = wasm_f64x2_add(c11_1, wasm_f64x2_mul(wasm_v128_load(sa), b0));
        c21_1 = wasm_f64x2_add(c21_1, wasm_f64x2_mul(wasm_v128_load(sa + 2), b0));
        c11_2 = wasm_f64x2_add(c11_2, wasm_f64x2_mul(wasm_v128_load(sa + 4), b1_dup));
        c21_2 = wasm_f64x2_add(c21_2, wasm_f64x2_mul(wasm_v128_load(sa + 6), b1_dup));
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

    v128_t alpha_v = wasm_f64x2_splat(alpha);
    store_m4n1(C, c11_1, c21_1, alpha_v);
}

/*===========================================================================
 * m2n8: 2 rows x 8 columns kernel
 *===========================================================================*/
static inline void dgemm_kernel_native_m2n8(
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
        v128_t b01 = wasm_v128_load(b1_);
        v128_t b23 = wasm_v128_load(b1_ + 2);
        b1_ += 4;

        v128_t b0 = dup_lo(b01);
        v128_t b1 = dup_hi(b01);
        v128_t b2 = dup_lo(b23);
        v128_t b3 = dup_hi(b23);

        c01 = wasm_f64x2_add(c01, wasm_f64x2_mul(a1, b0));
        c02 = wasm_f64x2_add(c02, wasm_f64x2_mul(a1, b1));
        c03 = wasm_f64x2_add(c03, wasm_f64x2_mul(a1, b2));
        c04 = wasm_f64x2_add(c04, wasm_f64x2_mul(a1, b3));

        /* Second 4 columns */
        b01 = wasm_v128_load(b2_);
        b23 = wasm_v128_load(b2_ + 2);
        b2_ += 4;

        b0 = dup_lo(b01);
        b1 = dup_hi(b01);
        b2 = dup_lo(b23);
        b3 = dup_hi(b23);

        c11 = wasm_f64x2_add(c11, wasm_f64x2_mul(a1, b0));
        c12 = wasm_f64x2_add(c12, wasm_f64x2_mul(a1, b1));
        c13 = wasm_f64x2_add(c13, wasm_f64x2_mul(a1, b2));
        c14 = wasm_f64x2_add(c14, wasm_f64x2_mul(a1, b3));
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
static inline void dgemm_kernel_native_m2n4(
    const FLOAT *sa, const FLOAT *sb, FLOAT *C,
    BLASLONG K, BLASLONG LDC, FLOAT alpha) {

    v128_t c1_1, c1_2, c2_1, c2_2, c3_1, c3_2, c4_1, c4_2;
    c1_1 = c1_2 = c2_1 = c2_2 = c3_1 = c3_2 = c4_1 = c4_2 = wasm_f64x2_splat(0.0);

    BLASLONG k = K;
    for (; k > 1; k -= 2) {
        v128_t a1 = wasm_v128_load(sa);
        v128_t a2 = wasm_v128_load(sa + 2);
        sa += 4;

        v128_t b01_1 = wasm_v128_load(sb);
        v128_t b23_1 = wasm_v128_load(sb + 2);
        v128_t b01_2 = wasm_v128_load(sb + 4);
        v128_t b23_2 = wasm_v128_load(sb + 6);
        sb += 8;

        /* Iteration 1 */
        v128_t b0 = dup_lo(b01_1);
        v128_t b1 = dup_hi(b01_1);
        v128_t b2 = dup_lo(b23_1);
        v128_t b3 = dup_hi(b23_1);

        c1_1 = wasm_f64x2_add(c1_1, wasm_f64x2_mul(a1, b0));
        c2_1 = wasm_f64x2_add(c2_1, wasm_f64x2_mul(a1, b1));
        c3_1 = wasm_f64x2_add(c3_1, wasm_f64x2_mul(a1, b2));
        c4_1 = wasm_f64x2_add(c4_1, wasm_f64x2_mul(a1, b3));

        /* Iteration 2 */
        b0 = dup_lo(b01_2);
        b1 = dup_hi(b01_2);
        b2 = dup_lo(b23_2);
        b3 = dup_hi(b23_2);

        c1_2 = wasm_f64x2_add(c1_2, wasm_f64x2_mul(a2, b0));
        c2_2 = wasm_f64x2_add(c2_2, wasm_f64x2_mul(a2, b1));
        c3_2 = wasm_f64x2_add(c3_2, wasm_f64x2_mul(a2, b2));
        c4_2 = wasm_f64x2_add(c4_2, wasm_f64x2_mul(a2, b3));
    }

    c1_1 = wasm_f64x2_add(c1_1, c1_2);
    c2_1 = wasm_f64x2_add(c2_1, c2_2);
    c3_1 = wasm_f64x2_add(c3_1, c3_2);
    c4_1 = wasm_f64x2_add(c4_1, c4_2);

    if (k) {
        v128_t a1 = wasm_v128_load(sa);
        sa += 2;
        v128_t b01 = wasm_v128_load(sb);
        v128_t b23 = wasm_v128_load(sb + 2);
        sb += 4;

        v128_t b0 = dup_lo(b01);
        v128_t b1 = dup_hi(b01);
        v128_t b2 = dup_lo(b23);
        v128_t b3 = dup_hi(b23);

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
 * m2n2: 2 rows x 2 columns kernel
 *===========================================================================*/
static inline void dgemm_kernel_native_m2n2(
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

        v128_t b0 = dup_lo(b1);
        v128_t b1_dup = dup_hi(b1);
        c1_1 = wasm_f64x2_add(c1_1, wasm_f64x2_mul(a1, b0));
        c2_1 = wasm_f64x2_add(c2_1, wasm_f64x2_mul(a1, b1_dup));

        b0 = dup_lo(b2);
        b1_dup = dup_hi(b2);
        c1_2 = wasm_f64x2_add(c1_2, wasm_f64x2_mul(a2, b0));
        c2_2 = wasm_f64x2_add(c2_2, wasm_f64x2_mul(a2, b1_dup));
    }

    c1_1 = wasm_f64x2_add(c1_1, c1_2);
    c2_1 = wasm_f64x2_add(c2_1, c2_2);

    if (k) {
        v128_t a1 = wasm_v128_load(sa);
        sa += 2;
        v128_t b1 = wasm_v128_load(sb);
        sb += 2;

        v128_t b0 = dup_lo(b1);
        v128_t b1_dup = dup_hi(b1);
        c1_1 = wasm_f64x2_add(c1_1, wasm_f64x2_mul(a1, b0));
        c2_1 = wasm_f64x2_add(c2_1, wasm_f64x2_mul(a1, b1_dup));
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
static inline void dgemm_kernel_native_m2n1(
    const FLOAT *sa, const FLOAT *sb, FLOAT *C,
    BLASLONG K, BLASLONG LDC, FLOAT alpha) {

    v128_t c1, c2, c3, c4;
    c1 = c2 = c3 = c4 = wasm_f64x2_splat(0.0);

    BLASLONG k = K;
    for (; k > 3; k -= 4) {
        v128_t b12 = wasm_v128_load(sb);
        v128_t b34 = wasm_v128_load(sb + 2);
        sb += 4;

        v128_t b0 = dup_lo(b12);
        v128_t b1 = dup_hi(b12);
        v128_t b2 = dup_lo(b34);
        v128_t b3 = dup_hi(b34);

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
static inline void dgemm_kernel_native_m1n8(
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

    store_m1n2(C, c1, alpha, LDC); C += LDC * 2;
    store_m1n2(C, c2, alpha, LDC); C += LDC * 2;
    store_m1n2(C, c3, alpha, LDC); C += LDC * 2;
    store_m1n2(C, c4, alpha, LDC);
}

/*===========================================================================
 * m1n4: 1 row x 4 columns kernel
 *===========================================================================*/
static inline void dgemm_kernel_native_m1n4(
    const FLOAT *sa, const FLOAT *sb, FLOAT *C,
    BLASLONG K, BLASLONG LDC, FLOAT alpha) {

    v128_t c1_1, c1_2, c2_1, c2_2;
    c1_1 = c1_2 = c2_1 = c2_2 = wasm_f64x2_splat(0.0);

    BLASLONG k = K;
    for (; k > 1; k -= 2) {
        v128_t a1 = wasm_v128_load(sa);
        sa += 2;

        v128_t a0 = dup_lo(a1);
        v128_t a1_dup = dup_hi(a1);

        c1_1 = wasm_f64x2_add(c1_1, wasm_f64x2_mul(wasm_v128_load(sb), a0));
        c2_1 = wasm_f64x2_add(c2_1, wasm_f64x2_mul(wasm_v128_load(sb + 2), a0));
        c1_2 = wasm_f64x2_add(c1_2, wasm_f64x2_mul(wasm_v128_load(sb + 4), a1_dup));
        c2_2 = wasm_f64x2_add(c2_2, wasm_f64x2_mul(wasm_v128_load(sb + 6), a1_dup));
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

    store_m1n2(C, c1_1, alpha, LDC); C += LDC * 2;
    store_m1n2(C, c2_1, alpha, LDC);
}

/*===========================================================================
 * m1n2: 1 row x 2 columns kernel
 *===========================================================================*/
static inline void dgemm_kernel_native_m1n2(
    const FLOAT *sa, const FLOAT *sb, FLOAT *C,
    BLASLONG K, BLASLONG LDC, FLOAT alpha) {

    v128_t c1, c2, c3, c4;
    c1 = c2 = c3 = c4 = wasm_f64x2_splat(0.0);

    BLASLONG k = K;
    for (; k > 3; k -= 4) {
        v128_t a12 = wasm_v128_load(sa);
        v128_t a34 = wasm_v128_load(sa + 2);
        sa += 4;

        v128_t a0 = dup_lo(a12);
        v128_t a1 = dup_hi(a12);
        v128_t a2 = dup_lo(a34);
        v128_t a3 = dup_hi(a34);

        c1 = wasm_f64x2_add(c1, wasm_f64x2_mul(wasm_v128_load(sb), a0));
        c2 = wasm_f64x2_add(c2, wasm_f64x2_mul(wasm_v128_load(sb + 2), a1));
        c3 = wasm_f64x2_add(c3, wasm_f64x2_mul(wasm_v128_load(sb + 4), a2));
        c4 = wasm_f64x2_add(c4, wasm_f64x2_mul(wasm_v128_load(sb + 6), a3));
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

    store_m1n2(C, c1, alpha, LDC);
}

/*===========================================================================
 * m1n1: 1 row x 1 column kernel (dot product)
 *===========================================================================*/
static inline void dgemm_kernel_native_m1n1(
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
int dgemm_kernel_native(BLASLONG M, BLASLONG N, BLASLONG K, FLOAT alpha,
                        FLOAT *sa, FLOAT *sb, FLOAT *C, BLASLONG LDC) {

    /* Process N in blocks of 8 */
    for (; N >= 8; N -= 8) {
        BLASLONG m_left = M;
        const FLOAT *a_ = sa;
        FLOAT *c_ = C;

        for (; m_left >= 4; m_left -= 4) {
            dgemm_kernel_native_m4n8(a_, sb, c_, K, LDC, alpha);
            c_ += 4;
            a_ += 4 * K;
        }
        if (m_left >= 2) {
            m_left -= 2;
            dgemm_kernel_native_m2n8(a_, sb, c_, K, LDC, alpha);
            c_ += 2;
            a_ += 2 * K;
        }
        if (m_left) {
            dgemm_kernel_native_m1n8(a_, sb, c_, K, LDC, alpha);
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
            dgemm_kernel_native_m4n4(a_, sb, c_, K, LDC, alpha);
            c_ += 4;
            a_ += 4 * K;
        }
        if (m_left >= 2) {
            m_left -= 2;
            dgemm_kernel_native_m2n4(a_, sb, c_, K, LDC, alpha);
            c_ += 2;
            a_ += 2 * K;
        }
        if (m_left) {
            dgemm_kernel_native_m1n4(a_, sb, c_, K, LDC, alpha);
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
            dgemm_kernel_native_m4n2(a_, sb, c_, K, LDC, alpha);
            c_ += 4;
            a_ += 4 * K;
        }
        if (m_left >= 2) {
            m_left -= 2;
            dgemm_kernel_native_m2n2(a_, sb, c_, K, LDC, alpha);
            c_ += 2;
            a_ += 2 * K;
        }
        if (m_left) {
            dgemm_kernel_native_m1n2(a_, sb, c_, K, LDC, alpha);
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
            dgemm_kernel_native_m4n1(a_, sb, c_, K, LDC, alpha);
            c_ += 4;
            a_ += 4 * K;
        }
        if (m_left >= 2) {
            m_left -= 2;
            dgemm_kernel_native_m2n1(a_, sb, c_, K, LDC, alpha);
            c_ += 2;
            a_ += 2 * K;
        }
        if (m_left) {
            dgemm_kernel_native_m1n1(a_, sb, c_, K, LDC, alpha);
        }
    }

    return 0;
}
