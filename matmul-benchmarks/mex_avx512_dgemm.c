/*
 * MEX AVX-512 DGEMM kernel for MATLAB
 *
 * Wraps the OpenBLAS SkylakeX AVX-512 DGEMM micro-kernel with packing routines
 * and a blocked driver loop. Compiled as a MATLAB MEX function to give MATLAB
 * access to AVX-512 matrix multiplication on CPUs that support it (including
 * AMD Zen 4/5 where Intel MKL deliberately disables AVX-512).
 *
 * Usage from MATLAB:
 *   C = mex_avx512_dgemm(A, B)
 *
 * Multi-threading: Uses OpenMP to parallelize over M-dimension tiles.
 * Thread count is controlled via OMP_NUM_THREADS environment variable.
 *
 * Based on OpenBLAS source files (BSD-3-Clause license):
 *   kernel/x86_64/dgemm_kernel_4x8_skylakex.c
 *   kernel/x86_64/dgemm_tcopy_8_skylakex.c
 *   kernel/x86_64/dgemm_ncopy_8_skylakex.c
 *
 * Copyright (c) 2015, The OpenBLAS Project. All rights reserved.
 * See OpenBLAS LICENSE file for full license terms.
 */

#include <string.h>
#include <stdlib.h>
#include <immintrin.h>
#include <omp.h>
#include "mex.h"

typedef long BLASLONG;

/* Blocking parameters from OpenBLAS param.h for SkylakeX DGEMM */
#define GEMM_P 192
#define GEMM_Q 384

/* ========================================================================
 * Pack A (row-major) into 8-row panels
 * Adapted from OpenBLAS dgemm_tcopy_8_skylakex.c
 *
 * Packs an m×n row-major matrix A (leading dimension lda) into buffer b.
 * The packed format groups 8 consecutive rows, storing their n columns
 * interleaved: row0[0..n-1], row1[0..n-1], ..., row7[0..n-1] per panel.
 * ======================================================================== */
static int pack_a(BLASLONG m, BLASLONG n, double * __restrict a, BLASLONG lda, double * __restrict b)
{
    BLASLONG i, j;
    double *aoffset, *aoffset1, *aoffset2, *aoffset3, *aoffset4;
    double *aoffset5, *aoffset6, *aoffset7, *aoffset8;
    double *boffset, *boffset1, *boffset2, *boffset3, *boffset4;
    double ctemp01, ctemp02, ctemp03, ctemp04;
    double ctemp05, ctemp06, ctemp07, ctemp08;

    aoffset = a;
    boffset = b;

    boffset2 = b + m * (n & ~7);
    boffset3 = b + m * (n & ~3);
    boffset4 = b + m * (n & ~1);

    j = (m >> 3);
    if (j > 0) {
        do {
            aoffset1 = aoffset;
            aoffset2 = aoffset1 + lda;
            aoffset3 = aoffset2 + lda;
            aoffset4 = aoffset3 + lda;
            aoffset5 = aoffset4 + lda;
            aoffset6 = aoffset5 + lda;
            aoffset7 = aoffset6 + lda;
            aoffset8 = aoffset7 + lda;
            aoffset += 8 * lda;

            boffset1 = boffset;
            boffset += 64;

            i = (n >> 3);
            if (i > 0) {
                do {
                    __m512d row1, row2, row3, row4, row5, row6, row7, row8;
                    row1 = _mm512_loadu_pd(aoffset1); aoffset1 += 8;
                    row2 = _mm512_loadu_pd(aoffset2); aoffset2 += 8;
                    row3 = _mm512_loadu_pd(aoffset3); aoffset3 += 8;
                    row4 = _mm512_loadu_pd(aoffset4); aoffset4 += 8;
                    row5 = _mm512_loadu_pd(aoffset5); aoffset5 += 8;
                    row6 = _mm512_loadu_pd(aoffset6); aoffset6 += 8;
                    row7 = _mm512_loadu_pd(aoffset7); aoffset7 += 8;
                    row8 = _mm512_loadu_pd(aoffset8); aoffset8 += 8;

                    _mm512_storeu_pd(boffset1 +  0, row1);
                    _mm512_storeu_pd(boffset1 +  8, row2);
                    _mm512_storeu_pd(boffset1 + 16, row3);
                    _mm512_storeu_pd(boffset1 + 24, row4);
                    _mm512_storeu_pd(boffset1 + 32, row5);
                    _mm512_storeu_pd(boffset1 + 40, row6);
                    _mm512_storeu_pd(boffset1 + 48, row7);
                    _mm512_storeu_pd(boffset1 + 56, row8);
                    boffset1 += m * 8;
                    i--;
                } while (i > 0);
            }

            if (n & 4) {
                __m256d row1, row2, row3, row4, row5, row6, row7, row8;
                row1 = _mm256_loadu_pd(aoffset1); aoffset1 += 4;
                row2 = _mm256_loadu_pd(aoffset2); aoffset2 += 4;
                row3 = _mm256_loadu_pd(aoffset3); aoffset3 += 4;
                row4 = _mm256_loadu_pd(aoffset4); aoffset4 += 4;
                row5 = _mm256_loadu_pd(aoffset5); aoffset5 += 4;
                row6 = _mm256_loadu_pd(aoffset6); aoffset6 += 4;
                row7 = _mm256_loadu_pd(aoffset7); aoffset7 += 4;
                row8 = _mm256_loadu_pd(aoffset8); aoffset8 += 4;

                _mm256_storeu_pd(boffset2 +  0, row1);
                _mm256_storeu_pd(boffset2 +  4, row2);
                _mm256_storeu_pd(boffset2 +  8, row3);
                _mm256_storeu_pd(boffset2 + 12, row4);
                _mm256_storeu_pd(boffset2 + 16, row5);
                _mm256_storeu_pd(boffset2 + 20, row6);
                _mm256_storeu_pd(boffset2 + 24, row7);
                _mm256_storeu_pd(boffset2 + 28, row8);
                boffset2 += 32;
            }

            if (n & 2) {
                __m128d row1, row2, row3, row4, row5, row6, row7, row8;
                row1 = _mm_loadu_pd(aoffset1); aoffset1 += 2;
                row2 = _mm_loadu_pd(aoffset2); aoffset2 += 2;
                row3 = _mm_loadu_pd(aoffset3); aoffset3 += 2;
                row4 = _mm_loadu_pd(aoffset4); aoffset4 += 2;
                row5 = _mm_loadu_pd(aoffset5); aoffset5 += 2;
                row6 = _mm_loadu_pd(aoffset6); aoffset6 += 2;
                row7 = _mm_loadu_pd(aoffset7); aoffset7 += 2;
                row8 = _mm_loadu_pd(aoffset8); aoffset8 += 2;

                _mm_storeu_pd(boffset3 +  0, row1);
                _mm_storeu_pd(boffset3 +  2, row2);
                _mm_storeu_pd(boffset3 +  4, row3);
                _mm_storeu_pd(boffset3 +  6, row4);
                _mm_storeu_pd(boffset3 +  8, row5);
                _mm_storeu_pd(boffset3 + 10, row6);
                _mm_storeu_pd(boffset3 + 12, row7);
                _mm_storeu_pd(boffset3 + 14, row8);
                boffset3 += 16;
            }

            if (n & 1) {
                ctemp01 = *(aoffset1 + 0); aoffset1++;
                ctemp02 = *(aoffset2 + 0); aoffset2++;
                ctemp03 = *(aoffset3 + 0); aoffset3++;
                ctemp04 = *(aoffset4 + 0); aoffset4++;
                ctemp05 = *(aoffset5 + 0); aoffset5++;
                ctemp06 = *(aoffset6 + 0); aoffset6++;
                ctemp07 = *(aoffset7 + 0); aoffset7++;
                ctemp08 = *(aoffset8 + 0); aoffset8++;

                *(boffset4 + 0) = ctemp01;
                *(boffset4 + 1) = ctemp02;
                *(boffset4 + 2) = ctemp03;
                *(boffset4 + 3) = ctemp04;
                *(boffset4 + 4) = ctemp05;
                *(boffset4 + 5) = ctemp06;
                *(boffset4 + 6) = ctemp07;
                *(boffset4 + 7) = ctemp08;
                boffset4 += 8;
            }

            j--;
        } while (j > 0);
    }

    if (m & 4) {
        aoffset1 = aoffset;
        aoffset2 = aoffset1 + lda;
        aoffset3 = aoffset2 + lda;
        aoffset4 = aoffset3 + lda;
        aoffset += 4 * lda;

        boffset1 = boffset;
        boffset += 32;

        i = (n >> 3);
        if (i > 0) {
            do {
                __m512d row1, row2, row3, row4;
                row1 = _mm512_loadu_pd(aoffset1); aoffset1 += 8;
                row2 = _mm512_loadu_pd(aoffset2); aoffset2 += 8;
                row3 = _mm512_loadu_pd(aoffset3); aoffset3 += 8;
                row4 = _mm512_loadu_pd(aoffset4); aoffset4 += 8;

                _mm512_storeu_pd(boffset1 +  0, row1);
                _mm512_storeu_pd(boffset1 +  8, row2);
                _mm512_storeu_pd(boffset1 + 16, row3);
                _mm512_storeu_pd(boffset1 + 24, row4);
                boffset1 += 8 * m;
                i--;
            } while (i > 0);
        }

        if (n & 4) {
            __m256d row1, row2, row3, row4;
            row1 = _mm256_loadu_pd(aoffset1); aoffset1 += 4;
            row2 = _mm256_loadu_pd(aoffset2); aoffset2 += 4;
            row3 = _mm256_loadu_pd(aoffset3); aoffset3 += 4;
            row4 = _mm256_loadu_pd(aoffset4); aoffset4 += 4;
            _mm256_storeu_pd(boffset2 +  0, row1);
            _mm256_storeu_pd(boffset2 +  4, row2);
            _mm256_storeu_pd(boffset2 +  8, row3);
            _mm256_storeu_pd(boffset2 + 12, row4);
            boffset2 += 16;
        }

        if (n & 2) {
            __m128d row1, row2, row3, row4;
            row1 = _mm_loadu_pd(aoffset1); aoffset1 += 2;
            row2 = _mm_loadu_pd(aoffset2); aoffset2 += 2;
            row3 = _mm_loadu_pd(aoffset3); aoffset3 += 2;
            row4 = _mm_loadu_pd(aoffset4); aoffset4 += 2;
            _mm_storeu_pd(boffset3 + 0, row1);
            _mm_storeu_pd(boffset3 + 2, row2);
            _mm_storeu_pd(boffset3 + 4, row3);
            _mm_storeu_pd(boffset3 + 6, row4);
            boffset3 += 8;
        }

        if (n & 1) {
            ctemp01 = *(aoffset1 + 0); aoffset1++;
            ctemp02 = *(aoffset2 + 0); aoffset2++;
            ctemp03 = *(aoffset3 + 0); aoffset3++;
            ctemp04 = *(aoffset4 + 0); aoffset4++;
            *(boffset4 + 0) = ctemp01;
            *(boffset4 + 1) = ctemp02;
            *(boffset4 + 2) = ctemp03;
            *(boffset4 + 3) = ctemp04;
            boffset4 += 4;
        }
    }

    if (m & 2) {
        aoffset1 = aoffset;
        aoffset2 = aoffset1 + lda;
        aoffset += 2 * lda;

        boffset1 = boffset;
        boffset += 16;

        i = (n >> 3);
        if (i > 0) {
            do {
                __m512d row1, row2;
                row1 = _mm512_loadu_pd(aoffset1); aoffset1 += 8;
                row2 = _mm512_loadu_pd(aoffset2); aoffset2 += 8;
                _mm512_storeu_pd(boffset1 + 0, row1);
                _mm512_storeu_pd(boffset1 + 8, row2);
                boffset1 += 8 * m;
                i--;
            } while (i > 0);
        }

        if (n & 4) {
            __m256d row1, row2;
            row1 = _mm256_loadu_pd(aoffset1); aoffset1 += 4;
            row2 = _mm256_loadu_pd(aoffset2); aoffset2 += 4;
            _mm256_storeu_pd(boffset2 + 0, row1);
            _mm256_storeu_pd(boffset2 + 4, row2);
            boffset2 += 8;
        }

        if (n & 2) {
            __m128d row1, row2;
            row1 = _mm_loadu_pd(aoffset1); aoffset1 += 2;
            row2 = _mm_loadu_pd(aoffset2); aoffset2 += 2;
            _mm_storeu_pd(boffset3 + 0, row1);
            _mm_storeu_pd(boffset3 + 2, row2);
            boffset3 += 4;
        }

        if (n & 1) {
            ctemp01 = *(aoffset1 + 0); aoffset1++;
            ctemp02 = *(aoffset2 + 0); aoffset2++;
            *(boffset4 + 0) = ctemp01;
            *(boffset4 + 1) = ctemp02;
            boffset4 += 2;
        }
    }

    if (m & 1) {
        aoffset1 = aoffset;

        boffset1 = boffset;

        i = (n >> 3);
        if (i > 0) {
            do {
                __m512d row1;
                row1 = _mm512_loadu_pd(aoffset1); aoffset1 += 8;
                _mm512_storeu_pd(boffset1 + 0, row1);
                boffset1 += 8 * m;
                i--;
            } while (i > 0);
        }

        if (n & 4) {
            __m256d row1;
            row1 = _mm256_loadu_pd(aoffset1); aoffset1 += 4;
            _mm256_storeu_pd(boffset2 + 0, row1);
        }

        if (n & 2) {
            __m128d row1;
            row1 = _mm_loadu_pd(aoffset1); aoffset1 += 2;
            _mm_storeu_pd(boffset3 + 0, row1);
        }

        if (n & 1) {
            ctemp01 = *(aoffset1 + 0);
            *(boffset4 + 0) = ctemp01;
        }
    }

    return 0;
}


/* ========================================================================
 * Pack B (column-major) into 8-column interleaved panels
 * Adapted from OpenBLAS dgemm_ncopy_8_skylakex.c
 *
 * Packs columns of an m-row × n-column column-major matrix into panels
 * of 8 columns interleaved by row position.
 * ======================================================================== */
static int pack_b(BLASLONG m, BLASLONG n, double * __restrict a, BLASLONG lda, double * __restrict b)
{
    BLASLONG i, j;
    double *aoffset, *aoffset1, *aoffset2, *aoffset3, *aoffset4;
    double *aoffset5, *aoffset6, *aoffset7, *aoffset8;
    double *boffset;
    double ctemp01, ctemp02, ctemp03, ctemp04;
    double ctemp05, ctemp06, ctemp07, ctemp08;
    double ctemp09, ctemp10, ctemp11, ctemp12;
    double ctemp13, ctemp14, ctemp15, ctemp16;
    double ctemp17, ctemp23, ctemp24;
    double ctemp25, ctemp31, ctemp32;
    double ctemp33, ctemp39, ctemp40;
    double ctemp41, ctemp47, ctemp48;
    double ctemp49, ctemp55, ctemp56;
    double ctemp57, ctemp63, ctemp64;

    aoffset = a;
    boffset = b;

    j = (n >> 3);
    if (j > 0) {
        do {
            aoffset1 = aoffset;
            aoffset2 = aoffset1 + lda;
            aoffset3 = aoffset2 + lda;
            aoffset4 = aoffset3 + lda;
            aoffset5 = aoffset4 + lda;
            aoffset6 = aoffset5 + lda;
            aoffset7 = aoffset6 + lda;
            aoffset8 = aoffset7 + lda;
            aoffset += 8 * lda;

            i = (m >> 3);
            if (i > 0) {
                do {
                    __m128d xmm0, xmm1;
                    xmm0 = _mm_load_pd1(aoffset2 + 0);
                    xmm0 = _mm_loadl_pd(xmm0, aoffset1 + 0);
                    _mm_storeu_pd(boffset + 0, xmm0);

                    ctemp07 = *(aoffset1 + 6);
                    ctemp08 = *(aoffset1 + 7);

                    xmm1 = _mm_load_pd1(aoffset4 + 0);
                    xmm1 = _mm_loadl_pd(xmm1, aoffset3 + 0);
                    _mm_storeu_pd(boffset + 2, xmm1);

                    xmm0 = _mm_load_pd1(aoffset6 + 0);
                    xmm0 = _mm_loadl_pd(xmm0, aoffset5 + 0);
                    _mm_storeu_pd(boffset + 4, xmm0);

                    xmm0 = _mm_load_pd1(aoffset8 + 0);
                    xmm0 = _mm_loadl_pd(xmm0, aoffset7 + 0);
                    _mm_storeu_pd(boffset + 6, xmm0);

                    ctemp15 = *(aoffset2 + 6);
                    ctemp16 = *(aoffset2 + 7);

                    xmm0 = _mm_load_pd1(aoffset2 + 1);
                    xmm0 = _mm_loadl_pd(xmm0, aoffset1 + 1);
                    _mm_storeu_pd(boffset + 8, xmm0);

                    xmm0 = _mm_load_pd1(aoffset4 + 1);
                    xmm0 = _mm_loadl_pd(xmm0, aoffset3 + 1);
                    _mm_storeu_pd(boffset + 10, xmm0);

                    xmm0 = _mm_load_pd1(aoffset6 + 1);
                    xmm0 = _mm_loadl_pd(xmm0, aoffset5 + 1);
                    _mm_storeu_pd(boffset + 12, xmm0);

                    xmm0 = _mm_load_pd1(aoffset8 + 1);
                    xmm0 = _mm_loadl_pd(xmm0, aoffset7 + 1);
                    _mm_storeu_pd(boffset + 14, xmm0);

                    xmm0 = _mm_load_pd1(aoffset2 + 2);
                    xmm0 = _mm_loadl_pd(xmm0, aoffset1 + 2);
                    _mm_storeu_pd(boffset + 16, xmm0);

                    xmm0 = _mm_load_pd1(aoffset4 + 2);
                    xmm0 = _mm_loadl_pd(xmm0, aoffset3 + 2);
                    _mm_storeu_pd(boffset + 18, xmm0);

                    xmm0 = _mm_load_pd1(aoffset6 + 2);
                    xmm0 = _mm_loadl_pd(xmm0, aoffset5 + 2);
                    _mm_storeu_pd(boffset + 20, xmm0);

                    xmm0 = _mm_load_pd1(aoffset8 + 2);
                    xmm0 = _mm_loadl_pd(xmm0, aoffset7 + 2);
                    _mm_storeu_pd(boffset + 22, xmm0);

                    ctemp23 = *(aoffset3 + 6);
                    ctemp24 = *(aoffset3 + 7);

                    xmm0 = _mm_load_pd1(aoffset2 + 3);
                    xmm0 = _mm_loadl_pd(xmm0, aoffset1 + 3);
                    _mm_storeu_pd(boffset + 24, xmm0);

                    xmm0 = _mm_load_pd1(aoffset4 + 3);
                    xmm0 = _mm_loadl_pd(xmm0, aoffset3 + 3);
                    _mm_storeu_pd(boffset + 26, xmm0);

                    xmm0 = _mm_load_pd1(aoffset6 + 3);
                    xmm0 = _mm_loadl_pd(xmm0, aoffset5 + 3);
                    _mm_storeu_pd(boffset + 28, xmm0);

                    xmm0 = _mm_load_pd1(aoffset8 + 3);
                    xmm0 = _mm_loadl_pd(xmm0, aoffset7 + 3);
                    _mm_storeu_pd(boffset + 30, xmm0);

                    ctemp31 = *(aoffset4 + 6);
                    ctemp32 = *(aoffset4 + 7);

                    xmm0 = _mm_load_pd1(aoffset2 + 4);
                    xmm0 = _mm_loadl_pd(xmm0, aoffset1 + 4);
                    _mm_storeu_pd(boffset + 32, xmm0);

                    xmm0 = _mm_load_pd1(aoffset4 + 4);
                    xmm0 = _mm_loadl_pd(xmm0, aoffset3 + 4);
                    _mm_storeu_pd(boffset + 34, xmm0);

                    xmm0 = _mm_load_pd1(aoffset6 + 4);
                    xmm0 = _mm_loadl_pd(xmm0, aoffset5 + 4);
                    _mm_storeu_pd(boffset + 36, xmm0);

                    xmm0 = _mm_load_pd1(aoffset8 + 4);
                    xmm0 = _mm_loadl_pd(xmm0, aoffset7 + 4);
                    _mm_storeu_pd(boffset + 38, xmm0);

                    ctemp39 = *(aoffset5 + 6);
                    ctemp40 = *(aoffset5 + 7);

                    xmm0 = _mm_load_pd1(aoffset2 + 5);
                    xmm0 = _mm_loadl_pd(xmm0, aoffset1 + 5);
                    _mm_storeu_pd(boffset + 40, xmm0);

                    xmm0 = _mm_load_pd1(aoffset4 + 5);
                    xmm0 = _mm_loadl_pd(xmm0, aoffset3 + 5);
                    _mm_storeu_pd(boffset + 42, xmm0);

                    xmm0 = _mm_load_pd1(aoffset6 + 5);
                    xmm0 = _mm_loadl_pd(xmm0, aoffset5 + 5);
                    _mm_storeu_pd(boffset + 44, xmm0);

                    xmm0 = _mm_load_pd1(aoffset8 + 5);
                    xmm0 = _mm_loadl_pd(xmm0, aoffset7 + 5);
                    _mm_storeu_pd(boffset + 46, xmm0);

                    ctemp47 = *(aoffset6 + 6);
                    ctemp48 = *(aoffset6 + 7);
                    ctemp55 = *(aoffset7 + 6);
                    ctemp56 = *(aoffset7 + 7);
                    ctemp63 = *(aoffset8 + 6);
                    ctemp64 = *(aoffset8 + 7);

                    *(boffset + 48) = ctemp07;
                    *(boffset + 49) = ctemp15;
                    *(boffset + 50) = ctemp23;
                    *(boffset + 51) = ctemp31;
                    *(boffset + 52) = ctemp39;
                    *(boffset + 53) = ctemp47;
                    *(boffset + 54) = ctemp55;
                    *(boffset + 55) = ctemp63;

                    *(boffset + 56) = ctemp08;
                    *(boffset + 57) = ctemp16;
                    *(boffset + 58) = ctemp24;
                    *(boffset + 59) = ctemp32;
                    *(boffset + 60) = ctemp40;
                    *(boffset + 61) = ctemp48;
                    *(boffset + 62) = ctemp56;
                    *(boffset + 63) = ctemp64;

                    aoffset1 += 8;
                    aoffset2 += 8;
                    aoffset3 += 8;
                    aoffset4 += 8;
                    aoffset5 += 8;
                    aoffset6 += 8;
                    aoffset7 += 8;
                    aoffset8 += 8;
                    boffset += 64;
                    i--;
                } while (i > 0);
            }

            i = (m & 7);
            if (i > 0) {
                do {
                    ctemp01 = *(aoffset1 + 0);
                    ctemp09 = *(aoffset2 + 0);
                    ctemp17 = *(aoffset3 + 0);
                    ctemp25 = *(aoffset4 + 0);
                    ctemp33 = *(aoffset5 + 0);
                    ctemp41 = *(aoffset6 + 0);
                    ctemp49 = *(aoffset7 + 0);
                    ctemp57 = *(aoffset8 + 0);

                    *(boffset + 0) = ctemp01;
                    *(boffset + 1) = ctemp09;
                    *(boffset + 2) = ctemp17;
                    *(boffset + 3) = ctemp25;
                    *(boffset + 4) = ctemp33;
                    *(boffset + 5) = ctemp41;
                    *(boffset + 6) = ctemp49;
                    *(boffset + 7) = ctemp57;

                    aoffset1++;
                    aoffset2++;
                    aoffset3++;
                    aoffset4++;
                    aoffset5++;
                    aoffset6++;
                    aoffset7++;
                    aoffset8++;
                    boffset += 8;
                    i--;
                } while (i > 0);
            }
            j--;
        } while (j > 0);
    }

    if (n & 4) {
        aoffset1 = aoffset;
        aoffset2 = aoffset1 + lda;
        aoffset3 = aoffset2 + lda;
        aoffset4 = aoffset3 + lda;
        aoffset += 4 * lda;

        i = (m >> 2);
        if (i > 0) {
            do {
                ctemp01 = *(aoffset1 + 0);
                ctemp02 = *(aoffset1 + 1);
                ctemp03 = *(aoffset1 + 2);
                ctemp04 = *(aoffset1 + 3);
                ctemp05 = *(aoffset2 + 0);
                ctemp06 = *(aoffset2 + 1);
                ctemp07 = *(aoffset2 + 2);
                ctemp08 = *(aoffset2 + 3);
                ctemp09 = *(aoffset3 + 0);
                ctemp10 = *(aoffset3 + 1);
                ctemp11 = *(aoffset3 + 2);
                ctemp12 = *(aoffset3 + 3);
                ctemp13 = *(aoffset4 + 0);
                ctemp14 = *(aoffset4 + 1);
                ctemp15 = *(aoffset4 + 2);
                ctemp16 = *(aoffset4 + 3);

                *(boffset +  0) = ctemp01;
                *(boffset +  1) = ctemp05;
                *(boffset +  2) = ctemp09;
                *(boffset +  3) = ctemp13;
                *(boffset +  4) = ctemp02;
                *(boffset +  5) = ctemp06;
                *(boffset +  6) = ctemp10;
                *(boffset +  7) = ctemp14;
                *(boffset +  8) = ctemp03;
                *(boffset +  9) = ctemp07;
                *(boffset + 10) = ctemp11;
                *(boffset + 11) = ctemp15;
                *(boffset + 12) = ctemp04;
                *(boffset + 13) = ctemp08;
                *(boffset + 14) = ctemp12;
                *(boffset + 15) = ctemp16;

                aoffset1 += 4;
                aoffset2 += 4;
                aoffset3 += 4;
                aoffset4 += 4;
                boffset += 16;
                i--;
            } while (i > 0);
        }

        i = (m & 3);
        if (i > 0) {
            do {
                ctemp01 = *(aoffset1 + 0);
                ctemp02 = *(aoffset2 + 0);
                ctemp03 = *(aoffset3 + 0);
                ctemp04 = *(aoffset4 + 0);

                *(boffset + 0) = ctemp01;
                *(boffset + 1) = ctemp02;
                *(boffset + 2) = ctemp03;
                *(boffset + 3) = ctemp04;

                aoffset1++;
                aoffset2++;
                aoffset3++;
                aoffset4++;
                boffset += 4;
                i--;
            } while (i > 0);
        }
    }

    if (n & 2) {
        aoffset1 = aoffset;
        aoffset2 = aoffset1 + lda;
        aoffset += 2 * lda;

        i = (m >> 1);
        if (i > 0) {
            do {
                ctemp01 = *(aoffset1 + 0);
                ctemp02 = *(aoffset1 + 1);
                ctemp03 = *(aoffset2 + 0);
                ctemp04 = *(aoffset2 + 1);

                *(boffset + 0) = ctemp01;
                *(boffset + 1) = ctemp03;
                *(boffset + 2) = ctemp02;
                *(boffset + 3) = ctemp04;

                aoffset1 += 2;
                aoffset2 += 2;
                boffset += 4;
                i--;
            } while (i > 0);
        }

        if (m & 1) {
            ctemp01 = *(aoffset1 + 0);
            ctemp02 = *(aoffset2 + 0);
            *(boffset + 0) = ctemp01;
            *(boffset + 1) = ctemp02;
            boffset += 2;
        }
    }

    if (n & 1) {
        aoffset1 = aoffset;

        i = m;
        if (i > 0) {
            do {
                ctemp01 = *(aoffset1 + 0);
                *(boffset + 0) = ctemp01;
                aoffset1++;
                boffset++;
                i--;
            } while (i > 0);
        }
    }

    return 0;
}


/* ========================================================================
 * AVX-512 DGEMM micro-kernel (SkylakeX)
 * Adapted from OpenBLAS dgemm_kernel_4x8_skylakex.c
 *
 * Computes: C += alpha * A_packed * B_packed
 * A_packed: m rows packed in 8-row panels (from pack_a)
 * B_packed: n columns packed in 8-column panels (from pack_b)
 * C: m×n output matrix, leading dimension ldc
 *
 * The kernel processes N in blocks of 8, then 4, 2, 1.
 * Within each N-block, M is processed in blocks of 24/16/8 (AVX-512 asm),
 * then 4 (AVX2 intrinsics), then 2 (SSE), then 1 (scalar).
 * ======================================================================== */

/* Macro definitions for the 4x8 sub-kernel (AVX2 intrinsics) */
#define INIT4x8() \
    ymm4 = _mm256_setzero_pd(); ymm5 = _mm256_setzero_pd(); \
    ymm6 = _mm256_setzero_pd(); ymm7 = _mm256_setzero_pd(); \
    ymm8 = _mm256_setzero_pd(); ymm9 = _mm256_setzero_pd(); \
    ymm10 = _mm256_setzero_pd(); ymm11 = _mm256_setzero_pd();

#define KERNEL4x8_SUB() \
    ymm0 = _mm256_loadu_pd(AO - 16); \
    ymm1 = _mm256_loadu_pd(BO - 12); \
    ymm2 = _mm256_loadu_pd(BO - 8); \
    ymm4 += ymm0 * ymm1; ymm8 += ymm0 * ymm2; \
    ymm0 = _mm256_permute4x64_pd(ymm0, 0xb1); \
    ymm5 += ymm0 * ymm1; ymm9 += ymm0 * ymm2; \
    ymm0 = _mm256_permute4x64_pd(ymm0, 0x1b); \
    ymm6 += ymm0 * ymm1; ymm10 += ymm0 * ymm2; \
    ymm0 = _mm256_permute4x64_pd(ymm0, 0xb1); \
    ymm7 += ymm0 * ymm1; ymm11 += ymm0 * ymm2; \
    AO += 4; BO += 8;

#define SAVE4x8(ALPHA) \
    ymm0 = _mm256_set1_pd(ALPHA); \
    ymm4 *= ymm0; ymm5 *= ymm0; ymm6 *= ymm0; ymm7 *= ymm0; \
    ymm8 *= ymm0; ymm9 *= ymm0; ymm10 *= ymm0; ymm11 *= ymm0; \
    ymm5 = _mm256_permute4x64_pd(ymm5, 0xb1); \
    ymm7 = _mm256_permute4x64_pd(ymm7, 0xb1); \
    ymm0 = _mm256_blend_pd(ymm4, ymm5, 0x0a); \
    ymm1 = _mm256_blend_pd(ymm4, ymm5, 0x05); \
    ymm2 = _mm256_blend_pd(ymm6, ymm7, 0x0a); \
    ymm3 = _mm256_blend_pd(ymm6, ymm7, 0x05); \
    ymm2 = _mm256_permute4x64_pd(ymm2, 0x1b); \
    ymm3 = _mm256_permute4x64_pd(ymm3, 0x1b); \
    ymm2 = _mm256_permute4x64_pd(ymm2, 0xb1); \
    ymm3 = _mm256_permute4x64_pd(ymm3, 0xb1); \
    ymm4 = _mm256_blend_pd(ymm2, ymm0, 0x03); \
    ymm5 = _mm256_blend_pd(ymm3, ymm1, 0x03); \
    ymm6 = _mm256_blend_pd(ymm0, ymm2, 0x03); \
    ymm7 = _mm256_blend_pd(ymm1, ymm3, 0x03); \
    ymm4 += _mm256_loadu_pd(CO1 + (0 * ldc)); \
    ymm5 += _mm256_loadu_pd(CO1 + (1 * ldc)); \
    ymm6 += _mm256_loadu_pd(CO1 + (2 * ldc)); \
    ymm7 += _mm256_loadu_pd(CO1 + (3 * ldc)); \
    _mm256_storeu_pd(CO1 + (0 * ldc), ymm4); \
    _mm256_storeu_pd(CO1 + (1 * ldc), ymm5); \
    _mm256_storeu_pd(CO1 + (2 * ldc), ymm6); \
    _mm256_storeu_pd(CO1 + (3 * ldc), ymm7); \
    ymm9  = _mm256_permute4x64_pd(ymm9,  0xb1); \
    ymm11 = _mm256_permute4x64_pd(ymm11, 0xb1); \
    ymm0 = _mm256_blend_pd(ymm8,  ymm9,  0x0a); \
    ymm1 = _mm256_blend_pd(ymm8,  ymm9,  0x05); \
    ymm2 = _mm256_blend_pd(ymm10, ymm11, 0x0a); \
    ymm3 = _mm256_blend_pd(ymm10, ymm11, 0x05); \
    ymm2 = _mm256_permute4x64_pd(ymm2, 0x1b); \
    ymm3 = _mm256_permute4x64_pd(ymm3, 0x1b); \
    ymm2 = _mm256_permute4x64_pd(ymm2, 0xb1); \
    ymm3 = _mm256_permute4x64_pd(ymm3, 0xb1); \
    ymm4 = _mm256_blend_pd(ymm2, ymm0, 0x03); \
    ymm5 = _mm256_blend_pd(ymm3, ymm1, 0x03); \
    ymm6 = _mm256_blend_pd(ymm0, ymm2, 0x03); \
    ymm7 = _mm256_blend_pd(ymm1, ymm3, 0x03); \
    ymm4 += _mm256_loadu_pd(CO1 + (4 * ldc)); \
    ymm5 += _mm256_loadu_pd(CO1 + (5 * ldc)); \
    ymm6 += _mm256_loadu_pd(CO1 + (6 * ldc)); \
    ymm7 += _mm256_loadu_pd(CO1 + (7 * ldc)); \
    _mm256_storeu_pd(CO1 + (4 * ldc), ymm4); \
    _mm256_storeu_pd(CO1 + (5 * ldc), ymm5); \
    _mm256_storeu_pd(CO1 + (6 * ldc), ymm6); \
    _mm256_storeu_pd(CO1 + (7 * ldc), ymm7); \
    CO1 += 4;

/* 2x8 sub-kernel (SSE) */
#define INIT2x8() \
    xmm4 = _mm_setzero_pd(); xmm5 = _mm_setzero_pd(); \
    xmm6 = _mm_setzero_pd(); xmm7 = _mm_setzero_pd(); \
    xmm8 = _mm_setzero_pd(); xmm9 = _mm_setzero_pd(); \
    xmm10 = _mm_setzero_pd(); xmm11 = _mm_setzero_pd();

#define KERNEL2x8_SUB() \
    xmm0 = _mm_loadu_pd(AO - 16); \
    xmm1 = _mm_set1_pd(*(BO - 12)); xmm2 = _mm_set1_pd(*(BO - 11)); \
    xmm3 = _mm_set1_pd(*(BO - 10)); \
    xmm4 += xmm0 * xmm1; xmm1 = _mm_set1_pd(*(BO - 9)); \
    xmm5 += xmm0 * xmm2; xmm2 = _mm_set1_pd(*(BO - 8)); \
    xmm6 += xmm0 * xmm3; xmm3 = _mm_set1_pd(*(BO - 7)); \
    xmm7 += xmm0 * xmm1; xmm1 = _mm_set1_pd(*(BO - 6)); \
    xmm8 += xmm0 * xmm2; xmm2 = _mm_set1_pd(*(BO - 5)); \
    xmm9 += xmm0 * xmm3; \
    xmm10 += xmm0 * xmm1; xmm11 += xmm0 * xmm2; \
    BO += 8; AO += 2;

#define SAVE2x8(ALPHA) \
    xmm0 = _mm_set1_pd(ALPHA); \
    xmm4 *= xmm0; xmm5 *= xmm0; xmm6 *= xmm0; xmm7 *= xmm0; \
    xmm8 *= xmm0; xmm9 *= xmm0; xmm10 *= xmm0; xmm11 *= xmm0; \
    xmm4 += _mm_loadu_pd(CO1 + (0 * ldc)); \
    xmm5 += _mm_loadu_pd(CO1 + (1 * ldc)); \
    xmm6 += _mm_loadu_pd(CO1 + (2 * ldc)); \
    xmm7 += _mm_loadu_pd(CO1 + (3 * ldc)); \
    _mm_storeu_pd(CO1 + (0 * ldc), xmm4); \
    _mm_storeu_pd(CO1 + (1 * ldc), xmm5); \
    _mm_storeu_pd(CO1 + (2 * ldc), xmm6); \
    _mm_storeu_pd(CO1 + (3 * ldc), xmm7); \
    xmm8  += _mm_loadu_pd(CO1 + (4 * ldc)); \
    xmm9  += _mm_loadu_pd(CO1 + (5 * ldc)); \
    xmm10 += _mm_loadu_pd(CO1 + (6 * ldc)); \
    xmm11 += _mm_loadu_pd(CO1 + (7 * ldc)); \
    _mm_storeu_pd(CO1 + (4 * ldc), xmm8); \
    _mm_storeu_pd(CO1 + (5 * ldc), xmm9); \
    _mm_storeu_pd(CO1 + (6 * ldc), xmm10); \
    _mm_storeu_pd(CO1 + (7 * ldc), xmm11); \
    CO1 += 2;

/* 1x8 sub-kernel (scalar) */
#define INIT1x8() \
    dbl4=0; dbl5=0; dbl6=0; dbl7=0; dbl8=0; dbl9=0; dbl10=0; dbl11=0;

#define KERNEL1x8_SUB() \
    dbl0 = *(AO - 16); \
    dbl1 = *(BO - 12); dbl2 = *(BO - 11); dbl3 = *(BO - 10); \
    dbl4 += dbl0 * dbl1; dbl1 = *(BO - 9); \
    dbl5 += dbl0 * dbl2; dbl2 = *(BO - 8); \
    dbl6 += dbl0 * dbl3; dbl3 = *(BO - 7); \
    dbl7 += dbl0 * dbl1; dbl1 = *(BO - 6); \
    dbl8 += dbl0 * dbl2; dbl2 = *(BO - 5); \
    dbl9 += dbl0 * dbl3; \
    dbl10 += dbl0 * dbl1; dbl11 += dbl0 * dbl2; \
    BO += 8; AO += 1;

#define SAVE1x8(ALPHA) \
    dbl0 = ALPHA; \
    dbl4 *= dbl0; dbl5 *= dbl0; dbl6 *= dbl0; dbl7 *= dbl0; \
    dbl8 *= dbl0; dbl9 *= dbl0; dbl10 *= dbl0; dbl11 *= dbl0; \
    dbl4 += *(CO1 + (0 * ldc)); dbl5 += *(CO1 + (1 * ldc)); \
    dbl6 += *(CO1 + (2 * ldc)); dbl7 += *(CO1 + (3 * ldc)); \
    *(CO1 + (0 * ldc)) = dbl4; *(CO1 + (1 * ldc)) = dbl5; \
    *(CO1 + (2 * ldc)) = dbl6; *(CO1 + (3 * ldc)) = dbl7; \
    dbl8  += *(CO1 + (4 * ldc)); dbl9  += *(CO1 + (5 * ldc)); \
    dbl10 += *(CO1 + (6 * ldc)); dbl11 += *(CO1 + (7 * ldc)); \
    *(CO1 + (4 * ldc)) = dbl8;  *(CO1 + (5 * ldc)) = dbl9; \
    *(CO1 + (6 * ldc)) = dbl10; *(CO1 + (7 * ldc)) = dbl11; \
    CO1 += 1;

/* Smaller N-block macros: 4x4, 8x4, 4x2, 8x2, 4x1, 8x1, etc. */
/* 8x4 */
#define INIT8x4() \
    ymm10 = _mm256_setzero_pd(); ymm11 = _mm256_setzero_pd(); \
    ymm12 = _mm256_setzero_pd(); ymm13 = _mm256_setzero_pd(); \
    ymm14 = _mm256_setzero_pd(); ymm15 = _mm256_setzero_pd(); \
    ymm16 = _mm256_setzero_pd(); ymm17 = _mm256_setzero_pd();

#define KERNEL8x4_SUB() \
    ymm0 = _mm256_loadu_pd(AO - 16); ymm1 = _mm256_loadu_pd(AO - 12); \
    ymm2 = _mm256_set1_pd(*(BO - 12)); ymm3 = _mm256_set1_pd(*(BO - 11)); \
    ymm4 = _mm256_set1_pd(*(BO - 10)); ymm5 = _mm256_set1_pd(*(BO - 9)); \
    ymm10 += ymm0 * ymm2; ymm11 += ymm1 * ymm2; \
    ymm12 += ymm0 * ymm3; ymm13 += ymm1 * ymm3; \
    ymm14 += ymm0 * ymm4; ymm15 += ymm1 * ymm4; \
    ymm16 += ymm0 * ymm5; ymm17 += ymm1 * ymm5; \
    BO += 4; AO += 8;

#define SAVE8x4(ALPHA) \
    ymm0 = _mm256_set1_pd(ALPHA); \
    ymm10 *= ymm0; ymm11 *= ymm0; ymm12 *= ymm0; ymm13 *= ymm0; \
    ymm14 *= ymm0; ymm15 *= ymm0; ymm16 *= ymm0; ymm17 *= ymm0; \
    ymm10 += _mm256_loadu_pd(CO1);          ymm11 += _mm256_loadu_pd(CO1 + 4); \
    ymm12 += _mm256_loadu_pd(CO1 + ldc);    ymm13 += _mm256_loadu_pd(CO1 + ldc + 4); \
    ymm14 += _mm256_loadu_pd(CO1 + ldc*2);  ymm15 += _mm256_loadu_pd(CO1 + ldc*2 + 4); \
    ymm16 += _mm256_loadu_pd(CO1 + ldc*3);  ymm17 += _mm256_loadu_pd(CO1 + ldc*3 + 4); \
    _mm256_storeu_pd(CO1,          ymm10); _mm256_storeu_pd(CO1 + 4,       ymm11); \
    _mm256_storeu_pd(CO1 + ldc,    ymm12); _mm256_storeu_pd(CO1 + ldc + 4, ymm13); \
    _mm256_storeu_pd(CO1 + ldc*2,  ymm14); _mm256_storeu_pd(CO1 + ldc*2 + 4, ymm15); \
    _mm256_storeu_pd(CO1 + ldc*3,  ymm16); _mm256_storeu_pd(CO1 + ldc*3 + 4, ymm17); \
    CO1 += 8;

/* 4x4 */
#define INIT4x4() \
    ymm4 = _mm256_setzero_pd(); ymm5 = _mm256_setzero_pd(); \
    ymm6 = _mm256_setzero_pd(); ymm7 = _mm256_setzero_pd();

#define KERNEL4x4_SUB() \
    ymm0 = _mm256_loadu_pd(AO - 16); \
    ymm1 = _mm256_broadcastsd_pd(_mm_load_sd(BO - 12)); ymm4 += ymm0 * ymm1; \
    ymm1 = _mm256_broadcastsd_pd(_mm_load_sd(BO - 11)); ymm5 += ymm0 * ymm1; \
    ymm1 = _mm256_broadcastsd_pd(_mm_load_sd(BO - 10)); ymm6 += ymm0 * ymm1; \
    ymm1 = _mm256_broadcastsd_pd(_mm_load_sd(BO -  9)); ymm7 += ymm0 * ymm1; \
    AO += 4; BO += 4;

#define SAVE4x4(ALPHA) \
    ymm0 = _mm256_set1_pd(ALPHA); \
    ymm4 *= ymm0; ymm5 *= ymm0; ymm6 *= ymm0; ymm7 *= ymm0; \
    ymm4 += _mm256_loadu_pd(CO1 + (0 * ldc)); \
    ymm5 += _mm256_loadu_pd(CO1 + (1 * ldc)); \
    ymm6 += _mm256_loadu_pd(CO1 + (2 * ldc)); \
    ymm7 += _mm256_loadu_pd(CO1 + (3 * ldc)); \
    _mm256_storeu_pd(CO1 + (0 * ldc), ymm4); \
    _mm256_storeu_pd(CO1 + (1 * ldc), ymm5); \
    _mm256_storeu_pd(CO1 + (2 * ldc), ymm6); \
    _mm256_storeu_pd(CO1 + (3 * ldc), ymm7); \
    CO1 += 4;

/* 2x4, 1x4 */
#define INIT2x4() xmm4 = _mm_setzero_pd(); xmm5 = _mm_setzero_pd(); \
    xmm6 = _mm_setzero_pd(); xmm7 = _mm_setzero_pd();

#define KERNEL2x4_SUB() \
    xmm0 = _mm_loadu_pd(AO - 16); \
    xmm1 = _mm_set1_pd(*(BO - 12)); xmm2 = _mm_set1_pd(*(BO - 11)); \
    xmm3 = _mm_set1_pd(*(BO - 10)); \
    xmm4 += xmm0 * xmm1; xmm1 = _mm_set1_pd(*(BO - 9)); \
    xmm5 += xmm0 * xmm2; xmm6 += xmm0 * xmm3; xmm7 += xmm0 * xmm1; \
    BO += 4; AO += 2;

#define SAVE2x4(ALPHA) \
    xmm0 = _mm_set1_pd(ALPHA); \
    xmm4 *= xmm0; xmm5 *= xmm0; xmm6 *= xmm0; xmm7 *= xmm0; \
    xmm4 += _mm_loadu_pd(CO1 + (0 * ldc)); xmm5 += _mm_loadu_pd(CO1 + (1 * ldc)); \
    xmm6 += _mm_loadu_pd(CO1 + (2 * ldc)); xmm7 += _mm_loadu_pd(CO1 + (3 * ldc)); \
    _mm_storeu_pd(CO1 + (0 * ldc), xmm4); _mm_storeu_pd(CO1 + (1 * ldc), xmm5); \
    _mm_storeu_pd(CO1 + (2 * ldc), xmm6); _mm_storeu_pd(CO1 + (3 * ldc), xmm7); \
    CO1 += 2;

#define INIT1x4() dbl4=0; dbl5=0; dbl6=0; dbl7=0;

#define KERNEL1x4_SUB() \
    dbl0 = *(AO - 16); dbl1 = *(BO - 12); dbl2 = *(BO - 11); \
    dbl3 = *(BO - 10); dbl8 = *(BO - 9); \
    dbl4 += dbl0 * dbl1; dbl5 += dbl0 * dbl2; \
    dbl6 += dbl0 * dbl3; dbl7 += dbl0 * dbl8; \
    BO += 4; AO += 1;

#define SAVE1x4(ALPHA) \
    dbl0 = ALPHA; dbl4 *= dbl0; dbl5 *= dbl0; dbl6 *= dbl0; dbl7 *= dbl0; \
    dbl4 += *(CO1 + (0 * ldc)); dbl5 += *(CO1 + (1 * ldc)); \
    dbl6 += *(CO1 + (2 * ldc)); dbl7 += *(CO1 + (3 * ldc)); \
    *(CO1 + (0 * ldc)) = dbl4; *(CO1 + (1 * ldc)) = dbl5; \
    *(CO1 + (2 * ldc)) = dbl6; *(CO1 + (3 * ldc)) = dbl7; \
    CO1 += 1;

/* 8x2, 4x2, 2x2, 1x2 */
#define INIT8x2() ymm4 = _mm256_setzero_pd(); ymm5 = _mm256_setzero_pd(); \
    ymm6 = _mm256_setzero_pd(); ymm7 = _mm256_setzero_pd();

#define KERNEL8x2_SUB() \
    ymm0 = _mm256_loadu_pd(AO - 16); ymm1 = _mm256_loadu_pd(AO - 12); \
    ymm2 = _mm256_set1_pd(*(BO - 12)); ymm3 = _mm256_set1_pd(*(BO - 11)); \
    ymm4 += ymm0 * ymm2; ymm5 += ymm1 * ymm2; \
    ymm6 += ymm0 * ymm3; ymm7 += ymm1 * ymm3; \
    BO += 2; AO += 8;

#define SAVE8x2(ALPHA) \
    ymm0 = _mm256_set1_pd(ALPHA); ymm4 *= ymm0; ymm5 *= ymm0; ymm6 *= ymm0; ymm7 *= ymm0; \
    ymm4 += _mm256_loadu_pd(CO1);       ymm5 += _mm256_loadu_pd(CO1 + 4); \
    ymm6 += _mm256_loadu_pd(CO1 + ldc); ymm7 += _mm256_loadu_pd(CO1 + ldc + 4); \
    _mm256_storeu_pd(CO1, ymm4);       _mm256_storeu_pd(CO1 + 4, ymm5); \
    _mm256_storeu_pd(CO1 + ldc, ymm6); _mm256_storeu_pd(CO1 + ldc + 4, ymm7); \
    CO1 += 8;

#define INIT4x2() xmm4 = _mm_setzero_pd(); xmm5 = _mm_setzero_pd(); \
    xmm6 = _mm_setzero_pd(); xmm7 = _mm_setzero_pd();

#define KERNEL4x2_SUB() \
    xmm0 = _mm_loadu_pd(AO - 16); xmm1 = _mm_loadu_pd(AO - 14); \
    xmm2 = _mm_set1_pd(*(BO - 12)); xmm3 = _mm_set1_pd(*(BO - 11)); \
    xmm4 += xmm0 * xmm2; xmm5 += xmm1 * xmm2; \
    xmm6 += xmm0 * xmm3; xmm7 += xmm1 * xmm3; \
    BO += 2; AO += 4;

#define SAVE4x2(ALPHA) \
    xmm0 = _mm_set1_pd(ALPHA); xmm4 *= xmm0; xmm5 *= xmm0; xmm6 *= xmm0; xmm7 *= xmm0; \
    xmm4 += _mm_loadu_pd(CO1);     xmm5 += _mm_loadu_pd(CO1 + 2); \
    xmm6 += _mm_loadu_pd(CO1+ldc); xmm7 += _mm_loadu_pd(CO1+ldc+2); \
    _mm_storeu_pd(CO1,     xmm4); _mm_storeu_pd(CO1 + 2,   xmm5); \
    _mm_storeu_pd(CO1+ldc, xmm6); _mm_storeu_pd(CO1+ldc+2, xmm7); \
    CO1 += 4;

#define INIT2x2() xmm4 = _mm_setzero_pd(); xmm6 = _mm_setzero_pd();
#define KERNEL2x2_SUB() \
    xmm2 = _mm_set1_pd(*(BO - 12)); xmm0 = _mm_loadu_pd(AO - 16); \
    xmm3 = _mm_set1_pd(*(BO - 11)); \
    xmm4 += xmm0 * xmm2; xmm6 += xmm0 * xmm3; \
    BO += 2; AO += 2;
#define SAVE2x2(ALPHA) \
    xmm0 = _mm_set1_pd(ALPHA); xmm4 *= xmm0; xmm6 *= xmm0; \
    xmm4 += _mm_loadu_pd(CO1); xmm6 += _mm_loadu_pd(CO1 + ldc); \
    _mm_storeu_pd(CO1, xmm4); _mm_storeu_pd(CO1 + ldc, xmm6); \
    CO1 += 2;

#define INIT1x2() dbl4 = 0; dbl5 = 0;
#define KERNEL1x2_SUB() \
    dbl0 = *(AO - 16); dbl1 = *(BO - 12); dbl2 = *(BO - 11); \
    dbl4 += dbl0 * dbl1; dbl5 += dbl0 * dbl2; \
    BO += 2; AO += 1;
#define SAVE1x2(ALPHA) \
    dbl0 = ALPHA; dbl4 *= dbl0; dbl5 *= dbl0; \
    dbl4 += *(CO1 + (0 * ldc)); dbl5 += *(CO1 + (1 * ldc)); \
    *(CO1 + (0 * ldc)) = dbl4; *(CO1 + (1 * ldc)) = dbl5; \
    CO1 += 1;

/* 8x1, 4x1, 2x1, 1x1 */
#define INIT8x1() zmm4 = _mm512_setzero_pd();
#define KERNEL8x1_SUB() \
    zmm2 = _mm512_set1_pd(*(BO - 12)); zmm0 = _mm512_loadu_pd(AO - 16); \
    zmm4 += zmm0 * zmm2; BO += 1; AO += 8;
#define SAVE8x1(ALPHA) \
    zmm0 = _mm512_set1_pd(ALPHA); zmm4 *= zmm0; \
    zmm4 += _mm512_loadu_pd(CO1); _mm512_storeu_pd(CO1, zmm4); CO1 += 8;

#define INIT4x1() ymm4 = _mm256_setzero_pd(); ymm5 = _mm256_setzero_pd(); \
    ymm6 = _mm256_setzero_pd(); ymm7 = _mm256_setzero_pd();
#define KERNEL4x1_SUB() \
    ymm2 = _mm256_set1_pd(*(BO - 12)); ymm0 = _mm256_loadu_pd(AO - 16); \
    ymm4 += ymm0 * ymm2; BO += 1; AO += 4;
#define SAVE4x1(ALPHA) \
    ymm0 = _mm256_set1_pd(ALPHA); \
    ymm4 += ymm5; ymm6 += ymm7; ymm4 += ymm6; ymm4 *= ymm0; \
    ymm4 += _mm256_loadu_pd(CO1); _mm256_storeu_pd(CO1, ymm4); CO1 += 4;

#define INIT2x1() xmm4 = _mm_setzero_pd();
#define KERNEL2x1_SUB() \
    xmm2 = _mm_set1_pd(*(BO - 12)); xmm0 = _mm_loadu_pd(AO - 16); \
    xmm4 += xmm0 * xmm2; BO += 1; AO += 2;
#define SAVE2x1(ALPHA) \
    xmm0 = _mm_set1_pd(ALPHA); xmm4 *= xmm0; \
    xmm4 += _mm_loadu_pd(CO1); _mm_storeu_pd(CO1, xmm4); CO1 += 2;

#define INIT1x1() dbl4 = 0;
#define KERNEL1x1_SUB() \
    dbl1 = *(BO - 12); dbl0 = *(AO - 16); dbl4 += dbl0 * dbl1; BO += 1; AO += 1;
#define SAVE1x1(ALPHA) \
    dbl0 = ALPHA; dbl4 *= dbl0; dbl4 += *CO1; *CO1 = dbl4; CO1 += 1;


static int __attribute__((noinline))
skylakex_kernel(BLASLONG m, BLASLONG n, BLASLONG k, double alpha,
                double * __restrict__ A, double * __restrict__ B,
                double * __restrict__ C, BLASLONG ldc)
{
    unsigned long M = m, N = n, K = k;

    if (M == 0 || N == 0 || K == 0)
        return 0;

    while (N >= 8) {
        double *CO1 = C;
        double *AO;
        int i;

        C += 8 * ldc;
        AO = A + 16;
        i = m;

        /* 24×8 hot path — AVX-512 inline asm */
        while (i >= 24) {
            double *BO = B + 12;
            double *A1 = AO + 8 * K;
            double *A2 = AO + 16 * K;
            int kloop = K;
            asm(
            "vxorpd  %%zmm1, %%zmm1, %%zmm1\n"
            "vmovapd %%zmm1, %%zmm2\n" "vmovapd %%zmm1, %%zmm3\n"
            "vmovapd %%zmm1, %%zmm4\n" "vmovapd %%zmm1, %%zmm5\n"
            "vmovapd %%zmm1, %%zmm6\n" "vmovapd %%zmm1, %%zmm7\n"
            "vmovapd %%zmm1, %%zmm8\n"
            "vmovapd %%zmm1, %%zmm11\n" "vmovapd %%zmm1, %%zmm12\n"
            "vmovapd %%zmm1, %%zmm13\n" "vmovapd %%zmm1, %%zmm14\n"
            "vmovapd %%zmm1, %%zmm15\n" "vmovapd %%zmm1, %%zmm16\n"
            "vmovapd %%zmm1, %%zmm17\n" "vmovapd %%zmm1, %%zmm18\n"
            "vmovapd %%zmm1, %%zmm21\n" "vmovapd %%zmm1, %%zmm22\n"
            "vmovapd %%zmm1, %%zmm23\n" "vmovapd %%zmm1, %%zmm24\n"
            "vmovapd %%zmm1, %%zmm25\n" "vmovapd %%zmm1, %%zmm26\n"
            "vmovapd %%zmm1, %%zmm27\n" "vmovapd %%zmm1, %%zmm28\n"
            "jmp .Lmex_label24\n"
            ".p2align 5\n"
            ".Lmex_label24:\n"
            "vmovupd     -128(%[AO]),%%zmm0\n"
            "vmovupd     -128(%[A1]),%%zmm10\n"
            "vmovupd     -128(%[A2]),%%zmm20\n"
            "vbroadcastsd -96(%[BO]), %%zmm9\n"
            "vfmadd231pd %%zmm9, %%zmm0,  %%zmm1\n"
            "vfmadd231pd %%zmm9, %%zmm10, %%zmm11\n"
            "vfmadd231pd %%zmm9, %%zmm20, %%zmm21\n"
            "vbroadcastsd -88(%[BO]), %%zmm9\n"
            "vfmadd231pd %%zmm9, %%zmm0,  %%zmm2\n"
            "vfmadd231pd %%zmm9, %%zmm10, %%zmm12\n"
            "vfmadd231pd %%zmm9, %%zmm20, %%zmm22\n"
            "vbroadcastsd -80(%[BO]), %%zmm9\n"
            "vfmadd231pd %%zmm9, %%zmm0,  %%zmm3\n"
            "vfmadd231pd %%zmm9, %%zmm10, %%zmm13\n"
            "vfmadd231pd %%zmm9, %%zmm20, %%zmm23\n"
            "vbroadcastsd -72(%[BO]), %%zmm9\n"
            "vfmadd231pd %%zmm9, %%zmm0,  %%zmm4\n"
            "vfmadd231pd %%zmm9, %%zmm10, %%zmm14\n"
            "vfmadd231pd %%zmm9, %%zmm20, %%zmm24\n"
            "vbroadcastsd -64(%[BO]), %%zmm9\n"
            "vfmadd231pd %%zmm9, %%zmm0,  %%zmm5\n"
            "vfmadd231pd %%zmm9, %%zmm10, %%zmm15\n"
            "vfmadd231pd %%zmm9, %%zmm20, %%zmm25\n"
            "vbroadcastsd -56(%[BO]), %%zmm9\n"
            "vfmadd231pd %%zmm9, %%zmm0,  %%zmm6\n"
            "vfmadd231pd %%zmm9, %%zmm10, %%zmm16\n"
            "vfmadd231pd %%zmm9, %%zmm20, %%zmm26\n"
            "vbroadcastsd -48(%[BO]), %%zmm9\n"
            "vfmadd231pd %%zmm9, %%zmm0,  %%zmm7\n"
            "vfmadd231pd %%zmm9, %%zmm10, %%zmm17\n"
            "vfmadd231pd %%zmm9, %%zmm20, %%zmm27\n"
            "vbroadcastsd -40(%[BO]), %%zmm9\n"
            "vfmadd231pd %%zmm9, %%zmm0,  %%zmm8\n"
            "vfmadd231pd %%zmm9, %%zmm10, %%zmm18\n"
            "vfmadd231pd %%zmm9, %%zmm20, %%zmm28\n"
            "add $64, %[AO]\n" "add $64, %[A1]\n"
            "add $64, %[A2]\n" "add $64, %[BO]\n"
            "prefetch 512(%[AO])\n" "prefetch 512(%[A1])\n"
            "prefetch 512(%[A2])\n" "prefetch 512(%[BO])\n"
            "subl $1, %[kloop]\n"
            "jg .Lmex_label24\n"
            "vbroadcastsd (%[alpha]), %%zmm9\n"
            "vfmadd213pd (%[C0]), %%zmm9, %%zmm1\n"
            "vfmadd213pd (%[C1]), %%zmm9, %%zmm2\n"
            "vfmadd213pd (%[C2]), %%zmm9, %%zmm3\n"
            "vfmadd213pd (%[C3]), %%zmm9, %%zmm4\n"
            "vfmadd213pd (%[C4]), %%zmm9, %%zmm5\n"
            "vfmadd213pd (%[C5]), %%zmm9, %%zmm6\n"
            "vfmadd213pd (%[C6]), %%zmm9, %%zmm7\n"
            "vfmadd213pd (%[C7]), %%zmm9, %%zmm8\n"
            "vmovupd %%zmm1, (%[C0])\n" "vmovupd %%zmm2, (%[C1])\n"
            "vmovupd %%zmm3, (%[C2])\n" "vmovupd %%zmm4, (%[C3])\n"
            "vmovupd %%zmm5, (%[C4])\n" "vmovupd %%zmm6, (%[C5])\n"
            "vmovupd %%zmm7, (%[C6])\n" "vmovupd %%zmm8, (%[C7])\n"
            "vfmadd213pd 64(%[C0]), %%zmm9, %%zmm11\n"
            "vfmadd213pd 64(%[C1]), %%zmm9, %%zmm12\n"
            "vfmadd213pd 64(%[C2]), %%zmm9, %%zmm13\n"
            "vfmadd213pd 64(%[C3]), %%zmm9, %%zmm14\n"
            "vfmadd213pd 64(%[C4]), %%zmm9, %%zmm15\n"
            "vfmadd213pd 64(%[C5]), %%zmm9, %%zmm16\n"
            "vfmadd213pd 64(%[C6]), %%zmm9, %%zmm17\n"
            "vfmadd213pd 64(%[C7]), %%zmm9, %%zmm18\n"
            "vmovupd %%zmm11, 64(%[C0])\n" "vmovupd %%zmm12, 64(%[C1])\n"
            "vmovupd %%zmm13, 64(%[C2])\n" "vmovupd %%zmm14, 64(%[C3])\n"
            "vmovupd %%zmm15, 64(%[C4])\n" "vmovupd %%zmm16, 64(%[C5])\n"
            "vmovupd %%zmm17, 64(%[C6])\n" "vmovupd %%zmm18, 64(%[C7])\n"
            "vfmadd213pd 128(%[C0]), %%zmm9, %%zmm21\n"
            "vfmadd213pd 128(%[C1]), %%zmm9, %%zmm22\n"
            "vfmadd213pd 128(%[C2]), %%zmm9, %%zmm23\n"
            "vfmadd213pd 128(%[C3]), %%zmm9, %%zmm24\n"
            "vfmadd213pd 128(%[C4]), %%zmm9, %%zmm25\n"
            "vfmadd213pd 128(%[C5]), %%zmm9, %%zmm26\n"
            "vfmadd213pd 128(%[C6]), %%zmm9, %%zmm27\n"
            "vfmadd213pd 128(%[C7]), %%zmm9, %%zmm28\n"
            "vmovupd %%zmm21, 128(%[C0])\n" "vmovupd %%zmm22, 128(%[C1])\n"
            "vmovupd %%zmm23, 128(%[C2])\n" "vmovupd %%zmm24, 128(%[C3])\n"
            "vmovupd %%zmm25, 128(%[C4])\n" "vmovupd %%zmm26, 128(%[C5])\n"
            "vmovupd %%zmm27, 128(%[C6])\n" "vmovupd %%zmm28, 128(%[C7])\n"
            : [AO] "+r" (AO), [A1] "+r" (A1), [A2] "+r" (A2),
              [BO] "+r" (BO), [C0] "+r" (CO1), [kloop] "+r" (kloop)
            : [alpha] "r" (&alpha),
              [C1] "r" (CO1 + 1*ldc), [C2] "r" (CO1 + 2*ldc),
              [C3] "r" (CO1 + 3*ldc), [C4] "r" (CO1 + 4*ldc),
              [C5] "r" (CO1 + 5*ldc), [C6] "r" (CO1 + 6*ldc),
              [C7] "r" (CO1 + 7*ldc)
            : "memory", "zmm0","zmm1","zmm2","zmm3","zmm4","zmm5","zmm6","zmm7","zmm8","zmm9",
              "zmm10","zmm11","zmm12","zmm13","zmm14","zmm15","zmm16","zmm17","zmm18",
              "zmm20","zmm21","zmm22","zmm23","zmm24","zmm25","zmm26","zmm27","zmm28"
            );
            CO1 += 24;
            AO += 16 * K;
            i -= 24;
        }

        /* 16×8 — AVX-512 inline asm */
        while (i >= 16) {
            double *BO = B + 12;
            double *A1 = AO + 8 * K;
            int kloop = K;
            asm(
            "vxorpd  %%zmm1, %%zmm1, %%zmm1\n"
            "vmovapd %%zmm1, %%zmm2\n" "vmovapd %%zmm1, %%zmm3\n"
            "vmovapd %%zmm1, %%zmm4\n" "vmovapd %%zmm1, %%zmm5\n"
            "vmovapd %%zmm1, %%zmm6\n" "vmovapd %%zmm1, %%zmm7\n"
            "vmovapd %%zmm1, %%zmm8\n"
            "vmovapd %%zmm1, %%zmm11\n" "vmovapd %%zmm1, %%zmm12\n"
            "vmovapd %%zmm1, %%zmm13\n" "vmovapd %%zmm1, %%zmm14\n"
            "vmovapd %%zmm1, %%zmm15\n" "vmovapd %%zmm1, %%zmm16\n"
            "vmovapd %%zmm1, %%zmm17\n" "vmovapd %%zmm1, %%zmm18\n"
            "jmp .Lmex_label16\n"
            ".p2align 5\n"
            ".Lmex_label16:\n"
            "vmovupd     -128(%[AO]),%%zmm0\n"
            "vmovupd     -128(%[A1]),%%zmm10\n"
            "vbroadcastsd -96(%[BO]), %%zmm9\n"
            "vfmadd231pd %%zmm9, %%zmm0,  %%zmm1\n"
            "vfmadd231pd %%zmm9, %%zmm10, %%zmm11\n"
            "vbroadcastsd -88(%[BO]), %%zmm9\n"
            "vfmadd231pd %%zmm9, %%zmm0,  %%zmm2\n"
            "vfmadd231pd %%zmm9, %%zmm10, %%zmm12\n"
            "vbroadcastsd -80(%[BO]), %%zmm9\n"
            "vfmadd231pd %%zmm9, %%zmm0,  %%zmm3\n"
            "vfmadd231pd %%zmm9, %%zmm10, %%zmm13\n"
            "vbroadcastsd -72(%[BO]), %%zmm9\n"
            "vfmadd231pd %%zmm9, %%zmm0,  %%zmm4\n"
            "vfmadd231pd %%zmm9, %%zmm10, %%zmm14\n"
            "vbroadcastsd -64(%[BO]), %%zmm9\n"
            "vfmadd231pd %%zmm9, %%zmm0,  %%zmm5\n"
            "vfmadd231pd %%zmm9, %%zmm10, %%zmm15\n"
            "vbroadcastsd -56(%[BO]), %%zmm9\n"
            "vfmadd231pd %%zmm9, %%zmm0,  %%zmm6\n"
            "vfmadd231pd %%zmm9, %%zmm10, %%zmm16\n"
            "vbroadcastsd -48(%[BO]), %%zmm9\n"
            "vfmadd231pd %%zmm9, %%zmm0,  %%zmm7\n"
            "vfmadd231pd %%zmm9, %%zmm10, %%zmm17\n"
            "vbroadcastsd -40(%[BO]), %%zmm9\n"
            "vfmadd231pd %%zmm9, %%zmm0,  %%zmm8\n"
            "vfmadd231pd %%zmm9, %%zmm10, %%zmm18\n"
            "add $64, %[AO]\n" "add $64, %[A1]\n" "add $64, %[BO]\n"
            "prefetch 512(%[AO])\n" "prefetch 512(%[A1])\n" "prefetch 512(%[BO])\n"
            "subl $1, %[kloop]\n"
            "jg .Lmex_label16\n"
            "vbroadcastsd (%[alpha]), %%zmm9\n"
            "vfmadd213pd (%[C0]), %%zmm9, %%zmm1\n"
            "vfmadd213pd (%[C1]), %%zmm9, %%zmm2\n"
            "vfmadd213pd (%[C2]), %%zmm9, %%zmm3\n"
            "vfmadd213pd (%[C3]), %%zmm9, %%zmm4\n"
            "vfmadd213pd (%[C4]), %%zmm9, %%zmm5\n"
            "vfmadd213pd (%[C5]), %%zmm9, %%zmm6\n"
            "vfmadd213pd (%[C6]), %%zmm9, %%zmm7\n"
            "vfmadd213pd (%[C7]), %%zmm9, %%zmm8\n"
            "vmovupd %%zmm1, (%[C0])\n" "vmovupd %%zmm2, (%[C1])\n"
            "vmovupd %%zmm3, (%[C2])\n" "vmovupd %%zmm4, (%[C3])\n"
            "vmovupd %%zmm5, (%[C4])\n" "vmovupd %%zmm6, (%[C5])\n"
            "vmovupd %%zmm7, (%[C6])\n" "vmovupd %%zmm8, (%[C7])\n"
            "vfmadd213pd 64(%[C0]), %%zmm9, %%zmm11\n"
            "vfmadd213pd 64(%[C1]), %%zmm9, %%zmm12\n"
            "vfmadd213pd 64(%[C2]), %%zmm9, %%zmm13\n"
            "vfmadd213pd 64(%[C3]), %%zmm9, %%zmm14\n"
            "vfmadd213pd 64(%[C4]), %%zmm9, %%zmm15\n"
            "vfmadd213pd 64(%[C5]), %%zmm9, %%zmm16\n"
            "vfmadd213pd 64(%[C6]), %%zmm9, %%zmm17\n"
            "vfmadd213pd 64(%[C7]), %%zmm9, %%zmm18\n"
            "vmovupd %%zmm11, 64(%[C0])\n" "vmovupd %%zmm12, 64(%[C1])\n"
            "vmovupd %%zmm13, 64(%[C2])\n" "vmovupd %%zmm14, 64(%[C3])\n"
            "vmovupd %%zmm15, 64(%[C4])\n" "vmovupd %%zmm16, 64(%[C5])\n"
            "vmovupd %%zmm17, 64(%[C6])\n" "vmovupd %%zmm18, 64(%[C7])\n"
            : [AO] "+r" (AO), [A1] "+r" (A1),
              [BO] "+r" (BO), [C0] "+r" (CO1), [kloop] "+r" (kloop)
            : [alpha] "r" (&alpha),
              [C1] "r" (CO1 + 1*ldc), [C2] "r" (CO1 + 2*ldc),
              [C3] "r" (CO1 + 3*ldc), [C4] "r" (CO1 + 4*ldc),
              [C5] "r" (CO1 + 5*ldc), [C6] "r" (CO1 + 6*ldc),
              [C7] "r" (CO1 + 7*ldc)
            : "memory", "zmm0","zmm1","zmm2","zmm3","zmm4","zmm5","zmm6","zmm7","zmm8","zmm9",
              "zmm10","zmm11","zmm12","zmm13","zmm14","zmm15","zmm16","zmm17","zmm18"
            );
            CO1 += 16;
            AO += 8 * K;
            i -= 16;
        }

        /* 8×8 — AVX-512 inline asm with broadcast-from-memory */
        while (i >= 8) {
            double *BO = B + 12;
            int kloop = K;
            asm(
            "vxorpd  %%zmm1, %%zmm1, %%zmm1\n"
            "vmovapd %%zmm1, %%zmm2\n" "vmovapd %%zmm1, %%zmm3\n"
            "vmovapd %%zmm1, %%zmm4\n" "vmovapd %%zmm1, %%zmm5\n"
            "vmovapd %%zmm1, %%zmm6\n" "vmovapd %%zmm1, %%zmm7\n"
            "vmovapd %%zmm1, %%zmm8\n"
            "vbroadcastsd (%[alpha]), %%zmm9\n"
            "jmp .Lmex_label1\n"
            ".p2align 5\n"
            ".Lmex_label1:\n"
            "vmovupd     -128(%[AO]),%%zmm0\n"
            "vfmadd231pd  -96(%[BO])%{1to8%}, %%zmm0, %%zmm1\n"
            "vfmadd231pd  -88(%[BO])%{1to8%}, %%zmm0, %%zmm2\n"
            "vfmadd231pd  -80(%[BO])%{1to8%}, %%zmm0, %%zmm3\n"
            "vfmadd231pd  -72(%[BO])%{1to8%}, %%zmm0, %%zmm4\n"
            "vfmadd231pd  -64(%[BO])%{1to8%}, %%zmm0, %%zmm5\n"
            "vfmadd231pd  -56(%[BO])%{1to8%}, %%zmm0, %%zmm6\n"
            "vfmadd231pd  -48(%[BO])%{1to8%}, %%zmm0, %%zmm7\n"
            "vfmadd231pd  -40(%[BO])%{1to8%}, %%zmm0, %%zmm8\n"
            "add $64, %[AO]\n" "add $64, %[BO]\n"
            "subl $1, %[kloop]\n"
            "jg .Lmex_label1\n"
            "vfmadd213pd (%[C0]), %%zmm9, %%zmm1\n"
            "vfmadd213pd (%[C1]), %%zmm9, %%zmm2\n"
            "vfmadd213pd (%[C2]), %%zmm9, %%zmm3\n"
            "vfmadd213pd (%[C3]), %%zmm9, %%zmm4\n"
            "vfmadd213pd (%[C4]), %%zmm9, %%zmm5\n"
            "vfmadd213pd (%[C5]), %%zmm9, %%zmm6\n"
            "vfmadd213pd (%[C6]), %%zmm9, %%zmm7\n"
            "vfmadd213pd (%[C7]), %%zmm9, %%zmm8\n"
            "vmovupd %%zmm1, (%[C0])\n" "vmovupd %%zmm2, (%[C1])\n"
            "vmovupd %%zmm3, (%[C2])\n" "vmovupd %%zmm4, (%[C3])\n"
            "vmovupd %%zmm5, (%[C4])\n" "vmovupd %%zmm6, (%[C5])\n"
            "vmovupd %%zmm7, (%[C6])\n" "vmovupd %%zmm8, (%[C7])\n"
            : [AO] "+r" (AO), [BO] "+r" (BO),
              [C0] "+r" (CO1), [kloop] "+r" (kloop)
            : [alpha] "r" (&alpha),
              [C1] "r" (CO1 + 1*ldc), [C2] "r" (CO1 + 2*ldc),
              [C3] "r" (CO1 + 3*ldc), [C4] "r" (CO1 + 4*ldc),
              [C5] "r" (CO1 + 5*ldc), [C6] "r" (CO1 + 6*ldc),
              [C7] "r" (CO1 + 7*ldc)
            : "memory", "zmm0","zmm1","zmm2","zmm3","zmm4","zmm5","zmm6","zmm7","zmm8","zmm9"
            );
            CO1 += 8;
            i -= 8;
        }

        /* 4×8 — AVX2 intrinsics */
        while (i >= 4) {
            double *BO;
            __m256d ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7, ymm8, ymm9, ymm10, ymm11;
            int kloop = K;
            BO = B + 12;
            INIT4x8()
            while (kloop > 0) { KERNEL4x8_SUB() kloop--; }
            SAVE4x8(alpha)
            i -= 4;
        }

        /* 2×8 — SSE intrinsics */
        while (i >= 2) {
            double *BO;
            __m128d xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8, xmm9, xmm10, xmm11;
            int kloop = K;
            BO = B + 12;
            INIT2x8()
            while (kloop > 0) { KERNEL2x8_SUB() kloop--; }
            SAVE2x8(alpha)
            i -= 2;
        }

        /* 1×8 — scalar */
        while (i >= 1) {
            double *BO;
            double dbl0, dbl1, dbl2, dbl3, dbl4, dbl5, dbl6, dbl7, dbl8, dbl9, dbl10, dbl11;
            int kloop = K;
            BO = B + 12;
            INIT1x8()
            while (kloop > 0) { KERNEL1x8_SUB() kloop--; }
            SAVE1x8(alpha)
            i -= 1;
        }
        B += K * 8;
        N -= 8;
    }

    if (N == 0) return 0;

    /* N=4 block */
    while (N >= 4) {
        double *CO1 = C;
        double *AO;
        int i;
        C += 4 * ldc;
        AO = A + 16;
        i = m;

        while (i >= 8) {
            double *BO;
            __m256d ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm10, ymm11, ymm12, ymm13, ymm14, ymm15, ymm16, ymm17;
            BO = B + 12;
            int kloop = K;
            INIT8x4()
            while (kloop > 0) { KERNEL8x4_SUB() kloop--; }
            SAVE8x4(alpha)
            i -= 8;
        }
        while (i >= 4) {
            double *BO;
            __m256d ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7;
            BO = B + 12;
            int kloop = K;
            INIT4x4()
            while (kloop > 0) { KERNEL4x4_SUB() kloop--; }
            SAVE4x4(alpha)
            i -= 4;
        }
        while (i >= 2) {
            double *BO;
            __m128d xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7;
            BO = B + 12;
            INIT2x4()
            int kloop = K;
            while (kloop > 0) { KERNEL2x4_SUB() kloop--; }
            SAVE2x4(alpha)
            i -= 2;
        }
        while (i >= 1) {
            double *BO;
            double dbl0, dbl1, dbl2, dbl3, dbl4, dbl5, dbl6, dbl7, dbl8;
            int kloop = K;
            BO = B + 12;
            INIT1x4()
            while (kloop > 0) { KERNEL1x4_SUB() kloop--; }
            SAVE1x4(alpha)
            i -= 1;
        }
        B += K * 4;
        N -= 4;
    }

    /* N=2 block */
    while (N >= 2) {
        double *CO1 = C;
        double *AO;
        int i;
        C += 2 * ldc;
        AO = A + 16;
        i = m;

        while (i >= 8) {
            double *BO;
            __m256d ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7;
            BO = B + 12;
            int kloop = K;
            INIT8x2()
            while (kloop > 0) { KERNEL8x2_SUB() kloop--; }
            SAVE8x2(alpha)
            i -= 8;
        }
        while (i >= 4) {
            double *BO;
            __m128d xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7;
            BO = B + 12;
            int kloop = K;
            INIT4x2()
            while (kloop > 0) { KERNEL4x2_SUB() kloop--; }
            SAVE4x2(alpha)
            i -= 4;
        }
        while (i >= 2) {
            double *BO;
            __m128d xmm0, xmm2, xmm3, xmm4, xmm6;
            int kloop = K;
            BO = B + 12;
            INIT2x2()
            while (kloop > 0) { KERNEL2x2_SUB() kloop--; }
            SAVE2x2(alpha)
            i -= 2;
        }
        while (i >= 1) {
            double *BO;
            double dbl0, dbl1, dbl2, dbl4, dbl5;
            int kloop = K;
            BO = B + 12;
            INIT1x2()
            while (kloop > 0) { KERNEL1x2_SUB() kloop--; }
            SAVE1x2(alpha)
            i -= 1;
        }
        B += K * 2;
        N -= 2;
    }

    /* N=1 block */
    while (N >= 1) {
        double *CO1 = C;
        double *AO;
        int i;
        C += ldc;
        AO = A + 16;
        i = m;

        while (i >= 8) {
            double *BO;
            __m512d zmm0, zmm2, zmm4;
            BO = B + 12;
            int kloop = K;
            INIT8x1()
            while (kloop > 0) { KERNEL8x1_SUB() kloop--; }
            SAVE8x1(alpha)
            i -= 8;
        }
        while (i >= 4) {
            double *BO;
            __m256d ymm0, ymm2, ymm4, ymm5, ymm6, ymm7;
            BO = B + 12;
            int kloop = K;
            INIT4x1()
            while (kloop > 0) { KERNEL4x1_SUB() kloop--; }
            SAVE4x1(alpha)
            i -= 4;
        }
        while (i >= 2) {
            double *BO;
            __m128d xmm0, xmm2, xmm4;
            int kloop = K;
            BO = B + 12;
            INIT2x1()
            while (kloop > 0) { KERNEL2x1_SUB() kloop--; }
            SAVE2x1(alpha)
            i -= 2;
        }
        while (i >= 1) {
            double *BO;
            double dbl0, dbl1, dbl4;
            int kloop = K;
            BO = B + 12;
            INIT1x1()
            while (kloop > 0) { KERNEL1x1_SUB() kloop--; }
            SAVE1x1(alpha)
            i -= 1;
        }
        B += K * 1;
        N -= 1;
    }

    return 0;
}


/* ========================================================================
 * Factor nthreads into a tm × tn grid matching the aspect ratio of
 * m_tiles × n_tiles.  Picks the factoring that minimises the ratio
 * distance so each thread gets a roughly square work rectangle.
 * ======================================================================== */
static void factor_threads(int nthreads, BLASLONG m_tiles, BLASLONG n_tiles,
                           int *tm_out, int *tn_out)
{
    int best_tm = 1, best_tn = nthreads;
    double best_diff = 1e30;
    double target = (m_tiles > 0 && n_tiles > 0)
                    ? (double)m_tiles / (double)n_tiles
                    : 1.0;

    int f;
    for (f = 1; f * f <= nthreads; f++) {
        if (nthreads % f != 0) continue;
        int g = nthreads / f;

        /* try (tm=f, tn=g) */
        double r = (double)f / (double)g;
        double diff = (r > target) ? r / target : target / r;
        if (diff < best_diff) { best_diff = diff; best_tm = f; best_tn = g; }

        /* try (tm=g, tn=f) */
        r = (double)g / (double)f;
        diff = (r > target) ? r / target : target / r;
        if (diff < best_diff) { best_diff = diff; best_tm = g; best_tn = f; }
    }

    *tm_out = best_tm;
    *tn_out = best_tn;
}


/* ========================================================================
 * DGEMM driver with cache blocking and OpenMP parallelism
 *
 * Computes C += A * B for row-major M×K (A) times K×N (B) → M×N (C)
 * using the SkylakeX AVX-512 micro-kernel with packing.
 *
 * Threading: 2D partitioning across M and N dimensions.  Each thread
 * owns a disjoint (M-range × N-range) rectangle of C, with per-thread
 * pack_a and pack_b buffers.  Zero barriers — threads are fully
 * independent after the fork.
 *
 * Loop order per thread: K (outer) → M (middle) → N (inner)
 * This ensures pack_a is called once per (K-tile, M-tile) pair and reused
 * across all N-tiles, matching the OpenBLAS level3.c driver strategy.
 * ======================================================================== */
static void dgemm_blocked(BLASLONG M, BLASLONG N, BLASLONG K,
                          double *A, BLASLONG lda,
                          double *B, BLASLONG ldb,
                          double *C, BLASLONG ldc)
{
    if (M == 0 || N == 0 || K == 0) return;

    int num_threads = omp_get_max_threads();

    /* Packing buffer sizes */
    BLASLONG sa_size = (BLASLONG)GEMM_P * GEMM_Q;   /* one packed A tile */
    BLASLONG sb_size = (BLASLONG)GEMM_Q * 8;         /* one packed B tile (8-col panel) */

    /* Per-thread buffers: one sa tile + one sb tile each */
    double *sa_all = (double *)malloc((BLASLONG)num_threads * sa_size * sizeof(double));
    double *sb_all = (double *)malloc((BLASLONG)num_threads * sb_size * sizeof(double));

    if (!sa_all || !sb_all) {
        free(sa_all);
        free(sb_all);
        return;
    }

    /* Zero C first (kernel accumulates) */
    { BLASLONG j; for (j = 0; j < N; j++) memset(C + j * ldc, 0, M * sizeof(double)); }

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();

        /* 2D thread grid matching tile aspect ratio */
        BLASLONG m_tiles = (M + GEMM_P - 1) / GEMM_P;
        BLASLONG n_tiles = (N + 7) / 8;

        int tm, tn;
        factor_threads(nthreads, m_tiles, n_tiles, &tm, &tn);

        int m_group = tid / tn;
        int n_group = tid % tn;

        /* M-range for this thread */
        BLASLONG mt_start = m_group * m_tiles / tm;
        BLASLONG mt_end   = (m_group + 1) * m_tiles / tm;
        BLASLONG ms_start = mt_start * GEMM_P;
        BLASLONG ms_end   = mt_end * GEMM_P;
        if (ms_end > M) ms_end = M;

        /* N-range for this thread */
        BLASLONG nt_start = n_group * n_tiles / tn;
        BLASLONG nt_end   = (n_group + 1) * n_tiles / tn;
        BLASLONG ns_start = nt_start * 8;
        BLASLONG ns_end   = nt_end * 8;
        if (ns_end > N) ns_end = N;

        /* Per-thread packing buffers */
        double *my_sa = sa_all + (BLASLONG)tid * sa_size;
        double *my_sb = sb_all + (BLASLONG)tid * sb_size;

        /* K (outer) → M (middle) → N (inner) */
        BLASLONG ks;
        for (ks = 0; ks < K; ks += GEMM_Q) {
            BLASLONG kk = K - ks;
            if (kk > GEMM_Q) kk = GEMM_Q;

            BLASLONG ms;
            for (ms = ms_start; ms < ms_end; ms += GEMM_P) {
                BLASLONG mm = ms_end - ms;
                if (mm > GEMM_P) mm = GEMM_P;

                pack_a(kk, mm, A + ms + ks * lda, lda, my_sa);

                BLASLONG ns;
                for (ns = ns_start; ns < ns_end; ns += 8) {
                    BLASLONG nn = ns_end - ns;
                    if (nn > 8) nn = 8;

                    pack_b(kk, nn, B + ks + ns * ldb, ldb, my_sb);

                    skylakex_kernel(mm, nn, kk, 1.0, my_sa, my_sb,
                                    C + ms + ns * ldc, ldc);
                }
            }
            /* No barrier: each thread owns a disjoint C rectangle */
        }
    }

    free(sa_all);
    free(sb_all);
}


/* ========================================================================
 * MEX gateway function
 *
 * Usage: C = mex_avx512_dgemm(A, B)
 *   A: M×K double matrix
 *   B: K×N double matrix
 *   C: M×N double matrix (output)
 *
 * Handles column-major ↔ row-major conversion by swapping operands:
 *   col-major C(M,N) = A(M,K) * B(K,N)
 *   ↔ row-major C^T(N,M) = B^T(N,K) * A^T(K,M)
 * ======================================================================== */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs != 2)
        mexErrMsgIdAndTxt("avx512dgemm:nrhs", "Two inputs required: C = mex_avx512_dgemm(A, B)");
    if (nlhs > 1)
        mexErrMsgIdAndTxt("avx512dgemm:nlhs", "One output required.");

    if (!mxIsDouble(prhs[0]) || mxIsComplex(prhs[0]))
        mexErrMsgIdAndTxt("avx512dgemm:notDouble", "A must be a real double matrix.");
    if (!mxIsDouble(prhs[1]) || mxIsComplex(prhs[1]))
        mexErrMsgIdAndTxt("avx512dgemm:notDouble", "B must be a real double matrix.");

    BLASLONG M = (BLASLONG)mxGetM(prhs[0]);
    BLASLONG K = (BLASLONG)mxGetN(prhs[0]);
    BLASLONG K2 = (BLASLONG)mxGetM(prhs[1]);
    BLASLONG N = (BLASLONG)mxGetN(prhs[1]);

    if (K != K2)
        mexErrMsgIdAndTxt("avx512dgemm:dimMismatch",
                          "Inner dimensions must agree: A is %ldx%ld, B is %ldx%ld.", M, K, K2, N);

    double *A = mxGetPr(prhs[0]);  /* M×K column-major = K×M row-major */
    double *B = mxGetPr(prhs[1]);  /* K×N column-major = N×K row-major */

    /* Create output matrix */
    plhs[0] = mxCreateDoubleMatrix(M, N, mxREAL);
    double *C = mxGetPr(plhs[0]);  /* M×N column-major = N×M row-major */

    /* Column-major trick: swap operands and dimensions
     * Row-major: C^T(N,M) = B^T(N,K) * A^T(K,M)
     * B_col is row-major B^T with lda = K
     * A_col is row-major A^T with lda = M
     * C_col is row-major C^T with ldc = M (but we use N for the driver's ldc since it sees N×M)
     */
    dgemm_blocked(N, M, K, B, K, A, M, C, N);
}
