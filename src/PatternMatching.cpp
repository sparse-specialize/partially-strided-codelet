/*
 * =====================================================================================
 *
 *       Filename:  PatternMatching.cpp
 *
 *    Description:  Pattern matching code for trace analysis
 *
 *        Version:  1.0
 *        Created:  2021-07-05 05:50:43 PM
 *       Revision:  1.0
 *       Compiler:  gcc
 *
 *         Author:  Zachary Cetinic,
 *   Organization:
 *
 * =====================================================================================
 */
#include "PatternMatching.h"

#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>

#include <chrono>
#include <iostream>
#include <vector>

int THRESHOLDS[4] = {0, 2, 100, 2};
const int TPD = 3;
int ZERO_MASK = 0xF000;

/**
 * Generate parallel first order differences
 *
 * @param ip Array of pointers to the start tuple of each iteration
 * @param ips Size of the ip array
 * @param tuples Array of padded tuples
 * @param differences Array of blank memory to store diffences
 *
 */
void computeParallelizedFOD(int **ip, int ips, int *differences) {
    auto t1 = std::chrono::steady_clock::now();
    for (int i = 0; i < ips - 1; i++) {
        computeFirstOrder(differences + (ip[i] - ip[0]) - (1 * i), ip[i], ip[i + 1] - ip[i]);
    }
    auto t2 = std::chrono::steady_clock::now();

    auto timeTaken = getTimeDifference(t1, t2);

    std::cout << "FOD Time: " << timeTaken << std::endl;
}

/**
 * Generate first order differences for tuples.
 *
 * @param differences Memory address to store differences
 * @param tuples Pointer to tuples which must be padded
 * to lengths of four
 * @param numTuples Number of tuples to process
 */
void computeFirstOrder(int *differences, int *tuples, int numTuples) {
    int i = 0;
    int to = 3;// Tuple offset

    for (; i < numTuples - 7; i += 8) {
        __m256i lhs = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(tuples + i));
        __m256i rhs = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(tuples + i + to));
        __m256i fod = _mm256_sub_epi32(rhs, lhs);
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(differences + i), fod);
    }
    for (; i < numTuples - 3; i += 4) {
        __m128i lhs = _mm_loadu_si128(reinterpret_cast<const __m128i *>(tuples + i));
        __m128i rhs = _mm_loadu_si128(reinterpret_cast<const __m128i *>(tuples + i + to));
        __m128i fod = _mm_sub_epi32(rhs, lhs);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(differences + i), fod);
    }
    for (; i < numTuples; i++) {
        differences[i] = *(tuples + i + to) - *(tuples + i);
    }
}

/**
 *
 * Generates codelets from differences and tuple information.
 *
 * @param ip Pointer to tuples at start of iterations
 * @param ips Number of pointers in ip
 * @param c Storage for codelet groupings
 *
 */
void mineDifferences(int **ip, int ips, Codelet *c) {
    auto t1 = std::chrono::steady_clock::now();
    for (int i = 0; i < ips - 1; i++) {
        auto lhscp = (ip[i] - ip[0]) / TPD;
        auto rhscp = (ip[i + 1] - ip[0]) / TPD;
        findCLCS(TPD, ip[i], ip[i + 1], (ip[i + 1] - ip[i]) / TPD, (ip[i + 2] - ip[i + 1]) / TPD, c + lhscp, c + rhscp);
    }
    auto t2 = std::chrono::steady_clock::now();
    auto timeTaken = getTimeDifference(t1, t2);

    std::cout << "Mine Time: " << timeTaken << std::endl;
}

/**
 *
 * Generate the longest common subsequence between two codelets
 *
 * Updates the pointers and values in lhscp and rhscp
 * to reflect the new codelet groupings.
 *
 * @param tpd Dimensionality of tuples
 * @param lhstp Left pointer to start of tuple grouping
 * @param rhstp Right pointer to start of tuple grouping
 * @param lhstps Size of tuples in left pointer to iterate
 * @param rhstps Size of tuples in right pointer to iterate
 *
 */
void findCLCS(int tpd, int *lhstp, int *rhstp, int lhstps, int rhstps, Codelet *lhscp, Codelet *rhscp) {
    __m128i lhs = _mm_loadu_si128(reinterpret_cast<const __m128i *>(lhstp));
    __m128i rhs = _mm_loadu_si128(reinterpret_cast<const __m128i *>(rhstp));
    __m128i thresholds = _mm_set_epi32(THRESHOLDS[3], THRESHOLDS[2], THRESHOLDS[1], THRESHOLDS[0]);

    bool isCodelet[2] = {false, false};

    for (int i = 0, j = 0; i < lhstps && j < rhstps;) {
        // Difference and comparison
        __m128i sub = _mm_sub_epi32(rhs, lhs);
        __m128i cmp = _mm_cmplt_epi32(_mm_abs_epi32(sub), thresholds);

        // Check if tuple is start of codelet
        isCodelet[0] = lhscp[i].pt != nullptr;
        isCodelet[1] = rhscp[i].pt != nullptr;

        // Mask upper 8 bits that aren't used
        auto mm = ~_mm_movemask_epi8(cmp) | 0xF000;

        if (ZERO_MASK == mm &&
            (!isCodelet[0] || (isCodelet[0] && tuplesHaveSameFOD(lhscp[i].pt, lhstp + i * tpd, rhstp + j * tpd)))) {

            __m128i lhsdv = _mm_loadu_si128(reinterpret_cast<const __m128i *>(lhstp + i * tpd));
            __m128i rhsdv = _mm_loadu_si128(reinterpret_cast<const __m128i *>(rhstp + j * tpd));
            __m128i xordv = _mm_xor_si128(lhsdv, rhsdv);

            // @TODO: if other section could contain codelets, update this code
            // if (isCodelet[1]) {
            //   tuplesHaveSameFOD();
            // }

            // @TODO: Add heuristic for partially strided
            int iStart = i;
            while ((i < lhstps && j < rhstps) && ZERO_MASK == (_mm_movemask_epi8(xordv) | 0xF000)) {
                // Calculate new differences
                lhsdv = _mm_loadu_si128(reinterpret_cast<const __m128i *>(lhstp + i * tpd));
                rhsdv = _mm_loadu_si128(reinterpret_cast<const __m128i *>(rhstp + j * tpd));
                xordv = _mm_xor_si128(lhsdv, rhsdv);
            }

            // Adjust pointers to form codelet
            // @TODO: Heuristic would go here
            int sz = i - iStart;
            if (sz == lhscp->sz) {
                rhscp->sz = sz;
                rhscp->pt = lhscp->ct;
            }
            if (!isCodelet[0]) {
                lhscp->sz = sz;
                lhscp->pt = lhscp->ct;
            }
        } else {
            if (sub[0] > 0) {
                // @TODO: turn into constexpr
                if (isCodelet[0]) {
                    i += lhscp->sz;
                } else {
                    i++;
                }
                lhs = _mm_loadu_si128(reinterpret_cast<const __m128i *>(lhstp + i * tpd));
            } else if (sub[0] < 0) {
                // @TODO: turn into constexpr
                if (isCodelet[1]) {
                    i += rhscp->sz;
                } else {
                    j++;
                }
                rhs = _mm_loadu_si128(reinterpret_cast<const __m128i *>(rhstp + j * tpd));
            }
        }
    }
}

/** 
 *
 * Determines if the diffence between mid-lhs is the same as rhs-mid
 *
 * @param lhs Memory location of start of lhs tuple
 * @param mid Memory location of start of mid tuple
 * @param rhs Memory location of start of rhs tuple
 * @return Returns true if distance is the same, otherwise false
 */
bool tuplesHaveSameFOD(int *lhs, int *mid, int *rhs) {
    // Load tuples into memory
    __m128i lhsv = _mm_loadu_si128(reinterpret_cast<const __m128i *>(lhs));
    __m128i midv = _mm_loadu_si128(reinterpret_cast<const __m128i *>(mid));
    __m128i rhsv = _mm_loadu_si128(reinterpret_cast<const __m128i *>(rhs));

    // Generate differences
    __m128i cmp = _mm_xor_si128(
            _mm_sub_epi32(midv, lhsv),
            _mm_sub_epi32(rhsv, midv));

    return ZERO_MASK == (_mm_movemask_epi8(cmp) | 0xF000);
}

uint32_t hsum_epi32_avx(__m128i x) {
    __m128i hi64 = _mm_unpackhi_epi64(x, x);
    __m128i sum64 = _mm_add_epi32(hi64, x);
    __m128i hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
    __m128i sum32 = _mm_add_epi32(sum64, hi32);
    return _mm_cvtsi128_si32(sum32);// movd
}

// only needs AVX2
uint32_t hsum_8x32(__m256i v) {
    __m128i sum128 = _mm_add_epi32(
            _mm256_castsi256_si128(v),
            _mm256_extracti128_si256(v, 1));
    return hsum_epi32_avx(sum128);
}

double getTimeDifference(std::chrono::steady_clock::time_point t1, std::chrono::steady_clock::time_point t2) {
    return std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
}
