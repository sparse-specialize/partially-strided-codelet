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
#include "DDT.h"
#include "PatternMatching.h"

#include <omp.h>
#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>

#include <chrono>
#include <iostream>
#include <vector>

int THRESHOLDS[4] = { 2, 100, 100, 0 };
const int TPD = 3;
int ZERO_MASK = 0xF000;
int FSC_MASK  = 0xF000;
int PSC1_MASK = 0xF0F0;
int PSC2_MASK = 0xF000;

/**
 * @brief Generate parallel first order differences
 *
 * @param ip Array of pointers to the start tuple of each iteration
 * @param ips Size of the ip array
 * @param tuples Array of padded tuples
 * @param differences Array of blank memory to store diffences
 *
 */
void computeParallelizedFOD(int **ip, int ips, int *differences) {
  auto t1 = std::chrono::steady_clock::now();
#pragma omp parallel for num_threads(2)
  for (int i = 0; i < ips - 1; i++) {
    computeFirstOrder(differences + (ip[i] - ip[0]), ip[i], ip[i + 1] - ip[i]);
  }
  auto t2 = std::chrono::steady_clock::now();

  auto timeTaken = getTimeDifference(t1, t2);

  //std::cout << "FOD Time: " << timeTaken << std::endl;
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
  int to = 3; // Tuple offset

  for (; i < numTuples - to*4; i += 8) {
    __m256i lhs = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(tuples + i));
    __m256i rhs = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(tuples + i + to));
    __m256i fod = _mm256_sub_epi32(rhs, lhs);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(differences + i), fod);
  }
  for (; i < numTuples - to*2; i += 4) {
    __m128i lhs = _mm_loadu_si128(reinterpret_cast<const __m128i *>(tuples + i));
    __m128i rhs = _mm_loadu_si128(reinterpret_cast<const __m128i *>(tuples + i + to));
    __m128i fod = _mm_sub_epi32(rhs, lhs);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(differences + i), fod);
  }
  for (; i < numTuples-to; i++) {
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
void mineDifferences(int **ip, int ips, DDT::PatternDAG *c, int* d) {
  // int bnd[9] = { 0, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000 };
//  int bnd[5] = { 0, ips/4, ips/2, ips*3/4, ips };
  auto t1 = std::chrono::steady_clock::now();
//#pragma omp parallel for num_threads(4)
//  for (int ii = 0; ii < 4; ++ii) {
    for (int i = 0; i < ips-1; i++) {
      auto lhscp = (ip[i] - ip[0]) / TPD;
      auto rhscp = (ip[i + 1] - ip[0]) / TPD;
      auto lhstps = (ip[i + 1] - ip[i]) / TPD;
      auto rhstps = (ip[i + 2] - ip[i + 1]) / TPD;
      findCLCS(TPD, ip[i], ip[i + 1], lhstps, rhstps, c + lhscp, c + rhscp, d + lhscp*TPD, d + rhscp*TPD);
    }
//  }
  auto t2 = std::chrono::steady_clock::now();
  auto timeTaken = getTimeDifference(t1, t2);

  //std::cout << "Mine Time: " << timeTaken << std::endl;
}

void printTuple(int* t, std::string&& s) {
  std::cout << s << ": (" << t[0] << "," << t[1] << "," << t[2] << ")" << std::endl;
}

/**
 * Determines if memory location is part of codelet
 *
 * @param c Codelet memory location associated with tuple
 * @return True if c->pt == nullptr
 */
inline bool isInCodelet(DDT::PatternDAG* c) {
    return c->pt != nullptr;
}

/**
 * Determines if DDT::Codelet is origin for codelet
 *
 * @param c Memory location of codelet pointer
 * @return True if codelet pointer is start of codelet
 */
inline bool isCodeletOrigin(DDT::PatternDAG* c) {
    return c->pt == c->ct;
}


/**
 *
 * @brief Generate the longest common subsequence between two codelet regions.
 *
 * Updates the pointers and values in lhscp and rhscp
 * to reflect the new codelet groupings.
 *
 * @param tpd    Dimensionality of tuples
 * @param lhstp  Left pointer to start of tuple grouping
 * @param rhstp  Right pointer to start of tuple grouping
 * @param lhstps Size of tuples in left pointer to iterate
 * @param rhstps Size of tuples in right pointer to iterate
 * @param lhscp  Left pointer to codelet
 * @param rhscp  Right pointer to codelet
 * @param lhstpd Left pointer to first order differences
 * @param rhstpd Right pointer to first order differences
 *
 */
void findCLCS(int tpd, int *lhstp, int *rhstp, int lhstps, int rhstps, DDT::PatternDAG
*lhscp, DDT::PatternDAG *rhscp, int* lhstpd, int* rhstpd) {
    __m128i thresholds = _mm_set_epi32(THRESHOLDS[3], THRESHOLDS[2], THRESHOLDS[1], THRESHOLDS[0]);
  auto lhs = _mm_loadu_si128(reinterpret_cast<const __m128i *>(lhstp));
  auto rhs = _mm_loadu_si128(reinterpret_cast<const __m128i *>(rhstp));

  for (int i = 0, j = 0; i < lhstps - 1 && j < rhstps - 1;) {
    // Difference and comparison
    __m128i sub = _mm_sub_epi32(rhs, lhs);
    __m128i cmp = _mm_cmplt_epi32(_mm_abs_epi32(sub), thresholds);

    // Mask upper 8 bits that aren't used
    uint16_t mm = _mm_movemask_epi8(cmp); 
    mm = ~mm | 0xf000;

    if (ZERO_MASK == mm) {
      __m128i lhsdv = _mm_loadu_si128(reinterpret_cast<const __m128i *>(lhstpd + i * tpd));
      __m128i rhsdv = _mm_loadu_si128(reinterpret_cast<const __m128i *>(rhstpd + j * tpd));
      __m128i xordv = _mm_xor_si128(rhsdv, lhsdv);

      auto odv = lhsdv; // originalDifferenceVector
      bool hsad = true; // hasSameAdjacentDifferences

      int iStart = i;
      int jStart = j;
      uint16_t imm = _mm_movemask_epi8(xordv);
      while ((i < lhstps - 1 && j < rhstps - 1) && ZERO_MASK == (imm | 0xF000)) {
        lhsdv = _mm_loadu_si128(reinterpret_cast<const __m128i *>(lhstpd + i++ * tpd));
        rhsdv = _mm_loadu_si128(reinterpret_cast<const __m128i *>(rhstpd + j++ * tpd));
        xordv = _mm_xor_si128(rhsdv, lhsdv);
        imm = _mm_movemask_epi8(xordv);

        // Compare FOD across iteration
        uint16_t MASK = _mm_movemask_epi8(_mm_xor_si128(odv, lhsdv));
        hsad = hsad && ZERO_MASK == (MASK | 0xF000);
      }

      // Determines codelet type
      if (isInCodelet(lhscp+iStart)) {
        uint16_t MASK = generateDifferenceMask(
            lhscp[iStart].pt, 
            lhstp + iStart * tpd, 
            rhstp + jStart * tpd, 
            ZERO_MASK);

        // Checks for PSC Type 1 and PSC Type 2
        if (MASK == PSC1_MASK && !hsad || MASK != (FSC_MASK|PSC2_MASK)) {
          i += lhscp[iStart].sz;
          continue;
        }

        // Update codelet type
        rhscp[jStart].t = hsad && MASK == PSC1_MASK ? DDT::TYPE_PSC1 : hsad ? DDT::TYPE_FSC : DDT::TYPE_PSC2;
      }

      // Adjust pointers to form codelet
      int sz = i - iStart;
      if (sz != 0) {
        if (!isInCodelet(lhscp + iStart)) {
          lhscp[iStart].sz = sz;
          lhscp[iStart].pt = lhscp[iStart].ct;
          rhscp[jStart].t  = hsad ? DDT::TYPE_FSC : DDT::TYPE_PSC2;
        }
        if (sz == lhscp[iStart].sz) {
          rhscp[jStart].sz = sz;
          rhscp[jStart].pt = lhscp[iStart].ct;
          lhs = _mm_loadu_si128(reinterpret_cast<const __m128i *>(lhstp + i * tpd));
          rhs = _mm_loadu_si128(reinterpret_cast<const __m128i *>(rhstp + j * tpd));
        }
      } else {
        if (lhstp[i*tpd+2] < rhstp[j*tpd+2]) {
          i += lhscp->sz + 1;
          lhs = _mm_loadu_si128(reinterpret_cast<const __m128i *>(lhstp + i * tpd));
        } else if (lhstp[i*tpd+2] >= rhstp[j*tpd+2]) {
          j += rhscp->sz + 1;
          rhs = _mm_loadu_si128(reinterpret_cast<const __m128i *>(rhstp + j * tpd));
        }
      }
    } else {
      if (lhstp[i*tpd+2] < rhstp[j*tpd+2]) {
        // Since lhscp->sz is number of tuples in codelet not including itself
        i += lhscp->sz + 1;
        lhs = _mm_loadu_si128(reinterpret_cast<const __m128i *>(lhstp + i * tpd));
      } else if (lhstp[i*tpd+2] >= rhstp[j*tpd+2]) {
        j += rhscp->sz + 1;
        rhs = _mm_loadu_si128(reinterpret_cast<const __m128i *>(rhstp + j * tpd));
      }
    }
  }
}

/** 
 *
 * @brief Generates a 16-bit bitmask describing (rhs-mid == mid-lhs)
 *
 * @param lhs Memory location of start of lhs tuple
 * @param mid Memory location of start of mid tuple
 * @param rhs Memory location of start of rhs tuple
 * @return Returns 16-bit bitmask
 */
inline uint16_t generateDifferenceMask(int *lhs, int *mid, int *rhs, int MASK) {
  // Load tuples into memory
  __m128i lhsv = _mm_loadu_si128(reinterpret_cast<const __m128i *>(lhs));
  __m128i midv = _mm_loadu_si128(reinterpret_cast<const __m128i *>(mid));
  __m128i rhsv = _mm_loadu_si128(reinterpret_cast<const __m128i *>(rhs));

  // Generate differences
  __m128i cmp = _mm_xor_si128(
      _mm_sub_epi32(midv, lhsv),
      _mm_sub_epi32(rhsv, midv));

  uint16_t mm = _mm_movemask_epi8(cmp);
  return (mm | 0xF000);
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
