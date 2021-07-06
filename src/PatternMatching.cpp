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
#include <chrono>
#include <immintrin.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "PatternMatching.h"

/**
 * @TODO: Information I will need:
 * - tuples per iteration
 * - a way to store merging results that is cheap
 *
 *
 */

/**
 * Generate parallel first order differences
 *
 * @param ip Array of pointers to the start tuple of each iteration
 * @param ips Size of the ip array
 * @param tuples Array of padded tuples
 * @param differences Array of blank memory to store diffences
 *
 */
void computeParallelizedFOD(int** ip, int ips, int* differences) {
  auto t1 = std::chrono::steady_clock::now();
  for (int i = 0; i < ips-1; i++) {
    computeFirstOrder(differences+(ip[i]-ip[0])-(1*i), ip[i], ip[i+1]-ip[i]);
  }
  auto t2 = std::chrono::steady_clock::now();

  auto timeTaken = std::chrono::duration_cast<std::chrono::duration<double>>(t2-t1).count();
}

/**
 * Generate first order differences for tuples.
 *
 * @param differences Memory address to store differences
 * @param tuples Pointer to tuples which must be padded
 * to lengths of four
 * @param numTuples Number of tuples to process
 */
void computeFirstOrder(int* differences, int* tuples, int numTuples) {
  int i = 0;
  int to = 3; // Tuple offset
  for (; i < numTuples-7; i+=8) {
    __m256i lhs = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(tuples+i));
    __m256i rhs = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(tuples + i + to));
    __m256i fod = _mm256_sub_epi32(rhs, lhs);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(differences + i), fod);
  }
  for (; i < numTuples-3; i+=4) {
    __m128i lhs = _mm_loadu_si128(reinterpret_cast<const __m128i *>(tuples + i));
    __m128i rhs = _mm_loadu_si128(reinterpret_cast<const __m128i *>(tuples + i + to));
    __m128i fod = _mm_sub_epi32(rhs, lhs);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(differences + i), fod);
  }
  for (; i < numTuples; i++) {
    differences[i] = *(tuples+i+to) - *(tuples+i);
  }
}

struct Codelet {
  int length;
  int* start;
};

/**
 * Generates longest common sub-arrays in O(lhstps+rhstps) time
 *
 * @param lhsd First order differences from iteration 'i'
 * @param rhsd First order differences from iteration 'i+1'
 * @param lhstp Pointer to tuple at start of iteration 'i'
 * @param rhstp Pointer to tuple at start of iteration 'i+1'
 * @param rhstps Number of tuples in iteration 'i'
 * @param rhstps Number of tuples in iteration 'i+1'
 *
 */
void findTLCS(int* lhsd, int* rhsd, int* lhstp, int* rhstp, int lhstps, int rhstps) {
  __m128i lhs = _mm_loadu_si128(reinterpret_cast<const __m128i *>(lhstp));
  __m128i rhs = _mm_loadu_si128(reinterpret_cast<const __m128i *>(rhstp));

  for (int i = 0, j = 0; i < lhstps && j < rhstps;) {
    // Difference and comparison
    __m128i sub = _mm_sub_epi32(rhs,lhs);
    __m128i cmp = _mm_cmplt_epi32(_mm_abs_epi32(sub), THRESHOLDS);

    // @TODO: Add mask for partially strided
    if (ZERO_MASK == _mm_movemask_epi8(cmp)) {
      __m128i lhsd = _mm_loadu_si128(reinterpret_cast<const __m128i *>(lhsd+i));
      __m128i rhsd = _mm_loadu_si128(reinterpret_cast<const __m128i *>(rhsd+j));
      __m128i vxor = _mm_xor_si128(lhsd, rhsd);

      while ((i < lhstps && j < rhstps) && ZERO_MASK == _mm_movemask_epi8(vxor)) {
        // Update codelet
        // codelet.d0 += 1;
        // codelet.p0 += 1;

        // Calculate new differences
        lhsd = _mm_loadu_si128(reinterpret_cast<const __m128i *>(lhsd + ++i));
        rhsd = _mm_loadu_si128(reinterpret_cast<const __m128i *>(rhsd + ++j));
        vxor = _mm_xor_si128(lhsd, rhsd);
      }

      if () {
        codeletPointer++;
      }
    } else {
      if (sub) {
        i++;
        __m128i lhs = _mm_loadu_si128(reinterpret_cast<const __m128i *>(lhstp+i));
      } else if(sub) {
        j++;
        __m128i rhs = _mm_loadu_si128(reinterpret_cast<const __m128i *>(rhstp+j));
      }
    }
  }
}

/**
 *
 * Generate the longest common subsequence between two codelets
 *
 */
void findCLCS() {
  // 1) Go over the codelets first
  //    a - Mark spots with codelets and if the spot is a codelet then do a
  //        codelet comparison with the memoized data.
  //    b - Need one additional array to tell if current spot has codelet. Before
  //        comparison, determine if other spot is compatible otherwise skip codelet elements
  //    c - 
  //
  //
  for (int i = 0, j = 0; i < && j < ;) {
    if (/* current trace(s) are associated with codelets */) {

    } else {

    }
  }
}

uint32_t hsum_epi32_avx(__m128i x)
{
  __m128i hi64  = _mm_unpackhi_epi64(x, x);           // 3-operand non-destructive AVX lets us save a byte without needing a movdqa
  __m128i sum64 = _mm_add_epi32(hi64, x);
  __m128i hi32  = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));    // Swap the low two elements
  __m128i sum32 = _mm_add_epi32(sum64, hi32);
  return _mm_cvtsi128_si32(sum32);       // movd
}

// only needs AVX2
uint32_t hsum_8x32(__m256i v)
{
  __m128i sum128 = _mm_add_epi32(
      _mm256_castsi256_si128(v),
      _mm256_extracti128_si256(v, 1)); // silly GCC uses a longer AXV512VL instruction if AVX512 is enabled :/
  return hsum_epi32_avx(sum128);
}
