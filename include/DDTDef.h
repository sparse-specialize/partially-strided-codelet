//
// Created by kazem on 7/12/21.
//

#ifndef DDT_DDTDEF_H
#define DDT_DDTDEF_H

#include<immintrin.h>
namespace DDT{

#ifdef __AVX2__
 typedef union
 {
  __m256d v;
  double d[4];
 } v4df_t;

 typedef union
 {
  __m128i v;
  int d[4];
 } v4if_t;

 union vector128
 {
  __m128i     i128;
  __m128d     d128;
  __m128      f128;
 };

#endif

#ifdef __AVX512CD__
 typedef union
 {
  __m512d v;
  double d[8];
 } v5df_t;
#endif
}

#endif //DDT_DDTDEF_H
