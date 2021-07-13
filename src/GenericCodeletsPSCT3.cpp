//
// Created by kazem on 7/12/21.
//

#include "DDTDef.h"

namespace DDT {
/**
 * From any location
 * @param y
 * @param Ax
 * @param Ai
 * @param x
 * @param offset
 * @param lb
 * @param fnl
 * @param cw
 */
 void psc_t3_1DAnyR(double *y, const double *Ax, const int *Ai,  const double
 *x, const int *offset, int lb, int fnl, int cw) {
  v4df_t Lx_reg, Lx_reg2, result, result2, x_reg, x_reg2;
  int i = lb;
  result.v = _mm256_setzero_pd();
  int ti = cw % 4;
  int k = fnl;
  for (int j = 0; j < cw-ti; j+=4, k+=4) {
   x_reg.v = _mm256_set_pd(x[Ai[offset[j+3]]], x[Ai[offset[j+2]]],
                           x[Ai[offset[j+1]]], x[Ai[offset[j]]]);
   Lx_reg.v = _mm256_set_pd(Ax[offset[j+3]], Ax[offset[j+2]],
                           Ax[offset[j+1]], Ax[offset[j]]);
   //Lx_reg.v = _mm256_loadu_pd((double *) (Ax + k)); // Skylake	7	0.5
   result.v = _mm256_fmadd_pd(Lx_reg.v,x_reg.v,result.v);//Skylake	4	0.5
  }
  double tail = 0;
  for (int j = cw-ti; j < cw; ++j, ++k) {
   tail += (Ax[offset[j]] * x[Ai[offset[j]]]);
  }
  auto h0 = _mm_hadd_pd(_mm256_extractf128_pd(result.v,0), _mm256_extractf128_pd(result.v,1));
  y[i] += (h0[0] + h0[1] + tail);
 }


 void psc_t3_1D1R(double *y, const double *Ax, const int *Ai,  const double
 *x, const int *offset, int lb, int fnl, int cw) {
  v4df_t Lx_reg, Lx_reg2, result, result2, x_reg, x_reg2;
  int i = lb;
  result.v = _mm256_setzero_pd();
  int ti = cw % 4;
  int k = fnl;
  for (int j = 0; j < cw-ti; j+=4, k+=4) {
   x_reg.v = _mm256_set_pd(x[Ai[k+3]], x[Ai[k+2]],
                           x[Ai[k+1]], x[Ai[k]]);
   Lx_reg.v = _mm256_loadu_pd((double *) (Ax + k)); // Skylake	7	0.5
   result.v = _mm256_fmadd_pd(Lx_reg.v,x_reg.v,result.v);//Skylake	4	0.5
  }
  double tail = 0;
  k= fnl+cw-ti;
  for (int j = cw-ti; j < cw; ++j, ++k) {
   tail += (Ax[k] * x[Ai[k]]);
  }
  auto h0 = _mm_hadd_pd(_mm256_extractf128_pd(result.v,0), _mm256_extractf128_pd(result.v,1));
  y[i] += (h0[0] + h0[1] + tail);
 }

}
