//
// Created by kazem on 7/12/21.
//

#include "def.h"

namespace DDT{

 void psc_t3_1D1R(double *y, const double *Ax, const int *Ai,  const double
 *x, const int *offset, int lb, int fnl, int cw){
  v4df_t Lx_reg, Lx_reg2, result, result2, x_reg, x_reg2;
  int i = lb;
  result.v = _mm256_setzero_pd();
  int ti = cw % 4;
  for (int j = 0, k=fnl; j < cw-ti; j+=4, k+=4) {
   x_reg.v = _mm256_set_pd(x[Ai[offset[j]]], x[Ai[offset[j+1]]],
                           x[Ai[offset[j+2]]], x[Ai[offset[j+3]]]);
   Lx_reg.v = _mm256_loadu_pd((double *) (Ax + k)); // Skylake	7	0.5
   result.v = _mm256_fmadd_pd(Lx_reg.v,x_reg.v,result.v);//Skylake	4	0.5
  }
  double tail = 0;
  int k = fnl-ti;
  for (int j = cw-ti; j < cw; ++j, ++k) {
   tail += (Ax[k] * x[Ai[offset[j]]]);
  }
  auto h0 = _mm_hadd_pd(_mm256_extractf128_pd(result.v,0), _mm256_extractf128_pd(result.v,1));
  y[i] += (h0[0] + h0[1] + tail);
 }
}
