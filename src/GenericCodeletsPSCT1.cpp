//
// Created by kazem on 7/12/21.
//
#include "def.h"
#include "GenericCodelets.h"

namespace DDT{
/**
 * Different implementations of PSC Type2 for SpMV ->
 * y[lb:ub]+=Ax[offset[lb]:offset[lb]+WDT, ...,
 * offset[ub-1]:offset[ub-1]+WDT]*x[lbc:ubc] where WDT=ubc-lbc
 *
 * @param y out solution
 * @param Ax in nonzero locations
 * @param x in the input vector
 * @param offset in the starting point of Ax in each row
 * @param lb in lower bound of rows
 * @param ub in upper bound of rows
 * @param lbc in lower bound of columns
 * @param ubc in upper bound of columns
 */
 void inline psc_t1_1D4C(double *y, const double *Ax, const double *x,
                         const int *offset, int lb, int ub, int lbc, int ubc){
  v4df_t Lx_reg, Lx_reg2, result, result2, x_reg, x_reg2;
  for (int i = lb, ii=0; i <ub; ++i, ++ii) {
   result.v = _mm256_setzero_pd();
   for (int j = lbc, k=offset[ii]; j < ubc; j+=4, k+=4) {
    //y[i] += Ax[k] * x[j];
    //_mm256_mask_i32gather_pd()
    // x_reg.d[0] = x[*aij]; /// TODO replaced with gather
    // x_reg.d[1] = x[*(aij+1)];
    // x_reg.d[2] = x[*(aij+2)];
    // x_reg.d[3] = x[*(aij+3)];
    //x_reg.v = _mm256_set_pd(x[j], x[j+1], x[j+2], x[j+3]);
    x_reg.v = _mm256_loadu_pd((double *) (x+j));
    Lx_reg.v = _mm256_loadu_pd((double *) (Ax + k)); // Skylake	7	0.5
    result.v = _mm256_fmadd_pd(Lx_reg.v,x_reg.v,result.v);//Skylake	4	0.5
    //x_reg2.v = _mm256_set_pd(x[*(aik)], x[*(aik+1)], x[*(aik+2)], x[*
    // (aik+3)]);
    //Lx_reg2.v = _mm256_loadu_pd((double *) (Ax + k + 4)); // Skylake	7	0.5
    //result2.v = _mm256_fmadd_pd(Lx_reg2.v,x_reg2.v,result2.v);//Skylake	4	0.5
   }
   auto h0 = _mm_hadd_pd(_mm256_extractf128_pd(result.v,0), _mm256_extractf128_pd(result.v,1));
   y[i] += (h0[0] + h0[1]);
  }
 }

 void inline psc_t1_1D8C(double *y, const double *Ax, const double *x,
                        const int *offset, int lb, int ub, int lbc, int ubc){
  v4df_t Lx_reg, Lx_reg2, result, result2, x_reg, x_reg2;
  for (int i = lb, ii=0; i <ub; ++i, ++ii) {
   result.v = _mm256_setzero_pd();
   for (int j = lbc, k=offset[ii]; j < ubc; j+=8, k+=8) {
    //y[i] += Ax[k] * x[j];
    //_mm256_mask_i32gather_pd()
    // x_reg.d[0] = x[*aij]; /// TODO replaced with gather
    // x_reg.d[1] = x[*(aij+1)];
    // x_reg.d[2] = x[*(aij+2)];
    // x_reg.d[3] = x[*(aij+3)];
    //x_reg.v = _mm256_set_pd(x[j], x[j+1], x[j+2], x[j+3]);
    x_reg.v = _mm256_loadu_pd((double *) (x+j));
    Lx_reg.v = _mm256_loadu_pd((double *) (Ax + k)); // Skylake	7	0.5
    result.v = _mm256_fmadd_pd(Lx_reg.v,x_reg.v,result.v);//Skylake	4	0.5
    //x_reg2.v = _mm256_set_pd(x[*(aik)], x[*(aik+1)], x[*(aik+2)], x[*
    // (aik+3)]);
    //Lx_reg2.v = _mm256_loadu_pd((double *) (Ax + k + 4)); // Skylake	7	0.5
    //result2.v = _mm256_fmadd_pd(Lx_reg2.v,x_reg2.v,result2.v);//Skylake	4	0.5
    x_reg2.v = _mm256_loadu_pd((double *) (x+4+j));
    Lx_reg2.v = _mm256_loadu_pd((double *) (Ax+4+k)); // Skylake	7	0.5
    result.v = _mm256_fmadd_pd(Lx_reg2.v,x_reg2.v,result.v);//Skylake	4	0.5
   }
   auto h0 = _mm_hadd_pd(_mm256_extractf128_pd(result.v,0), _mm256_extractf128_pd(result.v,1));
   y[i] += (h0[0] + h0[1]);
  }
 }

 void inline psc_t1_2D2R(double *y, const double *Ax, const double *x,
                            const int *offset, int lb, int ub, int lbc, int ubc){
  v4df_t Lx_reg, Lx_reg2, result, result2, x_reg, x_reg2;
  for (int i = lb, ii=0; i <ub; i+=2, ii+=2) {
   result.v = _mm256_setzero_pd();
   result2.v = _mm256_setzero_pd();
   for (int j = lbc, k=offset[ii], k1 = offset[ii+1]; j < ubc; j+=4, k+=4,
                                                               k1+=4) {
    //y[i] += Ax[k] * x[j];
    //_mm256_mask_i32gather_pd()
    // x_reg.d[0] = x[*aij]; /// TODO replaced with gather
    // x_reg.d[1] = x[*(aij+1)];
    // x_reg.d[2] = x[*(aij+2)];
    // x_reg.d[3] = x[*(aij+3)];
    //x_reg.v = _mm256_set_pd(x[j], x[j+1], x[j+2], x[j+3]);
    x_reg.v = _mm256_loadu_pd((double *) (x+j));
    Lx_reg.v = _mm256_loadu_pd((double *) (Ax + k)); // Skylake	7	0.5
    Lx_reg2.v = _mm256_loadu_pd((double *) (Ax + k1)); // Skylake	7
    // 	0.5
    result.v = _mm256_fmadd_pd(Lx_reg.v,x_reg.v,result.v);//Skylake	4	0.5
    result2.v = _mm256_fmadd_pd(Lx_reg2.v,x_reg.v,result2.v);//Skylake	4	0.5
   }
   //auto h0 = _mm_hadd_pd(_mm256_extractf128_pd(result.v,0),
   //                       _mm256_extractf128_pd(result.v,1));
   auto h0 = _mm256_hadd_pd(result.v, result2.v);
   y[i] += (h0[0] + h0[2]);
   y[i+1] += (h0[1] + h0[3]);
   //y[i] += (h0[0] + h0[1]);
  }
 }

 void inline psc_t1_2D4R(double *y, const double *Ax, const double *x,
                              const int *offset, int lb, int ub, int lbc, int ubc){
  v4df_t Lx_reg, Lx_reg2, Lx_reg3, Lx_reg4, result, result2, result3,
    result4, x_reg,
    x_reg2;
  for (int i = lb, ii=0; i <ub; i+=4, ii+=4) {
   result.v = _mm256_setzero_pd();
   result2.v = _mm256_setzero_pd();
   result3.v = _mm256_setzero_pd();
   result4.v = _mm256_setzero_pd();
   for (int j = lbc, k=offset[ii], k1=offset[ii], k2=offset[ii],
          k3=offset[ii]; j < ubc; j+=4, k+=4, k1+=4, k2+=4, k3+=4) {
    //y[i] += Ax[k] * x[j];
    //_mm256_mask_i32gather_pd()
    // x_reg.d[0] = x[*aij]; /// TODO replaced with gather
    // x_reg.d[1] = x[*(aij+1)];
    // x_reg.d[2] = x[*(aij+2)];
    // x_reg.d[3] = x[*(aij+3)];
    //x_reg.v = _mm256_set_pd(x[j], x[j+1], x[j+2], x[j+3]);
    x_reg.v = _mm256_loadu_pd((double *) (x+j));
    Lx_reg.v = _mm256_loadu_pd((double *) (Ax + k)); // Skylake	7	0.5
    Lx_reg2.v = _mm256_loadu_pd((double *) (Ax + k1)); // Skylake	7
    Lx_reg3.v = _mm256_loadu_pd((double *) (Ax + k2)); // Skylake	7
    Lx_reg4.v = _mm256_loadu_pd((double *) (Ax +k3)); // Skylake	7
    // 	0.5
    result.v = _mm256_fmadd_pd(Lx_reg.v,x_reg.v,result.v);//Skylake	4	0.5
    result2.v = _mm256_fmadd_pd(Lx_reg2.v,x_reg.v,result2.v);//Skylake	4	0.5
    result3.v = _mm256_fmadd_pd(Lx_reg3.v,x_reg.v,result3.v);//Skylake	4	0.5
    result4.v = _mm256_fmadd_pd(Lx_reg4.v,x_reg.v,result4.v);//Skylake	4	0.5
   }
   auto h0 = _mm256_hadd_pd(result.v, result2.v);
   y[i] += (h0[0] + h0[2]);
   y[i+1] += (h0[1] + h0[3]);
   h0 = _mm256_hadd_pd(result3.v, result4.v);
   y[i+2] += (h0[0] + h0[2]);
   y[i+3] += (h0[1] + h0[3]);
  }
 }
}