//
// Created by kazem on 7/12/21.
//

#ifndef DDT_GENERICCODELETS_H
#define DDT_GENERICCODELETS_H

#include <immintrin.h>
namespace DDT{
/// FSC

 void fsc_t2_2DC( double *y, const double *Ax, const double *x, const
 int axi, const int  axo, const int  lb, const int ub, const int cbl,
   const int cbu);

 /// PSCT1
 void psc_t1_1D4C(double *y, const double *Ax, const double *x,
                         const int *offset, int lb, int ub, int lbc, int ubc);


 void psc_t1_1D8C(double *y, const double *Ax, const double *x,
                         const int *offset, int lb, int ub, int lbc, int ubc);


 void psc_t1_2D2R(double *y, const double *Ax, const double *x,
                         const int *offset, int lb, int ub, int lbc, int ubc);


 void psc_t1_2D4R(double *y, const double *Ax, const double *x,
                         const int *offset, int lb, int ub, int lbc, int ubc);


 /// PSC T2

 void psc_t2_2DC( double *y, const double *Ax, const double *x, const int
 *offset, const int  axi, const  int axo, const   int lb, const int
   ub, const int cb);

 void psc_t2_2D4C( double *y, const double *Ax, const double *x,
   const int *offset, const int axi, const int axo, const int lb, const int
   ub, const int cb);


 /// PSC T3
 void psc_t3_1D1R(double *y, const double *Ax, const int *Ai,  const double
 *x, const int *offset, int lb, int fnl, int cw);
 }

#endif //DDT_GENERICCODELETS_H
