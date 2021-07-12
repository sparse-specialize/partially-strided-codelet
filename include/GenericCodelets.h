//
// Created by kazem on 7/12/21.
//

#ifndef DDT_GENERICCODELETS_H
#define DDT_GENERICCODELETS_H

namespace DDT{

 void inline psc_t1_1D4C(double *y, const double *Ax, const double *x,
                         const int *offset, int lb, int ub, int lbc, int ubc);


 void inline psc_t1_1D8C(double *y, const double *Ax, const double *x,
                         const int *offset, int lb, int ub, int lbc, int ubc);


 void inline psc_t1_2D2R(double *y, const double *Ax, const double *x,
                         const int *offset, int lb, int ub, int lbc, int ubc);


 void inline psc_t1_2D4R(double *y, const double *Ax, const double *x,
                         const int *offset, int lb, int ub, int lbc, int ubc);
 }

#endif //DDT_GENERICCODELETS_H
