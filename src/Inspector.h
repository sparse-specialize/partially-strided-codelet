//
// Created by kazem on 7/12/21.
//

#ifndef DDT_INSPECTOR_H
#define DDT_INSPECTOR_H

#include <DDT.h>
#include <vector>

namespace DDT{

 struct Codelet{

  Codelet();

 };

 struct FSCCodelet:public Codelet{
  /**
   * y[lbr:lbr+row_width] = Ax[FNL:FNL+CW, ..., FNL+RO:FNL+RO+CW]*x[lbc:lbc+CW];
   */
  int lbr, lbc, row_width, col_width; //LBR, RW, CW
  int first_nnz_loc, row_offset; //FNL, RO
 };

 struct PSCT1V1:public Codelet{
  /**
   * y[lbr:lbr+row_width] = Ax[RO[lbr]:RO[lbr]+CW, ...,
   * RO[lbr]:RO+CW]*x[lbc:lbc+CW];
   */
  int lbr, lbc, row_width, col_width; //LBR, RW, CW
  int *row_offsets; //RO
 };


 struct PSCT2V1:public Codelet{
  /**
   * y[lbr:lbr+row_width] = Ax[FNL:FNL+CW, ..., FNL+RO:FNL+RO+CW]
   * * x[CIO[0]:CIO[CW]];
   */
  int lbr, row_width; //LBR, RW
  int *coli_offset, col_width;// CIO, CW
  int first_nnz_loc, row_offset; //FNL, RO
 };

 struct PSCT3V1:public Codelet{
  /**
   * y[RS] = Ax[FNL:FNL+NN] * x[CIO[0],CIO[NN]];
   */
  int row_start, num_nnz; //RS, NN
  int *coli_offset;// CIO
  int first_nnz_loc; //FNL
 };


 void inspectCodelet(DDT::GlobalObject& d, DDT::PatternDAG* c,
                     const std::vector<Codelet>& cl);
 }


#endif //DDT_INSPECTOR_H
