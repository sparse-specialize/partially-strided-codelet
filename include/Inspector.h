//
// Created by kazem on 7/12/21.
//

#ifndef DDT_INSPECTOR_H
#define DDT_INSPECTOR_H

#include <DDT.h>
#include <vector>

namespace DDT{

 enum LOCATIONS{
  LBR=0, LBC, RW, CW, FNL, RO, CO
 };

 struct Codelet{
  int size;
  int *offsets;
  bool is_alloc;
  int lbr, lbc, row_width, col_width; //LBR, RW, CW
  int first_nnz_loc, row_offset; //FNL, RO

  Codelet(int br, int bc, int rw, int cw, int fnl, int ro, int *offs):lbr(br),
  lbc(bc),col_width(cw),first_nnz_loc(fnl),row_offset(ro), offsets(offs){}

  virtual CodeletType get_type()=0;
 // virtual void pack()=0;

 };

 struct FSCCodelet:public Codelet{
  /**
   * y[lbr:lbr+row_width] = Ax[FNL:FNL+CW, ..., FNL+RO:FNL+RO+CW]*x[lbc:lbc+CW];
   */
  FSCCodelet(int br, int bc, int rw, int cw, int fnl, int ro): Codelet(br,bc,
                                                                       rw,cw,fnl,ro,NULL){};

  CodeletType get_type() override{return CodeletType::TYPE_FSC;}
  //void pack()override;
 };


 struct PSCT1V1:public Codelet{
  /**
   * y[lbr:lbr+row_width] = Ax[RO[lbr]:RO[lbr]+CW, ...,
   * RO[lbr]:RO+CW]*x[lbc:lbc+CW];
   */
  //int lbr, lbc, row_width, col_width; //LBR, RW, CW
  //int *row_offsets; //RO
  PSCT1V1(int br, int bc, int rw, int cw, int *ros): Codelet(br,bc,rw,cw,-1,
                                                             -1,ros){};
  CodeletType get_type() override{return CodeletType::TYPE_PSC1;}
  //void pack()override;
 };


 struct PSCT2V1:public Codelet{
  /**
   * y[lbr:lbr+row_width] = Ax[FNL:FNL+CW, ..., FNL+RO:FNL+RO+CW]
   * * x[CIO[0]:CIO[CW]];
   */
  //int lbr, row_width, col_width; //LBR, RW, CW
  //int first_nnz_loc, row_offset; //FNL, RO
  //int *coli_offset;// CIO

  PSCT2V1(int br, int rw, int cw, int fnl, int ro, int *cio): Codelet(br, -1,
                                                                      rw, cw,
                                                                      fnl,
                                                                      ro, cio){};

  CodeletType get_type() override{return CodeletType::TYPE_PSC2;}
  //void pack()override;
 };

 struct PSCT3V1:public Codelet{
  /**
   * y[RS] = Ax[FNL:FNL+NN] * x[CIO[0],CIO[NN]];
   */
//  int row_start, num_nnz; //RS, NN
//  int first_nnz_loc; //FNL
//  int *coli_offset;// CIO

  PSCT3V1(int rs, int nn, int fnl, int *cio ): Codelet(rs, -1, 1, nn, fnl,
                                                       -1, cio){};

  CodeletType get_type() override{return CodeletType::TYPE_PSC3;}
  //void pack()override;
 };


 void inspectCodelet(DDT::GlobalObject& d, DDT::PatternDAG* c,
                     const std::vector<Codelet>& cl);
 }


#endif //DDT_INSPECTOR_H
