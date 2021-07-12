//
// Created by kazem on 7/12/21.
//

#ifndef DDT_INSPECTOR_H
#define DDT_INSPECTOR_H

#include "DDT.h"

#include <stdexcept>
#include <vector>

namespace DDT{

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
  FSCCodelet(int br, int bc, int rw, int cw, int fnl, int ro) : Codelet(br,bc,
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

    template <DDT::CodeletType Type>
    void generateCodeletType(DDT::GlobalObject& d, DDT::PatternDAG* c, std::vector<Codelet*>& cl) {
        int TPR = 3;

        // Get codelet type
        if constexpr (Type == DDT::TYPE_PSC3) {
            int oo = c->ct[0];
            int mo = c->ct[1];

            int colWidth = (c->ct - c->pt) / TPR;

            while (c->ct != c->pt) {
                d.o[--d.onz] = c->ct[2];
                c->ct -= TPR;
            }

            // Generate codelet
            cl.emplace_back(new DDT::PSCT3V1(oo,colWidth,mo,d.o+d.onz));
        }

        if constexpr (Type == DDT::TYPE_FSC) {
            int rowCnt = 0;

            // Loop array induction variable coefficients
            int mi = c->ct[1] - c->pt[1];
            int vi = c->ct[2] - c->pt[2];

            int mj = c->ct[4] - c->pt[1];
            int vj = c->ct[5] - c->pt[2];

            while (c->pt != c->ct) {
                int nc = (c->pt - d.mt.ip[0])/TPR;
                c->ct = nullptr;
                c = d.c + nc;
                rowCnt++;
            }

            // Loop Array Offsets
            int oo = c->ct[0];
            int mo = c->ct[1];
            int vo = c->ct[2];

            // Generate codelet
            cl.emplace_back(new DDT::FSCCodelet(oo,vo,rowCnt, c->sz,mo,mi));
        }

        if constexpr (Type == DDT::TYPE_PSC1) {
            int rowCnt = 0;

            while (c->pt != c->ct) {
                d.o[--d.onz] = c->ct[1];
                int nc = (c->pt - d.mt.ip[0])/TPR;
                c = d.c + nc;
                rowCnt++;
            }

            // Loop Array Offsets
            int oo = c->ct[0];
            int vo = c->ct[2];

            // Generate codelet
            cl.push_back(new DDT::PSCT1V1(oo,vo,rowCnt,c->sz,d.o+d.onz));
        }

        if constexpr (Type == DDT::TYPE_PSC2) {
            int rowCnt = 0;

            // Loop array induction variable coefficients
            int mi = c->ct[1] - c->pt[1];
            int vi = c->ct[2] - c->pt[2];

            int mj = c->ct[4] - c->pt[1];

            while (c->pt != c->ct) {
                int nc = (c->pt - d.mt.ip[0])/TPR;
                c = d.c + nc;
                rowCnt++;
            }

            // Loop Array Offsets
            int oo = c->ct[0];
            int mo = c->ct[1];
            int vo = c->ct[2];

            // Generate offset array into vector
            for (int i = c->sz; i >= 0; --i) {
                d.o[--d.onz] = c[i].ct[2];
            }

            // Generate codelet
            cl.emplace_back(new DDT::PSCT2V1(oo, rowCnt, c->sz, mo, mi, d.o+d.onz));
        }
    }

    void inspectCodelets(DDT::GlobalObject& d, std::vector<Codelet*>& cl);
}


#endif //DDT_INSPECTOR_H