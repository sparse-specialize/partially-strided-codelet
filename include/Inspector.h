//
// Created by kazem on 7/12/21.
//

#ifndef DDT_INSPECTOR_H
#define DDT_INSPECTOR_H

#include "DDT.h"

#include <iostream>
#include <stdexcept>
#include <vector>

#include <cassert>

namespace DDT {

 struct Codelet {
  int size;
  int *offsets;
  bool is_alloc;
  int lbr, lbc, row_width, col_width, col_offset; //LBR, RW, CW
  int first_nnz_loc, row_offset; //FNL, RO
  bool multi_stmt; //FIXME: we should remove this, each codelet has only one
  // statemet (operation)
  Codelet(int br, int bc, int rw, int cw, int fnl, int ro, int co, int *offs) : lbr(br), row_width(rw),
  lbc(bc),col_width(cw),first_nnz_loc(fnl),row_offset(ro), col_offset(co),
  offsets(offs), multi_stmt(false){}

  Codelet(int br, int bc, int rw, int cw, int fnl, int ro, int co, int *offs,
          bool ms
          ) : lbr(br), row_width(rw),lbc(bc),col_width(cw),first_nnz_loc(fnl),
          row_offset(ro), col_offset(co),offsets(offs), multi_stmt(ms){}

  virtual ~Codelet()= default;

  virtual CodeletType get_type()=0;
  virtual void print()=0;

 };

 struct FSCCodelet:public Codelet{
  /**
   * y[lbr:lbr+row_width] = Ax[FNL:FNL+CW, ..., FNL+RO:FNL+RO+CW]*x[lbc:lbc+CW];
   */
  FSCCodelet(int br, int bc, int rw, int cw, int fnl, int ro, int co) : Codelet(br,bc,
                                                                         rw,cw,fnl,ro,co,NULL){};

  FSCCodelet(int br, int bc, int rw, int cw, int fnl, int ro, int co, bool
  ms) : Codelet(br,bc,rw,cw,fnl,ro,co,NULL,ms){};
  CodeletType get_type() override{return CodeletType::TYPE_FSC;}
  void print()override;
 };


 struct PSCT1V1:public Codelet{
  /**
   * y[lbr:lbr+row_width] = Ax[RO[lbr]:RO[lbr]+CW, ...,
   * RO[lbr]:RO+CW]*x[lbc:lbc+CW];
   */
  //int lbr, lbc, row_width, col_width; //LBR, RW, CW
  //int *row_offsets; //RO
  PSCT1V1(int br, int bc, int rw, int cw, int co, int *ros): Codelet(br,bc,rw,cw,-1,
                                                             -1,co,ros){};

  PSCT1V1(int br, int bc, int rw, int cw, int co, int *ros, bool ms): Codelet
  (br,bc,rw,cw,-1,-1,co,ros, ms){};

  CodeletType get_type() override{return CodeletType::TYPE_PSC1;}
  void print()override;
 };


 struct PSCT2V1:public Codelet{
  /**
   * y[lbr:lbr+row_width] = Ax[FNL:FNL+CW, ..., FNL+RO:FNL+RO+CW]
   * * x[CIO[0]:CIO[CW]];
   */
  //int lbr, row_width, col_width; //LBR, RW, CW
  //int first_nnz_loc, row_offset; //FNL, RO
  //int *coli_offset;// CIO

  PSCT2V1(int br, int rw, int cw, int fnl, int ro, int co, int *cio): Codelet(br, -1,
                                                                      rw, cw,
                                                                      fnl,
                                                                      ro, co, cio){};

  PSCT2V1(int br, int rw, int cw, int fnl, int ro, int co, int *cio, bool ms):
  Codelet(br, -1, rw, cw, fnl, ro, co, cio, ms){};

  CodeletType get_type() override{return CodeletType::TYPE_PSC2;}
  void print()override;
 };

 struct PSCT3V1:public Codelet{
  /**
   * y[RS] = Ax[FNL:FNL+NN] * x[CIO[0],CIO[NN]];
   */
//  int row_start, num_nnz; //RS, NN
//  int first_nnz_loc; //FNL
//  int *coli_offset;// CIO

  PSCT3V1(int rs, int nn, int fnl, int *cio ): Codelet(rs, -1, 1, nn, fnl,
                                                       -1, 1, cio){};

  PSCT3V1(int rs, int nn, int fnl, int *cio, bool ms ): Codelet(rs, -1, 1, nn,
                                                              fnl,-1, 1, cio, ms){};

  CodeletType get_type() override{return CodeletType::TYPE_PSC3;}
  void print()override;
 };

    struct PSCT3V1_M:public Codelet{
        /**
         * y[RS] = Ax[FNL:FNL+NN] * x[CIO[0],CIO[NN]];
         */
//  int row_start, num_nnz; //RS, NN
//  int first_nnz_loc; //FNL
//  int *coli_offset;// CIO

        PSCT3V1_M(int rs, int nn, int fnl, int *cio ): Codelet(rs, -1, 1, nn, fnl,
                                                             -1, 1, cio, true){};

        CodeletType get_type() override{return CodeletType::TYPE_PSC3_M;}
        void print()override;
    };

    inline bool hasAdacentIteration(int i, int** ip);



    template <DDT::CodeletType Type>
    void generateCodeletType(DDT::GlobalObject& d, DDT::PatternDAG* c, std::vector<Codelet*>& cl) {
        int TPR = 3;

        // Get codelet type
        if constexpr (Type == DDT::TYPE_PSC3V2) {
            int colWidth = ((c->ct - c->pt) / TPR) + 1;

            while (c->ct != c->pt) {
                d.o[--d.onz] = c->ct[2];
                c->ct -= TPR;
            }
            d.o[--d.onz] = c->ct[2];

            int oo = c->ct[0];
            int mo = c->ct[1];

            auto cornerT = c->ct+(colWidth-1)*TPR;
            cl.emplace_back(new DDT::PSCT3V1(oo,colWidth,mo,d.o+d.onz,cornerT[0] == cornerT[2]));
        }
        if constexpr (Type == DDT::TYPE_PSC3) {
            int colWidth = ((c->ct - c->pt) / TPR) + 1;

            while (c->ct != c->pt) {
                d.o[--d.onz] = c->ct[2];
                c->ct -= TPR;
            }
            d.o[--d.onz] = c->ct[2];

            int oo = c->ct[0];
            int mo = c->ct[1];

            auto cornerT = c->ct+(colWidth-1)*TPR;
            cl.emplace_back(new DDT::PSCT3V1(oo,colWidth,mo,d.o+d.onz,cornerT[0] == cornerT[2]));
        }

        if constexpr (Type == DDT::TYPE_FSC) {
            int rowCnt = 1;

            // Loop array induction variable coefficients
            int mi = c->ct[1] - c->pt[1];
            int vi = c->ct[2] - c->pt[2];

            int mj = c->ct[4] - c->ct[1];
            int vj = c->ct[5] - c->ct[2];

            while (c->pt != c->ct) {
                int nc = (c->pt - d.mt.ip[0])/TPR;
                c->ct = nullptr;
                c = d.c + nc;
                rowCnt++;
            }

            if (vj != 1) { std::cout << c->ct[1] << std::endl; std::cout << vj << std::endl; assert(vj == 1); }

            // Loop Array Offsets
            int oo = c->ct[0];
            int mo = c->ct[1];
            int vo = c->ct[2];

            // Generate codelet
            cl.emplace_back(new DDT::FSCCodelet(oo,vo,rowCnt, c->sz+1,mo,mi,vi));
        }

        if constexpr (Type == DDT::TYPE_PSC1) {
            int rowCnt = 1;

            int vi = c->ct[2] - c->pt[2];
            int vj = c->ct[5] - c->ct[2];

            while (c->pt != c->ct) {
                d.o[--d.onz] = c->ct[1];
                int nc = (c->pt - d.mt.ip[0])/TPR;
                c->ct = nullptr;
                c = d.c + nc;
                rowCnt++;
            }
            d.o[--d.onz] = c->ct[1];

            assert(vi == 0);
            assert(vj == 1);

            // Loop Array Offsets
            int oo = c->ct[0];
            int vo = c->ct[2];

            // Generate codelet
            cl.push_back(new DDT::PSCT1V1(oo,vo,rowCnt,c->sz+1,vi,d.o+d.onz));
        }

        if constexpr (Type == DDT::TYPE_PSC2) {
            int rowCnt = 1;

            // Loop array induction variable coefficients
            int mi = c->ct[1] - c->pt[1];
            int vi = c->ct[2] - c->pt[2];

            int mj = c->ct[4] - c->pt[1];

            while (c->pt != c->ct) {
                assert((c->ct[1] - c->pt[1]) == mi);
                int nc = (c->pt - d.mt.ip[0])/TPR;
                c->ct = nullptr;
                c = d.c + nc;
                rowCnt++;
            }

            // Loop Array Offsets
            int oo = c->ct[0];
            int mo = c->ct[1];

            // Generate offset array into vector
            for (int i = c->sz; i >= 0; --i) {
                d.o[--d.onz] = c[i].ct[2];
            }

            // Generate codelet
            cl.emplace_back(new DDT::PSCT2V1(oo, rowCnt, c->sz+1, mo, mi, vi, d.o+d.onz));
        }
        c->ct = nullptr;
    }

    void inspectSerialTrace(DDT::GlobalObject& d, std::vector<Codelet*>* cl, const DDT::Config& cfg);

        void free(std::vector<DDT::Codelet*>& cl);
}


#endif //DDT_INSPECTOR_H
