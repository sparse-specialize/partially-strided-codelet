//
// Created by kazem on 7/12/21.
//

#ifndef DDT_INSPECTOR_H
#define DDT_INSPECTOR_H

#include "DDT.h"

#include <stdexcept>
#include <vector>

namespace DDT {
  struct Codelet{

    Codelet();

  };

  struct FSCCodelet:public Codelet{
    /**
     * y[lbr:lbr+row_width] = Ax[FNL:FNL+CW, ..., FNL+RO:FNL+RO+CW]*x[lbc:lbc+CW];
     */
    int lbr;
    int lbc;
    int row_width;
    int col_width; //LBR, RW, CW
    int first_nnz_loc;
    int row_offset; //FNL, RO
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

  template <DDT::CodeletType Type>
    void generateCodeletType(DDT::GlobalObject& d, DDT::PatternDAG* c, std::vector<Codelet>& cl) {
      int TPR = 3;

      // Get codelet type
      if constexpr (Type == DDT::TYPE_PSC3) {

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
        cl.emplace_back(DDT::FSCCodelet{ {}, oo, vo, rowCnt, c->sz, mo, mi });
      }

      if constexpr (Type == DDT::TYPE_PSC1) {
        int buf[40];
        int bufSz = 40;
        int rowCnt = 0;

        while (c->pt != c->ct) {
          buf[--bufSz] = c->ct[1];
          int nc = (c->pt - d.mt.ip[0])/TPR;
          c = d.c + nc;
          rowCnt++;
        }

        // Loop Array Offsets
        int oo = c->ct[0];
        int vo = c->ct[2];

        // Generate codelet
        cl.emplace_back(DDT::PSCT1V1{ {}, oo, vo, rowCnt, c->sz, nullptr });
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

        // Generate codelet
        cl.emplace_back(DDT::PSCT2V1{ {}, oo, vo, rowCnt, c->sz, nullptr });
      }
    }

  void inspectCodelets(DDT::GlobalObject& d, const std::vector<Codelet>& cl);
}


#endif //DDT_INSPECTOR_H
