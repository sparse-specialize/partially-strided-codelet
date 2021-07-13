//
// Created by Kazem on 7/12/21.
//

#include "GenericCodelets.h"
#include "SpMVGenericCode.h"

#include <vector>

namespace DDT {
    bool verifySpMV(const int n, const int *Ap, const int *Ai, const double
    *Ax, const double *x, double *y) {
        // Allocate memory
        auto yy = new double[n]();

        // Perform SpMV
        for (int i = 0; i < n; i++) {
            for (int j = Ap[i]; j < Ap[i+1]; j++) {
                yy[i] += Ax[j] * x[Ai[j]];
            }
        }

        // Compare outputs
        for (int i = 0; i < n; i++) {
            if (yy[i] != y[i]) {
                std::cout << "Wrong at 'i' = " << i << std::endl;
                return false;
            }
        }

        // Clean up memory
        delete[] yy;

        return true;
    }

  void spmv_generic(const int n, const int *Ap, const int *Ai, const double
      *Ax, const double *x, double *y, const std::vector<Codelet*>& lst) {
      // Perform SpMV
    for (const auto& c : lst) {
      switch (c->get_type()) {
        case CodeletType::TYPE_FSC:
          break;
        case CodeletType::TYPE_PSC1:
          psc_t1_2D4R(y,Ax,x,c->offsets,c->lbr,c->lbr+c->row_width,c->lbc,
              c->lbc+c->col_width);
          break;
        case CodeletType::TYPE_PSC2:
          psc_t2_2DC(y,Ax, x, c->offsets, c->row_offset, c->first_nnz_loc, c->lbr,
              c->lbr+c->row_width, c->col_width);
          break;
        case CodeletType::TYPE_PSC3:
          psc_t3_1D1R(y, Ax, Ai, x, c->offsets, c->lbr, c->first_nnz_loc,
              c->col_width);
          break;
        default:
          break;
      }
    }

    // Verify correctness
    if (!verifySpMV(n, Ap, Ai, Ax, x, y)) {
        std::cout << "Error: numerical operation was incorrect." << std::endl;
        exit(1);
    }
  }
}
