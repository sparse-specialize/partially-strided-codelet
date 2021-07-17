//
// Created by Kazem on 7/12/21.
//

#include "GenericCodelets.h"
#include "SpMVGenericCode.h"

#include <vector>

namespace DDT {
    template<class type>
    bool is_float_equal(const type x, const type y, double absTol, double relTol) {
        return std::abs(x - y) <= std::max(absTol, relTol * std::max(std::abs(x), std::abs(y)));
    }
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

        const double eps = 1e-8;


        // Compare outputs
        bool wrong = false;
        for (int i = 0; i < n; i++) {
            if (!is_float_equal(yy[i],y[i],eps,eps)) {
                std::cout << "Wrong at 'i' = " << i << std::endl;
                std::cout << "(" << yy[i] << "," << y[i] << ")" << std::endl;
                wrong = true;
            }
        }
        if (wrong)
            return false;

        // Clean up memory
        delete[] yy;

        return true;
    }

  void spmv_generic(const int n, const int *Ap, const int *Ai, const double
      *Ax, const double *x, double *y, const std::vector<Codelet*>* lst, const DDT::Config& cfg) {
      // Perform SpMV
#pragma omp parallel for num_threads(cfg.nThread)
for (int i = 0; i < cfg.nThread; i++) {
    for (const auto& c : lst[i]) {
      switch (c->get_type()) {
        case CodeletType::TYPE_FSC:
          fsc_t2_2DC(y, Ax, x, c->row_offset, c->first_nnz_loc, c->lbr, c->lbr+c->row_width, c->lbc, c->col_width+c->lbc, c->col_offset);
          break;
        case CodeletType::TYPE_PSC1:
          psc_t1_2D4R(y,Ax,x,c->offsets,c->lbr,c->lbr+c->row_width,c->lbc,
              c->lbc+c->col_width);
          break;
        case CodeletType::TYPE_PSC2:
          psc_t2_2DC(y,Ax, x, c->offsets, c->row_offset, c->first_nnz_loc, c->lbr,
              c->lbr+c->row_width, c->col_width, c->col_offset);
          break;
        case CodeletType::TYPE_PSC3:
          psc_t3_1D1R(y, Ax, Ai, x, c->offsets, c->lbr, c->first_nnz_loc,
              c->col_width);
          break;
        default:
          break;
      }
    }
}

    // Verify correctness
//    if (!verifySpMV(n, Ap, Ai, Ax, x, y)) {
//        std::cout << "Error: numerical operation was incorrect." << std::endl;
//        exit(1);
//    }
//    std::cout << "op correct..." << std::endl;
  }
}
