/*
 * =====================================================================================
 *
 *       Filename:  DDT.h
 *
 *    Description:  Header file for DDT.cpp 
 *
 *        Version:  1.0
 *        Created:  2021-07-08 02:16:50 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Zachary Cetinic, 
 *   Organization:  University of Toronto
 *
 * =====================================================================================
 */

#ifndef DDT_DDT
#define DDT_DDT

#include "ParseMatrixMarket.h"
#include "SpTRSVModel.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace DDT {
  enum NumericalOperation {
    OP_SPMV,
    OP_SPTRS
  };
 
  enum StorageFormat {
    CSR_SF,
    CSC_SF
  };

  struct MemoryTrace {
    int** ip;
    int ips;
  };

  struct Config {
    std::string matrixPath;
    NumericalOperation op;
    int header;
    int nThread;
    StorageFormat sf;
   int coarsening;
   bool bin_packing;
  };

  struct GlobalObject {
    MemoryTrace mt;
    PatternDAG* c;
    int* d;
    int* o;
    int onz;
    int* tb;
    sparse_avx::SpTRSVModel* sm;
    sparse_avx::Trace*** t;
  };

  void generateSource(DDT::GlobalObject& d);

  void printTuple(int* t, std::string&& s);

  template <typename M0, typename M1>
  DDT::GlobalObject allocateExternalSpTRSVMemoryTrace(const M0* m0, const
  M1* m1, const DDT::Config& cfg) {
      int lp = cfg.nThread, cp = cfg.coarsening, ic = 1, bp = cfg.bin_packing;
      auto *sm = new sparse_avx::SpTRSVModel(m0->m, m0->n, m0->nnz, m0->p,
                                             m0->i, m1->p, m1->i, lp, cp, ic,
                                             bp);
      auto trs = sm->generate_3d_trace(cfg.nThread);

      return GlobalObject{  {},  nullptr, nullptr, nullptr, 0, nullptr, sm, trs };
  }

    DDT::GlobalObject allocateSpTRSVMemoryTrace(const Matrix& m, int nThreads);

    template <typename M0>
  DDT::GlobalObject init(const M0* m0, const DDT::Config& cfg) {
        // Convert matrix into regular form
        Matrix m{};
        copySymLibMatrix(m, m0);

        // Allocate memory and generate trace
        DDT::GlobalObject d;
        if (cfg.op == OP_SPMV) {
//            d = DDT::allocateSpMVMemoryTrace(m, cfg.nThread);
        } else if (cfg.op == OP_SPTRS) {
            d = DDT::allocateSpTRSVMemoryTrace(m, cfg.nThread);
        } else {
            throw std::runtime_error("Error: Operation not currently supported");
        }

        return d;
  }

  template <typename M0, typename M1>
  DDT::GlobalObject init(const M0* m0, const M1* m1, const DDT::Config& cfg) {
      // Allocate memory and generate trace
      DDT::GlobalObject d;
      if (cfg.op == OP_SPTRS) {
          d = DDT::allocateExternalSpTRSVMemoryTrace(m0, m1, cfg);
      } else {
          throw std::runtime_error("Error: Operation not currently supported");
      }

      return d;
  }

    GlobalObject init(const DDT::Config& config);

  void free(DDT::GlobalObject d);

  /// Used for testing executor
  struct Args {
  double *x, *y;
  int r; int* Lp; int* Li; double* Lx;
 };

}

#endif
