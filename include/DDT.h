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

  enum CodeletType {
    TYPE_FSC,
    TYPE_PSC1,
    TYPE_PSC2,
    TYPE_PSC3,
    TYPE_PSC3_M
  };

  struct PatternDAG {
    int sz;
    int* ct;
    int* pt;
    CodeletType t;
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
  };

  struct GlobalObject {
    MemoryTrace mt;
    PatternDAG* c;
    int* d;
    int* o;
    int onz;
    int* tb;
  };

  void generateSource(DDT::GlobalObject& d);

  void printTuple(int* t, std::string&& s);

  template <typename Matrix>
  DDT::GlobalObject allocateExternalSpTRSVMemoryTrace(const Matrix* m, const DDT::Config& cfg) {
      int lp = cfg.nThread, cp = 2, ic = 1;
      auto *sm = new sparse_avx::SpTRSVModel(m->m, m->n, m->nnz, m->p, m->i, lp, cp, ic);
      auto trs = sm->generate_3d_trace(cfg.nThread);
  }

  template <typename Matrix>
  DDT::GlobalObject init(const Matrix* m, const DDT::Config& cfg) {
      // Allocate memory and generate trace
      DDT::GlobalObject d;
      if (cfg.op == OP_SPTRS) {
          d = DDT::allocateExternalSpTRSVMemoryTrace(m, cfg);
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
