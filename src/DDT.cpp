/*
 * =====================================================================================
 *
 *       Filename:  DDT.cpp
 *
 *    Description:  File containing main DDT functionality 
 *
 *        Version:  1.0
 *        Created:  2021-07-08 02:15:12 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Zachary Cetinic, 
 *   Organization:  University of Toronto
 *
 * =====================================================================================
 */
#include "DDT.h"
#include "ParseMatrixMarket.h"

#include <iostream>
#include <tuple>
#include <vector>

namespace DDT {
  DDT::GlobalObject allocateMemory(Matrix& m) {
    // TPR
    int tpr = 3;
    // Allocate memory
    int nd = m.nz*tpr;
    auto* tuples = new int[nd%4+nd+1];
    auto* codelets = new DDT::Codelet[m.nz]();
    int** dp = new int*[m.r+1];
    auto df = new int[nd-(tpr*m.r)+1]();
    // posix_memalign(reinterpret_cast<void **>(d), 32, nd);

    // Convert matrix into SpMV Trace
    int dps = 0;
    for (int i = 0; i < m.nz; i++) {
      auto& t = m.m[i];
      tuples[i*3] = std::get<0>(t);
      tuples[i*3+1] = i;
      tuples[i*3+2] = std::get<1>(t);

      codelets[i].ct = tuples+i*3;

      if (i == 0 || (tuples[(i-1)*3] != tuples[i*3])) {
        dp[dps++] = tuples+i*3;
      }
    }
    dp[dps] = tuples + nd;

    return GlobalObject{ MemoryTrace{dp, dps}, codelets, df };
  }

  DDT::GlobalObject init(DDT::Config& config) {
    // Parse matrix
    auto m = readSparseMatrix<CSR>(config.matrixPath);

    // Allocate memory and generate trace
    auto d = DDT::allocateMemory(m);

    return d;
  }

  void free(DDT::GlobalObject d) {
      delete d.mt.ip[0];
      delete d.d;
      delete d.mt.ip;
      delete d.c;
  }
}
