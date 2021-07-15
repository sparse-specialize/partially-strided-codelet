/*
 * =====================================================================================
 *
 *       Filename:  Executor.cpp
 *
 *    Description:  Executes patterns found in codes from a differentiated matrix 
 *
 *        Version:  1.0
 *        Created:  2021-07-13 09:25:02 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Zachary Cetinic, 
 *   Organization:  University of Toronto
 *
 * =====================================================================================
 */

#include "DDT.h"
//#include "smp_def.h"
//#include "io.h"
#include "Executor.h"
#include "Inspector.h"
#include "SpMVGenericCode.h"
#include "ParseMatrixMarket.h"

#include <chrono>
#include <vector>

namespace DDT {
void executeSpTRSCodelets(const std::vector<DDT::Codelet*>& cl, const DDT::Config c) {
}

void executeSPMVCodelets(const std::vector<DDT::Codelet*>& cl, const DDT::Config c) {
  // Read matrix
  auto m = readSparseMatrix<CSR>(c.matrixPath);

  // Setup memory
  auto x = new double[m.c]();
  for (int i = 0; i < m.c; i++) {
      x[i] = 1;
  }
  auto y = new double[m.r]();

  // Execute SpMV
  spmv_generic(m.r, m.Lp, m.Li, m.Lx, x, y, cl);

  // Clean up memory
  delete[] x;
  delete[] y;
}


void executeSPMVCodelets(const std::vector<DDT::Codelet*>& cl, const
DDT::Config c, const int r, const int* Lp, const int *Li, const double*Lx,
const double* x, double* y) {
  // Read matrix
  //CSR m = readSparseMatrix<CSR>(c.matrixPath);

  // Execute SpMV
  spmv_generic(r, Lp, Li, Lx, x, y, cl);

 }
/**
 * @brief Executes codelets found in a matrix performing a computation
 *
 * @param cl List of codelets to perform computation on
 * @param c  Configuration object for setting up executor
 */
void executeCodelets(const std::vector<DDT::Codelet*>& cl, const DDT::Config c) {
  switch (c.op) {
    case DDT::OP_SPMV:
      executeSPMVCodelets(cl, c);
      break;
    case DDT::OP_SPTRS:
      executeSpTRSCodelets(cl, c);
    default:
      break;
  }
}
 void executeCodelets(const std::vector<DDT::Codelet*>& cl, const DDT::Config
 c, Args args) {
  switch (c.op) {
   case DDT::OP_SPMV:
    executeSPMVCodelets(cl, c,args.r, args.Lp, args.Li, args.Lx, args.x, args
    .y);
    break;
   case DDT::OP_SPTRS:
    executeSpTRSCodelets(cl, c);
   default:
    break;
  }
 }

}
