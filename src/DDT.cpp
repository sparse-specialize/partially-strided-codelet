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

#define MAX_THREADS 1028

#include "DDT.h"
#include "ParseMatrixMarket.h"
#include "SpTRSVModel.h"

#include <chrono>
#include <iostream>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace DDT {
  int closest_row(int nnz_num, const int *Ap, int init_row = 0){
    int i = init_row;
    while ( Ap[i] <= nnz_num )
      i++;
    return i-1;
  }

  void getSpMVIterationThreadBounds(int* bnd_row_array, int nThreads, const Matrix& m) {
    int nnz_bounds[1028];
    int nnz_part = m.nz/nThreads;
    int bnd_row = closest_row(nnz_part, m.Lp, 0);
    bnd_row_array[0] = 0;
    bnd_row_array[1] = bnd_row;
    nnz_bounds[0] = m.Lp[bnd_row];
    for (int i = 1; i < nThreads-1; ++i) {
      nnz_bounds[i] = nnz_bounds[i-1] + nnz_part;
      bnd_row = closest_row(nnz_bounds[i], m.Lp, bnd_row);
      bnd_row_array[i+1] = bnd_row;
      nnz_bounds[i] = m.Lp[bnd_row];
    }
    nnz_bounds[nThreads-1] = m.nz;
    bnd_row_array[nThreads] = m.r;
  }

  void getSpTRSVIterationThreadBounds(int* bnd_row_array, int nThreads, const Matrix& m) {
    getSpMVIterationThreadBounds(bnd_row_array, nThreads, m);
  }


  /**
   * @brief Generates memory trace divided up into level sets for SpTRSV
   *
   * @param m Matrix to be used to generate level set memory trace for SpTRSV
   * @param nThreads Number of threads to be used by inspector/executor
   *
   * @return Global object containing memory allocations
   */
    DDT::GlobalObject allocateLevelSetSpTRSVMemoryTrace(const Matrix& m, int nThreads) {
        int lp = nThreads, cp = 2, ic = 1;
        auto *sm = new sparse_avx::SpTRSVModel(m.r, m.c, m.nz, m.Lp, m.Li, lp, cp, ic);
        auto tdt = sm->generate_3d_trace(nThreads);

         for (auto const& bnd : sm->_wp_bounds) {
             std::cout << bnd << std::endl;
         }
    }

  /**
   * @brief Allocates memory trace for serial SpTRSV calculation
   * @param m Matrix
   * @param nThreads
   * @return
   */
  DDT::GlobalObject allocateSpTRSVMemoryTrace(const Matrix& m, int nThreads) {
    // Calculate memory needed
    int TPR = 3;
    int dps = m.r;
    int nd = m.nz*TPR;
    int nTuples = nd%4+nd+1;
    int* iba = new int[nThreads+1]();  // Iteration bounds array

    // Allocate memory
    auto mem = new int[nTuples*2+m.nz]();
    auto* codelets = new DDT::PatternDAG[m.nz]();
    int** dp = new int*[m.r+1];
    auto tuples = mem;
    auto df = mem+nTuples;
    auto o = mem+nTuples*2;
  
    // Determine bounds
    getSpTRSVIterationThreadBounds(iba, nThreads, m);

    // Fill Memory
    for (int t = 0; t < nThreads; t++) {
      for (int i = iba[t]; i < iba[t+1]; i++) {
        int j = m.Lp[i];
        for (; j < m.Lp[i+1] - 1; j++) {
          tuples[j*TPR] = i;
          tuples[j*TPR+1] = j;
          tuples[j*TPR+2] = m.Li[j];

          codelets[j].ct = tuples+j*TPR;
        }
        tuples[j*TPR] = i;
        tuples[j*TPR+1] = j;
        tuples[j*TPR+2] = m.Li[j];
        codelets[j].ct = tuples+j*TPR;

        dp[i] = tuples + m.Lp[i] * 3;
      }
      dp[iba[nThreads]] = tuples + m.Lp[iba[nThreads]] * 3;
    }

    return GlobalObject{ MemoryTrace{dp, dps}, codelets, df, o, m.nz, iba };
  }

  DDT::GlobalObject allocateSpMVMemoryTrace(const Matrix& m, int nThreads) {
    // Calculate memory needed
    int TPR = 3;
    int dps = m.r;
    int nd = m.nz*TPR;
    int nTuples = nd%4+nd+1;

    int* iba = new int[nThreads+1]();  // Iteration bounds array

    // Allocate memory
    auto mem = new int[nTuples*2+m.nz]();
    auto* codelets = new DDT::PatternDAG[m.nz]();
    int** dp = new int*[m.r+1];
    auto tuples = mem;
    auto df = mem+nTuples;
    auto o = mem+nTuples*2;

    // Determine bounds
    getSpMVIterationThreadBounds(iba, nThreads, m);

    // Convert matrix into Parallel SPMV Trace
#pragma omp parallel for num_threads(nThreads)
    for (int t = 0; t < nThreads; t++) {
      for (int i = iba[t]; i < iba[t+1]; i++) {
        for (int j = m.Lp[i]; j < m.Lp[i+1]; j++) {
          tuples[j*TPR] = i;
          tuples[j*TPR+1] = j;
          tuples[j*TPR+2] = m.Li[j];

          codelets[j].ct = tuples+j*TPR;
        }
        dp[i] = tuples + m.Lp[i] * 3;
      }
    }
    dp[iba[nThreads]] = tuples + m.Lp[iba[nThreads]] * 3;

    return GlobalObject{ MemoryTrace{dp, dps}, codelets, df, o, m.nz, iba };
  }

  void runSpTRSVModel(const Matrix& m) {

  }

  void runSpMVModel(const Matrix& m) {
  }

  void printTuple(int* t, std::string&& s) {
    std::cout << s << ": (" << t[0] << "," << t[1] << "," << t[2] << ")" << std::endl;
  }


  /** 
   * @brief Initializes memory and trace for framework
   *
   * @param cfg Configuration for operation, matrix and threading
   * @return GlobalObject containing memory for inspector
   */
  DDT::GlobalObject init(const DDT::Config& cfg) {
    // Parse matrix file
    Matrix m;
    if (cfg.sf == CSR_SF) {
       m = readSparseMatrix<DDT::CSR>(cfg.matrixPath);
    } else if (cfg.sf == CSC_SF) {
      m = readSparseMatrix<DDT::CSC>(cfg.matrixPath);
    }

    // Allocate memory and generate trace
    DDT::GlobalObject d;
    if (cfg.op == OP_SPMV) {
      d = DDT::allocateSpMVMemoryTrace(m, cfg.nThread);
    } else if (cfg.op == OP_SPTRS) {
      d = DDT::allocateSpTRSVMemoryTrace(m, cfg.nThread);
    } else {
      throw std::runtime_error("Error: Operation not currently supported");
    }

    return d;
  }

  /** 
   * @brief Deallocates memory in global object
   *
   * @param d Global object containing inspector memory
   */
  void free(DDT::GlobalObject d) {
    delete d.mt.ip[0];
    delete d.mt.ip;
    delete d.c;
    delete d.tb;
  }

}
