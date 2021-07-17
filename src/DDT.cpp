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
#include "SpMVModel.h"

#include <chrono>
#include <iostream>
#include <tuple>
#include <vector>

namespace DDT {
  int closest_row(int nnz_num, const int *Ap, int init_row=0){
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

  DDT::GlobalObject allocateSpMVMemoryTrace(const Matrix& m, int nThreads) {
    // Calculate memory needed
    int TPR = 3;
    int nd = m.nz*TPR;
    int nTuples = nd%4+nd+1;

    int* iba = new int[nThreads+1]();  // Iteration bound array

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

//          std::cout << tuples[j*TPR] << "," << tuples[j*TPR+1] << "," << tuples[j*TPR+2] << std::endl;

          codelets[j].ct = tuples+j*TPR;
        }
        dp[i] = tuples + m.Lp[i] * 3;
      }
    }
    dp[iba[nThreads]] = tuples + m.Lp[iba[nThreads]] * 3;

    int dps = m.r;

//     Convert matrix into SpMV Trace
//     int dps = 0, cnt = 0;
//     for (int i = 0; i < m.nz; i++) {
//       if (m.Lp[cnt] == i) {
//         cnt++;
//       }
//       tuples[i*3+0] = cnt-1;
//       tuples[i*3+1] = i;
//       tuples[i*3+2] = m.Li[i];
//
//       codelets[i].ct = tuples+i*3;
//
//       if (i == 0 || (tuples[(i-1)*3] != tuples[i*3])) {
//         dp[dps++] = tuples+i*3;
//       }
//     }
//     dp[dps] = tuples + nd;

    return GlobalObject{ MemoryTrace{dp, dps}, codelets, df, o, m.nz, iba };
  }

  void runSpMVModel(const Matrix& m) {
    // // Make matrix to 
    int num_threads = 1;
    auto sm = new sparse_avx::SpMVModel(m.r, m.c, m.nz, m.Lp, m.Li);
    auto trs = sm->generate_trace(num_threads);
    // auto trace_array = trs[0]->MemAddr(); 
    //
    //
    //  return GlobalObject{ MemoryTrace{dp, dps}, codelets, df, o, m.nz };
  }

  void generateSingleStatement(std::stringstream& ss, int* ct) {
    ss << "y[" << ct[0] << "] += Lx[" << ct[1] << "] * x[" << ct[2] << "];\n";
  }

  void generateCodelet(std::stringstream& ss, DDT::GlobalObject& d, DDT::PatternDAG*
      c) {
    int TPR = 3;
    // Get codelet type
    auto TYPE = c->t;

    if (TYPE == DDT::TYPE_PSC1) {
      int buf[40];
      int rowCnt = 0;

      while (c->pt != c->ct) {
        buf[rowCnt++] = c->ct[0];
        int nc = (c->pt - d.mt.ip[0])/TPR;
        c->ct = nullptr;
        c = d.c + nc;
      }
      int oo = c->ct[0];
      int mo = c->ct[1];
      int vo = c->ct[2];

      ss << "{\n";
      ss << "auto yy = y+" << oo << ";";
      ss << "auto mm = Lx+" << mo << ";";
      ss << "auto xx = x+" << vo << ";\n";
      ss << "int of[] = {";
      for (int i = 0; i < rowCnt; i++) {
        ss << buf[rowCnt-i] << ",";
      }
      ss << "};\n";

      // Generated Code
      ss << "for (int i = 0; i < " << rowCnt+1 << "; i++) {\n";
      ss << "\tfor (int j = 0; j < " << c->sz+1 << "; j++) {\n";
      ss << "\t\tyy[i] += mm[of[i]+j] * x[i*j];\n";
      ss << "\t}\n";
      ss << "}\n";
      ss << "}\n";
      c->ct = nullptr;
    } else if (TYPE == DDT::TYPE_PSC2) {
      int buf[40];
      int rowCnt = 0;

      int mi = c->ct[1] - c->pt[1];
      int vi = c->ct[2] - c->pt[2];

      while (c->pt != c->ct) {
        int nc = (c->pt - d.mt.ip[0])/TPR;
        c->ct = nullptr;
        c = d.c + nc;
        rowCnt++;
      }
      //        std::cout << "Codelet At: (" << c->ct[0] << "," << c->ct[1]<< "," << c->ct[2] << ")\n";

      int oo = c->ct[0];
      int mo = c->ct[1];
      int vo = c->ct[2];

      ss << "{\n";
      ss << "auto yy = y+" << oo << ";";
      ss << "auto mm = Lx+" << mo << ";";
      ss << "auto xx = x+" << vo << ";\n";
      ss << "int of[] = {";
      for (int i = 0; i < c->sz+1; i++) {
        ss << c[i].ct[2] << ",";
      }
      ss << "};\n";

      // Generated Code
      ss << "for (int i = 0; i < " << rowCnt+1 << "; i++) {\n";
      ss << "\tfor (int j = 0; j < " << c->sz+1 << "; j++) {\n";
      ss << "\t\tyy[i] += mm[i*" << mi << "+j] * x[i*"<< vi << "+of[j]];\n";
      ss << "\t}\n";
      ss << "}\n";
      ss << "}\n";

      c->ct = nullptr;
    } else if (TYPE == DDT::TYPE_FSC) {
      int rowCnt = 0;

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

      int oo = c->ct[0];
      int mo = c->ct[1];
      int vo = c->ct[2];

      ss << "{\n";
      ss << "auto yy = y+" << oo << ";";
      ss << "auto mm = Lx+" << mo << ";";
      ss << "auto xx = x+" << vo << ";\n";
      ss << "for (int i = 0; i < " << rowCnt+1 << "; i++) {\n";
      ss << "\tfor (int j = 0; j < " << c->sz+1 << "; j++) {\n";
      ss << "\t\tyy[i] += mm[i*" << mi << "+j*" << mj << "] * xx[i*" << vi << "+j*"<<vj<<"];\n";
      ss << "\t}\n";
      ss << "}\n";
      ss << "}\n";

      c->ct = nullptr;
    }
  }

  void printTuple(int* t, std::string&& s) {
    std::cout << s << ": (" << t[0] << "," << t[1] << "," << t[2] << ")" << std::endl;
  }

  void generateSource(DDT::GlobalObject& d) {
    int TPR = 3;

    auto t0 = std::chrono::steady_clock::now();

    std::stringstream ss;

    ss << "void f0(double* y, double* Lx, double* x) {\n";

    // Iterate through codelets
    for (int i = d.mt.ips-1; i >= 0; i--) {
      for (int j = 0; j < d.mt.ip[i+1]-d.mt.ip[i];) {
        int cn = ((d.mt.ip[i]+j)-d.mt.ip[0]) / TPR;
        if (d.c[cn].pt != nullptr && d.c[cn].ct != nullptr) {
          generateCodelet(ss, d, d.c+cn);
          j += (d.c[cn].sz+1) * TPR;
        } else if (d.c[cn].ct != nullptr) {
          // Regular codelet
          generateSingleStatement(ss, d.c[cn].ct);
          j += TPR;
        } else {
          j += TPR * (d.c[cn].sz + 1);
        }
      }
    }
    ss << "}\n";

    std::ofstream file("output.cpp");
    file << ss.str();
    file.close();

    auto t1 = std::chrono::steady_clock::now();
    auto td = std::chrono::duration_cast<std::chrono::duration<double>>(t1-t0).count();
    std::cout << "Generation Time: " << td << std::endl;
  }

  DDT::GlobalObject init(DDT::Config& cfg) {
    // Parse matrix
    auto m = readSparseMatrix<CSR>(cfg.matrixPath);

    // Allocate memory and generate trace
    auto d = DDT::allocateSpMVMemoryTrace(m, cfg.nThread);

    return d;
  }

  void free(DDT::GlobalObject d) {
    delete d.mt.ip[0];
    delete d.mt.ip;
    delete d.c;
    delete d.tb;
  }

}
