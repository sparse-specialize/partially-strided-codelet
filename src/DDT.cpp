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
#include "SpMVModel.h"

#include <chrono>
#include <iostream>
#include <tuple>
#include <vector>

namespace DDT {
  enum CodeletType {
    FSC = 0,
    PSC1 = 1,
    PSC2 = 2
  };

  DDT::GlobalObject allocateMemory(Matrix& m) {
    // TPR
    int tpr = 3;
    // Allocate memory
    int nd = m.nz*tpr;
    auto* tuples = new int[nd%4+nd+1];
    auto* codelets = new DDT::Codelet[m.nz]();
    int** dp = new int*[m.r+1];
    auto df = new int[nd%4+nd+1]();
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

  void runSpMVModel(Matrix& m) {
    auto ap = new int[m.r+1];
    auto ai = new int[m.nz];
    int rn = 0;
    for (int i = 0; i < m.nz; ++i) {
      ai[i] = std::get<1>(m.m[i]);
      if (i == 0 || std::get<0>(m.m[i]) != std::get<0>(m.m[i-1])) {
        ap[rn++] = i;
      }
    }
    ap[rn] = m.nz;
    // make matrix to specs
    int num_thread = 1;
    auto *sm = new sparse_avx::SpMVModel(m.r, m.c, m.nz, ap, ai);
    auto trs = sm->generate_trace(num_thread);
    auto trace_array = trs[0]->MemAddr(); // This is the array that has traces
    // for (int i = 0; i < trs[0]->_num_partitions; ++i) {
      // trs[i]->print();
      // std::cout << "\n\n";
    // }
    delete[] ap;
    delete[] ai;
  }

  void generateSingleStatement(std::stringstream& ss, int* ct) {
    ss << "y[" << ct[0] << "] += Lx[" << ct[1] << "] * x[" << ct[2] << "];\n";
  }

  void generateCodelet(std::stringstream& ss, DDT::GlobalObject& d, DDT::Codelet* c) {
    int TPR = 3;
    // Get codelet type
    auto TYPE = FSC;

    int df = 0;
    for (int i = 0; i < c->sz; i++) {
      if (i == 0)
        df = c[i+1].ct[2] - c[i].ct[2];
      else if (df != (c[i+1].ct[2] - c[i].ct[2])) {
        TYPE = PSC2;
        break;
      }
    }

    if (TYPE == PSC1) {
      int buf[40];
      int rowCnt = 0;

      while (c->pt != c->ct) {
        buf[rowCnt++] = c->ct[0];
        int nc = (c->pt - d.mt.ip[0])/TPR;
        c = d.c + nc;
      }
    } else if (TYPE == PSC2) {
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

      int oo = c->ct[0];
      int mo = c->ct[1];
      int vo = c->ct[2];

      ss << "{\n";
      ss << "auto yy = y+" << oo << ";";
      ss << "auto mm = Lx+" << mo << ";";
      ss << "auto xx = x+" << vo << ";\n";
      ss << "int of[] = {";
      for (int i = 0; i < c->sz; i++) {
        ss << c[i].ct[2] << ",";
      }
      ss << "};\n";

      // Generated Code
      ss << "for (int i = 0; i < " << rowCnt+1 << "; i++) {\n";
      ss << "\tfor (int j = 0; j < " << c->sz+1 << "; j++) {\n";
      ss << "\t\tyy[i] += mm[i*" << mi << "+j] * xx[i*"<< vi << "+of[j]];\n";
      ss << "\t}\n";
      ss << "}\n";
      ss << "}\n";
    } else if (TYPE == FSC) {
      int rowCnt = 0;

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
    }
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

  DDT::GlobalObject init(DDT::Config& config) {
    // Parse matrix
    auto m = readSparseMatrix<CSR>(config.matrixPath);

    // Allocate memory and generate trace
    runSpMVModel(m);
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
