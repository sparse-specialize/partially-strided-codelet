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

#include <string>
#include <vector>

namespace DDT {
  enum NumericalOperation {
    OP_SPMV,
    OP_SPTRS
  };

  enum CodeletType {
    TYPE_FSC,
    TYPE_PSC1,
    TYPE_PSC2,
    TYPE_PSC3
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

  GlobalObject init(DDT::Config& config);

  void free(DDT::GlobalObject d);

  /// Used for testing executor
  struct Args {
  double *x, *y;
  int r; int* Lp; int* Li; double* Lx;
 };

}

#endif
