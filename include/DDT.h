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
#include <string>

#ifndef DDT_DDT
#define DDT_DDT

namespace DDT {
  struct PatternDAG {
    int sz;
    int* ct;
    int* pt;
  };

  struct MemoryTrace {
    int** ip;
    int ips;
  };

  struct Config {
    std::string matrixPath;
  };

  struct GlobalObject {
    MemoryTrace mt;
    PatternDAG* c;
    int* d;
  };

  void generateSource(DDT::GlobalObject& d);

  GlobalObject init(DDT::Config& config);

  void free(DDT::GlobalObject d);
}

#endif
