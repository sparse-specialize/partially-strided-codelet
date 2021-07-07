//
// Created by cetinicz on 2021-07-06.
//
#include <chrono>

#ifndef DDT_PATTERNMATCHING_H
#define DDT_PATTERNMATCHING_H

struct Codelet {
  int sz;
  int* ct;
  int* pt;
};

void computeFirstOrder(int* differences, int* tuples, int numTuples);

void computeParallelizedFOD(int** ip, int ips, int* differences);

bool tuplesHaveSameFOD(int* lhs, int* mid, int* rhs);

void mineDifferences(int** ip, int ips, Codelet* c);

double getTimeDifference(std::chrono::steady_clock::time_point t1, std::chrono::steady_clock::time_point t2);

void findCLCS(int tpd, int* lhstp, int* rhstp, int lhstps, int rhstps, Codelet* lhscp, Codelet* rhscp);

#endif  // DDT_PATTERNMATCHING_H
