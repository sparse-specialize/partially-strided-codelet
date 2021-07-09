//
// Created by cetinicz on 2021-07-06.
//

#ifndef DDT_PATTERNMATCHING_H
#define DDT_PATTERNMATCHING_H

#include "DDT.h"

#include <chrono>

void computeFirstOrder(int* differences, int* tuples, int numTuples);

void computeParallelizedFOD(int** ip, int ips, int* differences);

bool tuplesHaveSameFOD(int* lhs, int* mid, int* rhs);

void mineDifferences(int** ip, int ips, DDT::Codelet* c, int* d);

double getTimeDifference(std::chrono::steady_clock::time_point t1, std::chrono::steady_clock::time_point t2);

void findCLCS(int tpd, int* lhstp, int* rhstp, int lhstps, int rhstps, DDT::Codelet* lhscp, DDT::Codelet* rhscp, int* lhstpd, int* rhstpd);

void printTuple(int* t, std::string&& s);

#endif  // DDT_PATTERNMATCHING_H
