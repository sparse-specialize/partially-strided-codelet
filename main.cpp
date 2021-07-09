#include "DDT.h"
#include "Input.h"
#include "PatternMatching.h"

#include <iostream>

int main(int argc, char** argv) {
  // Parse program arguments
  auto config = DDT::parseInput(argc, argv);

  // Allocate memory and generate global object
  auto d = DDT::init(config);

  // Compute and Mark regions for PSC-3
  computeParallelizedFOD(d.mt.ip, d.mt.ips, d.d);

  // Mine trace and profile
  mineDifferences(d.mt.ip, d.mt.ips, d.c, d.d);

  // Compact Codes
  int cnt = 0;
  for (int i = 0; i < d.mt.ips; ++i) {
      for (int j = 0; j < d.mt.ip[i+1] - d.mt.ip[i]; j+=3) {
          if (d.c[cnt].ct == d.c[cnt].pt) {
              std::cout << "Size: " << d.c[cnt].sz + 1 << std::endl;
              printTuple(d.c[cnt].ct, "Start: ");
          }
          else if (nullptr != d.c[cnt].pt) {
              printTuple(d.c[cnt].ct, "Middle: ");
          }
          cnt++;
      }
  }

  // Generate Codes

  // Clean up
  DDT::free(d);

  return 0;
}
