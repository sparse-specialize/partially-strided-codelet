#include "DDT.h"
#include "Executor.h"
#include "Input.h"
#include "Inspector.h"
#include "PatternMatching.h"

#include <iostream>

/**
 *
 * Different iteration sections processed differently
 *
 * 1) Prune iterations completely out of processing
 * 2) Mark single iterations
 * 3) Something to enable PSC (?)
 * 4) Di
 */
void pruneIterations(DDT::MemoryTrace mt, int density) {
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < mt.ips; i++) {
    int oneD = 0;
    while (mt.ip[i+1] - mt.ip[i] == 1) {
      ++oneD;
    }
//    mt.a.nt = i+oneD;
//    mt.a.t  = 0;
  }
  auto t1 = std::chrono::steady_clock::now();
  std::cout << "Pruning Time: " << getTimeDifference(t0,t1) << std::endl;
}

int main(int argc, char** argv) {
  // Parse program arguments
  auto config = DDT::parseInput(argc, argv);

  // Allocate memory and generate global object
  auto d = DDT::init(config);

  // Prune memory trace
  pruneIterations(d.mt, 10);

  // Compute and Mark regions for PSC-3
  computeParallelizedFOD(d.mt.ip, d.mt.ips, d.d);

  // Mine trace and profile
  mineDifferences(d.mt.ip, d.mt.ips, d.c, d.d);

//  for (int i = 0; i < d.mt.ips; i++) {
//      for (int j = 0; j < d.mt.ip[i+1]-d.mt.ip[i]; j+=3) {
//          int cn = ((d.mt.ip[i]+j)-d.mt.ip[0]) / 3;
////          std::cout << d.d[cn*3] << "," << d.d[cn*3+1] << "," << d.d[cn*3+2] << std::endl;
//          if (d.c[cn].ct == d.c[cn].pt && d.c[cn].ct[0] == 2463) {
//              DDT::printTuple(d.c[cn].ct, "Tuple Start");
////              std::cout << "Tuple Size: " << d.c[cn].sz << std::endl;
////              std::cout << "Tuple I: " << i << std::endl;
//          }
//          if (d.c[cn].pt != nullptr && d.c[cn].ct[0] == 2463) {
//              DDT::printTuple(d.c[cn].ct, "Row Tuple");
//          }
//      }
//  }

  // Generate Codes
//  DDT::generateSource(d);

  // Parse into run-time Codelets
  std::vector<DDT::Codelet*> cl;
   DDT::inspectCodelets(d, cl);

//   for (auto const& c : cl) {
//       if (c->lbr == 399) {
//           std::cout << c->get_type() << std::endl;
//           std::cout << c->first_nnz_loc << std::endl;
//           if (c->get_type() == 1) {
//               std::cout << c->offsets[0] << std::endl;
//           }
//           std::cout << c->col_width << std::endl;
//           std::cout << c->row_width << std::endl;
//           std::cout << std::endl;
//       }
//   }

  // Execute codes
   DDT::executeCodelets(cl, config);

  // Clean up
  DDT::free(d);

  return 0;
}
