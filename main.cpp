#include "DDT.h"
#include "Executor.h"
#include "Input.h"
#include "Inspector.h"

#include <algorithm>
#include <iostream>

int main(int argc, char** argv) {
  // Parse program arguments
  auto config = DDT::parseInput(argc, argv);

  // Allocate memory and generate global object
  auto d = DDT::init(config);

  // Parse into run-time Codelets
  auto cl = new std::vector<DDT::Codelet*> [config.nThread]();
  DDT::inspectSerialTrace(d, cl, config);

  // Execute codes
  DDT::executeCodelets(cl, config);

  // Clean up
  DDT::free(d);

  return 0;
}
