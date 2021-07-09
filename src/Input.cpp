/*
 * =====================================================================================
 *
 *       Filename:  Input.cpp
 *
 *    Description:  Parses input for DDT 
 *
 *        Version:  1.0
 *        Created:  2021-07-08 12:06:45 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Zachary Cetinic, 
 *   Organization:  University of Toronto
 *
 * =====================================================================================
 */
#include "Input.h"

#include <iostream>
#include <stdlib.h>

#include "cxxopts.hpp"

namespace DDT {
    Config parseInput(int argc, char** argv) {
      cxxopts::Options options("DDT", "Generates vectorized code from memory streams");

      options.add_options()
        ("h,help", "Help")
        ("m,matrix", "Path to matrix market file.", cxxopts::value<std::string>());

      auto result = options.parse(argc, argv);

      if (result.count("help")) {
        std::cout << options.help() << std::endl;
        exit(0);
      }

      if (!result.count("matrix")) {
        std::cout << "Matrix is manditory argument. Use --help" << std::endl;
        exit(0);
      }

      auto matrixPath = result["matrix"].as<std::string>();

      return Config{ matrixPath };
    }
}
