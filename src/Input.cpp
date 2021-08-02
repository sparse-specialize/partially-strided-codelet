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
#include "DDTDef.h"
#include "DDT.h"
#include "Input.h"

#include <stdlib.h>

#include <iostream>

#include "cxxopts.hpp"

namespace DDT {
  /**
   * @brief Parses commandline input for the program
   *
   * @param argc 
   * @param argv
   */
  Config parseInput(int argc, char** argv) {
    cxxopts::Options options("DDT", "Generates vectorized code from memory streams");

    options.add_options()
      ("h,help", "Prints help text")
      ("m,matrix", "Path to matrix market file.", cxxopts::value<std::string>())
      ("n,numerical_operation", "Numerical operation being performed on "
                                "matrix.", cxxopts::value<std::string>())
      ("s,storage_format", "Storage format for matrix", cxxopts::value<std::string>())
      ("t,threads", "Number of parallel threads", cxxopts::value<int>()->default_value("1"))
      ("c,coarsening", "coarsening levels", cxxopts::value<int>()->default_value("5"))
      ("p,packing", "bin-packing", cxxopts::value<int>()->default_value("1"))
      ("u,tuning", "Tuning enabled", cxxopts::value<int>()->default_value("0"))
      ("iteration_limit", "Max length of periodic iteration space to find", cxxopts::value<int>()->default_value("0"))
      ("fsc_only", "Max length of periodic iteration space to find", cxxopts::value<bool>()->default_value("false"))
      ("col_th", "Max length of periodic iteration space to find", cxxopts::value<int>()->default_value("0"))
      ("clt_width", "Max length of periodic iteration space to find", cxxopts::value<int>()->default_value("0"))
      ("d,header", "prints header or not.");

    auto result = options.parse(argc, argv);

    if (result.count("help")) {
      std::cout << options.help() << std::endl;
      exit(0);
    }

    if (!result.count("matrix")) {
      std::cout << "'matrix' is manditory argument. Use --help" << std::endl;
      exit(0);
    }
    if (!result.count("numerical_operation")) {
      std::cout << "'numerical_operation' is manditory argument. Use --help" << std::endl;
      exit(0);
    }
    if (!result.count("storage_format")) {
      std::cout << "'storage_format' is manditory argument. Use --help" << std::endl;
      exit(0);
    }

    auto matrixPath = result["matrix"].as<std::string>();
    auto operation = result["numerical_operation"].as<std::string>();
    auto storageFormat = result["storage_format"].as<std::string>();
    auto nThreads = result["threads"].as<int>();
    auto coarsening = result["coarsening"].as<int>();
    auto bpacking = result["packing"].as<int>();
    auto tuning_en = result["tuning"].as<int>();
    auto lim = result["iteration_limit"].as<int>();
    auto fsc_only = result["fsc_only"].as<bool>();
    auto col_th = result["col_th"].as<int>();
    auto clt_width = result["clt_width"].as<int>();

    assert(lim <= MAX_LIM);

    DDT::clt_width = clt_width;
    DDT::col_th = col_th;
    DDT::fsc_only = fsc_only;

    NumericalOperation op;
    if (operation == "SPMV") {
      op = OP_SPMV;
    } else if (operation == "SPTRS") {
      op = OP_SPTRS;
    } else {
      std::cout << "'numerical_operation' must be passed in as one of: ['SPMV', 'SPTRS']" << std::endl;
      exit(0);
    }

    DDT::StorageFormat sf;
    if (storageFormat == "CSR") {
      sf = DDT::CSR_SF;
    } else if (storageFormat == "CSC") {
      sf = DDT::CSC_SF;
    } else {
      std::cout << "'storage_format' must be passed in as one of: ['CSC', 'CSR']" << std::endl;
      exit(0);
    }
    int header = 0;
   if (result.count("header")) {
    header = 1;
   }

   return Config{ matrixPath, op, header, nThreads, sf, coarsening, bpacking,
    tuning_en, lim };
  }
}
