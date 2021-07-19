//
// Created by kazem on 7/16/21.
//

#include <iostream>
#include <sparse_io.h>
#include <test_utils.h>
#include <sparse_utilities.h>
#include <omp.h>
#include <lbc.h>
#include <fstream>

#include "SPTRSV_demo_utils.h"

#ifdef METIS
#include "metis_interface.h"
#endif

using namespace sparse_avx;
int main(int argc, char* argv[]){
 auto config = DDT::parseInput(argc, argv);
 int num_threads = 6;
 int coarsening_p = 2; int initial_cut=1;
 sym_lib::CSC *A;
 A = sym_lib::read_mtx(config.matrixPath);
 auto  *sol = new double[A->n]();
 int  *perm;
 std::fill_n(sol, A->n, 1);
 sym_lib::CSC *A_full=NULLPNTR;

#ifdef METISA
 //We only reorder L since dependency matters more in l-solve.
 A_full = sym_lib::make_full(A);
 sym_lib::metis_perm_general(A_full, perm);
 sym_lib::CSC *Lt = transpose_symmetric(A, perm);
 sym_lib::CSC *L1_ord = transpose_symmetric(Lt, NULLPNTR);
 delete Lt;
 delete A_full;
 delete[]perm;
#endif

 sym_lib::CSR *L1_ord_csr = sym_lib::csc_to_csr(A);


 auto *sps = new SpTRSVSerial(L1_ord_csr, A, NULLPNTR, "Baseline");
 auto sptrsv_baseline =  sps->evaluate();
 double *sol_sptrsv = sps->solution();
 //std::copy(sol_spmv, sol_spmv+A->n, )

 auto *spsp = new SpTRSVParallel(L1_ord_csr, A, sol_sptrsv, "Parallel "
                    "LBC",num_threads,coarsening_p, initial_cut);
 auto sptrsv_par =  spsp->evaluate();


 auto *ddtsptrsv = new SpTRSVDDT(L1_ord_csr, A, sol_sptrsv, config,
                                 "SpMV DDT", num_threads, coarsening_p, initial_cut);
 auto ddt_exec =  ddtsptrsv->evaluate();
 auto ddt_analysis = ddtsptrsv->get_analysis_bw();


 if (config.header || true) {
  std::cout<<"Matrix,";
  std::cout<<"SpMV Base,SpMV DDT Executor,Prune Time,FOD "
             "Time,Mining Time,";
  std::cout<<"\n";
 }

 std::cout<<config.matrixPath <<","<<
          sptrsv_baseline.elapsed_time<<",";
 std::cout<<sptrsv_par.elapsed_time<<",";
 std::cout <<         ddt_exec.elapsed_time<<",";
 ddt_analysis.print_t_array();
 std::cout<<"\n";

 delete A;
// delete L1_ord;
 delete L1_ord_csr;
 delete []sol;

 delete sps;
 delete spsp;
 delete ddtsptrsv;

 return 0;
}


