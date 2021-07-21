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
 int num_threads = 8;
 int coarsening_p = 4; int initial_cut=1;
 sym_lib::CSC *A;
 A = sym_lib::read_mtx(config.matrixPath);
 auto  *sol = new double[A->n]();
 int  *perm;
 std::fill_n(sol, A->n, 1);
 sym_lib::CSC *A_full=NULLPNTR;

#ifdef METIS
    //We only reorder L since dependency matters more in l-solve.
    A_full = sym_lib::make_full(A);
    sym_lib::metis_perm_general(A_full, perm);
    sym_lib::CSC *Lt = transpose_symmetric(A, perm);
    sym_lib::CSC *L1_ord = transpose_symmetric(Lt, NULLPNTR);
    delete Lt;
    delete A_full;
    delete[]perm;
#endif
   sym_lib::CSR *L1_ord_csr = sym_lib::csc_to_csr(L1_ord);


 auto *sps = new SpTRSVSerial(L1_ord_csr, L1_ord, NULLPNTR, "Baseline");
 auto sptrsv_baseline =  sps->evaluate();
 double *sol_sptrsv = sps->solution();
 //std::copy(sol_spmv, sol_spmv+A->n, )

 auto *spsp = new SpTRSVParallel(L1_ord_csr, L1_ord, sol_sptrsv, "Parallel "
                    "LBC",num_threads,coarsening_p, initial_cut);
 auto sptrsv_par =  spsp->evaluate();

    auto *spspv2 = new SpTRSVParallelVec2(L1_ord_csr, L1_ord, sol_sptrsv, "Parallel Vec2"
                                                                    "LBC",num_threads,coarsening_p, initial_cut);
    auto sptrsv_parv2 =  spspv2->evaluate();


 auto *ddtsptrsv = new SpTRSVDDT(L1_ord_csr, L1_ord, sol_sptrsv, config,
                                 "SpMV DDT", num_threads, coarsening_p, initial_cut);
 auto ddt_exec =  ddtsptrsv->evaluate();
 auto ddt_analysis = ddtsptrsv->get_analysis_bw();

 auto* sptrsv_vec1 = new SpTRSVSerialVec1(L1_ord_csr, L1_ord, NULLPNTR, "SpTRSV_Vec1");
 auto sptrsv_vec1_exec = sptrsv_vec1->evaluate();

    auto* sptrsv_vec2 = new SpTRSVSerialVec2(L1_ord_csr, L1_ord, NULLPNTR, "SpTRSV_Vec2");
    auto sptrsv_vec2_exec = sptrsv_vec2->evaluate();


 if (config.header || true) {
  std::cout<<"Matrix,";
  std::cout<<"SpTRSV Base,SpTRSV Vec1, SpTRSV Vec2, SpTRSV Parallel,SpTRSV Vec2 Parallel, SpTRSV DDT Executor,Prune Time,FOD "
             "Time,Mining Time,";
  std::cout<<"\n";
 }

 std::cout<<config.matrixPath <<","<<
          sptrsv_baseline.elapsed_time<<"," << sptrsv_vec1_exec.elapsed_time << ","
                                                        << sptrsv_vec2_exec.elapsed_time << ",";
 std::cout<<sptrsv_par.elapsed_time<<",";
 std::cout << sptrsv_parv2.elapsed_time <<",";
// std::cout <<         ddt_exec.elapsed_time<<",";
// ddt_analysis.print_t_array();
 std::cout<<"\n";

 delete A;
 delete L1_ord;
 delete L1_ord_csr;
 delete []sol;

 delete sps;
 delete spsp;
// delete ddtsptrsv;

 return 0;
}


