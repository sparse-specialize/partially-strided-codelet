//
// Created by kazem on 7/16/21.
//

#include <sparse_utilities.h>
#include <sparse_io.h>
#include <def.h>
#include "SPTRSV_demo_utils.h"

#ifdef METIS
#include "metis.h"
#include "metis_interface.h"
#endif
using namespace sparse_avx;
int main(int argc, char* argv[]){
 auto config = DDT::parseInput(argc, argv);

 //std::ifstream f(config.matrixPath);

 sym_lib::CSC *A;
 A = sym_lib::read_mtx(config.matrixPath);
 auto  *sol = new double[A->n]();
 int  *perm;
 std::fill_n(sol, A->n, 1);
 sym_lib::CSC *A_full=NULLPNTR;
 sym_lib::CSR *B=NULLPNTR, *L_csr=NULLPNTR;

#ifdef METIS
 //We only reorder L since dependency matters more in l-solve.
 //perm = new int[n]();
 perm = new int[A->n]();
 A_full = sym_lib::make_full(A);
 sym_lib::metis_perm_general(A_full, perm);
 sym_lib::CSC *Lt = transpose_symmetric(A, perm);
 sym_lib::CSC *L1_ord = transpose_symmetric(Lt, NULLPNTR);
 delete Lt;
 delete A_full;
 delete[]perm;
#endif


 auto *sps = new SpTRSVSerial(B, A, NULLPNTR, "Baseline SpMV");
 auto sptrsv_baseline =  sps->evaluate();
 double *sol_spmv = sps->solution();
 //std::copy(sol_spmv, sol_spmv+A->n, )


 auto *ddtsptrsv = new SpTRSVDDT(B, A, sol_spmv, config, "SpMV DDT");
 auto ddt_exec =  ddtsptrsv->evaluate();
 auto ddt_analysis = ddtsptrsv->get_analysis_bw();

 if (config.header){
  std::cout<<"Matrix,";
  std::cout<<"SpMV Base,SpMV DDT Executor,Prune Time,FOD "
             "Time,Mining Time,"
    ;
  std::cout<<"\n";
 }

 std::cout<<config.matrixPath <<","<<
          sptrsv_baseline.elapsed_time<<","<<
          ddt_exec.elapsed_time<<",";
 ddt_analysis.print_t_array();
 std::cout<<"\n";

 delete A;
 delete B;
 delete L_csr;
 delete []sol;

 delete sps;
 delete ddtsptrsv;


 return 0;
}
