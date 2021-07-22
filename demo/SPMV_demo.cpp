//
// Created by kazem on 7/13/21.
//
#include <sparse_utilities.h>
#include <sparse_io.h>
#include <def.h>
#include "SPMV_demo_utils.h"

using namespace sparse_avx;
int main(int argc, char* argv[]){
 auto config = DDT::parseInput(argc, argv);

 //std::ifstream f(config.matrixPath);

 sym_lib::CSC *A;
 A = sym_lib::read_mtx(config.matrixPath);
 auto  *sol = new double[A->n]();
 for (int i = 0; i < A->n; i++) {
     sol[i] = 1;
 }
// std::fill_n(sol, A->n, 1);
 sym_lib::CSC *A_full=NULLPNTR;
 sym_lib::CSR *B=NULLPNTR, *L_csr=NULLPNTR;
 if (A->stype < 0) {
  A_full = sym_lib::make_full(A);
  B = sym_lib::csc_to_csr(A_full);
 } else{
  B = sym_lib::csc_to_csr(A);
 }
 L_csr = sym_lib::csc_to_csr(A);



 auto *sps = new SpMVSerial(B, A, NULLPNTR, "Baseline SpMV");
 auto spmv_baseline =  sps->evaluate();
 double *sol_spmv = sps->solution();
 //std::copy(sol_spmv, sol_spmv+A->n, )

 auto *spsp = new SpMVParallel(B, A, sol_spmv, "SpMV Parallel");
 spsp->set_num_threads(config.nThread);
 auto spmv_p =  spsp->evaluate();

 int nThread = config.nThread;
 config.nThread = 1;
 auto *ddtspmvst = new SpMVDDT(B, A, sol_spmv, config, "SpMV DDT ST");
 auto ddt_execst =  ddtspmvst->evaluate();

    config.nThread = nThread;
    auto *ddtspmv = new SpMVDDT(B, A, sol_spmv, config, "SpMV DDT MT");
    auto ddt_exec =  ddtspmv->evaluate();

 if (config.header){
  std::cout<<"Matrix,Threads,";
  std::cout<<"SpMV Base,SpMV Parallel Base,SpMV DDT Serial Executor, SpMV DDT Parallel Executor";
  std::cout<<"\n";
 }

 std::cout<<config.matrixPath <<","<<config.nThread<<","<<
          spmv_baseline.elapsed_time<<","<<
          spmv_p.elapsed_time<<","<<
          ddt_execst.elapsed_time <<","<<
          ddt_exec.elapsed_time<<",";
 std::cout<<"\n";

 delete A;
 delete B;
 delete A_full;
 delete L_csr;
 delete []sol;

 delete spsp;
 delete sps;
 delete ddtspmv;


 return 0;
}
