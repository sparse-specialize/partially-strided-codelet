//
// Created by kazem on 7/13/21.
//

#include "SPMV_demo_utils.h"
#include "smp/io.h"

using namespace sparse_avx;
int main(int argc, char* argv[]){
 auto config = DDT::parseInput(argc, argv);

 std::ifstream f(config.matrixPath);

 format::CSC *A;
 format::read_mtx_csc_real(f, A);
 auto  *sol = new double[A->n]();
 std::fill_n(sol, A->n, 1);
 CSC *A_full=NULLPNTR;
 format::CSR *B=NULLPNTR, *L_csr=NULLPNTR;
 if(A->stype < 0){
//  A_full = sym_lib::make_full(A);
  B = sym_lib::csc_to_csr(A);
 } else{
  B = sym_lib::csc_to_csr(A);
 }
 L_csr = sym_lib::csc_to_csr(A);



 auto *sps = new SpMVSerial(B, A, NULLPNTR, "Baseline SpMV");
 auto spmv_baseline =  sps->evaluate();
 double *sol_spmv = sps->solution();
 //std::copy(sol_spmv, sol_spmv+A->n, )

 auto *spsp = new SpMVParallel(B, A, sol_spmv, "SpMV Parallel");
 auto spmv_p =  spsp->evaluate();

 auto *ddtspmv = new SpMVDDT(B, A, sol_spmv, config, "SpMV DDT");
 auto ddt_exec =  ddtspmv->evaluate();
 auto ddt_analysis = ddtspmv->get_analysis_bw();

 if (true){
  std::cout<<"Matrix,";
  std::cout<<"SpMV Base,SpMV Parallel Base,SpMV DDT Executor,Prune Time,FOD "
             "Time,Mining Time,"
             ;
  std::cout<<"\n";
 }

 std::cout<<config.matrixPath <<","<<
          spmv_baseline.elapsed_time<<","<<
          spmv_p.elapsed_time<<","<<
          ddt_exec.elapsed_time<<",";
 ddt_analysis.print_t_array();

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
