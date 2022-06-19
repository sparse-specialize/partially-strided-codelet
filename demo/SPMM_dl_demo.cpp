//
// Created by kazem on 6/17/22.
//
#include "SPMM_dl_demo_utils.h"
#include <Input.h>
#include <def.h>
#include <sparse_io.h>
#include <sparse_utilities.h>
#include <SparseMatrixIO.h>


#ifdef PAPI
#include "Profiler.h"
#endif

template<class type>
sym_lib::CSR *convert_dcsr_scsr(type A){
 int r = A.r;
 int c = A.c;
 int nnz = A.nz;
 sym_lib::CSR *scsr = new sym_lib::CSR(r,c,nnz);
 sym_lib::copy_vector(0,r, A.Lp, scsr->p);
 sym_lib::copy_vector(0, nnz, A.Li, scsr->i);
 sym_lib::copy_vector(0, nnz, A.Lx, scsr->x);
 scsr->stype = A.stype;
 return scsr;
}

int main(int argc, char *argv[]) {
 auto config = DDT::parseInput(argc, argv);
 //auto A = sym_lib::read_mtx(config.matrixPath);
 auto As = DDT::readSparseMatrix<DDT::CSR>(config.matrixPath);
 auto B = convert_dcsr_scsr<DDT::Matrix>(As);
 sym_lib::CSC *A = new sym_lib::CSC(B->m, B->n, B->nnz);
#ifdef PROFILE
 /// Profiling
    int event_limit = 1, instance_per_run = 5;
    auto event_list = sym_lib::get_available_counter_codes();
#endif
 int bCols = config.bMatrixCols;
 auto final_solution = new float[B->n*bCols]();
 int num_thread = config.nThread;
 MKL_Set_Num_Threads(num_thread);
 MKL_Domain_Set_Num_Threads(num_thread, MKL_DOMAIN_BLAS);

 spmm_config sc_in{}; //sc_in.m_tile = config.mTileSize;

 auto *sps = new GEMMSpMM(B, A, num_thread, bCols, NULLPNTR, sc_in, "SpMM "
                                                                    "Serial");
 auto mkl_gemm = sps->evaluate();
 auto *sol_gemm = sps->get_Cx();
 std::copy(sol_gemm,sol_gemm+B->n*bCols,final_solution);
 delete sps;


 auto *gemmt = new GEMMSpMMTuned(B, A, num_thread, bCols, final_solution, sc_in,
                                 "SpMM Serial");
 auto mkl_gemm_t1 = gemmt->evaluate();
 delete gemmt;



 if (config.header) {
  std::cout << "Matrix,nRows,nCols,NNZ,mTileSize,nTileSize,"
               "Number of Threads,B Cols,MKL GEMM,GEMM Tuned1,";
  std::cout << "\n";
 }
 std::cout << config.matrixPath << "," << B->m << "," << B->n
           << "," << B->nnz << "," << config.mTileSize
           << "," << config.nTileSize << "," <<config.nThread  << "," << config
           .bMatrixCols << "," <<mkl_gemm.elapsed_time << ","<<
           mkl_gemm_t1.elapsed_time<<",";
 std::cout << "\n";

 delete A;
 delete B;
 delete[] final_solution;
}
