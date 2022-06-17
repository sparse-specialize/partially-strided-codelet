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

 int bRows = B->n;
 int bCols = config.bMatrixCols;
 auto final_solution = new double[B->m*bCols]();

 auto *sps = new GEMMSpMM(B, A, bRows, bCols, "SpMM Serial");
 auto spmm_baseline = sps->evaluate();
 double *sol_spmm = sps->solution();
 std::copy(sol_spmm,sol_spmm+B->m*bCols,final_solution);
 delete sps;


 delete A;
 delete B;
 delete[] final_solution;
}
