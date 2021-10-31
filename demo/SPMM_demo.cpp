//
// Created by cetinicz on 2021-10-30.
//

#include "SPMM_demo_utils.h"
#include <Input.h>
#include <def.h>
#include <sparse_io.h>
#include <sparse_utilities.h>


int main(int argc, char *argv[]) {
    auto config = DDT::parseInput(argc, argv);
    auto A = sym_lib::read_mtx(config.matrixPath);
    sym_lib::CSC *A_full = NULLPNTR;
    sym_lib::CSR *B = NULLPNTR, *L_csr = NULLPNTR;
    if (A->stype < 0) {
        A_full = sym_lib::make_full(A);
        B = sym_lib::csc_to_csr(A_full);
    } else {
        A_full = A;
        B = sym_lib::csc_to_csr(A);
    }

    int cbb = B->m;
    int cbd = B->n;
    auto correct_cx = new double[cbb*cbd]();

    auto *sps = new sparse_avx::SpMMSerial(B, A_full, correct_cx, cbb, cbd, "SpMM Serial");
    auto spmm_baseline = sps->evaluate();
    double *sol_spmm = sps->solution();

    std::cout << "nice" << std::endl;

    auto *spmm_ddt = new sparse_avx::SpMMDDT(B, A_full, sol_spmm, config, cbb, cbd,"SpMM DDT");
    auto spmm_ddt_eval = spmm_ddt->evaluate();

    std::cout << "SpMM Baseline, SpMM Parallel Baseline, SpMM DDT" << "\n";
    std::cout << spmm_baseline.elapsed_time << "," << spmm_ddt_eval.elapsed_time << "\n";
}