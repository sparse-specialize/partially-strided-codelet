//
// Created by cetinicz on 2021-10-30.
//

#include "SPMM_demo_utils.h"
#include <Input.h>
#include <def.h>
#include <sparse_io.h>
#include <sparse_utilities.h>

#define PROFILE

#ifdef PAPI
#include "Profiler.h"
#endif

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

#ifdef PROFILE
    /// Profiling
    int event_limit = 1, instance_per_run = 5;
    auto event_list = sym_lib::get_available_counter_codes();
#endif

    int bRows = B->n;
    int bCols = config.bMatrixCols;

    auto final_solution = new double[A_full->m*bCols]();

    
#ifdef PROFILE
    auto *sps = new sparse_avx::SpMMSerial(B, A_full, bRows, bCols, "SpMM Serial");
    auto spmm_baseline = sps->evaluate();
    double *sol_spmm = sps->solution();
    std::copy(sol_spmm,sol_spmm+A_full->m*bCols,final_solution);
    delete sps;

    auto *spmm_profiler = new sym_lib::ProfilerWrapper<sparse_avx::SpMMTiledParallel>(event_list, event_limit, 1,B, A_full, final_solution, bRows, bCols,config.mTileSize,config.nTileSize,"SpMM Tiled Parallel");
    spmm_profiler->d->set_num_threads(config.nThread);
    spmm_profiler->profile(config.nThread);
    if (config.header) {
      std::cout << "Matrix,mTileSize,nTileSize,Matrix B Columns,";spmm_profiler->print_headers();std::cout << '\n';
    }
    std::cout << config.matrixPath << "," << config.mTileSize << "," << config.nTileSize << "," << config.bMatrixCols << ",";
    spmm_profiler->print_counters();
    std::cout << "\n";
#else
    auto *sps = new sparse_avx::SpMMSerial(B, A_full, bRows, bCols, "SpMM Serial");
    auto spmm_baseline = sps->evaluate();
    double *sol_spmm = sps->solution();
    std::copy(sol_spmm,sol_spmm+A_full->m*bCols,final_solution);
    delete sps;

    auto *spsp = new sparse_avx::SpMMParallel(B, A_full, final_solution, bRows, bCols, "SpMM Parallel");
    spsp->set_num_threads(config.nThread);
    auto spmm_parallel_baseline = spsp->evaluate();
    auto spmm_parallel_baseline_elapsed = spmm_parallel_baseline.elapsed_time;
    delete spsp;

    auto *sp_tiled = new sparse_avx::SpMMTiledParallel(B, A_full, final_solution, bRows, bCols,config.mTileSize,config.nTileSize,"SpMM Tiled Parallel");
    sp_tiled->set_num_threads(config.nThread);
    auto spmm_tiled_parallel_baseline = sp_tiled->evaluate();
    auto spmm_tiled_parallel_baseline_elapsed = spmm_tiled_parallel_baseline.elapsed_time;
    delete sp_tiled;

    auto *spmkl = new sparse_avx::SpMMMKL(B, A_full, final_solution, bRows, bCols, "SpMM MKL");
    spmkl->set_num_threads(config.nThread);
    auto spmm_mkl_eval = spmkl->evaluate();
    auto spmm_mkl_eval_elapsed = spmm_mkl_eval.elapsed_time;
    delete spmkl;


    if (config.header) {
        std::cout << "Matrix,mTileSize,nTileSize,Matrix B Columns,SpMM "
                     "Baseline,SpMM Parallel Baseline,SpMM Tiled Parallel "
                     "Baseline,SpMM MKL"
                  << "\n";
    }
    std::cout << config.matrixPath << "," << config.mTileSize << "," << config.nTileSize << "," << config.bMatrixCols << "," <<spmm_baseline.elapsed_time << "," << spmm_parallel_baseline_elapsed << "," << spmm_tiled_parallel_baseline_elapsed << "," << spmm_mkl_eval_elapsed << "\n";
#endif

    delete A;
    delete A_full;
    delete B;
    delete[] final_solution;
}
