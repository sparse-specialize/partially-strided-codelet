//
// Created by cetinicz on 2021-10-30.
//

#include "SPMM_demo_utils.h"
#include <Input.h>
#include <def.h>
#include <sparse_io.h>
#include <sparse_utilities.h>

#ifdef PAPI
#include "Profiler.h"
#endif

int main(int argc, char *argv[]) {
    auto config = DDT::parseInput(argc, argv);
    auto A = sym_lib::read_mtx(config.matrixPath);
    sym_lib::CSR *B = sym_lib::csc_to_csr(A);

#ifdef PROFILE
    /// Profiling
    int event_limit = 1, instance_per_run = 5;
    auto event_list = sym_lib::get_available_counter_codes();
#endif

    int bRows = B->n;
    int bCols = config.bMatrixCols;

    auto final_solution = new double[A->m*bCols]();

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

    auto *sps = new sparse_avx::SpMMSerial(B, A, bRows, bCols, "SpMM Serial");
    auto spmm_baseline = sps->evaluate();
    double *sol_spmm = sps->solution();
    std::copy(sol_spmm,sol_spmm+A->m*bCols,final_solution);
    delete sps;

    auto *spsp = new sparse_avx::SpMMParallel(B, A, final_solution, bRows, bCols, "SpMM Parallel");
    spsp->set_num_threads(config.nThread);
    auto spmm_parallel_baseline = spsp->evaluate();
    auto spmm_parallel_baseline_elapsed = spmm_parallel_baseline.elapsed_time;
    delete spsp;

#ifdef PERMUTED
    auto *spPermuted = new sparse_avx::SpMMPermutedParallel(B, A_full, final_solution, bRows, bCols, "SpMM Permuted Parallel");
    spPermuted->set_num_threads(config.nThread);
    auto spmm_permuted_parallel_baseline = spPermuted->evaluate();
    auto spmm_permuted_parallel_baseline_elapsed = spmm_permuted_parallel_baseline.elapsed_time;
    delete spPermuted;
#endif

    auto *sp_tiled = new sparse_avx::SpMMTiledParallel(B, A, final_solution, bRows, bCols,config.mTileSize,config.nTileSize,"SpMM Tiled Parallel");
    sp_tiled->set_num_threads(config.nThread);
    auto spmm_tiled_parallel_baseline = sp_tiled->evaluate();
    auto spmm_tiled_parallel_baseline_elapsed = spmm_tiled_parallel_baseline.elapsed_time;
    delete sp_tiled;

    auto *spmkl = new sparse_avx::SpMMMKL(B, A, final_solution, bRows, bCols, "SpMM MKL");
    spmkl->set_num_threads(config.nThread);
    auto spmm_mkl_eval = spmkl->evaluate();
    auto spmm_mkl_eval_elapsed = spmm_mkl_eval.elapsed_time;
    delete spmkl;

    auto *spddt = new sparse_avx::SpMMDDT(B, A, final_solution, config, bRows, bCols, "SpMM DDT");
    spddt->set_num_threads(config.nThread);
    auto spmm_ddt_eval = spddt->evaluate();
    auto spmm_ddt_eval_elapsed = spmm_ddt_eval.elapsed_time;
    delete spddt;


    if (config.header) {
      std::cout << "Matrix,mTileSize,nTileSize,Matrix B Columns,SpMM "
        "Baseline,SpMM Parallel Baseline,SpMM Tiled Parallel "
        "Baseline,SpMM MKL,SpMM DDT";

#ifdef PERMUTED
      std::cout << ",SpMM Permuted Parallel"
#endif
        std::cout << "\n";
    }
    std::cout << config.matrixPath << "," << config.mTileSize << "," << config.nTileSize << "," << config.bMatrixCols << "," <<spmm_baseline.elapsed_time << "," << spmm_parallel_baseline_elapsed << "," << spmm_tiled_parallel_baseline_elapsed << "," << spmm_mkl_eval_elapsed << "," << spmm_ddt_eval_elapsed;

#ifdef PERMUTED
    std::cout << "," << spmm_permuted_parallel_baseline_elapsed;
#endif
    std::cout << "\n";
#endif

    delete A;
    delete B;
    delete[] final_solution;
}
