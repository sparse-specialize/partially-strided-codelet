//
// Created by kazem on 7/13/21.
//
#include "SPMV_demo_utils.h"
#include "SerialSpMVExecutor.h"

#include "DDTDef.h"
#include <def.h>
#include <sparse_io.h>
#include <sparse_utilities.h>

#ifdef PAPI
#include "Profiler.h"
#endif


using namespace sparse_avx;
int main(int argc, char *argv[]) {
    auto config = DDT::parseInput(argc, argv);
    auto A = sym_lib::read_mtx(config.matrixPath);
    auto *sol = new double[A->n]();
    for (int i = 0; i < A->n; i++) { sol[i] = 1; }
    // std::fill_n(sol, A->n, 1);
    sym_lib::CSC *A_full = NULLPNTR;
    sym_lib::CSR *B = NULLPNTR, *L_csr = NULLPNTR;
    if (A->stype < 0) {
        A_full = sym_lib::make_full(A);
        B = sym_lib::csc_to_csr(A_full);
    } else {
        B = sym_lib::csc_to_csr(A);
    }
    L_csr = sym_lib::csc_to_csr(A);


#ifdef PROFILE
    /// Profiling
    int event_limit =1, instance_per_run = 5;
//    auto event_list = get_available_counter_codes();
#endif


    auto *sps = new SpMVSerial(B, A, NULLPNTR, "Baseline SpMV");
    auto spmv_baseline = sps->evaluate();
    double *sol_spmv = sps->solution();
    //std::copy(sol_spmv, sol_spmv+A->n, )

//    auto *spsp = new SpMVParallel(B, A, sol_spmv, "SpMV Parallel");
//    spsp->set_num_threads(config.nThread);
//    auto spmv_p = spsp->evaluate();
//
//    auto spmvp = new SpMVVec1Parallel(B,A,sol_spmv, "SpMVVec1_Parallel");
//    spmvp->set_num_threads(config.nThread);
//    auto spmv1pe = spmvp->evaluate();
//
//    auto *spmv1 = new SpMVVec1(B, A, sol_spmv, "SpMVVec1_4");
//    auto spmv1e = spmv1->evaluate();
//
//    auto *spmv1_8 = new SpMVVec1_8(B, A, sol_spmv, "SpMVVec1_8");
//    auto spmv1_8e = spmv1_8->evaluate();
//
//    auto *spmv1_16 = new SpMVVec1_16(B, A, sol_spmv, "SpMVVec1_16");
//    auto spmv1_16e = spmv1_16->evaluate();
//
//    auto *spmv2 = new SpMVVec2(B, A, sol_spmv, "SpMVVec2");
//    auto spmv2e = spmv2->evaluate();

//    auto *spmv_wathen120 = new SpMVWathen(B, A, sol_spmv, "SpMVVec2");
//    auto spmv_wathen120e = spmv_wathen120->evaluate();

#ifdef MKL
    auto mklspmvst = new SpMVMKL(1, B, A, sol_spmv, "MKL SPMV ST");
    auto mkl_exec_st = mklspmvst->evaluate();

    auto mklspmvmt = new SpMVMKL(config.nThread, B, A, sol_spmv, "MKL SPMV MT");
    mklspmvmt->set_num_threads(config.nThread);
    auto mkl_exec_mt = mklspmvmt->evaluate();
#endif

    int nThread = config.nThread;
    config.nThread = 1;
#undef PROFILE
#ifdef PROFILE
    auto matrixName = config.matrixPath;
    auto *ddt_profiler = new sym_lib::ProfilerWrapper<SpMVDDT>(event_list, event_limit, 1,B,A, sol_spmv, config, "SpMV DDT ST");
    ddt_profiler->profile(1);
    matrixName.erase(std::remove(matrixName.begin(), matrixName.end(), '\n'), matrixName.end());
    if (config.header) { std::cout << "MATRIX_NAME" << "," << "KERNEL_TYPE,"; ddt_profiler->print_headers(); }
    std::cout << matrixName << "," << "DDT,";
    ddt_profiler->print_counters();
    std::cout << "\n";

    auto *spmv_profiler = new sym_lib::ProfilerWrapper<SpMVSerial>(event_list, event_limit, 1,B,A, sol_spmv, "CSR SPMV BASELINE");
    spmv_profiler->profile(1);
    std::cout << matrixName << "," << "BASELINE,";
    spmv_profiler->print_counters();
    std::cout << "\n";

    auto *mkl_spmv_profiler = new sym_lib::ProfilerWrapper<SpMVMKL>(event_list, event_limit, 1,1,B,A, sol_spmv, "CSR SPMV BASELINE");
    mkl_spmv_profiler->profile(1);
    std::cout << matrixName << "," << "MKL,";
    mkl_spmv_profiler->print_counters();
    std::cout << "\n";

    delete ddt_profiler;
    delete spmv_profiler;
    delete mkl_spmv_profiler;

    exit(0);
#endif
    auto *ddtspmvst = new SpMVDDT(B, A, sol_spmv, config, "SpMV DDT ST");
    auto ddt_execst = ddtspmvst->evaluate();

    config.nThread = nThread;
    auto *ddtspmvmt = new SpMVDDT(B, A, sol_spmv, config, "SpMV DDT MT");
    ddtspmvmt->set_num_threads(config.nThread);
    auto ddt_execmt = ddtspmvmt->evaluate();

//    auto *tlspmv = new TightLoopSpMV(1, B, A, sol_spmv, "SpMV Tight Loop");
//    auto tlspmv_exec = tlspmv->evaluate();

    if (config.header || true) {
        std::cout << "Matrix,Threads, fsc_only, size_cutoff, col_threshold,";
        std::cout << "SpMV Base,"; //SpMV Parallel Base, SpMV Vec 1_4 Parallel, SpMV Vec 1_4, SpMV Vec 1_8, SpMV Vec 1_16, SpMV Vec 2,";
#ifdef MKL
//        std::cout << "SpMV MKL Serial Executor, SpMV MKL Parallel Executor,";
#endif
        std::cout <<
                     "SpMVDDT Serial Executor, SpMV DDT Parallel Executor";
        std::cout << "\n";
    }

    std::cout << config.matrixPath << "," << config.nThread << "," << DDT::fsc_only << "," << DDT::clt_width << "," << DDT::col_th << ","
               << spmv_baseline.elapsed_time << ",";//<< "," << tlspmv_exec.elapsed_time << ","; //<< spmv_p.elapsed_time << ",";
    // std::cout << spmv1pe.elapsed_time << ",";
    // std::cout << spmv1e.elapsed_time << "," << spmv2e.elapsed_time << ",";
    // std::cout << spmv1_8e.elapsed_time << "," << spmv1_16e.elapsed_time << ",";
#ifdef MKL
//     std::cout << mkl_exec_st.elapsed_time << "," << mkl_exec_mt.elapsed_time << ",";
#endif
    std::cout
              << ddt_execst.elapsed_time << ",";
                    std::cout << ddt_execmt.elapsed_time;
    std::cout << "\n";

    delete A;
    delete B;
    delete A_full;
    delete L_csr;
    delete[] sol;

//    delete spsp;
    delete sps;
//    delete ddtspmv;


    return 0;
}
