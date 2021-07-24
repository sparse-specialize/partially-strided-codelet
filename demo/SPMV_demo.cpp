//
// Created by kazem on 7/13/21.
//
#include "SPMV_demo_utils.h"

#include <def.h>
#include <sparse_io.h>
#include <sparse_utilities.h>

#ifdef PAPI
#include "papi.h"
#endif

//#define PROFILE

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
 std::vector<int> event_list = {PAPI_L1_DCM, PAPI_L1_TCM, PAPI_L2_DCM,
                                PAPI_L2_DCA,PAPI_L2_TCM,PAPI_L2_TCA,
                                PAPI_L3_TCM, PAPI_L3_TCA,
                                PAPI_TLB_DM, PAPI_TOT_CYC, PAPI_LST_INS,
                                PAPI_TOT_INS, PAPI_REF_CYC, PAPI_RES_STL,
                                PAPI_BR_INS, PAPI_BR_MSP};
#endif


    auto *sps = new SpMVSerial(B, A, NULLPNTR, "Baseline SpMV");
    auto spmv_baseline = sps->evaluate();
    double *sol_spmv = sps->solution();
    //std::copy(sol_spmv, sol_spmv+A->n, )

    auto *spsp = new SpMVParallel(B, A, sol_spmv, "SpMV Parallel");
    spsp->set_num_threads(config.nThread);
    auto spmv_p = spsp->evaluate();

#ifdef MKL
    auto mklspmvst = new SpMVMKL(1, B, A, sol_spmv, "MKL SPMV ST");
    auto mkl_exec_st = mklspmvst->evaluate();

    auto mklspmvmt = new SpMVMKL(config.nThread, B, A, sol_spmv, "MKL SPMV MT");
    auto mkl_exec_mt = mklspmvmt->evaluate();
#endif

    int nThread = config.nThread;
    config.nThread = 1;
#ifdef PROFILE
    auto *ddt_profiler = new sym_lib::ProfilerWrapper<SpMVDDT>(event_list, event_limit, 1,B,A, sol_spmv, config, "SpMV DDT ST");
    ddt_profiler->profile(1);
    ddt_profiler->print_headers();
    std::cout << "\n";
    ddt_profiler->print_counters();
    std::cout << "\n\n";
#endif
    auto *ddtspmvst = new SpMVDDT(B, A, sol_spmv, config, "SpMV DDT ST");
    auto ddt_execst = ddtspmvst->evaluate();

    config.nThread = nThread;
    auto *ddtspmv = new SpMVDDT(B, A, sol_spmv, config, "SpMV DDT MT");
    auto ddt_exec = ddtspmv->evaluate();

    if (config.header) {
        std::cout << "Matrix,Threads,";
        std::cout << "SpMV Base,SpMV Parallel Base,";
#ifdef MKL
        std::cout << "SpMV MKL Serial Executor, SpMV MKL Parallel Executor,";
#endif
        std::cout <<
                     "SpMVDDT Serial Executor,SpMV DDT Parallel Executor";
        std::cout << "\n";
    }

    std::cout << config.matrixPath << "," << config.nThread << ","
              << spmv_baseline.elapsed_time << "," << spmv_p.elapsed_time << ",";
#ifdef MKL
    std::cout << mkl_exec_st.elapsed_time << "," << mkl_exec_mt.elapsed_time << ",";
#endif
    std::cout
              << ddt_execst.elapsed_time << "," << ddt_exec.elapsed_time << ",";
    std::cout << "\n";

    delete A;
    delete B;
    delete A_full;
    delete L_csr;
    delete[] sol;

//    delete spsp;
    delete sps;
    delete ddtspmv;


    return 0;
}
