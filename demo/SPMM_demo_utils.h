//
// Created by cetinicz on 2021-10-30.
//

#ifndef DDT_SPMM_DEMO_UTILS_H
#define DDT_SPMM_DEMO_UTILS_H

#include "FusionDemo.h"
#include <DDT.h>
#include <DDTCodelets.h>
#include <Executor.h>
namespace sparse_avx {

    void spmm_csr(double* Cx, double* Ax, double* Bx, int m, int* Ap, int* Ai, int cbb, int cbd) {
        for (int i = 0; i < m; ++i) {
            for (int k = 0; k < cbb; ++k) {
                for (int j = Ap[i]; j < Ap[i+1]; ++j) {
                    Cx[i+k*cbd] += Ax[j] * Bx[Ai[j]+k*cbd];
                }
            }
        }
    }

    void spmm_csr_parallel(double* Cx, double* Ax, double* Bx, int m, int* Ap, int* Ai, int cbb, int cbd) {
#pragma omp parallel for
        for (int i = 0; i < m; ++i) {
            for (int k = 0; k < cbb; ++k) {
                for (int j = Ap[i]; j < Ap[i+1]; ++j) {
                    Cx[i+k*cbd] += Ax[j] * Bx[Ai[j]+k*cbd];
                }
            }
        }
    }

    class SpMMSerial : public sym_lib::FusionDemo {
    protected:
        double* Bx, *Cx, *correct_cx_;
        int cbb, cbd;
        sym_lib::timing_measurement fused_code() override {
            sym_lib::timing_measurement t1;
            t1.start_timer();
            spmm_csr(Cx, L1_csr_->x, Bx, L1_csr_->m, L1_csr_->p, L1_csr_->i, cbb, cbd);
            t1.measure_elapsed_time();
            //copy_vector(0,n_,x_in_,x_);
            return t1;
        }

    public:
        SpMMSerial(sym_lib::CSR *L, sym_lib::CSC *L_csc, double *correct_cx, int cbb, int cbd,
                   std::string name)
                   : FusionDemo(L_csc->n, name) {
            L1_csr_ = L;
            L1_csc_ = L_csc;
            correct_cx_ = correct_cx;
            this->Bx = new double[cbb*cbd]();
            this->Cx = new double[cbb*cbd]();
            this->cbb = cbb;
            this->cbd = cbd;
        };

        ~SpMMSerial() override = default;
    };

    class SpMMParallel : public SpMMSerial {
    protected:
        sym_lib::timing_measurement fused_code() override {
            sym_lib::timing_measurement t1;
            t1.start_timer();
            spmm_csr_parallel(Cx, L1_csr_->x, Bx, L1_csr_->m, L1_csr_->p, L1_csr_->i, cbb, cbd);
            t1.measure_elapsed_time();
            //copy_vector(0,n_,x_in_,x_);
            return t1;
        }

    public:
        SpMMParallel(sym_lib::CSR *L, sym_lib::CSC *L_csc, double *correct_cx, int cbb, int cbd,
                     std::string name)
                     : SpMMSerial(L,L_csc,correct_cx,cbb,cbd,name) {
            L1_csr_ = L;
            L1_csc_ = L_csc;
            correct_cx_ = correct_cx;
        };

        ~SpMMParallel() override = default;
    };

    class SpMMDDT : public SpMMSerial {
    protected:
        std::vector<DDT::Codelet*>* cl;
        DDT::Config cfg;
        DDT::Args args;
        DDT::GlobalObject d;
        sym_lib::timing_measurement analysis_breakdown;

        void build_set() override {
            // Allocate memory and generate global object
            if (this->cl == nullptr) {
                this->cl = new std::vector<DDT::Codelet *>[cfg.nThread];
                analysis_breakdown.start_timer();
                d = DDT::init(this->L1_csr_, cfg);
                analysis_breakdown.start_timer();
                DDT::inspectSerialTrace(d, cl, cfg);
                analysis_breakdown.measure_elapsed_time();
            }
        }

        sym_lib::timing_measurement fused_code() override {
            sym_lib::timing_measurement t1;
            t1.start_timer();
            DDT::executeCodelets(cl, cfg, args);
            t1.measure_elapsed_time();
            //copy_vector(0,n_,x_in_,x_);
            return t1;
        }

    public:
        SpMMDDT(sym_lib::CSR *L, sym_lib::CSC *L_csc, double *correct_cx, DDT::Config& cfg, int cbb, int cbd,
                std::string name)
                : SpMMSerial(L,L_csc,correct_cx,cbb,cbd,name) {
            L1_csr_ = L;
            L1_csc_ = L_csc;
            correct_cx_ = correct_cx;
            this->cfg = cfg;
        };

        ~SpMMDDT() override = default;
    };
}

#endif//DDT_SPMM_DEMO_UTILS_H
