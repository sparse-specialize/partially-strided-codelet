//
// Created by kazem on 6/13/21.
//

#ifndef SPARSE_AVX512_DEMO_SPMVVEC_H
#define SPARSE_AVX512_DEMO_SPMVVEC_H

#include "FusionDemo.h"

#include "DDT.h"
#include "Executor.h"
#include "Input.h"
#include "Inspector.h"
#include "PatternMatching.h"

#include <iostream>
#ifdef MKL
#include <mkl.h>
#endif
#ifdef PAPI
#include "PAPIWrapper.h"
#include "Profiler.h"
#endif

namespace sparse_avx{

 ///// SPMV
 void spmv_csr(int n, const int *Ap, const int *Ai, const double *Ax,
               const double *x, double *y) {
  int i, j;
  for (i = 0; i < n; i++) {
   for (j = Ap[i]; j < Ap[i + 1]; j++) {
    y[i] += Ax[j] * x[Ai[j]];
   }
  }
 }

 void spmv_csr_parallel(int n, const int *Ap, const int *Ai, const double *Ax,
               const double *x, double *y, int nThreads) {
#pragma omp parallel for num_threads(nThreads)
  for (int i = 0; i < n; i++) {
   for (int j = Ap[i]; j < Ap[i + 1]; j++) {
    y[i] += Ax[j] * x[Ai[j]];
   }
  }
 }


    inline double hsum_double_avx(__m256d v) {
        __m128d vlow = _mm256_castpd256_pd128(v);
        __m128d vhigh = _mm256_extractf128_pd(v, 1);  // high 128
        vlow = _mm_add_pd(vlow, vhigh);      // reduce down to 128

        __m128d high64 = _mm_unpackhi_pd(vlow, vlow);
        return _mm_cvtsd_f64(_mm_add_sd(vlow, high64));  // reduce to scalar
    }

void spmv_vec1(int n, const int *Ap, const int *Ai, const double *Ax,
                           const double *x, double *y, int nThreads) {
        for (int i = 0; i < n; i++) {
            auto r0 = _mm256_setzero_pd();
            int j = Ap[i];
            for (; j < Ap[i+1] - 3; j += 4) {
                auto xv = _mm256_set_pd(x[Ai[j+3]],x[Ai[j+2]],x[Ai[j+1]],x[Ai[j]]);
                auto axv0 = _mm256_loadu_pd(Ax + j);
                r0 = _mm256_fmadd_pd(axv0, xv, r0);
            }

            // Compute tail
            double tail = 0.;
            for (; j < Ap[i+1]; j++) { tail += Ax[j] * x[Ai[j]]; }

            // H-Sum
            y[i] += tail + hsum_double_avx(r0);
        }
    }
    void spmv_vec1_16(int n, const int *Ap, const int *Ai, const double *Ax,
                   const double *x, double *y, int nThreads) {
        for (int i = 0; i < n; i++) {
            auto r0 = _mm256_setzero_pd();
            int j = Ap[i];
            for (; j < Ap[i+1] - 15; j += 16) {
                auto xv0 = _mm256_set_pd(x[Ai[j+3]],x[Ai[j+2]],x[Ai[j+1]],x[Ai[j]]);
                auto xv1 = _mm256_set_pd(x[Ai[j+7]],x[Ai[j+6]],x[Ai[j+5]],x[Ai[j+4]]);
                auto xv2 = _mm256_set_pd(x[Ai[j+11]],x[Ai[j+10]],x[Ai[j+9]],x[Ai[j+8]]);
                auto xv3 = _mm256_set_pd(x[Ai[j+15]],x[Ai[j+14]],x[Ai[j+13]],x[Ai[j+12]]);

                auto axv0 = _mm256_loadu_pd(Ax + j);
                auto axv1 = _mm256_loadu_pd(Ax + j + 4);
                auto axv2 = _mm256_loadu_pd(Ax + j + 8);
                auto axv3 = _mm256_loadu_pd(Ax + j + 12);

                r0 = _mm256_fmadd_pd(axv0, xv0, r0);
                r0 = _mm256_fmadd_pd(axv1, xv1, r0);
                r0 = _mm256_fmadd_pd(axv2, xv2, r0);
                r0 = _mm256_fmadd_pd(axv3, xv3, r0);
            }

            // Compute tail
            double tail = 0.;
            for (; j < Ap[i+1]; j++) { tail += Ax[j] * x[Ai[j]]; }

            // H-Sum
            y[i] += tail + hsum_double_avx(r0);
        }
    }
    void spmv_vec1_8(int n, const int *Ap, const int *Ai, const double *Ax,
                      const double *x, double *y, int nThreads) {
        for (int i = 0; i < n; i++) {
            auto r0 = _mm256_setzero_pd();
            int j = Ap[i];
            for (; j < Ap[i+1] - 7; j += 8) {
                auto xv0 = _mm256_set_pd(x[Ai[j+3]],x[Ai[j+2]],x[Ai[j+1]],x[Ai[j]]);
                auto xv1 = _mm256_set_pd(x[Ai[j+8]],x[Ai[j+7]],x[Ai[j+6]],x[Ai[j+5]]);

                auto axv0 = _mm256_loadu_pd(Ax + j);
                auto axv1 = _mm256_loadu_pd(Ax + j + 4);

                r0 = _mm256_fmadd_pd(axv0, xv0, r0);
                r0 = _mm256_fmadd_pd(axv1, xv1, r0);
            }

            // Compute tail
            double tail = 0.;
            for (; j < Ap[i+1]; j++) { tail += Ax[j] * x[Ai[j]]; }

            // H-Sum
            y[i] += tail + hsum_double_avx(r0);
        }
    }
    void spmv_vec2(int n, const int *Ap, const int *Ai, const double *Ax,
                   const double *x, double *y, int nThreads) {
        int i = 0;
        for (; i < n-1; i += 2) {
            auto r0 = _mm256_setzero_pd();
            auto r1 = _mm256_setzero_pd();

            int j0 = Ap[i];
            int j1 = Ap[i+1];
            for (; j0 < Ap[i+1] - 3 && j1 < Ap[i+2] - 3; j0 += 4, j1 += 4) {
                auto xv0 = _mm256_set_pd(x[Ai[j0+3]],x[Ai[j0+2]],x[Ai[j0+1]],x[Ai[j0]]);
                auto xv1 = _mm256_set_pd(x[Ai[j1+3]],x[Ai[j1+2]],x[Ai[j1+1]],x[Ai[j1]]);

                auto axv0 = _mm256_loadu_pd(Ax + j0);
                auto axv1 = _mm256_loadu_pd(Ax + j1);

                r0 = _mm256_fmadd_pd(axv0, xv0, r0);
                r1 = _mm256_fmadd_pd(axv1, xv1, r1);
            }

            // Compute tail
            __m128d tail = _mm_loadu_pd(y+i);
            for (; j0 < Ap[i+1]; j0++) {
                tail[0] += Ax[j0] * x[Ai[j0]];
            }
            for (; j1 < Ap[i+2]; j1++) {
                tail[1] += Ax[j1] * x[Ai[j1]];
            }

            // H-Sum
            auto h0 = _mm256_hadd_pd(r0, r1);
            __m128d vlow = _mm256_castpd256_pd128(h0);
            __m128d vhigh = _mm256_extractf128_pd(h0, 1);// high 128
            vlow = _mm_add_pd(vlow, vhigh);              // reduce down to 128
            vlow = _mm_add_pd(vlow, tail);
            // Store
            _mm_storeu_pd(y + i, vlow);
        }
        for (; i < n; i++) {
            auto r0 = _mm256_setzero_pd();
            int j = Ap[i];
            for (; j < Ap[i+1] - 3; j += 4) {
                auto xv = _mm256_set_pd(x[Ai[j+3]],x[Ai[j+2]],x[Ai[j+1]],x[Ai[j]]);
                auto axv0 = _mm256_loadu_pd(Ax + j);
                r0 = _mm256_fmadd_pd(axv0, xv, r0);
            }

            // Compute tail
            double tail = 0.;
            for (; j < Ap[i+1]; j++) { tail += Ax[j] * x[Ai[j]]; }

            // H-Sum
            y[i] += tail + hsum_double_avx(r0);
        }
    }

 class SpMVSerial : public sym_lib::FusionDemo {
 protected:
  void setting_up() override {
   std::fill_n(x_,n_,0.0);
   std::fill_n(x_in_,n_,1.0);
  }

  sym_lib::timing_measurement fused_code() override {
   sym_lib::timing_measurement t1;
   t1.start_timer();
   spmv_csr(n_, L1_csr_->p, L1_csr_->i, L1_csr_->x, x_in_, x_);
   t1.measure_elapsed_time();
   //copy_vector(0,n_,x_in_,x_);
   return t1;
  }

 public:
  SpMVSerial(sym_lib::CSR *L, sym_lib::CSC *L_csc,
             double *correct_x,
             std::string name) :
    FusionDemo(L_csc->n, name) {
   L1_csr_ = L;
   L1_csc_ = L_csc;
   correct_x_ = correct_x;
   this->pw_ = nullptr;
  };

  ~SpMVSerial() override = default;
 };

 class SpMVParallel : public SpMVSerial {
 protected:
  sym_lib::timing_measurement fused_code() override {
   sym_lib::timing_measurement t1;
   t1.start_timer();
   spmv_csr_parallel(n_, L1_csr_->p, L1_csr_->i, L1_csr_->x, x_in_, x_, num_threads_);
   t1.measure_elapsed_time();
   //copy_vector(0,n_,x_in_,x_);
   return t1;
  }

 public:
  SpMVParallel(sym_lib::CSR *L, sym_lib::CSC *L_csc,
           double *correct_x,
           std::string name) :
    SpMVSerial(L, L_csc, correct_x, name) {
   L1_csr_ = L;
   L1_csc_ = L_csc;
   correct_x_ = correct_x;
  };
 };



    class SpMVVec1 : public SpMVSerial {
    protected:
        sym_lib::timing_measurement fused_code() override {
            sym_lib::timing_measurement t1;
            t1.start_timer();
            spmv_vec1(n_, L1_csr_->p, L1_csr_->i, L1_csr_->x, x_in_, x_, num_threads_);
            t1.measure_elapsed_time();
            //copy_vector(0,n_,x_in_,x_);
            return t1;
        }

    public:
        SpMVVec1(sym_lib::CSR *L, sym_lib::CSC *L_csc,
                     double *correct_x,
                     std::string name) :
                SpMVSerial(L, L_csc, correct_x, name) {
            L1_csr_ = L;
            L1_csc_ = L_csc;
            correct_x_ = correct_x;
        };
    };

    class SpMVVec1_8 : public SpMVSerial {
    protected:
        sym_lib::timing_measurement fused_code() override {
            sym_lib::timing_measurement t1;
            t1.start_timer();
            spmv_vec1_8(n_, L1_csr_->p, L1_csr_->i, L1_csr_->x, x_in_, x_, num_threads_);
            t1.measure_elapsed_time();
            //copy_vector(0,n_,x_in_,x_);
            return t1;
        }

    public:
        SpMVVec1_8(sym_lib::CSR *L, sym_lib::CSC *L_csc,
                 double *correct_x,
                 std::string name) :
                SpMVSerial(L, L_csc, correct_x, name) {
            L1_csr_ = L;
            L1_csc_ = L_csc;
            correct_x_ = correct_x;
        };
    };

    class SpMVVec1_16 : public SpMVSerial {
    protected:
        sym_lib::timing_measurement fused_code() override {
            sym_lib::timing_measurement t1;
            t1.start_timer();
            spmv_vec1_16(n_, L1_csr_->p, L1_csr_->i, L1_csr_->x, x_in_, x_, num_threads_);
            t1.measure_elapsed_time();
            //copy_vector(0,n_,x_in_,x_);
            return t1;
        }

    public:
        SpMVVec1_16(sym_lib::CSR *L, sym_lib::CSC *L_csc,
                 double *correct_x,
                 std::string name) :
                SpMVSerial(L, L_csc, correct_x, name) {
            L1_csr_ = L;
            L1_csc_ = L_csc;
            correct_x_ = correct_x;
        };
    };

    class SpMVVec2 : public SpMVSerial {
    protected:
        sym_lib::timing_measurement fused_code() override {
            sym_lib::timing_measurement t1;
            t1.start_timer();
            spmv_vec2(n_, L1_csr_->p, L1_csr_->i, L1_csr_->x, x_in_, x_, num_threads_);
            t1.measure_elapsed_time();
            //copy_vector(0,n_,x_in_,x_);
            return t1;
        }

    public:
        SpMVVec2(sym_lib::CSR *L, sym_lib::CSC *L_csc,
                 double *correct_x,
                 std::string name) :
                SpMVSerial(L, L_csc, correct_x, name) {
            L1_csr_ = L;
            L1_csc_ = L_csc;
            correct_x_ = correct_x;
        };
    };

#ifdef MKL
 class SpMVMKL : public SpMVSerial {
     sparse_matrix_t m;
     matrix_descr d;
     MKL_INT* LLI;
     int num_threads;

     void build_set() override {
         if (this->LLI != nullptr)
             return;
         d.type = SPARSE_MATRIX_TYPE_GENERAL;
//         d.diag = SPARSE_DIAG_NON_UNIT;
//         d.mode = SPARSE_FILL_MODE_FULL;

         MKL_INT expected_calls = 5;

         LLI = new MKL_INT[this->L1_csr_->m+1]();
         for (int l = 0; l < this->L1_csr_->m+1; ++l) {
             LLI[l] = this->L1_csr_->p[l];
         }

         auto  stat = mkl_sparse_d_create_csr(&m, SPARSE_INDEX_BASE_ZERO, this->L1_csr_->m, this->L1_csr_->n,
                                 LLI, LLI+ 1, this->L1_csr_->i, this->L1_csr_->x);
         mkl_sparse_set_mv_hint(m, SPARSE_OPERATION_NON_TRANSPOSE, d, expected_calls);
         mkl_sparse_set_memory_hint(m, SPARSE_MEMORY_AGGRESSIVE);

         mkl_set_num_threads(num_threads);
         mkl_set_num_threads_local(num_threads);
     }
     sym_lib::timing_measurement fused_code() override {
         sym_lib::timing_measurement t1;
         t1.start_timer();
         mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1, m, this->d, this->x_in_, 1, this->x_);
         t1.measure_elapsed_time();
         return t1;
     }

 public:
     SpMVMKL(int nThreads, sym_lib::CSR *L, sym_lib::CSC *L_csc,
     double *correct_x,
     std::string name) :
     SpMVSerial(L, L_csc, correct_x, name), num_threads(nThreads), LLI(nullptr) {}

     ~SpMVMKL() override {
         delete[] LLI;
     }
 };
#endif

 class SpMVDDT : public SpMVSerial {
 protected:
  DDT::Config config;
  std::vector<DDT::Codelet*>* cl;
  DDT::GlobalObject d;
  sym_lib::timing_measurement analysis_breakdown;

  void build_set() override {
   // Allocate memory and generate global object
   if (this->cl == nullptr) {
       this->cl = new std::vector<DDT::Codelet *>[config.nThread];
       d = DDT::init(config);
       analysis_breakdown.start_timer();
       DDT::inspectSerialTrace(d, cl, config);
       analysis_breakdown.measure_elapsed_time();
   }
  }

  sym_lib::timing_measurement fused_code() override {
   sym_lib::timing_measurement t1;
   DDT::Args args; args.x = x_in_; args.y = x_;
   args.r = L1_csr_->m; args.Lp=L1_csr_->p; args.Li=L1_csr_->i;
   args.Lx = L1_csr_->x;
   t1.start_timer();
   // Execute codes
   DDT::executeCodelets(cl, config, args);
   t1.measure_elapsed_time();
   //copy_vector(0,n_,x_in_,x_);
   return t1;
  }

 public:
  SpMVDDT(sym_lib::CSR *L, sym_lib::CSC *L_csc,
           double *correct_x, DDT::Config &conf,
           std::string name) :
    SpMVSerial(L, L_csc, correct_x, name), config(conf), cl(nullptr) {
   L1_csr_ = L;
   L1_csc_ = L_csc;
   correct_x_ = correct_x;
  };

  sym_lib::timing_measurement get_analysis_bw(){return analysis_breakdown;}

  ~SpMVDDT(){
   DDT::free(d);
   delete[] cl;
  }
 };



}


#endif //SPARSE_AVX512_DEMO_SPMVVEC_H
