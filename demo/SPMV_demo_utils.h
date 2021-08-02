//
// Created by kazem on 6/13/21.
//

#ifndef SPARSE_AVX512_DEMO_SPMVVEC_H
#define SPARSE_AVX512_DEMO_SPMVVEC_H

#include "FusionDemo.h"
#include "SerialSpMVExecutor.h"

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

    inline void hsum_double_avx_avx(__m256d v0, __m256d v1, __m128d v2, double* y, int i) {
        auto h0 = _mm256_hadd_pd(v0,v1);
        __m128d vlow = _mm256_castpd256_pd128(h0);
        __m128d vhigh = _mm256_extractf128_pd(h0, 1);  // high 128
        vlow = _mm_add_pd(vlow, vhigh);
        vlow = _mm_add_pd(vlow, v2);
        _mm_storeu_pd(y+i, vlow);
    }

    inline double hsum_double_avx(__m256d v) {
        __m128d vlow = _mm256_castpd256_pd128(v);
        __m128d vhigh = _mm256_extractf128_pd(v, 1);  // high 128
        vlow = _mm_add_pd(vlow, vhigh);      // reduce down to 128

        __m128d high64 = _mm_unpackhi_pd(vlow, vlow);
        return _mm_cvtsd_f64(_mm_add_sd(vlow, high64));  // reduce to scalar
    }

    inline double hsum_add_double_avx(__m256d v, __m128d a) {
        __m128d vlow = _mm256_castpd256_pd128(v);
        __m128d vhigh = _mm256_extractf128_pd(v, 1);  // high 128
        vlow = _mm_add_pd(vlow, vhigh);      // reduce down to 128
        vlow = _mm_add_pd(vlow, a);

        __m128d high64 = _mm_unpackhi_pd(vlow, vlow);
        return _mm_cvtsd_f64(_mm_add_sd(vlow, high64));  // reduce to scalar
    }

    __m256i MASK_3 = _mm256_set_epi64x(0,-1,-1,-1);
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
    void spmv_vec1_parallel(int n, const int *Ap, const int *Ai, const double *Ax,
                   const double *x, double *y, int nThreads) {
#pragma omp parallel for num_threads(nThreads)
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

    inline double wathen3_2_3vec_1d(const int* Ap, const int* Ai, const double* Ax, const double* x, int i) {
        int j = Ap[i];
        int o0 = Ai[j], o1 = Ai[j+3], o2 = Ai[j+5];

        auto r0 = _mm256_setzero_pd();
        r0 = _mm256_fmadd_pd(_mm256_loadu_pd(Ax+j), _mm256_maskload_pd(x+o0, MASK_3), r0);
        r0 = _mm256_fmadd_pd(_mm256_loadu_pd(Ax+j+5), _mm256_maskload_pd(x+o2, MASK_3), r0);

        return hsum_add_double_avx(r0, _mm_mul_pd(_mm_loadu_pd(Ax+j+3), _mm_loadu_pd(x+o1)));
    }

    inline double wathen3_2_3_2_3vec1d(const int* Ap, const int* Ai, const double* Ax, const double* x, int i) {
        int j = Ap[i];
        int o0 = Ai[j], o1 = Ai[j+3], o2 = Ai[j+5], o3 = Ai[j+8], o4 = Ai[j+10];

        auto r0 = _mm256_setzero_pd();
        auto r1 = _mm_setzero_pd();
        r0 = _mm256_fmadd_pd(_mm256_loadu_pd(Ax+j), _mm256_maskload_pd(x+o0, MASK_3), r0);
        r1 = _mm_fmadd_pd(_mm_loadu_pd(Ax+j+3), _mm_loadu_pd(x+o1), r1);
        r0 = _mm256_fmadd_pd(_mm256_loadu_pd(Ax+j+5), _mm256_maskload_pd(x+o2, MASK_3), r0);
        r1 = _mm_fmadd_pd(_mm_loadu_pd(Ax+j+8), _mm_loadu_pd(x+o3), r1);
        r0 = _mm256_fmadd_pd(_mm256_loadu_pd(Ax+j+10), _mm256_maskload_pd(x+o4, MASK_3), r0);

        return hsum_add_double_avx(r0, r1);
    }

    inline double wathen_5_3_5_3_5vec_1d(const int* Ap, const int* Ai, const double* Ax, const double* x, int i) {
        int j = Ap[i];
        int o0 = Ai[j], o1 = Ai[j+5], o2 = Ai[j+8], o3 = Ai[j+13], o4 = Ai[j+16];

        auto r0 = _mm256_setzero_pd();
        r0 = _mm256_fmadd_pd(_mm256_loadu_pd(Ax+j), _mm256_loadu_pd(x+o0), r0);
        double tail = Ax[j+4] * x[o0+4];
        r0 = _mm256_fmadd_pd(_mm256_loadu_pd(Ax+j+5), _mm256_maskload_pd(x+o1, MASK_3), r0);
        r0 = _mm256_fmadd_pd(_mm256_loadu_pd(Ax+j+8), _mm256_loadu_pd(x+o2), r0);
        tail += Ax[j+12] * x[o2+4];
        r0 = _mm256_fmadd_pd(_mm256_loadu_pd(Ax+j+13), _mm256_maskload_pd(x+o3, MASK_3), r0);
        r0 = _mm256_fmadd_pd(_mm256_loadu_pd(Ax+j+16), _mm256_loadu_pd(x+o4), r0);
        tail += Ax[j+20] * x[o4+4];
        return hsum_double_avx(r0) + tail;
    }

    inline double wathen3_5_vec_1d(const int* Ap, const int* Ai, const double* Ax, const double* x, int i) {
        int j = Ap[i];
        int o0 = Ai[j], o1 = Ai[j+3];

        auto r0 = _mm256_setzero_pd();
        r0 = _mm256_fmadd_pd(_mm256_loadu_pd(Ax+j), _mm256_maskload_pd(x+o0, MASK_3), r0);
        r0 = _mm256_fmadd_pd(_mm256_loadu_pd(Ax+j+3), _mm256_loadu_pd(x+o1), r0);
        double tail = Ax[j+7] * x[o1+4];

        return hsum_double_avx(r0) + tail;
    }

    inline void wathen5_3_5vec_2d(const int* Ap, const int* Ai, const double* Ax, const double* x, double* y, int i) {
        int j0 = Ap[i], j1 = Ap[i+1];
        int o0 = Ai[j0], o1 = Ai[j0+5], o2 = Ai[j0+8];
        int o3 = Ai[j1], o4 = Ai[j1+5], o5 = Ai[j1+8];

        auto r0 = _mm256_setzero_pd();
        auto r1 = _mm256_setzero_pd();
        auto tail = _mm_loadu_pd(y+i);

        // Row 1
        r0 = _mm256_fmadd_pd(_mm256_loadu_pd(Ax+j0), _mm256_loadu_pd(x+o0), r0);
        tail[0] += Ax[j0+4] * x[o0+4];
        r0 = _mm256_fmadd_pd(_mm256_loadu_pd(Ax+j0+5), _mm256_maskload_pd(x+o1, MASK_3), r0);
        r0 = _mm256_fmadd_pd(_mm256_loadu_pd(Ax+j0+8), _mm256_loadu_pd(x+o2), r0);
        tail[0] += Ax[j0+12] * x[o2+4];

        // Row 2
        r1 = _mm256_fmadd_pd(_mm256_loadu_pd(Ax+j1), _mm256_loadu_pd(x+o3), r1);
        tail[1] += Ax[j1+4] * x[o3+4];
        r1 = _mm256_fmadd_pd(_mm256_loadu_pd(Ax+j1+5), _mm256_maskload_pd(x+o4, MASK_3), r1);
        r1 = _mm256_fmadd_pd(_mm256_loadu_pd(Ax+j1+8), _mm256_loadu_pd(x+o5), r1);
        tail[1] += Ax[j1+12] * x[o5+4];

        // H-Sum
        auto h0 = _mm256_hadd_pd(r0,r1);
        __m128d vlow = _mm256_castpd256_pd128(h0);
        __m128d vhigh = _mm256_extractf128_pd(h0, 1);  // high 128
        vlow = _mm_add_pd(vlow, vhigh);
        vlow = _mm_add_pd(vlow, tail);
        _mm_storeu_pd(y+i, vlow);
    }

    inline double wathen5_3_5vec_1d(const int* Ap, const int* Ai, const double* Ax, const double* x, int i) {
        int j = Ap[i];
        int o0 = Ai[j], o1 = Ai[j+5], o2 = Ai[j+8];

        auto r0 = _mm256_setzero_pd();
        r0 = _mm256_fmadd_pd(_mm256_loadu_pd(Ax+j), _mm256_loadu_pd(x+o0), r0);
        double tail = Ax[j+4] * x[o0+4];
        r0 = _mm256_fmadd_pd(_mm256_loadu_pd(Ax+j+5), _mm256_maskload_pd(x+o1, MASK_3), r0);
        r0 = _mm256_fmadd_pd(_mm256_loadu_pd(Ax+j+8), _mm256_loadu_pd(x+o2), r0);
        tail += Ax[j+12] * x[o2+4];
        return hsum_double_avx(r0) + tail;
    }

    void spmv_wathen_clean(int n, const int *Ap, const int *Ai, const double *Ax,
                     const double *x, double *y, int nThreads) {
        int i = 1, j;
        y[0] += wathen3_5_vec_1d(Ap, Ai, Ax, x, 0);
        for (int k = 0; k < 99; k++, i+=2) {
            y[i] += wathen3_2_3vec_1d(Ap, Ai, Ax, x, i);
            y[i+1] += wathen5_3_5vec_1d(Ap, Ai, Ax, x, i+1);
        }
        for (int ii = 0; ii < 119; ++ii) {
            for (int k = 0; k < 4; k++, i++) {
                if (Ap[i+1] - Ap[i] == 8)
                    y[i] += wathen3_2_3vec_1d(Ap, Ai, Ax, x, i);
                else
                    y[i] += wathen5_3_5vec_1d(Ap, Ai, Ax, x, i);
            }
            for (int k = 0; k < 49; k++, i+=2) {
                y[i] += wathen5_3_5vec_1d(Ap, Ai, Ax, x, i);
                y[i + 1] += wathen5_3_5vec_1d(Ap, Ai, Ax, x, i + 1);
            }
            for (int k = 0; k < 4; k++, i++) {
                for (j = Ap[i]; j < Ap[i + 1]; j++) {
                    y[i] += Ax[j] * x[Ai[j]];
                }
            }
            for (int k = 0; k < 98; k++, i+=2) {
                y[i] += wathen5_3_5vec_1d(Ap, Ai, Ax, x, i);
                y[i + 1] += wathen_5_3_5_3_5vec_1d(Ap, Ai, Ax, x, i + 1);
            }
        }
        for (int k = 0; k < 4; k++, i++) {
            if (Ap[i+1] - Ap[i] == 8)
                y[i] += wathen3_2_3vec_1d(Ap, Ai, Ax, x, i);
            else
                y[i] += wathen5_3_5vec_1d(Ap, Ai, Ax, x, i);
        }
        for (int k = 0; k < 49; k++, i+=2) {
                y[i] += wathen5_3_5vec_1d(Ap, Ai, Ax, x, i);
                y[i+1] += wathen5_3_5vec_1d(Ap, Ai, Ax, x, i+1);
        }
        for (int k = 0; k < 4; k++, i++) {
            for (j = Ap[i]; j < Ap[i + 1]; j++) {
                y[i] += Ax[j] * x[Ai[j]];
            }
        }
        for (int k = 0; k < 99; k++, i+=2) {
            y[i] += wathen3_5_vec_1d(Ap, Ai, Ax, x, i);
            y[i+1] += wathen5_3_5vec_1d(Ap, Ai, Ax, x, i+1);
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
   spmv_csr(L1_csr_->m, L1_csr_->p, L1_csr_->i, L1_csr_->x, x_in_, x_);
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
#ifdef PROFILE
   this->pw_ = nullptr;
#endif
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

    class SpMVVec1Parallel : public SpMVSerial {
    protected:
        sym_lib::timing_measurement fused_code() override {
            sym_lib::timing_measurement t1;
            t1.start_timer();
            spmv_vec1_parallel(n_, L1_csr_->p, L1_csr_->i, L1_csr_->x, x_in_, x_, num_threads_);
            t1.measure_elapsed_time();
            //copy_vector(0,n_,x_in_,x_);
            return t1;
        }

    public:
        SpMVVec1Parallel(sym_lib::CSR *L, sym_lib::CSC *L_csc,
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

    class SpMVWathen : public SpMVSerial {
    protected:
        sym_lib::timing_measurement fused_code() override {
            sym_lib::timing_measurement t1;
            t1.start_timer();
            spmv_wathen_clean(n_, L1_csr_->p, L1_csr_->i, L1_csr_->x, x_in_, x_, num_threads_);
            t1.measure_elapsed_time();
            //copy_vector(0,n_,x_in_,x_);
            return t1;
        }

    public:
        SpMVWathen(sym_lib::CSR *L, sym_lib::CSC *L_csc,
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

         MKL_INT expected_calls = 1;

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

    class TightLoopSpMV : public SpMVSerial {
        typedef void (*FunctionPtr)();
        void reset_pointers() {
            // Assign pointers
            axi = this->L1_csr_->i;
            xp = this->x_in_;
            axp = this->L1_csr_->x;
            app = this->L1_csr_->p;
            yp = this->x_;
            m = this->L1_csr_->m;
        }
        void build_set() override {
            // Init templates
            build_templates();

            // Init memory
            using FunctionPtr2 = void (*)();
            fp = new FunctionPtr2[this->L1_csr_->m+1];

            for (int i = 0; i < this->L1_csr_->m; ++i) {
                auto diff = this->L1_csr_->p[i+1] - this->L1_csr_->p[i];
                int sbs[6] = { 0 };
                int cnt = 0;
                int cntr = 0;
                for (int j = this->L1_csr_->p[i]; j < this->L1_csr_->p[i+1]-1; ++j) {
                    cntr++;
                    if ((this->L1_csr_->i[j+1] - this->L1_csr_->i[j]) != 1) {
                        sbs[cnt++] = cntr;
                        cntr = 0;
                    }
                }
                sbs[cnt++] = cntr+1;

                if (cnt == 1) {
                    if (sbs[0] == 2) {
                        fp[i] = v2_;
                    } else if (sbs[0] == 3) {
                        fp[i] = v3_;
                    } else {
                        goto flag;
                    }
                } else if (cnt == 2) {
                    if (sbs[0] == 2 && sbs[1] == 1) {
                        fp[i] = v2_1;
                    } else if (sbs[0] == 1 && sbs[1] == 2) {
                        fp[i] = v3;
                    } else if (sbs[0] == 3 && sbs[1] == 1) {
                        fp[i] = v3_1;
                    } else if (sbs[0] == 1 && sbs[1] == 3) {
                        fp[i] = v1_3;
                    } else {
                       goto flag;
                    }
                } else if (cnt == 3) {
                   if (sbs[0] == 1 && sbs[1] == 2 && sbs[2] == 1) {
                       fp[i] = v1_2_1;
                   } else if (sbs[0] == 1 && sbs[1] == 3 && sbs[2] == 1) {
                       fp[i] = v1_3_1;
                   } else {
                       goto flag;
                   }
                } else if (cnt == 4) {
                    if (sbs[0] == 1 && sbs[1] == 1 && sbs[2] == 2 && sbs[3] == 1) {
                        fp[i] = v1_1_2_1;
                    } else if (sbs[0] == 1 && sbs[1] == 1 && sbs[2] == 3 && sbs[3] == 1) {
                        fp[i] = v1_1_3_1;
                    } else {
                        goto flag;
                    }
                } else {
                    flag:
                    std::cout << cnt << ": ";
                    for (int k = 0; k < cnt; k++) {
                        std::cout << sbs[k] << ",";
                    }
                    std::cout << std::endl;
                    exit(1);
                    fp[i] = mapped_fp[diff];
                }

                if (diff > 14) {
                    throw std::runtime_error("Error: template specialization too large... exiting...");
                }
            }

            // Assign pointers
            reset_pointers();
        }
        sym_lib::timing_measurement fused_code() override {
            auto m = this->L1_csr_->m;
            auto p = this->L1_csr_->p;
            auto ai = this->L1_csr_->i;
            auto ax = this->L1_csr_->x;
            sym_lib::timing_measurement t1;
            t1.start_timer();
            for (int i = 0; i < m; ++i) {
                for (int j = p[i]; j < p[i+1]; ++j) {
                    x_[i] += ax[j] * x_in_[ai[j]];
                }
            }
            t1.measure_elapsed_time();
            return t1;
        }

    public:
        TightLoopSpMV(int nThreads, sym_lib::CSR *L, sym_lib::CSC *L_csc,
                double *correct_x,
                std::string name) :
                SpMVSerial(L, L_csc, correct_x, name) {}

        ~TightLoopSpMV() override {}
    };

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
