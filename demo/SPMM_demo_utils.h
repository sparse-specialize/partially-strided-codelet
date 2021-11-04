//
// Created by cetinicz on 2021-10-30.
//

#ifndef DDT_SPMM_DEMO_UTILS_H
#define DDT_SPMM_DEMO_UTILS_H

#include "FusionDemo.h"
#include <DDT.h>
#include <DDTCodelets.h>
#include <Executor.h>

#ifdef MKL
#include <mkl.h>
#endif

#define loadmx(v,ai,o) \
_mm256_set_pd(v[ai[3]+o],v[ai[2]+o],v[ai[1]+o],v[ai[0]+o]);

namespace sparse_avx {

    void spmm_csr_csc(double* Cx, double* Ax, double* Bx, int m, int* Ap, int* Ai, int bRows, int bCols) {
        for (int i = 0; i < m; ++i) {
            for (int k = 0; k < bCols; ++k) {
                for (int j = Ap[i]; j < Ap[i+1]; ++j) {
                    Cx[i+k*bRows] += Ax[j] * Bx[Ai[j]+k*bRows];
                }
            }
        }
    }

    void spmm_csr_csr(double* Cx, double* Ax, double* Bx, int m, int* Ap, int* Ai, int bRows, int bCols) {
#pragma omp parallel for
        for (int i = 0; i < m; ++i) {
            for (int j = Ap[i]; j < Ap[i+1]; ++j) {
                for (int k = 0; k < bCols; ++k) {
                    Cx[i*bCols+k] += Ax[j] * Bx[Ai[j]*bCols+k];
                }
            }
        }
    }

    void spmm_csr_csc_parallel(double* Cx, double* Ax, double* Bx, int m, int* Ap, int* Ai, int bRows, int bCols, int nThreads) {
#pragma omp parallel for num_threads(nThreads)
        for (int i = 0; i < m; ++i) {
            for (int k = 0; k < bCols; ++k) {
                for (int j = Ap[i]; j < Ap[i+1]; ++j) {
                    Cx[i+k*bRows] += Ax[j] * Bx[Ai[j]+k*bRows];
                }
            }
        }
    }

    void spmm_csr_csr_parallel(double* Cx, double* Ax, double* Bx, int m, int* Ap, int* Ai, int bRows, int bCols, int nThreads) {
#pragma omp parallel for num_threads(nThreads)
        for (int i = 0; i < m; ++i) {
            for (int j = Ap[i]; j < Ap[i+1]; ++j) {
                for (int k = 0; k < bCols; ++k) {
                    Cx[i*bCols+k] += Ax[j] * Bx[Ai[j]*bCols+k];
                }
            }
        }
    }


    void spmm_csr_csc_tiled_parallel(double *Cx, double *Ax, double *Bx, int m,
                                 int *Ap, int *Ai, int bRows, int bCols,
                                 int nThreads) {
        const int mTileSize = 256;
        const int nTileSize = 32;

        int mTile = m / mTileSize;
        int nTile = bCols / nTileSize;
        int mTileRemainder = m % mTileSize;
        int nTileRemainder = bCols % nTileSize;

#pragma omp parallel for num_threads(nThreads)
        for (int ii = 0; ii < mTile; ++ii) {
            for (int kk = 0; kk < nTile; ++kk) {
                for (int i = 0; i < mTileSize; ++i) {
                    for (int k = 0; k < nTileSize; ++k) {
                        for (int j = Ap[i+ii*mTileSize]; j < Ap[i+ii*mTileSize + 1]; ++j) {
                            Cx[i+ii*mTileSize + (k+kk*nTileSize) * bRows] += Ax[j] * Bx[Ai[j] + (k+kk*nTileSize) * bRows];
                        }
                    }
                }
            }
            for (int k = bCols - nTileRemainder; k < bCols; ++k) {
                for (int i = 0; i < mTileSize; ++i) {
                    for (int j = Ap[i+ii*mTileSize]; j < Ap[i+ii*mTileSize + 1]; ++j) {
                        Cx[i+ii*mTileSize + k * bRows] += Ax[j] * Bx[Ai[j] + k * bRows];
                    }
                }
            }
        }
        for (int i = m-mTileRemainder; i < m; ++i) {
            for (int k = 0; k < bCols; ++k) {
                for (int j = Ap[i]; j < Ap[i + 1]; ++j) {
                    Cx[i + k * bRows] += Ax[j] * Bx[Ai[j] + k * bRows];
                }
            }
        }
    }
    void spmm_csr_csr_4x4_kernel(double* Cx, double* Ax, double* Bx, int* Ai, int bCols) {
        // Get pointers
        auto bxp0 = Bx + Ai[0]*bCols;
        auto bxp1 = Bx + Ai[1]*bCols;
        auto bxp2 = Bx + Ai[2]*bCols;
        auto bxp3 = Bx + Ai[3]*bCols;

        // Pack vectors
        auto axv0 = _mm256_set1_pd(Ax[0]);
        auto axv1 = _mm256_set1_pd(Ax[1]);
        auto axv2 = _mm256_set1_pd(Ax[2]);
        auto axv3 = _mm256_set1_pd(Ax[3]);

        // Load vectors
        auto bxv0 = _mm256_loadu_pd(bxp0);
        auto bxv1 = _mm256_loadu_pd(bxp1);
        auto bxv2 = _mm256_loadu_pd(bxp2);
        auto bxv3 = _mm256_loadu_pd(bxp3);

        // Load output
        auto r0 = _mm256_loadu_pd(Cx);

        // Compute
        r0 = _mm256_fmadd_pd(axv0,bxv0,r0);
        r0 = _mm256_fmadd_pd(axv1,bxv1,r0);
        r0 = _mm256_fmadd_pd(axv2,bxv2,r0);
        r0 = _mm256_fmadd_pd(axv3,bxv3,r0);

        // Store
        _mm256_storeu_pd(Cx,r0);
    }

    void spmm_csr_csr_tiled_parallel(double *Cx, double *Ax, double *Bx, int m,
                                 int *Ap, int *Ai, int bRows, int bCols,
                                 int nThreads,int mts,int nts) {
      int mTileSize= mts;
      int nTileSize = nts;
      int mTile = m / mTileSize;
      int nTile = bCols / nTileSize;
      int mTileRemainder = m % mTileSize;
      int nTileRemainder = bCols % nTileSize;

#pragma omp parallel for schedule(dynamic,1) num_threads(nThreads) 
      for (int ii = 0; ii < mTile; ++ii) {
        auto iOffset = ii*mTileSize;
        for (int kk = 0; kk < nTile; ++kk) {
          auto kOffset = kk*nTileSize;
          for (int i = 0; i < mTileSize; ++i) {
            int j = Ap[i+iOffset];
            for (; j < Ap[i+iOffset+1]-3; j+=4) {
              int k = 0;
              for (; k < nTileSize-3; k+=4) {
                spmm_csr_csr_4x4_kernel(Cx+(i+iOffset)*bCols+kOffset+k,Ax+j,Bx+kOffset+k,Ai+j,bCols);
              }
              for (; k < nTileSize; ++k) {
                Cx[(i+iOffset)*bCols+kOffset + k] += Ax[j] * Bx[Ai[j]*bCols+kOffset+k];
                Cx[(i+iOffset)*bCols+kOffset + k] += Ax[j+1] * Bx[Ai[j]*bCols+kOffset+k];
                Cx[(i+iOffset)*bCols+kOffset + k] += Ax[j+2] * Bx[Ai[j]*bCols+kOffset+k];
                Cx[(i+iOffset)*bCols+kOffset + k] += Ax[j+3] * Bx[Ai[j]*bCols+kOffset+k];
              }
            }
            for (; j < Ap[i+iOffset + 1]; ++j) {
              for (int k = 0; k < nTileSize; ++k) {
                Cx[(i+iOffset)*bCols+kOffset + k] += Ax[j] * Bx[Ai[j]*bCols+kOffset+k];
              }
            }
          }
        }
        // nTile Remainder
        for (int k = bCols-nTileRemainder; k < bCols; ++k) {
          for (int i = 0; i < mTileSize; ++i) {
            for (int j = Ap[i + iOffset]; j < Ap[i + iOffset + 1]; ++j) {
              Cx[(i+iOffset)*bCols + k] += Ax[j] * Bx[Ai[j]*bCols+k];
            }
          }
        }
      }
      // mTile Remainder
      for (int i = m-mTileRemainder; i < m; ++i) {
        for (int j = Ap[i]; j < Ap[i+1]; ++j) {
          for (int k = 0; k < bCols; ++k) {
            Cx[i*bCols + k] += Ax[j] * Bx[Ai[j]*bCols+k];
          }
        }
      }
    }

    class SpMMSerial : public sym_lib::FusionDemo {
      protected:
        double* Bx;
        int bRows, bCols;
        sym_lib::timing_measurement fused_code() override {
          sym_lib::timing_measurement t1;
          t1.start_timer();
          spmm_csr_csr(x_, L1_csr_->x, Bx, L1_csr_->m, L1_csr_->p, L1_csr_->i, bRows, bCols);
          t1.measure_elapsed_time();
          //            std::copy(x_,x_+L1_csr_->m*cbb,x_);
          return t1;
        }

      public:
        SpMMSerial(sym_lib::CSR *L, sym_lib::CSC *L_csc, int bRows, int bCols,
            std::string name)
          : FusionDemo(L->m*bCols, name) {
            L1_csr_ = L;
            L1_csc_ = L_csc;
            this->Bx = new double[bRows*bCols];

            for (int i = 0; i < bRows*bCols; i++) {
              this->Bx[i] = 1;
            }
            this->bRows = bRows;
            this->bCols = bCols;
          };

        ~SpMMSerial() override {
          delete[] Bx;
        }
    };

    class SpMMParallel : public SpMMSerial {
      protected:
        sym_lib::timing_measurement fused_code() override {
          sym_lib::timing_measurement t1;
          t1.start_timer();
          spmm_csr_csr_parallel(x_, L1_csr_->x, Bx, L1_csr_->m, L1_csr_->p, L1_csr_->i, bRows, bCols, num_threads_);
          t1.measure_elapsed_time();
          //copy_vector(0,n_,x_in_,x_);
          return t1;
        }

      public:
        SpMMParallel(sym_lib::CSR *L, sym_lib::CSC *L_csc, double *correct_cx, int bRows, int bCols,
            std::string name)
          : SpMMSerial(L,L_csc,bRows,bCols,name) {
            L1_csr_ = L;
            L1_csc_ = L_csc;
            correct_x_ = correct_cx;
          };

        ~SpMMParallel() override = default;
    };

    class SpMMTiledParallel : public SpMMSerial {
      protected:
        int mTileSize;
        int nTileSize;
        sym_lib::timing_measurement fused_code() override {
          sym_lib::timing_measurement t1;
          t1.start_timer();
          spmm_csr_csr_tiled_parallel(x_, L1_csr_->x, Bx, L1_csr_->m, L1_csr_->p, L1_csr_->i, bRows, bCols, num_threads_,mTileSize,nTileSize);
          t1.measure_elapsed_time();
          //copy_vector(0,n_,x_in_,x_);
          return t1;
        }

      public:
        SpMMTiledParallel(sym_lib::CSR *L, sym_lib::CSC *L_csc, double *correct_cx, int bRows, int bCols,int mTileSize,int nTileSize,
            std::string name)
          : mTileSize(mTileSize), nTileSize(nTileSize), SpMMSerial(L,L_csc,bRows,bCols,name) {
            L1_csr_ = L;
            L1_csc_ = L_csc;
            correct_x_ = correct_cx;
          };

        ~SpMMTiledParallel() override = default;
    };

    class SpMMDDT : public SpMMSerial {
      protected:
        std::vector<DDT::Codelet*>* cl;
        DDT::Config cfg;
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
          DDT::Args args;
          args.x = x_in_;
          args.y = x_;
          args.r = L1_csr_->m;
          args.Lp = L1_csr_->p;
          args.Li = L1_csr_->i;
          args.Lx = L1_csr_->x;
          args.bRows = bRows;
          args.bCols = bCols;
          args.Ax = L1_csr_->x;
          args.Bx = Bx;
          args.Cx = x_;
          t1.start_timer();
          DDT::executeCodelets(cl, cfg, args);
          t1.measure_elapsed_time();
          //copy_vector(0,n_,x_in_,x_);
          return t1;
        }

      public:
        SpMMDDT(sym_lib::CSR *L, sym_lib::CSC *L_csc, double *correct_cx, DDT::Config& cfg, int bRows, int bCols,
            std::string name)
          : cl(nullptr), SpMMSerial(L,L_csc,bRows,bCols,name) {
            L1_csr_ = L;
            L1_csc_ = L_csc;
            correct_x_ = correct_cx;
            this->cfg = cfg;
          };

        ~SpMMDDT() override = default;
    };

    class SpMMMKL : public SpMMSerial {
      protected:
        sym_lib::timing_measurement analysis_breakdown;
        matrix_descr d;
        sparse_matrix_t m;
        MKL_INT *LLI;

        void build_set() override {
          if (LLI == nullptr) {
            analysis_breakdown.start_timer();
            d.type = SPARSE_MATRIX_TYPE_GENERAL;
            //         d.diag = SPARSE_DIAG_NON_UNIT;
            //         d.mode = SPARSE_FILL_MODE_FULL;

            MKL_INT expected_calls = 5;

            LLI = new MKL_INT[this->L1_csr_->m + 1]();
            for (int l = 0; l < this->L1_csr_->m + 1; ++l) {
              LLI[l] = this->L1_csr_->p[l];
            }

            mkl_sparse_d_create_csr(&m, SPARSE_INDEX_BASE_ZERO,
                this->L1_csr_->m, this->L1_csr_->n, LLI,
                LLI + 1, this->L1_csr_->i,
                this->L1_csr_->x);
            //         mkl_sparse_set_mv_hint(m, SPARSE_OPERATION_NON_TRANSPOSE, d, expected_calls);
            //         mkl_sparse_set_memory_hint(m, SPARSE_MEMORY_AGGRESSIVE);

            mkl_set_num_threads(num_threads_);
            mkl_set_num_threads_local(num_threads_);
            analysis_breakdown.measure_elapsed_time();
          }
        }

        sym_lib::timing_measurement fused_code() override {
          sym_lib::timing_measurement t1;
          t1.start_timer();
          mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1, m, this->d, SPARSE_LAYOUT_ROW_MAJOR, Bx, bCols, bCols, 0, x_, bCols);
          t1.measure_elapsed_time();
          return t1;
        }

      public:
        SpMMMKL(sym_lib::CSR *L, sym_lib::CSC *L_csc, double *correct_cx, int bRows, int bCols,
            std::string name)
          : LLI(nullptr), SpMMSerial(L,L_csc,bRows,bCols,name) {
            L1_csr_ = L;
            L1_csc_ = L_csc;
            correct_x_ = correct_cx;
          };
        ~SpMMMKL() override {
          //            mkl_free(LLI);
        };
    };
}

#endif//DDT_SPMM_DEMO_UTILS_H
