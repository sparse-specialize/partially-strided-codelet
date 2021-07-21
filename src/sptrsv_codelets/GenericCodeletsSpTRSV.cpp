//
// Created by cetinicz on 2021-07-17.
//
#include "DDT.h"
#include "GenericCodeletsSpTRSV.h"
#include "Inspector.h"

#include <immintrin.h>

#include <DDTDef.h>
#include <vector>

#ifdef _mm256_i32gather_pd
#define loadvx(of, x, ind, xv, vindex) \
  auto (vindex) = _mm_loadu_si128(reinterpret_cast<const __m128i *>(of+ind)); \
  (xv) = _mm256_i32gather_pd(x, vindex, 8);
#else
#define loadvx(of, x, ind, xv, msk) \
  xv = _mm256_set_pd(x[of[ind+3]], x[of[ind+2]], x[of[ind+1]], x[of[ind]]);
#endif


inline double hsum_double_avx(__m256d v) {
    __m128d vlow = _mm256_castpd256_pd128(v);
    __m128d vhigh = _mm256_extractf128_pd(v, 1); // high 128
    vlow = _mm_add_pd(vlow, vhigh);     // reduce down to 128

    __m128d high64 = _mm_unpackhi_pd(vlow, vlow);
    return _mm_cvtsd_f64(_mm_add_sd(vlow, high64));  // reduce to scalar
}

namespace DDT {
    void psc_t3_1D1R_sptrs(double *x, const double *Ax, const int *offset,
                           int lb, int fnl, int cw, bool ms) {
        if (ms) {
            auto ax0 = Ax+fnl;
            int i = 0;
            auto r0 = _mm256_setzero_pd();
            for (; i < cw - 4; i+=4) {
                auto axv0 = _mm256_loadu_pd(ax0+i);
                __m256d xv0;
                loadvx(offset, x, i, xv0, msk)
                r0 = _mm256_fmadd_pd(axv0, xv0, r0);
            }
            double tail = 0.;
            for (; i < cw - 1; ++i) {
                tail += ax0[i] * x[offset[i]];
            }
            x[lb] -= tail + hsum_double_avx(r0);
            x[lb] /= Ax[fnl + cw-1];
        } else {
            int i = 0;
            auto ax0 = Ax + fnl;
            auto r0 = _mm256_setzero_pd();
            for (; i < cw-3; i+=4) {
                auto axv0 = _mm256_loadu_pd(ax0+i);
                __m256d xv0;
                loadvx(offset, x, i, xv0, msk)
                r0 = _mm256_fmadd_pd(axv0, xv0, r0);
            }
            double tail = 0.;
            for (; i < cw; ++i) {
                tail += ax0[i] * x[offset[i]];
            }
            x[lb] -= tail + hsum_double_avx(r0);
        }
    }

    void psc_t2_2D2R_sptrs(double *x, const double *Ax, const int *offset,
                           const int axi, const int axo, const int lb,
                           const int ub, const int cb, const int cof) {
        auto ax0 = Ax + axo + axi * 0;
        auto ax1 = Ax + axo + axi * 1;
        auto x0  = x;
        int i = lb;

        // Preamble
        if (i == offset[cb-1]) {
            for (int j = 0; j < cb - 1; ++j) {
                x[i] -= ax0[j] * x0[offset[j]];
            }
            x[i] /= ax0[cb - 1];
            i++;
            ax0+=axi;
            ax1+=axi;
        }

        // SpMV
//        for (; i < ub; ++i) {
//            for (int j = 0; j < cb; ++j) {
//                x[i] -= Ax[axi * (i-lb) + axo + j] * x[cof * (i-lb) + offset[j]];
//            }
//        }


        int co = (ub-i) % 2;
        for (; i < ub-co; i += 2) {
            auto r0 = _mm256_setzero_pd();
            auto r1 = _mm256_setzero_pd();

            int j = 0;
            for (; j < cb-3; j += 4) {
                __m256d xv0;
                loadvx(offset, x0, j, xv0, msk0);

                auto axv0 = _mm256_loadu_pd(ax0 + j);
                auto axv1 = _mm256_loadu_pd(ax1 + j);

                r0 = _mm256_fmadd_pd(axv0, xv0, r0);
                r1 = _mm256_fmadd_pd(axv1, xv0, r1);
            }

            // Compute tail
            __m128d tail = _mm_loadu_pd(x+i);

            for (; j < cb; j++) {
                tail[0] -= ax0[j] * x0[offset[j]];
                tail[1] -= ax1[j] * x0[offset[j]];
            }

            // H-Sum
            auto h0 = _mm256_hadd_pd(r0, r1);
            __m128d vlow = _mm256_castpd256_pd128(h0);
            __m128d vhigh = _mm256_extractf128_pd(h0, 1);  // high 128
            vlow = _mm_add_pd(vlow, vhigh);     // reduce down to 128
            vlow = _mm_sub_pd(tail, vlow);

            // Store
            _mm_storeu_pd(x + i, vlow);

            // Load new addresses
            ax0 += axi * 2;
            ax1 += axi * 2;
    }
        if (co) {
            // Compute last iteration
            auto r0 = _mm256_setzero_pd();
            __m256d xv;
            int j = 0;
            for (; j < cb - 3; j += 4) {
                loadvx(offset, x0, j, xv, msk);
                auto axv0 = _mm256_loadu_pd(ax0 + j);
                r0 = _mm256_fmadd_pd(axv0, xv, r0);
            }

            // Compute tail
            double tail = 0.;
            for (; j < cb; j++) { tail += *(ax0 + j) * x0[offset[j]]; }

            // H-Sum
            x[ub - 1] -= tail + hsum_double_avx(r0);
        }
    }

    void psc_t1_2D2R_sptrs(double *x, const double *Ax, const int *offset,
                           int lb, int ub, int lbc, int ubc) {
        // Preamble
        int i = lb;
        if (i == ubc) {
            for (int j = lbc; j < ubc; ++j) {
                x[i] -= Ax[offset[0] + j - lbc] * x[j];
            }
            x[i] /= Ax[offset[0] + ubc - lbc];
            ++i;
        }

        // SpMV
//        for (; i < ub; ++i) {
//            for (int j = lbc; j < ubc; ++j) {
//                x[i] -= Ax[offset[i - lb] + i] * x[i];
//            }
//        }
        v4df_t Lx_reg, Lx_reg2, Lx_reg3, Lx_reg4, result, result2, result3,
                result4, x_reg, x_reg2;

        int tii = (ub-i)%4;

        for (int ii=0; i < ub-tii; i+=4, ii+=4) {
            result.v = _mm256_setzero_pd();
            result2.v = _mm256_setzero_pd();
            result3.v = _mm256_setzero_pd();
            result4.v = _mm256_setzero_pd();
            int ti = (ubc-lbc)%4;
            for (int j = lbc, k=offset[ii], k1=offset[ii+1], k2=offset[ii+2],
                         k3=offset[ii+3]; j < ubc-ti; j+=4, k+=4, k1+=4, k2+=4, k3+=4) {
                //y[i] += Ax[k] * x[j];
                //_mm256_mask_i32gather_pd()
                // x_reg.d[0] = x[*aij]; /// TODO replaced with gather
                // x_reg.d[1] = x[*(aij+1)];
                // x_reg.d[2] = x[*(aij+2)];
                // x_reg.d[3] = x[*(aij+3)];
                //x_reg.v = _mm256_set_pd(x[j], x[j+1], x[j+2], x[j+3]);
                x_reg.v = _mm256_loadu_pd((double *) (x+j));
                Lx_reg.v = _mm256_loadu_pd((double *) (Ax + k)); // Skylake	7	0.5
                Lx_reg2.v = _mm256_loadu_pd((double *) (Ax + k1)); // Skylake	7
                Lx_reg3.v = _mm256_loadu_pd((double *) (Ax + k2)); // Skylake	7
                Lx_reg4.v = _mm256_loadu_pd((double *) (Ax +k3)); // Skylake	7
                // 	0.5
                result.v = _mm256_fmadd_pd(Lx_reg.v,x_reg.v,result.v);//Skylake	4	0.5
                result2.v = _mm256_fmadd_pd(Lx_reg2.v,x_reg.v,result2.v);//Skylake	4	0.5
                result3.v = _mm256_fmadd_pd(Lx_reg3.v,x_reg.v,result3.v);//Skylake	4	0.5
                result4.v = _mm256_fmadd_pd(Lx_reg4.v,x_reg.v,result4.v);//Skylake	4	0.5
            }
            double t0=0, t1=0, t2=0, t3=0;
            int jt = ubc-ti-lbc;
            for ( int j=ubc-ti, k=offset[ii]+jt, k1=offset[ii+1]+jt, k2=offset[ii+2]+jt,
                          k3=offset[ii+3]+jt; j < ubc; j++, k++, k1++, k2++, k3++) {
                double xj = x[j];
                t0 += Ax[k] * xj;
                t1 += Ax[k1] * xj;
                t2 += Ax[k2] * xj;
                t3 += Ax[k3] * xj;
            }
            auto h0 = _mm256_hadd_pd(result.v, result2.v);
            x[i] -= (h0[0] + h0[2] + t0);
            x[i+1] -= (h0[1] + h0[3] + t1);
            h0 = _mm256_hadd_pd(result3.v, result4.v);
            x[i+2] -= (h0[0] + h0[2] + t2);
            x[i+3] -= (h0[1] + h0[3] + t3);
        }
        /** the rest **/
        for (int i = ub-tii, ii=ub-lb-tii; i < ub; i++, ii++) {
            result.v = _mm256_setzero_pd();
            int ti = (ubc-lbc) % 4;
            for (int j = lbc, k=offset[i-lb]; j < ubc-ti; j+=4, k+=4) {
                x_reg.v = _mm256_loadu_pd((double *) (x+j));
                Lx_reg.v = _mm256_loadu_pd((double *) (Ax + k)); // Skylake	7	0.5
                result.v = _mm256_fmadd_pd(Lx_reg.v,x_reg.v,result.v);//Skylake	4	0.5
            }
            double t0=0;
            int jt = ubc-lbc-ti;
            for (int j = ubc-ti, k=offset[i-lb]+jt; j < ubc; j++, k++) {
                t0 += Ax[k] * x[j];
            }
            auto h0 = hsum_double_avx(result.v);
            x[i] -= h0 + t0;
        }
    }

    void fsc_2D2R_sptrs(double *x, const double *Ax, const int axi,
                        const int axo, const int lb, const int ub,
                        const int cbl, const int cbu, const int co) {
        int i = lb;
        auto ax0 = Ax + axo + axi * 0;
        auto ax1 = Ax + axo + axi * 1;
        auto x0 = x + cbl;

        // Preamble
        if (cbu == lb) {
            for (int j = cbl; j < cbu - 1; ++j) {
                x[lb] -= ax0[j - cbl] * x0[j-cbl];
            }
            x[lb] /= Ax[axo + cbu - cbl];
            i++;
            ax0 += axi;
            ax1 += axi;
        }

        // SpMV
//        for (; i < ub; ++i) {
//            for (int j = cbl; j < cbu; ++j) {
//                x[i] -= Ax[axi * (i - lb) + axo + j - cbl] *
//                        x[co * (i - lb) + j];
//            }
//        }



        int cr = (ub - i) % 2;
        for (; i < ub - cr; i += 2) {
            auto r0 = _mm256_setzero_pd();
            auto r1 = _mm256_setzero_pd();

            int j = 0;
            for (; j < (cbu - cbl) - 3; j += 4) {
                auto xv0 = _mm256_loadu_pd(x0 + j);

                auto axv0 = _mm256_loadu_pd(ax0 + j);
                auto axv1 = _mm256_loadu_pd(ax1 + j);

                r0 = _mm256_fmadd_pd(axv0, xv0, r0);
                r1 = _mm256_fmadd_pd(axv1, xv0, r1);
            }

            // Compute tail
            __m128d tail = _mm_loadu_pd(x+i);
            for (; j < (cbu - cbl); j++) {
                tail[0] -= ax0[j] * x0[j];
                tail[1] -= ax1[j] * x0[j];
            }

            // H-Sum
            auto h0 = _mm256_hadd_pd(r0, r1);
            __m128d vlow = _mm256_castpd256_pd128(h0);
            __m128d vhigh = _mm256_extractf128_pd(h0, 1);// high 128
            vlow = _mm_add_pd(vlow, vhigh);              // reduce down to 128
            vlow = _mm_sub_pd(tail,vlow);
            // Store
            _mm_storeu_pd(x + i, vlow);

            // Load new addresses
            ax0 += axi * 2;
            ax1 += axi * 2;
//            x0 += co*2;
        }

        // Compute last iteration
        if (cr) {
            auto r0 = _mm256_setzero_pd();
            int j = 0;
            for (; j < (cbu - cbl) - 3; j += 4) {
                auto xv = _mm256_loadu_pd(x0 + j);
                auto axv0 = _mm256_loadu_pd(ax0 + j);
                r0 = _mm256_fmadd_pd(axv0, xv, r0);
            }

            // Compute tail
            double tail = 0.;
            for (; j < cbu - cbl; j++) { tail += *(ax0 + j) * x0[j]; }

            // H-Sum
            x[ub - 1] -= tail + hsum_double_avx(r0);
        }
    }

    void psc_t3_v1_base(double *x, const double *Ax, const int *Ai,
                          const int *Ap,
                          const int *rs_off, const int *rl_off,
                          const int lb, const int ub, int min_iter){
     for (int i = lb; i < ub; i++) {
      for (int j = rs_off[i]; j < rs_off[i] + rl_off[i]; j++) {
       x[i] -= Ax[j] * x[Ai[j]];
      }
      x[i] /= Ax[Ap[i + 1] - 1];
     }
    }

 void psc_t3_v1_2D1D(double *x, const double *Ax, const int *Ai,
                     const int *Ap,
                     const int *rs_off, const int *rl_off,
                     const int lb, const int ub, int min_iter){
  v4df_t Lx_reg, Lx_reg2, result, result2, x_reg, x_reg2;
  for (int i = lb; i < ub; i+=2) {
   result.v = _mm256_setzero_pd();
   result2.v = _mm256_setzero_pd();
   for (int j1 = rs_off[i], j2 = rs_off[i+1]; j1 < rs_off[i] + min_iter;
   j1+=4, j2+=4) {
    //x[i] += Ax[j1] * x[Ai[j1]];
    x_reg.v = _mm256_loadu_pd((double *) (x+Ai[j1]));
    x_reg2.v = _mm256_loadu_pd((double *) (x+Ai[j2]));
    Lx_reg.v = _mm256_loadu_pd((double *) (Ax + j1)); // Skylake	7	0.5
    Lx_reg2.v = _mm256_loadu_pd((double *) (Ax + j2)); // Skylake	7
    // 	0.5
    result.v = _mm256_fmadd_pd(Lx_reg.v,x_reg.v,result.v);//Skylake	4	0.5
    result2.v = _mm256_fmadd_pd(Lx_reg2.v,x_reg.v,result2.v);//Skylake	4	0.5
   }
   // 1D row i
   int rs_p_rl_i = rs_off[i] + rl_off[i];
   int m_i = rs_p_rl_i % 4;
   for (int j = rs_off[i] + min_iter; j < rs_p_rl_i - m_i; j+=4) {
    //x[i] += Ax[j] * x[Ai[j]];
    x_reg.v = _mm256_loadu_pd((double *) (x+Ai[j]));
    Lx_reg.v = _mm256_loadu_pd((double *) (Ax + j));
    result.v = _mm256_fmadd_pd(Lx_reg.v,x_reg.v,result.v);
   }
   double t=0;
   for (int j = rs_p_rl_i - m_i; j < rs_off[i] + rl_off[i]; j++) {
    t += Ax[j] * x[Ai[j]];
   }
   x[i] -= t + hsum_double_avx(result.v);
   x[i] /= Ax[Ap[i + 1] - 1];

   // 1D row i+1
   int rs_p_rl_ip = rs_off[i+1] + rl_off[i+1];
   int m_ip = rs_p_rl_ip % 4;
   for (int j = rs_off[i+1] + min_iter; j < rs_p_rl_ip + m_ip; j+=4) {
    //x[i+1] -= Ax[j] * x[Ai[j]];
    x_reg.v = _mm256_loadu_pd((double *) (x+Ai[j]));
    Lx_reg.v = _mm256_loadu_pd((double *) (Ax + j));
    result2.v = _mm256_fmsub_pd(Lx_reg.v,x_reg.v,result.v);
   }
   t=0;
   for (int j = rs_p_rl_ip - m_ip; j < rs_off[i] + rl_off[i]; j++) {
    t += Ax[j] * x[Ai[j]];
   }
   x[i+1] -= t + hsum_double_avx(result2.v);
   x[i+1] /= Ax[Ap[i + 2] - 1];
  }
 }

 void psc_t3_v2_base(double *x, const double *Ax, const int *Ai,
                     const int *Ap, const int *row_id,
                     const int *rs_off, const int *rl_off,
                     const int iter_len, int min_iter){
    for (int l = 0; l < iter_len; l++) {
     int i = row_id[l];
     for (int j = rs_off[i]; j < rs_off[i] + rl_off[i]; j++) {
      x[i] -= Ax[j] * x[Ai[j]];
     }
     x[i] /= Ax[Ap[i + 1] - 1];
    }
   }

 template<class type>
 bool is_float_equal(const type x, const type y, double absTol, double relTol) {
  return std::abs(x - y) <= std::max(absTol, relTol * std::max(std::abs(x), std::abs(y)));
 }

    void verify_sptrsv(int n, double* x, const int* Lp, const int* Li, const double* Lx) {
        // Allocate and fill
        auto xx = new double[n]();
        std::fill(xx, xx+n, 1);

        for (int i = 0; i < n; ++i) {
            for (int j = Lp[i]; j < Lp[i+1]-1; ++j) {
                xx[i] -= Lx[j] * xx[Li[j]];
            }
            xx[i] /= Lx[Lp[i+1]-1];
        }

        for (int i = 0; i < n; ++i) {
            if (!is_float_equal(x[i], xx[i], 1e-4, 1e-4)) {
                std::cout << "'x' not equal at i = " << i << std::endl;
                std::cout << "(x,x_cpy): (" << x[i] << "," << xx[i] << ")" << std::endl;
                exit(1);
            }
        }

        delete[] xx;
    }

    void sptrsv_generic(const int n, const int* Lp, const int* Li, const double *Ax, double *x,
                        const std::vector<DDT::Codelet *> *lst,
                        const DDT::Config &cfg) {
        //#pragma omp parallel for num_threads(cfg.nThread)
        for (int i = 0; i < cfg.nThread; i++) {
            for (auto const& c : lst[i]) {
                switch (c->get_type()) {
#ifdef O3
                    case DDT::CodeletType::TYPE_FSC:
                        fsc_2D2R_sptrs(x, Ax, c->row_offset, c->first_nnz_loc,
                                       c->lbr, c->lbr + c->row_width, c->lbc,
                                       c->col_width + c->lbc, c->col_offset);
                        break;
                    case DDT::CodeletType::TYPE_PSC1:
                        psc_t1_2D2R_sptrs(x, Ax, c->offsets, c->lbr,
                                          c->lbr + c->row_width, c->lbc,
                                          c->lbc + c->col_width);
                        break;
                    case DDT::CodeletType::TYPE_PSC2:
                        psc_t2_2D2R_sptrs(x, Ax, c->offsets, c->row_offset,
                                          c->first_nnz_loc, c->lbr,
                                          c->lbr + c->row_width, c->col_width,
                                          c->col_offset);
                        break;
#endif
                    case DDT::CodeletType::TYPE_PSC3:
                        psc_t3_1D1R_sptrs(x, Ax, c->offsets, c->lbr,
                                          c->first_nnz_loc, c->col_width,c->multi_stmt);
                        break;
                    case DDT::CodeletType::TYPE_PSC3_V1:
                        psc_t3_v1_base(x, Ax, Li, Lp,  c->offsets,
                                       c->offsets2, c->lbr, c->lbr +
                                       c->row_width, -1);
                        break;
                    case DDT::CodeletType::TYPE_PSC3_V2:
                        psc_t3_v2_base(x, Ax, Li, Lp, c->offset3,  c->offsets,
                                    c->offsets2, c->row_width, -1);
                        break;
                 default:
                        break;
                }
            }
        }
//        verify_sptrsv(n,x,Lp,Li,Ax);
    }
}