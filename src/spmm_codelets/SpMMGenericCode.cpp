//
// Created by cetinicz on 2021-10-30.
//

#include "SpMMGenericCode.h"
#include <vector>

#define loadbx(of, x, ind, xv, msk, cbd) \
xv = _mm256_set_pd(x[of[ind+3]+cbd], x[of[ind+2]+cbd], x[of[ind+1]+cbd], x[of[ind]+cbd]);

namespace DDT {
    inline double hsum_double_avx(__m256d v) {
        __m128d vlow = _mm256_castpd256_pd128(v);
        __m128d vhigh = _mm256_extractf128_pd(v, 1);// high 128
        vlow = _mm_add_pd(vlow, vhigh);             // reduce down to 128

        __m128d high64 = _mm_unpackhi_pd(vlow, vlow);
        return _mm_cvtsd_f64(_mm_add_sd(vlow, high64));// reduce to scalar
    }

    void spmm_generic(const int n, const int *Ap, const int *Ai,
                      const double *Ax, const double *Bx, double *Cx, int bRows, int bCols,
                      const std::vector<Codelet *> *lst,
                      const DDT::Config &cfg) {
        // Perform SpMV
        auto a = std::chrono::steady_clock::now();
#pragma omp parallel for num_threads(cfg.nThread)
        for (int i = 0; i < cfg.nThread; i++) {
            for (const auto &c : lst[i]) {
                switch (c->get_type()) {
                    case CodeletType::TYPE_FSC:
                        fsc_t2_2DC_gemm(Cx, Ax, Bx, c->row_offset,
                                        c->first_nnz_loc, c->lbr,
                                        c->lbr + c->row_width, c->lbc,
                                        c->col_width + c->lbc, c->col_offset,bRows,bCols);
                        break;
                    case CodeletType::TYPE_PSC1:
                        psc_t1_2D4R_gemm(Cx, Ax, Bx, c->offsets, c->lbr,
                                         c->lbr + c->row_width, c->lbc,
                                         c->lbc + c->col_width,bRows,bCols);
                        break;
                    case CodeletType::TYPE_PSC2:
                        psc_t2_2DC_gemm(Cx, Ax, Bx, c->offsets, c->row_offset,
                                        c->first_nnz_loc, c->lbr,
                                        c->lbr + c->row_width, c->col_width,
                                        c->col_offset,bRows,bCols);
                        break;
                    case CodeletType::TYPE_PSC3:
                        psc_t3_1D1R_gemm(Cx, Ax, Ai, Bx, c->offsets, c->lbr,
                                         c->first_nnz_loc, c->col_width,bRows,bCols);
                        break;
                    default:
                        break;
                }
            }
        }
    }


    /**
* Computes y[lb:ub] += Ax[axo+axi:axo+axi*cb] * Bx[cbl:cbu]
*
* @param Cx output matrix solution - col storage
* @param Ax in nonzero locations
* @param Bx in the input matrix - col storage
* @param axi distance to next row in matrix
* @param axo distance to first element in matrix
* @param lb in lower bound of rows
* @param ub in upper bound of rows
* @param cbl in number of columns to compute
* @param cbu in number of columns to compute
* @param cbb in number of columns in matrix Bx
* @param cbd in number of rows in matrix Bx
*/
    void fsc_t2_2DC_gemm(double *Cx, const double *Ax, const double *Bx,
                         const int axi, const int axo, const int lb,
                         const int ub, const int cbl, const int cbu,
                         const int co, const int bRows, const int bCols) {
        auto ax0 = Ax + axo + axi * 0;
        auto ax1 = Ax + axo + axi * 1;
        auto x0 = Bx + cbl;
        auto x1 = x0 + co;

        int cr = (ub - lb) % 2;
        for (int i = lb; i < ub - 1; i += 2) {
            for (int k = 0; k < bCols; k++) {
                auto r0 = _mm256_setzero_pd();
                auto r1 = _mm256_setzero_pd();

                int j = 0;
                for (; j < (cbu - cbl) - 3; j += 4) {
                    auto xv0 = _mm256_loadu_pd(x0 + bRows * k + j);
                    auto xv1 = _mm256_loadu_pd(x1 + bRows * k + j);

                    auto axv0 = _mm256_loadu_pd(ax0 + j);
                    auto axv1 = _mm256_loadu_pd(ax1 + j);

                    r0 = _mm256_fmadd_pd(axv0, xv0, r0);
                    r1 = _mm256_fmadd_pd(axv1, xv1, r1);
                }

                // Compute tail
                __m128d tail = _mm_loadu_pd(Cx + i + bRows * k);
                for (; j < (cbu - cbl); j++) {
                    tail[0] += ax0[j] * x0[j + bRows * k];
                    tail[1] += ax1[j] * x1[j + bRows * k];
                }

                // H-Sum
                auto h0 = _mm256_hadd_pd(r0, r1);
                __m128d vlow = _mm256_castpd256_pd128(h0);
                __m128d vhigh = _mm256_extractf128_pd(h0, 1);// high 128
                vlow = _mm_add_pd(vlow, vhigh);// reduce down to 128
                vlow = _mm_add_pd(vlow, tail);
                // Store
                _mm_storeu_pd(Cx + i + bRows * k, vlow);
            }
            // Load new addresses
            ax0 += axi * 2;
            ax1 += axi * 2;
            x0 += co * 2;
            x1 += co * 2;
        }

        // Compute last iteration
        if (cr) {
            for (int k = 0; k < bCols; k++) {
                auto r0 = _mm256_setzero_pd();
                int j = 0;
                for (; j < (cbu - cbl) - 3; j += 4) {
                    auto xv = _mm256_loadu_pd(x0 + j + bRows * k);
                    auto axv0 = _mm256_loadu_pd(ax0 + j);
                    r0 = _mm256_fmadd_pd(axv0, xv, r0);
                }

                // Compute tail
                double tail = 0.;
                for (; j < cbu - cbl; j++) {
                    tail += *(ax0 + j) * x0[j + bRows * k];
                }

                // H-Sum
                Cx[ub - 1 + bRows * k] += tail + hsum_double_avx(r0);
            }
        }
    }

    void psc_t1_2D4R_gemm(double *Cx, const double *Ax, const double *Bx,
                          const int *offset, int lb, int ub, int lbc, int ubc,
                          const int bRows, const int bCols) {
        v4df_t Lx_reg, Lx_reg2, Lx_reg3, Lx_reg4, result, result2, result3,
                result4, x_reg, x_reg2;

        int tii = (ub - lb) % 4;
        for (int i = lb, ii = 0; i < ub - tii; i += 4, ii += 4) {
            for (int kk = 0; kk < bCols; kk++) {
                result.v = _mm256_setzero_pd();
                result2.v = _mm256_setzero_pd();
                result3.v = _mm256_setzero_pd();
                result4.v = _mm256_setzero_pd();
                int ti = (ubc - lbc) % 4;
                for (int j = lbc, k = offset[ii], k1 = offset[ii + 1],
                         k2 = offset[ii + 2], k3 = offset[ii + 3];
                     j < ubc - ti; j += 4, k += 4, k1 += 4, k2 += 4, k3 += 4) {
                    x_reg.v = _mm256_loadu_pd((double *) (Bx + j + kk * bRows));
                    Lx_reg.v = _mm256_loadu_pd(
                            (double *) (Ax + k));// Skylake	7	0.5
                    Lx_reg2.v = _mm256_loadu_pd(
                            (double *) (Ax + k1));// Skylake	7
                    Lx_reg3.v = _mm256_loadu_pd(
                            (double *) (Ax + k2));// Skylake	7
                    Lx_reg4.v = _mm256_loadu_pd(
                            (double *) (Ax + k3));// Skylake	7

                    result.v = _mm256_fmadd_pd(Lx_reg.v, x_reg.v,
                                               result.v);//Skylake	4	0.5
                    result2.v = _mm256_fmadd_pd(Lx_reg2.v, x_reg.v,
                                                result2.v);//Skylake	4	0.5
                    result3.v = _mm256_fmadd_pd(Lx_reg3.v, x_reg.v,
                                                result3.v);//Skylake	4	0.5
                    result4.v = _mm256_fmadd_pd(Lx_reg4.v, x_reg.v,
                                                result4.v);//Skylake	4	0.5
                }
                double t0 = 0, t1 = 0, t2 = 0, t3 = 0;
                int jt = ubc - ti - lbc;
                for (int j = ubc - ti, k = offset[ii] + jt,
                         k1 = offset[ii + 1] + jt, k2 = offset[ii + 2] + jt,
                         k3 = offset[ii + 3] + jt;
                     j < ubc; j++, k++, k1++, k2++, k3++) {
                    double xj = Bx[j + k * bRows];
                    t0 += Ax[k] * xj;
                    t1 += Ax[k1] * xj;
                    t2 += Ax[k2] * xj;
                    t3 += Ax[k3] * xj;
                }
                auto h0 = _mm256_hadd_pd(result.v, result2.v);
                Cx[i + kk * bRows] += (h0[0] + h0[2] + t0);
                Cx[i + 1 + kk * bRows] += (h0[1] + h0[3] + t1);
                h0 = _mm256_hadd_pd(result3.v, result4.v);
                Cx[i + 2 + kk * bRows] += (h0[0] + h0[2] + t2);
                Cx[i + 3 + kk * bRows] += (h0[1] + h0[3] + t3);
            }
        }
        /** the rest **/
        for (int i = ub - tii, ii = ub - lb - tii; i < ub; i++, ii++) {
            for (int kk = 0; kk < bCols; kk++) {
                result.v = _mm256_setzero_pd();
                int ti = (ubc - lbc) % 4;
                for (int j = lbc, k = offset[i - lb]; j < ubc - ti;
                     j += 4, k += 4) {
                    x_reg.v = _mm256_loadu_pd((double *) (Bx + j + kk * bRows));
                    Lx_reg.v = _mm256_loadu_pd(
                            (double *) (Ax + k));// Skylake	7	0.5
                    result.v = _mm256_fmadd_pd(Lx_reg.v, x_reg.v,
                                               result.v);//Skylake	4	0.5
                }
                double t0 = 0;
                int jt = ubc - lbc - ti;
                for (int j = ubc - ti, k = offset[i - lb] + jt; j < ubc;
                     j++, k++) {
                    t0 += Ax[k] * Bx[j + kk * bRows];
                }
                auto h0 = hsum_double_avx(result.v);
                Cx[i + kk * bRows] += h0 + t0;
            }
        }
    }


    /**
* Computes y[lb:ub] += Lx[axo+axi:axo+axi*cb] * x[offset[0:cb]]
*
* @param Cx out solution
* @param Ax in nonzero locations
* @param Bx in the input vector
* @param offset in the starting point of each load from x
* @param axi distance to next row in matrix
* @param axo distance to first element in matrix
* @param lb in lower bound of rows
* @param ub in upper bound of rows
* @param cb in number of columns to compute
* @param cbb number of columns in matrix Bx
* @param cbd number of rows in matrix Bx
*/
    void psc_t2_2DC_gemm(double *Cx, const double *Ax, const double *Bx,
                         const int *offset, const int axi, const int axo,
                         const int lb, const int ub, const int cb,
                         const int cof, const int bRows, const int bCols) {
        auto ax0 = Ax + axo + axi * 0;
        auto ax1 = Ax + axo + axi * 1;
        auto x0 = Bx;
        auto x1 = x0 + cof;

        int co = (ub - lb) % 2;
        for (int i = lb; i < ub - co; i += 2) {
            for (int k = 0; k < bCols; k++) {
                auto r0 = _mm256_setzero_pd();
                auto r1 = _mm256_setzero_pd();

                int j = 0;
                for (; j < cb - 3; j += 4) {
                    __m256d xv0, xv1;
                    loadbx(offset, x0, j, xv0, msk0, bRows * k);
                    loadbx(offset, x1, j, xv1, msk1, bRows * k);

                    auto axv0 = _mm256_loadu_pd(ax0 + j);
                    auto axv1 = _mm256_loadu_pd(ax1 + j);

                    r0 = _mm256_fmadd_pd(axv0, xv0, r0);
                    r1 = _mm256_fmadd_pd(axv1, xv1, r1);
                }

                // Compute tail
                __m128d tail = _mm_loadu_pd(Cx + i + bRows * k);

                for (; j < cb; j++) {
                    tail[0] += ax0[j] * x0[offset[j] + bRows * k];
                    tail[1] += ax1[j] * x1[offset[j] + bRows * k];
                }

                // H-Sum
                auto h0 = _mm256_hadd_pd(r0, r1);
                __m128d vlow = _mm256_castpd256_pd128(h0);
                __m128d vhigh = _mm256_extractf128_pd(h0, 1);
                vlow = _mm_add_pd(vlow, vhigh);
                vlow = _mm_add_pd(vlow, tail);

                // Store
                _mm_storeu_pd(Cx + i + k * bRows, vlow);
            }

            // Load new addresses
            ax0 += axi * 2;
            ax1 += axi * 2;
            x0 += cof * 2;
            x1 += cof * 2;
        }

        if (co) {
            for (int k = 0; k < bCols; k++) {
                // Compute last iteration
                auto r0 = _mm256_setzero_pd();
                __m256d xv;
                int j = 0;
                for (; j < cb - 3; j += 4) {
                    loadbx(offset, x0, j, xv, msk, bRows * k);
                    auto axv0 = _mm256_loadu_pd(ax0 + j);
                    r0 = _mm256_fmadd_pd(axv0, xv, r0);
                }

                // Compute tail
                double tail = 0.;
                for (; j < cb; j++) {
                    tail += *(ax0 + j) * x0[offset[j] + bRows * k];
                }

                // H-Sum
                Cx[ub - 1 + bRows * k] += tail + hsum_double_avx(r0);
            }
        }
    }

    void psc_t3_1D1R_gemm(double *Cx, const double *Ax, const int *Ai,
                          const double *Bx, const int *offset, int lb, int fnl,
                          int cw, const int bRows, const int bCols) {
        v4df_t Lx_reg, result, x_reg;
        int i = lb;
        for (int kk = 0; kk < bCols; kk++) {
            int j = 0, k = fnl;
            result.v = _mm256_setzero_pd();
            for (; j < cw - 3; j += 4, k += 4) {
                x_reg.v = _mm256_set_pd(Bx[offset[j + 3] + kk * bRows],
                                        Bx[offset[j + 2] + kk * bRows],
                                        Bx[offset[j + 1] + kk * bRows],
                                        Bx[offset[j] + kk * bRows]);
                Lx_reg.v = _mm256_loadu_pd((double *) (Ax + k));
                result.v = _mm256_fmadd_pd(Lx_reg.v, x_reg.v, result.v);
            }
            double tail = 0;
            for (; j < cw; ++j, ++k) {
                tail += (Ax[k] * Bx[offset[j] + kk * bRows]);
            }
            auto h0 = hsum_double_avx(result.v);
            Cx[i + kk * bRows] += h0 + tail;
        }
    }
}