//
// Created by Kazem on 7/12/21.
//

#include "Inspector.h"
#include "DDT.h"
#include "PatternMatching.h"
#include <algorithm>

namespace DDT {

    void FSCCodelet::print() {
      std::cout<<"FSC: y["<<lbr<<":"<<lbr+row_width<<"] = Ax["<<first_nnz_loc<<","
      <<row_offset<<"]*x["<<lbc<<":"<<lbc+col_width<<"]\n";
    }

    void PSCT1V1::print() {
     std::cout<<"T1: y["<<lbr<<":"<<lbr+row_width<<"] = Ax["<<offsets[0]<<","
              <<offsets[row_width]<<"]*x["<<lbc<<":"<<lbc+col_width<<"]\n";
    }

    void PSCT2V1::print() {
      std::cout<<"T2: y["<<lbr<<":"<<lbr+row_width<<"] = Ax["<<first_nnz_loc<<","
              <<first_nnz_loc+row_offset<<"]*x["<<offsets[0]<<":"<<offsets
              [col_width]<<"]\n";
    }

    void PSCT3V1::print() {
     std::cout<<"T3: y["<<lbr<<"] = Ax["<<first_nnz_loc<<","
              <<first_nnz_loc+col_width<<"]*x["<<offsets[0]<<":"<<offsets
              [col_width]<<"]\n";
    }

 void PSCT3V2::print() {
  std::cout<<"NOT IMPLEMENTED\n";
 }

 void PSCT3V3::print() {
  std::cout<<"NOT IMPLEMENTED\n";
 }

    void generateFullRowCodeletType(int i, int** ip, int ips, DDT::PatternDAG* c, DDT::PatternDAG* cc, std::vector<Codelet*>& cl) {
        int TPR = 3;
        auto type = cc->t;
        // Get codelet type
        if (type == DDT::TYPE_PSC3) {
            int colWidth = ((cc->pt - cc->ct) / TPR) + 1;

            int oo = cc->ct[0];
            int mo = cc->ct[1];
            auto cornerT = cc->ct+(colWidth-1)*TPR;
            assert(cornerT[0] == cornerT[2]);

            int cnt = 0;
            auto o = new int[colWidth]();
            while (cc->ct != cc->pt) {
                o[cnt++] = cc->ct[2];
                cc->ct += TPR;
            }
            o[cnt] = cc->ct[2];


            cl.emplace_back(new DDT::PSCT3V1(oo, colWidth, mo,o,true));
        }

        if (type == DDT::TYPE_PSC3_V1) {
            int rowCnt = 1;
            auto ccc = cc;
            while (ccc->ct != ccc->pt) {
                int nc = (ccc->pt - ip[0])/TPR;
                ccc = c + nc;
                rowCnt++;
            }
            auto ro = new int[rowCnt]();
            auto rls = new int[rowCnt]();

            assert((ip[i+1] - ip[i]) / TPR > 4);

            int cnt = rowCnt-1;
            while (cc->ct != cc->pt) {
                ro[cnt] = cc->ct[1];
                rls[cnt--] = (ip[i+1] - ip[i])/TPR;
                int nc = (cc->pt - ip[0])/TPR;
                cc = c + nc;
                --i;
            }
            ro[cnt] = cc->ct[1];
            rls[cnt] = (ip[i+1] - ip[i])/TPR;

            int oo = cc->ct[0];
            int vo = cc->ct[2];

            cl.emplace_back(new DDT::PSCT3V2(oo, vo, rowCnt, cc->sz, 0, ro, rls));
        }

        if (type == DDT::TYPE_PSC3_V2) {
            int rowCnt = 1;
            auto ccc = cc;
            while (ccc->ct != ccc->pt) {
                int nc = (ccc->pt - ip[0])/TPR;
                ccc = c + nc;
                rowCnt++;
            }
            auto ro = new int[rowCnt]();
            auto rls = new int[rowCnt]();
            auto rid = new int[rowCnt]();

            int cnt = rowCnt-1;
            while (cc->ct != cc->pt) {
                rid[cnt] = cc->ct[0];
                ro[cnt] = cc->ct[1];
                rls[cnt--] = (ip[i+1] - ip[i])/TPR;
                int nc = (cc->pt - ip[0])/TPR;
                cc = c + nc;
                --i;
            }
            rid[cnt] = cc->ct[0];
            ro[cnt] = cc->ct[1];
            rls[cnt] = (ip[i+1] - ip[i])/TPR;

            int oo = cc->ct[0];
            int vo = cc->ct[2];

            cl.emplace_back(new DDT::PSCT3V3(oo, vo, rowCnt, cc->sz, 0, ro, rls, rid));
        }
    }

 int TPR = 3;
 /**
  * @brief Generates run-time codelet object based on type in DDT::PatternDAG
  *
  * @param d  Global DDT object containing pattern information
  * @param c  Codelet to turn into run-time object
  * @param cl List of runtime codelet descriptions
  */
  void generateCodelet(DDT::GlobalObject& d, DDT::PatternDAG* c, std::vector<Codelet*>& cl) {
    switch (c->t) {
      case DDT::TYPE_FSC:
        generateCodeletType<DDT::TYPE_FSC>(d,c,cl);
        break;
      case DDT::TYPE_PSC1:
        generateCodeletType<DDT::TYPE_PSC1>(d,c,cl);
        break;
      case DDT::TYPE_PSC2:
        generateCodeletType<DDT::TYPE_PSC2>(d,c,cl);
        break;
      case DDT::TYPE_PSC3:
        generateCodeletType<DDT::TYPE_PSC3>(d,c,cl);
        break;
      case DDT::TYPE_PSC3_V1:
        generateCodeletType<DDT::TYPE_PSC3_V1>(d,c,cl);
        break;
      case DDT::TYPE_PSC3_V2:
        generateCodeletType<DDT::TYPE_PSC3_V2>(d,c,cl);
        break;
      default:
        break;
    }
  }

  inline int nnzLessThan(int ii, int** ip, int ub) {
      auto tr = (ip[ii+1]-ip[ii])/TPR;
      for (int i = 0; i < tr; i++) {
          auto ro = ip[ii][i*3+2];
          if (ub <= ro) {
              return i;
          }
      }
      return ip[ii+1]-ip[ii];
  }

  /**
   * @brief Finds bounds for PSC3v3
   * @param ip
   * @param c
   * @param i
   * @return
   */
  int findType3VBounds(int** ip, DDT::PatternDAG* c, int i) {
      int cnl = ((ip[i])-ip[0]) / TPR;
      int cnlt = cnl;
      int VW = 4; // Vector width

      if (0 == i) {
          int cn = ((ip[i + 1]) - ip[0]) / TPR;
          c[cnl].t = DDT::TYPE_PSC3;
          c[cnl].pt = c[cn-1].ct;
          return i;
      }


      int NNZ_P_R = 5;
      int ii = i;
      bool hai = hasAdacentIteration(ii, ip);
      auto TH = nnzLessThan(ii, ip,ip[ii][-1]);
      TH = TH - (TH % VW);

      if (hai && TH) {
          while (ii > 0 &&
                 nnzInIteration(ii-1, ip) > TH &&
                 hasAdacentIteration(ii, ip) &&
                 nnzLessThan(ii, ip, ip[ii][-1]) > TH
                 /* && nCodelets == 0 */ ) {
              ii--;
          }
          if (i != ii) {
              int iii = i;
              ii += (i - (ii - 1)) % 2;
              while (iii-- != ii) {
                  int cn = ((ip[iii]) - ip[0]) / TPR;
                  c[cnl].pt = c[cn].ct;
                  cnl = cn;
              }
              assert((i-ii+1)%2 == 0);
          }
      } else if (TH) {
          while (ii > 0 &&
                 nnzInIteration(ii-1, ip) > TH  &&
                 nnzLessThan(ii, ip, ip[ii][-1]) >TH
                 /* && nCodelets == 0 */) {
              ii--;
          }
          if (i != ii) {
              int iii = i;
              ii += (i - (ii - 1)) % 2;
              while (iii-- != ii) {
                  int cn = ((ip[iii]) - ip[0]) / TPR;
                  c[cnl].pt = c[cn].ct;
                  cnl = cn;
              }
              assert((i-ii+1)%2 == 0);
          }
      }

      if (ii == i) {
          // TODO: FIX HERE FOR PREVIOUS TYPE
          int cn = ((ip[ii+1]) - ip[0]) / TPR;
          c[cnlt].t = DDT::TYPE_PSC3;
          c[cnl].pt = c[cn-1].ct;
      } else if (hai) {
          c[cnl].pt = c[cnl].ct;
          c[cnlt].t  = DDT::TYPE_PSC3_V1;
          c[cnl].sz = TH;
      } else {
          c[cnl].pt = c[cnl].ct;
          c[cnlt].t  = DDT::TYPE_PSC3_V2;
          c[cnl].sz = TH;
      }

      return ii;
  }

  /**
   * Finds the continuous bounds of a TYPE_PSC3
   * @param d
   * @param i
   * @param j
   * @param jBound
   * @return
   */
  int findType3Bounds(DDT::GlobalObject& d, int i, int j, int jBound) {
      int cn = ((d.mt.ip[i]+j)-d.mt.ip[0]) / TPR;
      auto pscb = d.c[cn].ct;
      int jStart = j;

      while (j < jBound && d.c[cn].ct != nullptr && d.c[cn].pt == nullptr) {
          j += TPR;
          cn = ((d.mt.ip[i]+j)-d.mt.ip[0]) / TPR;
      }
      cn = ((d.mt.ip[i]+(j-TPR))-d.mt.ip[0]) / TPR;
      d.c[cn].t = DDT::TYPE_PSC3;
      d.c[cn].pt = pscb;

      return j - jStart;
  }

/**
 * @brief Different iteration sections processed differently
 *
 */
    void pruneIterations(int** ip, int ips) {
        // @TODO: CALCULATE MIN-MAX OVERLAP TO GET DIMENSION OF REUSE
        for (int i = 0; i < ips-1; i++) {
            if (ip[i+1]-ip[i] == 1) {
//                adj[i+1] = true;
            }
//            while (ip[i+1] - ip[i] == 1) {
//            }
        }
    }

    inline bool hasAdacentIteration(int i, int** ip) {
        return (ip[i][0] - ip[i-1][0]) == 1;
    }

    inline int nnzInIteration(int i, int** ip) {
        return (ip[i+1] - ip[i]) / TPR;
    }

    void generateCodeletsFromParallelDag(const sparse_avx::Trace* tr, std::vector<DDT::Codelet*>& cc, const DDT::Config& cfg) {
                for (int i = tr->_ni-1; i >= 0; --i) {
                    for (int j = 0; j < tr->_iter_pt[i+1]-tr->_iter_pt[i]; ++j) {
                        int cn = ((tr->_iter_pt[i] + j) - tr->_iter_pt[0]) / TPR;
                        if (tr->_c[cn].pt != nullptr && tr->_c[cn].ct != nullptr) {
                            // Generate (TYPE_FSC | TYPE_PSC1 | TYPE_PSC2)
                            //  generateCodelet(d, d.c + cn, cc);
                            // j += (d.c[cn].sz + 1) * TPR;
                        } else if (tr->_c[cn].ct != nullptr) { // Generate (TYPE_PSC3)
                            if (/* nCodelets == 0*/ true) {
                                int iEnd = findType3VBounds(tr->_iter_pt, tr->_c, i);
                                cn = ((tr->_iter_pt[i]) - tr->_iter_pt[0]) / TPR;
                                generateFullRowCodeletType(i, tr->_iter_pt, tr->_ni, tr->_c, tr->_c + cn, cc);
                                i = iEnd;
                                break;
                            } else {
                                // j = findType3Bounds(d, i, j, jbound)
                            }
                        } else {
                            j += TPR * (tr->_c[cn].sz + 1);
                        }
                    }
                }
                std::sort(cc.begin(), cc.end(), [](DDT::Codelet* lhs, DDT::Codelet* rhs) {
                  return lhs->first_nnz_loc < rhs->first_nnz_loc;
                });
            }

    /**
     * @brief Generates runtime information for executor codes
     *
     * @param d Object containing inspector information
     * @param cl Pointer to containers for codelets for each thread
     * @param cfg Configuration object for inspector/executor
     */
    void generateCodeletsFromSerialDAG(DDT::GlobalObject& d, std::vector<Codelet*>* cl, const DDT::Config& cfg) {
        //#pragma omp parallel for num_threads(c.nThread) default(none) shared(TPR, cl, d, c, std::cout)
        for (int t = 0; t < cfg.nThread; t++) {
            auto& cc = cl[t];
            for (int i = d.tb[t+1]-1; i >= d.tb[t]; i--) {
                for (int j = 0; j < d.mt.ip[i + 1] - d.mt.ip[i];) {
                    int cn = ((d.mt.ip[i] + j) - d.mt.ip[0]) / TPR;
                    if (d.c[cn].pt != nullptr && d.c[cn].ct != nullptr) {
                        // Generate (TYPE_FSC | TYPE_PSC1 | TYPE_PSC2)
                        generateCodelet(d, d.c + cn, cc);
                        j += (d.c[cn].sz + 1) * TPR;
                    } else if (d.c[cn].ct != nullptr) {
                        // Generate (TYPE_PSC3)
                        if (/* nCodelets == 0*/ false) {
                            int iEnd = findType3VBounds(d.mt.ip, d.c, i);
                            cn = ((d.mt.ip[i]) - d.mt.ip[0]) / TPR;
                            generateFullRowCodeletType(i, d.mt.ip, d.mt.ips, d.c, d.c + cn, cc);
                            i = iEnd;
                            break;
                        } else {
                            j += findType3Bounds(d, i, j,
                                                 d.mt.ip[i + 1] - d.mt.ip[i]);
                            cn = ((d.mt.ip[i] + (j - TPR)) - d.mt.ip[0]) / TPR;
                            generateCodelet(d, d.c + cn, cc);
                        }
                    } else {
                        j += TPR * (d.c[cn].sz + 1);
                    }
                }
            }
            std::sort(cc.begin(), cc.end(), [](DDT::Codelet* lhs, DDT::Codelet* rhs) {
              return lhs->first_nnz_loc < rhs->first_nnz_loc;
            });
        }
    }

    /**
     * @brief Inspects a trace partitioned into parallel segments
     *
     * @param d
     * @param cl
     * @param cfg
     */
    void inspectParallelTrace(DDT::GlobalObject& d, const DDT::Config& cfg) {
        for (int i = 0; i < d.sm->_final_level_no; i++) {
            for (int j = 0; j < d.sm->_wp_bounds[i]; ++j) {
                auto tr = d.t[i][j];
                auto& cc = d.sm->_cl[i][j];

                // Calculate overlap for each iteration
                // DDT::pruneIterations(d.t[i][j]->_iter_pt, d.t[i][j]->_ni);
#ifdef O3
                // Compute first order differences
                DDT::computeParallelizedFOD(d.t[i][j]->_iter_pt, d.t[i][j]->_ni, d.d, 1);

                // Mine trace for codelets
                DDT::mineDifferences(d.mt.ip, d.c, d.d, cfg.nThread, d.tb);
#endif
                // Generate codelets from pattern DAG
                DDT::generateCodeletsFromParallelDag(tr, cc, cfg);
            }
        }
    }

  /** 
   * @brief Generates the pattern DAG and creates run-time codelets
   *
   * @param d  Global DDT object containing pattern information
   * @param cl List of runtime codelet descriptions 
   */
  void inspectSerialTrace(DDT::GlobalObject& d, std::vector<Codelet*>* cl, const DDT::Config& cfg) {
      // Calculate overlap for each iteration
      DDT::pruneIterations(d.mt.ip, d.mt.ips);
//#ifdef O3
      if (cfg.op == DDT::OP_SPMV) {
          // Compute first order differences
          DDT::computeParallelizedFOD(d.mt.ip, d.mt.ips, d.d, cfg.nThread);

          // Mine trace for codelets
          DDT::mineDifferences(d.mt.ip, d.c, d.d, cfg.nThread, d.tb);
      }
//#endif
      // Generate codelets from pattern DAG
      DDT::generateCodeletsFromSerialDAG(d, cl, cfg);

  }


 void free(std::vector<DDT::Codelet*>& cl){
  for (auto & i : cl) {
   delete i;
  }
 }
}
