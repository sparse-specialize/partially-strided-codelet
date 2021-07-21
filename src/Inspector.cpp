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
      case DDT::TYPE_PSC3V2:
        generateCodeletType<DDT::TYPE_PSC3V2>(d,c,cl);
        break;
      default:
        break;
    }
  }

  /**
   * @brief Finds bounds for type3v2 codelets
   *
   * @param ip
   * @param c
   * @param i
   * @param adj
   * @return
   */
  int findType3v2Bounds(int** ip, DDT::PatternDAG* c, int i) {
      int cnl = ((ip[i])-ip[0]) / TPR;
      while (hasAdacentIteration(i--, ip)/* && nCodelets == 0 */) {
          int cn = ((ip[i])-ip[0]) / TPR;
          c[cnl].pt = c[cn].ct;
          cnl = cn;
      }
      c[cnl].t  = DDT::TYPE_PSC3V2;

      return i;
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
    void pruneIterations(int** ip, int ips, bool* adj) {
        // @TODO: CALCULATE MIN-MAX OVERLAP TO GET DIMENSION OF REUSE
        for (int i = 0; i < ips-1; i++) {
            if (ip[i+1]-ip[i] == 1) {
                adj[i+1] = true;
            }
            while (ip[i+1] - ip[i] == 1) {
            }
        }
    }

    inline bool hasAdacentIteration(int i, int** ip) {
        return ip[i][0] - ip[i-1][0];
    }

    void generateCodeletsFromParallelDag(DDT::GlobalObject& d, const DDT::Config& cfg) {
        //#pragma omp parallel for num_threads(c.nThread) default(none) shared(TPR, cl, d, c, std::cout)
        for (int t = 0; t < d.sm->_final_level_no; t++) {
            for (int ii = 0; ii < d.sm->_wp_bounds[t]; ++ii) {
                auto tr = d.t[t][ii];
                auto& cc = d.sm->_cl[t][ii];
                for (int i = tr->_ni-1; i >= 0; --i) {
                    for (int j = 0; j < tr->_iter_pt[i+1]-tr->_iter_pt[i]; ++j) {
                        int cn = ((tr->_iter_pt[i] + j) - tr->_iter_pt[0]) / TPR;
                        if (tr->_c[cn].pt != nullptr && tr->_c[cn].ct != nullptr) {
                            // Generate (TYPE_FSC | TYPE_PSC1 | TYPE_PSC2)
                            //  generateCodelet(d, d.c + cn, cc);
                            // j += (d.c[cn].sz + 1) * TPR;
                        } else if (tr->_c[cn].ct != nullptr) { // Generate (TYPE_PSC3)
                            if (/*nCodelets == 0 && */ hasAdacentIteration(i, tr->_iter_pt)) {
                                i  = findType3v2Bounds(tr->_iter_pt, tr->_c, i);
                                cn = ((tr->_iter_pt[i] + j) - tr->_iter_pt[0]) / TPR;
                                generateCodelet(d, tr->_c + cn, cc);
                            } else {
                                j += findType3Bounds(
                                        d, i, j, d.mt.ip[i + 1] - d.mt.ip[i]);
                                cn = ((tr->_iter_pt[i] + j) - tr->_iter_pt[0]) / TPR;
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
    }

    /**
     * @brief Generates runtime information for executor codes
     *
     * @param d Object containing inspector information
     * @param cl Pointer to containers for codelets for each thread
     * @param cfg Configuration object for inspector/executor
     */
    void generateCodeletsFromPattern(DDT::GlobalObject& d, std::vector<Codelet*>* cl, const DDT::Config& cfg) {
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
                        j += findType3Bounds(d, i, j, d.mt.ip[i + 1] - d.mt.ip[i]);
                        cn = ((d.mt.ip[i] + (j - TPR)) - d.mt.ip[0]) / TPR;
                        generateCodelet(d, d.c + cn, cc);
                    } else {
                        j += TPR * (d.c[cn].sz + 1);
                    }
                }
            }
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
                // Calculate overlap for each iteration
                DDT::pruneIterations(d.t[i][j]->_iter_pt, d.t[i][j]->_ni, 0);
#ifdef O3
                // Compute first order differences
                DDT::computeParallelizedFOD(d.t[i][j]->_iter_pt, d.t[i][j]->_ni, d.d, 1);

                // Mine trace for codelets
                DDT::mineDifferences(d.mt.ip, d.c, d.d, cfg.nThread, d.tb);
#endif
                // Generate codelets from pattern DAG
                DDT::generateCodeletsFromParallelDag(d, cfg);

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
      DDT::pruneIterations(d.mt, 10);

      // Compute first order differences
      DDT::computeParallelizedFOD(d.mt.ip, d.mt.ips, d.d, cfg.nThread);

      // Mine trace for codelets
      DDT::mineDifferences(d.mt.ip, d.c, d.d, cfg.nThread, d.tb);

      // Generate codelets from pattern DAG
      DDT::generateCodeletsFromPattern(d, cl, cfg);

      std::sort(cl[0].begin(), cl[0].end(), [](DDT::Codelet* lhs, DDT::Codelet* rhs) {
          return lhs->first_nnz_loc < rhs->first_nnz_loc;
      });
  }


 void free(std::vector<DDT::Codelet*>& cl){
  for (auto & i : cl) {
   delete i;
  }
 }
}
