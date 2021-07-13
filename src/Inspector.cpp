//
// Created by Kazem on 7/12/21.
//

#include "DDT.h"
#include "Inspector.h"

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
      default:
        break;
    }
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
   * @brief Inspects the pattern DAG and creates run-time codelet structs
   *
   * @param d  Global DDT object containing pattern information
   * @param cl List of runtime codelet descriptions 
   */
  void inspectCodelets(DDT::GlobalObject& d, std::vector<Codelet*>& cl) {
    int TPR = 3;

    for (int i = d.mt.ips-1; i >= 0; i--) {
      for (int j = 0; j < d.mt.ip[i+1]-d.mt.ip[i];) {
        int cn = ((d.mt.ip[i]+j)-d.mt.ip[0]) / TPR;
        if (d.c[cn].pt != nullptr && d.c[cn].ct != nullptr) {
          // Generate (TYPE_FSC|TYPE_PSC1|TYPE_PSC2)
          generateCodelet(d, d.c+cn, cl);
          j += (d.c[cn].sz+1) * TPR;
        } else if (d.c[cn].ct != nullptr) {
          // Generate (TYPE_PSC3)
          j += findType3Bounds(d,i,j,d.mt.ip[i+1]-d.mt.ip[i]);
          cn = ((d.mt.ip[i]+(j-TPR))-d.mt.ip[0]) / TPR;
          generateCodelet(d, d.c+cn, cl);
        } else {
          j += TPR * (d.c[cn].sz + 1);
        }
      }
    }
  }


 void free(std::vector<DDT::Codelet*>& cl){
  for (auto & i : cl) {
   delete i;
  }
 }
}
