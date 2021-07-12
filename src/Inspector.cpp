//
// Created by kazem on 7/12/21.
//

#include "DDT.h"
#include "Inspector.h"

namespace DDT {
    void generateCodelet(DDT::GlobalObject& d, DDT::PatternDAG* c, std::vector<Codelet>& cl) {
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
            default:
                break;
        }
    }

  void inspectCodelets(DDT::GlobalObject& d, std::vector<Codelet>& cl) {
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
          j += TPR;
        } else {
          j += TPR * (d.c[cn].sz + 1);
        }
      }
    }
  }
}
