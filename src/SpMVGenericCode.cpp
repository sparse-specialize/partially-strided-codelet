//
// Created by kazem on 7/12/21.
//

#include <vector>
#include <GenericCodelets.h>
#include "SpMVGenericCode.h"

namespace DDT{

 void spmv_generic(const int n, const int *Ap, const int *Ai, const double
 *Ax, const double *x, double *y, const std::vector<Codelet*>& lst){
  for(auto c : lst){
   switch (c->get_type()) {
    case CodeletType::TYPE_FSC:
     break;
    case CodeletType::TYPE_PSC1:

//     psc_t1_2D4R(y,Ax,x,c->offsets,c->lbr,c->lbr+c->row_width,c->lbc,
//                 c->lbc+c->col_width);
      break;
    case CodeletType::TYPE_PSC2:
     break;
    case CodeletType::TYPE_PSC3:
     break;
    default:
     break;
   }
  }
 }

}
