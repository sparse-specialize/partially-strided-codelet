//
// Created by kazem on 7/13/21.
//

#ifndef DDT_SPTRSVMODEL_H
#define DDT_SPTRSVMODEL_H

#include "PolyModel.h"

namespace sparse_avx{

 class SpTRSVModel : public SectionPolyModel {
  int _num_rows{}, _num_cols{}, _nnz{};
  int *_Ap{}, *_Ai{};

 public:
  SpTRSVModel();
  SpTRSVModel(int n, int m, int nnz, int *Ap, int *Ai);

  Trace* generate_trace() override;
  Trace** generate_trace(int num_threads) override;
 };

}


#endif //DDT_SPTRSVMODEL_H
