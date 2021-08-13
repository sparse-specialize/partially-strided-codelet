//
// Created by cetinicz on 2021-08-07.
//

#ifndef DDT_ANALYZER_H
#define DDT_ANALYZER_H

#include "DDT.h"
#include "DDTDef.h"

namespace DDT {
    void analyzeData(const DDT::GlobalObject& d,  std::vector<DDT::Codelet*>** cll, const DDT::Config& config);
    void analyzeData(const DDT::GlobalObject& d, const std::vector<DDT::Codelet*>* cll, const DDT::Config& config);
}
#endif//DDT_ANALYZER_H
