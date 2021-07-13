/*
 * =====================================================================================
 *
 *       Filename:  Executor.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2021-07-13 09:31:19 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */

#include "DDT.h"
#include "Inspector.h"

#include <vector>


#ifndef DDT_EXECUTOR
#define DDT_EXECUTOR

namespace DDT {

void executeCodelets(const std::vector<DDT::Codelet*>& cl, const DDT::Config c);

}

#endif  // DDT_EXECUTOR
