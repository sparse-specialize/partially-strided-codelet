//
// Created by kazem on 6/17/22.
//

#ifndef DDT_SPMM_DL_DEMO_UTILS_H
#define DDT_SPMM_DL_DEMO_UTILS_H
#include "FusionDemo.h"
#include <DDT.h>
#include <DDTCodelets.h>
#include <Executor.h>



class GEMMSpMM : public sym_lib::FusionDemo {
protected:
 float * Bx;
 int bRows, bCols;
 float *d_A, *Cx;
 float *correct_cx;

 void setting_up() override{
  auto dense_size = L1_csc_->m * L1_csc_->n;
  d_A = new float[dense_size]();
  sym_lib::csc2dense<float>(L1_csc_, d_A);
 }

 sym_lib::timing_measurement fused_code() override {
  sym_lib::timing_measurement t1;
  t1.start_timer();
  // spmm_csr_csr(x_, L1_csr_->x, Bx, L1_csr_->m, L1_csr_->p, L1_csr_->i,
  //              bRows, bCols);
  t1.measure_elapsed_time();
  //            std::copy(x_,x_+L1_csr_->m*cbb,x_);
  return t1;
 }

 virtual void testing() override{
  if(correct_cx)
   if (!sym_lib::is_equal<float>(0, bCols*bRows, correct_cx, Cx, 1e-6))
     PRINT_LOG(name_ + " code != reference solution.\n");
 }

public:
 GEMMSpMM(sym_lib::CSR *L, sym_lib::CSC *L_csc, int bRows, int bCols,
            std::string name)
   : FusionDemo(1, name) {
  L1_csr_ = L;
  L1_csc_ = L_csc;
  this->Bx = new float[bRows*bCols];
  this->Cx = new float[bRows*bCols];
  for (int i = 0; i < bRows*bCols; i++) {
   this->Bx[i] = 1;
  }
  this->bRows = bRows;
  this->bCols = bCols;
 };

 ~GEMMSpMM() override {
  delete[] Bx;
  delete[] Cx;
 }


};
#endif //DDT_SPMM_DL_DEMO_UTILS_H
