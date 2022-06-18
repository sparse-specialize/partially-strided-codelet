//
// Created by kazem on 6/17/22.
//

#ifndef DDT_SPMM_DL_DEMO_UTILS_H
#define DDT_SPMM_DL_DEMO_UTILS_H
#include "FusionDemo.h"
#include <DDT.h>
#include <DDTCodelets.h>
#include <Executor.h>
#include <sparse_io.h>

#include <mkl.h>

int tmp = 9;

class GEMMSpMM : public sym_lib::FusionDemo {
protected:
 float * Bx;
 int aRows, aCols, bCols;
 float *d_A, *Cx;
 float *correct_cx;

 void setting_up() override{
  auto dense_size = aRows * aCols;
  d_A = new float[dense_size]();
  sym_lib::compressed2dense<float,double>(L1_csr_->m, L1_csr_->n, L1_csr_->p,
                                          L1_csr_->i, L1_csr_->x,
                                          L1_csr_->stype, d_A);
  //sym_lib::print_vec("A: \n", 0, tmp, d_A);
  //sym_lib::print_vec("\n B: \n", 0, tmp, Bx);
 }

 sym_lib::timing_measurement fused_code() override {
  mkl_set_num_threads(num_threads_);
  mkl_set_num_threads_local(num_threads_);
  sym_lib::timing_measurement t1;
  t1.start_timer();
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                           aRows,  bCols, aCols,
                           1.,              // alpha
                           d_A,aCols,  // lda = t.k()
                           Bx, bCols,  // ldb = t.n()
                           0.,              // beta
                           Cx,bCols   // ldc = t.n()
  );
  t1.measure_elapsed_time();
  //sym_lib::print_vec("\n C: \n", 0, tmp, Cx);
  //            std::copy(x_,x_+L1_csr_->m*cbb,x_);
  return t1;
 }

 void testing() override{
  if(correct_cx)
   if (!sym_lib::is_equal<float>(0, aCols*bCols, correct_cx, Cx, 1e-6))
     PRINT_LOG(name_ + " code != reference solution.\n");
 }

public:
 GEMMSpMM(sym_lib::CSR *L, sym_lib::CSC *L_csc, int num_threads, int bCols,
            float *correct, std::string name)
   : FusionDemo(1, name) {
  L1_csr_ = L;
  L1_csc_ = L_csc;
  this->aRows = L_csc->m;
  this->aCols = L_csc->n;
  this->bCols = bCols;
  correct_cx = d_A = NULLPNTR;
  this->Bx = new float[aCols*bCols]();
  this->Cx = new float[aCols*bCols]();
  for (int i = 0; i < aCols*bCols; i++) {
   this->Bx[i] = 1;
  }
  num_threads_ = num_threads;
  num_test_ = 15;
  correct_cx = correct;
 };

 ~GEMMSpMM() override {
  delete[] Bx;
  delete[] Cx;
 }

 float *get_Cx(){
  return Cx;
 }
};

void gemm_base(const int m, const int bcol, const int n, const float *a,
                  const float *b, float *c){
 for(int i=0; i<m; i++){
  for(int j=0; j<bcol; j++){
   c[i*bcol+j] = 0;
   for(int k=0; k<n; k++)
    c[i*bcol+j]+=a[i*n+k]*b[k+bcol+j];//c[i][j]+=a[i][k]*b[k][j];
  }
 }
}
struct spmm_config{// mxn -> A in C = A*B
 int m_tile, bcol_tile, n_tile;
};
void gemm_tuned_1(const int m, const int bcol, const int n, const float *a,
                  const float *b, float *c){
 for(int i=0; i<m; i++){
  for(int j=0; j<bcol; j++){
   c[i*bcol+j] = 0;
   for(int k=0; k<n; k++){
    c[i*bcol+j]+=a[i*n+k]*b[k+bcol+j];//c[i][j]+=a[i][k]*b[k][j];
   }
  }
 }
}

class GEMMSpMMTuned : public GEMMSpMM{
 sym_lib::timing_measurement fused_code() override {
  mkl_set_num_threads(num_threads_);
  mkl_set_num_threads_local(num_threads_);
  sym_lib::timing_measurement t1;
  t1.start_timer();
  gemm_tuned_1(aRows, bCols, aCols, d_A, Bx, Cx);
  t1.measure_elapsed_time();
  //sym_lib::print_vec("\n C2: \n", 0, tmp, Cx);
  return t1;
 }
public:
 GEMMSpMMTuned(sym_lib::CSR *L, sym_lib::CSC *L_csc, int num_threads, int bCols,
   float *correct, std::string name): GEMMSpMM(L,L_csc,num_threads,bCols,
                                              correct, name){}
};

#endif //DDT_SPMM_DL_DEMO_UTILS_H
