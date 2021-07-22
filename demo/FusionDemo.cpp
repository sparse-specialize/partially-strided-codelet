//
// Created by Kazem on 11/10/19.
//
#define DBG_LOG
#define CSV_LOG
#define PROFILE

#include <cstring>
#include <cmath>
#include <utils.h>
#include "FusionDemo.h"
#ifdef METIS
#include <metis_interface.h>
#endif


namespace sym_lib{



 template<class type>
 bool is_float_equal(const type x, const type y, double absTol, double relTol) {
     return std::abs(x - y) <= std::max(absTol, relTol * std::max(std::abs(x), std::abs(y)));
 }

 template<class type>
 bool is_generic_equal(const type x, const type y, double eps) {
     return std::abs(x - y) > eps;
 }

 template<class type>
 bool is_equal(int beg_idx, int end_idx, const type* vec1, const type* vec2,
               double eps=1e-8){
  for (int i = beg_idx; i < end_idx; ++i) {
   if (std::isnan(vec1[i]) || std::isnan(vec2[i]))
    return false;
   if constexpr (std::is_same_v<type, double> || std::is_same_v<type, float>) {
        if (!is_float_equal(vec1[i],vec2[i], eps, eps)) { std::cout << i << '\n' << vec1[i] << "," << vec2[i] << std::endl; return false; }
   } else {
       if (!is_generic_equal(vec1[i],vec2[i], eps))
           return false;
   }
  }
  return true;
 }


 FusionDemo::FusionDemo():L1_csr_(NULLPNTR), L1_csc_(NULLPNTR),
                          L2_csr_(NULLPNTR), L2_csc_(NULLPNTR),
                          A_csr_(NULLPNTR),A_csc_(NULLPNTR),
                          x_(NULLPNTR),
                          x_in_(NULLPNTR), correct_x_(NULLPNTR){
  num_test_=9;
  redundant_nodes_=0;
#ifdef PROFILE
  pw_ = NULLPNTR;
#endif
  }

  FusionDemo::FusionDemo(int n, std::string name):FusionDemo() {
  n_ = n;
  name_ = name;
  x_in_ = new double[n]();
  x_ = new double[n]();
 }

#ifdef PROFILE
  FusionDemo::FusionDemo(int n, std::string name, PAPIWrapper *pw):FusionDemo(n, name){
   pw_ = pw;
 }

 FusionDemo::FusionDemo(CSR *L, CSC* L_csc, CSR *A, CSC *A_csc,
                        double *correct_x, std::string name, PAPIWrapper *pw):
   FusionDemo(L->n,name, pw){
  L1_csr_ = L;
  L1_csc_ = L_csc;
  A_csr_ = A;
  A_csc_ = A_csc;
  correct_x_ = correct_x;
 }
#endif

  FusionDemo::~FusionDemo() {
  delete []x_in_;
  delete []x_;
 }

 void FusionDemo::setting_up() {
  std::fill_n(x_in_,n_,1);
  std::fill_n(x_,n_,0.0);
 }

 void FusionDemo::testing() {
  if(correct_x_)
   if (!is_equal(0, n_, correct_x_, x_,1e-8))
    PRINT_LOG(name_ + " code != reference solution.\n");
 }


 timing_measurement FusionDemo::evaluate() {
  timing_measurement median_t;
  std::vector<timing_measurement> time_array;
  analysis_time_.start_timer();
  build_set();
  analysis_time_.measure_elapsed_time();
  for (int i = 0; i < num_test_; ++i) {
   setting_up();
#ifdef PROFILE
   if(pw_)
    pw_->begin_profiling();
#endif
   timing_measurement t1 = fused_code();
#ifdef PROFILE
   if(pw_)
    pw_->finish_profiling();
#endif
   time_array.emplace_back(t1);
  }
  testing();

  median_t = sym_lib::time_median(time_array);
/*  for (int j = 0; j < time_array.size(); ++j) {
   std::cout<<" :"<<time_array[j].elapsed_time<<";";
  }
  std::cout<<"\n";*/
  return median_t;
 }


 timing_measurement FusionDemo::analysisTime() {
  return analysis_time_;
 }



 void print_common_header(){
 PRINT_CSV("Matrix Name,A Dimension,A Nonzero,L Nonzero,Code Type,Data Type,"
           "Metis Enabled,Number of Threads");
 }

 void print_common(std::string matrix_name, std::string variant, std::string strategy,
   CSC *B, CSC *L, int num_threads){
  PRINT_CSV(matrix_name);
  PRINT_CSV(B->m);
  PRINT_CSV(B->nnz);
  if(L)
   PRINT_CSV(L->nnz);
  PRINT_CSV(variant);
  PRINT_CSV(strategy);
#ifdef METIS
  PRINT_CSV("Metis");
#else
  PRINT_CSV("No Metis");
#endif
  PRINT_CSV(num_threads);
 }



}
