//
// Created by Kazem on 11/10/19.
//
#define DBG_LOG
#define CSV_LOG

#include <cstring>
#include "FusionDemo.h"
#ifdef METIS
#include <metis_interface.h>
#endif


namespace sym_lib{


 bool time_cmp(timing_measurement a, timing_measurement b){
  return a.elapsed_time<b.elapsed_time;}

 timing_measurement
 time_median(std::vector<timing_measurement> time_array){
  size_t n = time_array.size();
  if(n==0){
   timing_measurement t;
   t.elapsed_time=-1;
   return t;
  }
  std::sort(time_array.begin(),time_array.end(),time_cmp);
  if(n==1)
   return time_array[0];
  return time_array[n/2];
 }

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
   if(std::isnan(vec1[i]) || std::isnan(vec2[i]))
    return false;
   if constexpr (std::is_same_v<type, double> || std::is_same_v<type, float>) {
        if (!is_float_equal(vec1[i],vec2[i], eps, eps)) { std::cout << i << std::endl; return false; }
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
   if (!is_equal(0, n_, correct_x_, x_,1e-1))
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

  median_t = time_median(time_array);
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


 // TODO eventually use CSC / CSR class
 // TODO test for these two format change
 int
 csc_to_csr(int nrow, int ncol, int *Ap, int *Ai, double *Ax, int *&rowptr,
            int *&colind, double *&values) {
  // count row entries to generate row ptr
  int nnz = Ap[ncol];
  int *rowCnt = new int[nrow]();
  for (int i = 0; i < nnz; i++)
   rowCnt[Ai[i]]++;

  rowptr = new int[nrow + 1]();
  int counter = 0;
  for (int i = 0; i < nrow; i++) {
   rowptr[i] = counter;
   counter += rowCnt[i];
  }
  rowptr[nrow] = nnz;

  colind = new int[nnz]();
  values = new double[nnz]();

  memset(rowCnt, 0, sizeof(int) * nrow);
  for (int i = 0; i < ncol; i++) {
   for (int j = Ap[i]; j < Ap[i + 1]; j++) {
    int row = Ai[j];
    int index = rowptr[row] + rowCnt[row];
    colind[index] = i;
    values[index] = Ax[j];
    rowCnt[row]++;
   }
  }
  delete[]rowCnt;

  return 0;
 }


 CSR* csc_to_csr(CSC* A) {
  // count row entries to generate row ptr
  int nnz = A->p[A->n];
  int *rowCnt = new int[A->n]();
  for (int i = 0; i < nnz; i++)
   rowCnt[A->i[i]]++;

  CSR *B = new CSR(A->n,A->m,A->nnz,A->is_pattern);
  int *rowptr = B->p; //new int[nrow + 1]();
  size_t ncol = B->n;
  size_t nrow = B->m;
  int counter = 0;
  for (int i = 0; i < (int)nrow; i++) {
   rowptr[i] = counter;
   counter += rowCnt[i];
  }
  rowptr[nrow] = nnz;

  int *colind = B->i;
  double *values = B->x;

  memset(rowCnt, 0, sizeof(int) * nrow);
  for (int i = 0; i < (int)ncol; i++) {
   for (int j = A->p[i]; j < A->p[i + 1]; j++) {
    int row = A->i[j];
    int index = rowptr[row] + rowCnt[row];
    colind[index] = i;
    if(!B->is_pattern)
     values[index] = A->x[j];
    rowCnt[row]++;
   }
  }
  delete[]rowCnt;
  return B;
 }


 CSC *make_full(CSC *A) {
  if(A->stype == 0) {
   std::cerr << "Not symmetric\n";
   return nullptr;
  }

  CSC *Afull = new CSC(A->m, A->n, A->nnz * 2 - A->n);
  auto ind = new int[A->n]();

  for(size_t i = 0; i < A->n; i++) {
   for(size_t p = A->p[i]; p < A->p[i+1]; p++) {
    int row = A->i[p];
    ind[i]++;
    if(row != i)
     ind[row]++;
   }
  }
  Afull->p[0] = 0;
  for(size_t i = 0; i < A->n; i++)
   Afull->p[i+1] = Afull->p[i] + ind[i];

  for(size_t i = 0; i < A->n; i++)
   ind[i] = 0;
  for(size_t i = 0; i < A->n; i++) {
   for(size_t p = A->p[i]; p < A->p[i+1]; p++) {
    int row = A->i[p];
    int index = Afull->p[i] + ind[i];
    Afull->i[index] = row;
    Afull->x[index] = A->x[p];
    ind[i]++;
    if(row != i) {
     index = Afull->p[row] + ind[row];
     Afull->i[index] = i;
     Afull->x[index] = A->x[p];
     ind[row]++;
    }
   }
  }
  delete[]ind;

  return Afull;
 }



}
