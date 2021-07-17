//
// Created by kazem on 7/16/21.
//

#ifndef DDT_SPTRSV_DEMO_UTILS_H
#define DDT_SPTRSV_DEMO_UTILS_H

#include "FusionDemo.h"

#include "DDT.h"
#include "Executor.h"
#include "Input.h"
#include "Inspector.h"
#include "PatternMatching.h"

#include <iostream>
#include <SpTRSVModel.h>
#include <lbc.h>

namespace sparse_avx{

 ///// SPTRSV
 void sptrsv_csc(int n, int *Lp, int *Li, double *Lx, double *x) {
  int i, j;
  for (i = 0; i < n; i++) {
   x[i] /= Lx[Lp[i]];
   for (j = Lp[i] + 1; j < Lp[i + 1]; j++) {
    x[Li[j]] -= Lx[j] * x[i];
   }
  }
 }

 void sptrsv_csr(int n, int *Lp, int *Li, double *Lx, double *x) {
  int i, j;
  for (i = 0; i < n; i++) {
   for (j = Lp[i]; j < Lp[i + 1] - 1; j++) {
    x[i] -= Lx[j] * x[Li[j]];
   }
   x[i] /= Lx[Lp[i + 1] - 1];
  }
 }


 void sptrsv_csr_lbc(int n, int *Lp, int *Li, double *Lx, double *x,
                     int level_no, int *level_ptr,
                     int *par_ptr, int *partition) {
  for (int i1 = 0; i1 < level_no; ++i1) {
#pragma omp  parallel
   {
#pragma omp for schedule(auto)
    for (int j1 = level_ptr[i1]; j1 < level_ptr[i1 + 1]; ++j1) {
     for (int k1 = par_ptr[j1]; k1 < par_ptr[j1 + 1]; ++k1) {
      int i = partition[k1];
      for (int j = Lp[i]; j < Lp[i + 1] - 1; j++) {
       x[i] -= Lx[j] * x[Li[j]];
      }
      x[i] /= Lx[Lp[i + 1] - 1];
     }
    }
   }
  }
 }

 class SpTRSVSerial : public sym_lib::FusionDemo {
 protected:
  void setting_up() override {
   std::fill_n(x_,n_,0.0);
   std::fill_n(x_in_,n_,1.0);
  }

  sym_lib::timing_measurement fused_code() override {
   std::copy(x_in_, x_in_+n_, x_);
   sym_lib::timing_measurement t1;
   t1.start_timer();
   sptrsv_csr(n_, L1_csr_->p, L1_csr_->i, L1_csr_->x, x_);
   t1.measure_elapsed_time();
   //copy_vector(0,n_,x_in_,x_);
   return t1;
  }

 public:
  SpTRSVSerial(sym_lib::CSR *L, sym_lib::CSC *L_csc,
             double *correct_x,
             std::string name) :
    FusionDemo(L_csc->n, name) {
   L1_csr_ = L;
   L1_csc_ = L_csc;
   correct_x_ = correct_x;
  };

  ~SpTRSVSerial() override = default;
 };


 class SpTRSVParallel : public SpTRSVSerial {
 protected:

  int final_level_no, *fina_level_ptr, *final_part_ptr, *final_node_ptr;
  int part_no;
  int lp_, cp_, ic_;
  void build_set() override {
   auto *cost = new double[n_]();
   for (int i = 0; i < n_; ++i) {
    cost[i] = L1_csr_->p[i+1] - L1_csr_->p[i];
   }
   sym_lib::get_coarse_levelSet_DAG_CSC(n_, L1_csc_->p, L1_csc_->i,
                                    final_level_no,
                                    fina_level_ptr,part_no,
                                    final_part_ptr,final_node_ptr,
                                    lp_,cp_, ic_, cost);

   delete []cost;
  }

  sym_lib::timing_measurement fused_code() override {
   sym_lib::timing_measurement t1;
   t1.start_timer();
   sptrsv_csr_lbc(n_, L1_csr_->p, L1_csr_->i, L1_csr_->x, x_in_,
                  final_level_no, fina_level_ptr,
                  final_part_ptr, final_node_ptr);
   t1.measure_elapsed_time();
   sym_lib::copy_vector(0,n_,x_in_,x_);
   return t1;
  }

 public:
  SpTRSVParallel(sym_lib::CSR *L, sym_lib::CSC *L_csc,
               double *correct_x,
               std::string name, int lp, int cp, int ic) :
    SpTRSVSerial(L, L_csc, correct_x, name), lp_(lp), cp_(cp), ic_(ic) {
   L1_csr_ = L;
   L1_csc_ = L_csc;
   correct_x_ = correct_x;
  };

  ~SpTRSVParallel(){
   delete []fina_level_ptr;
   delete []final_node_ptr;
   delete []final_part_ptr;
  }
 };


 void pruneIterations(DDT::MemoryTrace mt, int density) {
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < mt.ips; i++) {
   int oneD = 0;
   while (mt.ip[i+1] - mt.ip[i] == 1) {
    ++oneD;
   }
//    mt.a.nt = i+oneD;
//    mt.a.t  = 0;
  }
  auto t1 = std::chrono::steady_clock::now();
 }

 class SpTRSVDDT : public SpTRSVSerial {
 protected:
  DDT::Config config;
  std::vector<DDT::Codelet*> cl;
  DDT::GlobalObject d;
  sym_lib::timing_measurement analysis_breakdown;
  int lp_, cp_, ic_;

  void build_set() override{
   // Allocate memory and generate global object
   d = DDT::init(config);
   analysis_breakdown.start_timer();
   auto *sm = new SpTRSVModel(L1_csc_->n, L1_csc_->m, L1_csc_->nnz, L1_csc_->p,
                              L1_csc_->i, lp_, cp_, ic_);
   sym_lib::timing_measurement t1;
   t1.start_timer();
   auto trs = sm->generate_3d_trace(lp_);
   auto tgen = t1.measure_elapsed_time();
   auto trace_array = trs[0][0]->MemAddr(); // This is the array that has traces
#ifdef PRINT
   std::cout<<"\n---- Trace list: ---\n";
   for (int i = 0; i < sm->_final_level_no; ++i) {
    for (int j = 0; j < sm->_wp_bounds[i]; ++j) {
     std::cout<<i<<" , "<<j<<" :\n";
     trs[i][j]->print();
     std::cout<<"\n\n";
    }
   }
   std::cout<<": "<<tgen<<"\n";
#endif
   // Prune memory trace
   //pruneIterations(d.mt, 10);

   analysis_breakdown.start_timer();
   // Compute and Mark regions for PSC-3
   //computeParallelizedFOD(d.mt.ip, d.mt.ips, d.d);

   analysis_breakdown.start_timer();
   // Mine trace and profile
   //mineDifferences(d.mt.ip, d.mt.ips, d.c, d.d);
   // Generate Codes
   // DDT::generateSource(d);
   //DDT::inspectCodelets(d, cl);

   // for(auto ii : cl){
   //  ii->print();
   // }

   analysis_breakdown.measure_elapsed_time();
   free_trace_list(trs, sm->_final_level_no, sm->_wp_bounds);
   delete sm;
  }

  sym_lib::timing_measurement fused_code() override {
   sym_lib::timing_measurement t1;
   DDT::Args args; args.x = x_in_; args.y = x_;
   args.r = L1_csr_->m; args.Lp=L1_csr_->p; args.Li=L1_csr_->i;
   args.Lx = L1_csr_->x;
   t1.start_timer();
   // Execute codes
   //DDT::executeCodelets(cl, config, args);
   t1.measure_elapsed_time();
   //copy_vector(0,n_,x_in_,x_);
   return t1;
  }

 public:
  SpTRSVDDT(sym_lib::CSR *L, sym_lib::CSC *L_csc,
          double *correct_x, DDT::Config &conf,
          std::string name, int lp, int cp, int ic) :
    SpTRSVSerial(L, L_csc, correct_x, name), config(conf), lp_(lp), cp_(cp),
    ic_(ic) {
  };

  sym_lib::timing_measurement get_analysis_bw(){return analysis_breakdown;}

  ~SpTRSVDDT(){
   DDT::free(d);
   DDT::free(cl);
  }
 };


}

#endif //DDT_SPTRSV_DEMO_UTILS_H
