//
// Created by kazem on 7/13/21.
//

#include <cassert>
#include <lbc.h>
#include "SpTRSVModel.h"
namespace sparse_avx{

 SpTRSVModel::SpTRSVModel():SectionPolyModel() {}


 SpTRSVModel::~SpTRSVModel() {
  delete []_final_level_ptr;
  delete []_final_part_ptr;
  delete []_final_node_ptr;
 }


 SpTRSVModel::SpTRSVModel(int n, int m, int nnz, int *Ap, int *Ai):_num_rows
                                                                 (n),_num_cols(m),
                                                                 _nnz(nnz),
                                                                 _Ap(Ap), _Ai(Ai) {}


 Trace* SpTRSVModel::generate_trace() {
  auto *trace = new Trace(_nnz+_num_cols);
  int cnt = 0;
  for (int i = 0; i < _num_rows; ++i) {
   auto cur_adr = trace->_mem_addr + 3*cnt;
   cur_adr[0] = i; cur_adr[1] = _Ap[i]; cur_adr[2] = i;
   cnt++;
   for (int j = _Ap[i]; j < _Ap[i+1]; ++j) {
    cur_adr = trace->_mem_addr + 3*cnt;
    cur_adr[0] = i; cur_adr[1] = j; cur_adr[2] = _Ai[j];
    //trace->_tuple[j] = new AddMul(cur_adr);
    cnt++;
   }
  }
  return trace;
 }


 void SpTRSVModel::iteration_space_prunning(int parts){
  int lp = parts, cp = 2, ic= 3;
  auto *cost = new double[_num_cols]();
  for (int i = 0; i < _num_cols; ++i) {
   cost[i] = _Ap[i+1] - _Ap[i];
  }
  sym_lib::get_coarse_levelSet_DAG_CSC(_num_cols, _Ap, _Ai,
                                   _final_level_no,
                                   _final_level_ptr,parts,
                                   _final_part_ptr,_final_node_ptr,
                                   lp, cp, ic, cost);

  delete []cost;
 }


 Trace** SpTRSVModel::generate_trace(int num_threads) { //FIXME
  Trace** trace_list = new Trace*[num_threads];
  auto *tr_list_mm_array = new int[3*_nnz + 3*_num_cols]();
  auto *tr_list_oc_array = new int[_nnz + _num_cols]();
  std::fill_n(tr_list_oc_array, _nnz, TRACE_OP::AddM);
  auto *nnz_bounds = new int[num_threads];
  std::vector<int> bnd_row_array(num_threads+1);
  int nnz_part = _nnz/num_threads;
  int bnd_row = closest_row(nnz_part, _Ap, 0);
  bnd_row_array[0] = 0;
  bnd_row_array[1] = bnd_row;
  nnz_bounds[0] = _Ap[bnd_row];
  trace_list[0] = new Trace(nnz_bounds[0], tr_list_mm_array, tr_list_oc_array,
                            num_threads);
  for (int i = 1; i < num_threads-1; ++i) {
   nnz_bounds[i] = nnz_bounds[i-1] + nnz_part;
   bnd_row = closest_row(nnz_bounds[i], _Ap, bnd_row);
   bnd_row_array[i+1] = bnd_row;
   nnz_bounds[i] = _Ap[bnd_row];
   trace_list[i] = new Trace(nnz_bounds[i], tr_list_mm_array+3*nnz_bounds[i-1],
                             tr_list_oc_array+nnz_bounds[i-1], num_threads);
  }
  nnz_bounds[num_threads-1] = _nnz;
  bnd_row_array[num_threads] = _num_rows;
  trace_list[num_threads-1] = new Trace(_nnz-nnz_bounds[num_threads-2],
                                        tr_list_mm_array+3*nnz_bounds[num_threads-2],
                                        tr_list_oc_array+nnz_bounds[num_threads-2], num_threads);
#pragma omp parallel for //default(none) shared(num_threads, bnd_row_array, \
  trace_list)
  for (int ii = 0; ii < num_threads; ++ii) {
   int cnt = 0;
   for (int i = bnd_row_array[ii]; i < bnd_row_array[ii+1]; ++i) {
    assert(i <_num_rows);
    for (int j = _Ap[i]; j < _Ap[i+1]; ++j) {
     auto cur_adr = trace_list[ii]->_mem_addr + 3*cnt;
     cur_adr[0] = i; cur_adr[1] = j; cur_adr[2] = _Ai[j];
     //trace->_tuple[j] = new AddMul(cur_adr);
     cnt++;
     //std::cout<<ii<<" - "<<cnt<<" : "<<i <<", "<<j<<", "<<_Ai[j]<<"\n";
    }
   }
  }
  delete []nnz_bounds;
  return trace_list;
 }


 Trace*** SpTRSVModel::generate_3d_trace(int num_threads) {
  iteration_space_prunning(num_threads);
  Trace*** trace_list = new Trace**[_final_level_no];
  for (int i = 0; i < _final_level_no; ++i) {
   trace_list[i] = new Trace*[num_threads];
  }
  auto *tr_list_mm_array = new int[3*_nnz + 3*_num_cols]();
  auto *tr_list_oc_array = new int[_nnz + _num_cols]();
  //std::fill_n(tr_list_oc_array, _nnz, TRACE_OP::AddM);
  auto *nnz_bounds = new int[_final_level_no*num_threads+1]();
  int n_part = 1;
  for (int i = 0; i < _final_level_no; ++i) {
   for (int j = _final_level_ptr[i], wp=0; j < _final_level_ptr[i + 1]; ++j,
     ++wp) {
    int cols_wp = _final_part_ptr[j+1] - _final_part_ptr[j];
    int nnz_wp = 0;
    //trace_list[i][wp] = new Trace()
    for (int k = _final_part_ptr[j]; k < _final_part_ptr[j + 1]; ++k) {
     int cn = _final_node_ptr[k];
     nnz_wp += _Ap[cn+1] - _Ap[cn];
    }
    nnz_bounds[n_part] = nnz_wp + cols_wp;
    trace_list[i][wp] = new Trace(nnz_bounds[n_part],
                                  tr_list_mm_array+3*nnz_bounds[n_part-1],
                               tr_list_oc_array+nnz_bounds[n_part-1],
                              num_threads);
   }
  }
//#pragma omp parallel for //default(none) shared(num_threads, bnd_row_array, \
  trace_list)
  for (int ii = 0; ii < _final_level_no; ++ii) {
   for (int l = _final_level_ptr[ii], wp=0; l < _final_level_ptr[ii + 1];
   ++l, ++wp) {
    int cnt = 0;
    for (int r = _final_part_ptr[l]; r < _final_part_ptr[l + 1]; ++r) {
     int i = _final_node_ptr[r];
     assert(i < _num_rows);
     for (int j = _Ap[i]; j < _Ap[i + 1]; ++j) {
      auto cur_adr = trace_list[ii][wp]->_mem_addr + 3 * cnt;
      cur_adr[0] = i;
      cur_adr[1] = j;
      cur_adr[2] = _Ai[j];
      //trace->_tuple[j] = new AddMul(cur_adr);
      cnt++;
      //std::cout<<ii<<" - "<<cnt<<" : "<<i <<", "<<j<<", "<<_Ai[j]<<"\n";
     }
    }
   }
  }
  delete []nnz_bounds;
  return trace_list;
 }

}
