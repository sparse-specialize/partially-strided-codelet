//
// Created by Zachary Cetinic on 2021-08-24.
//

#ifndef DDT_SPTRSV_SUPERNODAL_SYMPILER_H
#define DDT_SPTRSV_SUPERNODAL_SYMPILER_H

#include "SPTRSV_demo_utils.h"
#include "SuperNodalTools.h"
#include "BLAS.h"

//================================ SpTrSv Block Vectorized LBC LL ================================
class SpTrSv_LL_Blocked_LBC : public SpTRSVSerial {
protected:
  int nthreads;
  sym_lib::timing_measurement Scheduling_time1, Scheduling_time2;
  std::vector<int> blocked_level_set, blocked_level_ptr, blocked_node_to_level;
  int blocked_levelNo, num_supernodes, num_dep_edges;
  std::vector<int> supernode_ptr;
  std::vector<int> LBC_level_ptr, LBC_w_ptr, LBC_node_ptr;
  int LBC_level_no;
  int num_w_part;
  int lp_, cp_, ic_;
  std::vector<int> DAG_ptr;
  std::vector<int> DAG_set;
  std::vector<double> cost;
  supernodaltools::SuperNodal* supernode_obj;

  void build_set() override {
    Scheduling_time2.start_timer();
    assert(L1_csc_ != nullptr);
    assert(L1_csr_ != nullptr);
    LBC_level_ptr.resize(n_ + 1);
    LBC_w_ptr.resize(n_ + 1);
    LBC_node_ptr.resize(n_);
    sym_lib::get_coarse_Level_set_DAG_CSC03_parallel_NOLEVELSET_V2(
        num_supernodes, DAG_ptr.data(), DAG_set.data(),
        LBC_level_no, LBC_level_ptr.data(), num_w_part,
        LBC_w_ptr.data(), LBC_node_ptr.data(),
        lp_, cp_, ic_, nthreads,
        cost.data(),
        blocked_levelNo, blocked_level_set.data(), blocked_level_ptr.data(),
        blocked_node_to_level.data(),
        false);
    Scheduling_time2.measure_elapsed_time();
  }

  sym_lib::timing_measurement fused_code() override {
    //sym_lib::rhs_init(L1_csc_->n, L1_csc_->p, L1_csc_->i, L1_csc_->x, x_); // x is b
    sym_lib::timing_measurement t1;
    auto Lp = L1_csr_->p;
    auto Li = L1_csr_->i;
    auto Lx = L1_csr_->x;
    auto x = x_in_;

    t1.start_timer();
            #pragma omp parallel
    {
      std::vector<double> tempvec(n_);
      for (int lvl = 0; lvl < LBC_level_no; ++lvl) {
                    #pragma omp for schedule(static)
        for (int w_ptr = LBC_level_ptr[lvl]; w_ptr < LBC_level_ptr[lvl + 1]; ++w_ptr) {
          for(int super_ptr = LBC_w_ptr[w_ptr]; super_ptr < LBC_w_ptr[w_ptr + 1]; super_ptr++){
            int super = LBC_node_ptr[super_ptr];
            int start_row = supernode_ptr[super];
            int end_row = supernode_ptr[super + 1];
            //ncol is the number of columns in the off-diagonal block
            int num_off_diag_col = Lp[start_row + 1] - Lp[start_row] - 1;
            assert(num_off_diag_col >= 0);
            int nrows = end_row - start_row;
            assert(nrows > 0);
            //Solving the independent part
            //Copy x[Li[col]] into a continues buffer
            for (int col_ptr = Lp[supernode_ptr[super]], k = 0;
            col_ptr < Lp[supernode_ptr[super]] + num_off_diag_col; col_ptr++, k++) {
              tempvec[k] = x[Li[col_ptr]];
            }
            custom_blas::SpTrSv_MatVecCSR_BLAS(nrows, num_off_diag_col, &Lx[Lp[start_row]], tempvec.data(), &x[start_row]);
            custom_blas::SpTrSv_LSolveCSR_BLAS(nrows, num_off_diag_col, &Lx[Lp[start_row] + num_off_diag_col], &x[start_row]);
          }
        }
      }
    }
    t1.measure_elapsed_time();
    sym_lib::copy_vector(0, n_, x_in_, x_);
    return t1;
  }

public:
  /*
   * @brief Class constructor
   * @param L Sparse Matrix with CSR format
   * @param L_csc Sparse Matrix with CSC format
   * @param correct_x The right answer for x
   * @param name The name of the algorithm
   * @param nt number of threads
   */
  SpTrSv_LL_Blocked_LBC(sym_lib::CSR *L, sym_lib::CSC *L_csc,
                        double *correct_x, std::string name,
                        int lp)
                        : SpTRSVSerial(L, L_csc, correct_x, name) {
    L1_csr_ = L;
    L1_csc_ = L_csc;
    correct_x_ = correct_x;
    lp_=lp;
    nthreads = lp;


    //Calculating The Dependency DAG and the levelset
    sym_lib::timing_measurement block_LBC_time;
    Scheduling_time1.start_timer();
    supernodaltools::SuperNodal super_node_obj(supernodaltools::CSR_TYPE,
                                               L1_csc_, L1_csr_, L1_csc_, 0);

    super_node_obj.getDependencyDAGCSCformat(DAG_ptr, DAG_set,num_supernodes, num_dep_edges);

    sym_lib::computingLevelSet_CSC(num_supernodes, DAG_ptr, DAG_set,
                                   blocked_level_ptr, blocked_level_set, blocked_levelNo);

    sym_lib::computingNode2Level(blocked_level_ptr, blocked_level_set,
                                 blocked_levelNo, num_supernodes, blocked_node_to_level);

    cost.resize(num_supernodes, 0);
    super_node_obj.getSuperNodeGroups(supernode_ptr);
    for (int super = 0; super < num_supernodes; ++super) {
      for(int i = supernode_ptr[super]; i < supernode_ptr[super + 1]; i++){
        cost[super] += L1_csr_->p[i + 1] - L1_csr_->p[i];
      }
    }
    Scheduling_time1.start_timer();
  };

  double getSchedulingTime() { return Scheduling_time1.elapsed_time + Scheduling_time2.elapsed_time; }

  void setP2_P3(int p2, int p3) {
    this->cp_ = p2;
    this->ic_ = p3;
  }
  ~SpTrSv_LL_Blocked_LBC() = default;
};

#endif // DDT_SPTRSV_SUPERNODAL_SYMPILER_H
